"""
Kaggle专用训练脚本 - 微多普勒DALLE2先验网络训练
适配Kaggle环境和数据集结构: /kaggle/input/dataset/ID_1 到 ID_31
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm

from dalle2_pytorch import OpenClipAdapter
from dalle2_pytorch.micro_doppler_dalle2 import (
    UserConditionedPriorNetwork, 
    UserConditionedDiffusionPrior
)
from dalle2_pytorch.dataloaders import create_micro_doppler_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train Micro-Doppler DALLE2 Prior on Kaggle')
    
    # Kaggle数据路径 (固定)
    parser.add_argument('--data_root', type=str, default='/kaggle/input/dataset',
                        help='Kaggle dataset root directory')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/outputs',
                        help='Output directory for models and logs')
    
    # 数据参数
    parser.add_argument('--num_users', type=int, default=31,
                        help='Number of users (ID_1 to ID_31)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (assumes square images)')
    
    # 模型参数
    parser.add_argument('--dim', type=int, default=512,
                        help='Dimension for prior network')
    parser.add_argument('--user_embed_dim', type=int, default=64,
                        help='User embedding dimension')
    parser.add_argument('--depth', type=int, default=6,
                        help='Depth of causal transformer')
    parser.add_argument('--dim_head', type=int, default=64,
                        help='Dimension per attention head')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--sample_timesteps', type=int, default=64,
                        help='Number of timesteps for sampling')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (Kaggle GPU memory limited)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (reduced for Kaggle)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--sample_every', type=int, default=5,
                        help='Generate samples every N epochs')
    
    # 条件参数
    parser.add_argument('--cond_drop_prob', type=float, default=0.2,
                        help='Conditioning dropout probability for classifier-free guidance')
    
    # CLIP参数
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        help='CLIP model to use')
    
    # 实验参数
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    
    return parser.parse_args()


def check_kaggle_environment():
    """检查Kaggle环境和数据集"""
    print("🔍 Checking Kaggle environment...")
    
    # 检查是否在Kaggle环境
    if not os.path.exists('/kaggle'):
        print("⚠️  Warning: Not running in Kaggle environment")
        return False
    
    # 检查数据集路径
    data_path = Path('/kaggle/input/dataset')
    if not data_path.exists():
        print(f"❌ Dataset not found at {data_path}")
        return False
    
    # 检查用户文件夹
    user_folders = []
    total_images = 0
    for i in range(1, 32):  # ID_1 to ID_31
        folder = data_path / f"ID_{i}"
        if folder.exists():
            user_folders.append(folder)
            # 统计图像数量
            image_count = len(list(folder.glob('*.png'))) + len(list(folder.glob('*.jpg')))
            total_images += image_count
            print(f"✅ Found ID_{i} with {image_count} images")
        else:
            print(f"⚠️  Missing ID_{i}")
    
    print(f"📊 Total user folders: {len(user_folders)}/31")
    print(f"📊 Total images: {total_images}")
    return len(user_folders) > 0


def create_model(args):
    """创建用户条件扩散先验模型"""
    print("🏗️  Creating user-conditioned diffusion prior...")
    
    # 创建CLIP适配器
    clip = OpenClipAdapter(args.clip_model)
    
    # 创建先验网络
    prior_network = UserConditionedPriorNetwork(
        dim=args.dim,
        num_users=args.num_users,
        user_embed_dim=args.user_embed_dim,
        depth=args.depth,
        dim_head=args.dim_head,
        heads=args.heads,
        timesteps=args.timesteps,
        rotary_emb=True,
        cosine_sim=True
    )
    
    # 创建扩散先验
    diffusion_prior = UserConditionedDiffusionPrior(
        net=prior_network,
        clip=clip,
        timesteps=args.timesteps,
        sample_timesteps=args.sample_timesteps,
        cond_drop_prob=args.cond_drop_prob,
        num_users=args.num_users
    )
    
    return diffusion_prior


def create_dataloader(args):
    """创建数据加载器"""
    print("📁 Creating data loader...")
    
    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 创建数据加载器
    dataloader = create_micro_doppler_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        num_users=args.num_users,
        shuffle=True,
        transform=transform,
        flat_structure=False  # 使用层次结构
    )
    
    return dataloader


def save_samples(diffusion_prior, epoch, output_dir, num_users=5, samples_per_user=2):
    """生成并保存样本图像embeddings"""
    print(f"🎨 Generating prior samples for epoch {epoch}...")
    
    device = next(diffusion_prior.parameters()).device
    
    # 从不同用户采样
    user_ids = torch.arange(min(num_users, 31), device=device)
    user_ids = user_ids.repeat_interleave(samples_per_user)
    
    # 生成图像embeddings
    with torch.no_grad():
        image_embeds = diffusion_prior.sample(
            user_ids=user_ids,
            num_samples_per_batch=1,
            cond_scale=2.0
        )
    
    # 保存embeddings
    samples_dir = Path(output_dir) / 'prior_samples'
    samples_dir.mkdir(exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'user_ids': user_ids.cpu(),
        'image_embeds': image_embeds.cpu()
    }, samples_dir / f'prior_samples_epoch_{epoch:03d}.pt')
    
    print(f"✅ Saved prior samples to {samples_dir}")


class PriorTrainer:
    """简单的先验网络训练器"""
    
    def __init__(self, model, lr=3e-4, weight_decay=1e-2, accelerator=None):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.accelerator = accelerator
        
    def train_step(self, batch):
        """单个训练步骤"""
        images = batch['image']
        user_ids = batch['user_id']
        
        # 从CLIP获取图像embeddings
        with torch.no_grad():
            image_embeds = self.model.clip.embed_image(images).image_embed
        
        # 前向传播
        loss = self.model(image_embeds, user_ids=user_ids)
        
        # 反向传播
        if self.accelerator:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])


def main():
    args = parse_args()
    
    # 检查Kaggle环境
    if not check_kaggle_environment():
        print("❌ Environment check failed!")
        return 1
    
    # 设置输出目录
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("prior_%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存参数
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"📂 Output directory: {output_dir}")
    
    # 初始化accelerator
    accelerator = Accelerator(mixed_precision='fp16')  # 使用混合精度节省内存
    
    # 创建模型和数据加载器
    diffusion_prior = create_model(args)
    dataloader = create_dataloader(args)
    
    print(f"📊 Dataset size: {len(dataloader.dataset)} images")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📈 Total batches per epoch: {len(dataloader)}")
    
    # 创建训练器
    trainer = PriorTrainer(
        model=diffusion_prior,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accelerator=accelerator
    )
    
    # 准备分布式训练
    trainer.model, trainer.optimizer, dataloader = accelerator.prepare(
        trainer.model, trainer.optimizer, dataloader
    )
    
    print(f"🚀 Starting prior training for {args.epochs} epochs")
    
    # 训练循环
    for epoch in range(args.epochs):
        trainer.model.train()
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch in progress_bar:
            # 训练步骤
            loss = trainer.train_step(batch)
            
            epoch_loss += loss
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        print(f'📊 Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}')
        
        # 生成样本
        if (epoch + 1) % args.sample_every == 0:
            if accelerator.is_main_process:
                save_samples(trainer.model, epoch + 1, output_dir)
        
        # 保存模型
        if (epoch + 1) % args.save_every == 0:
            if accelerator.is_main_process:
                checkpoint = {
                    'epoch': epoch + 1,
                    'trainer_state_dict': trainer.state_dict(),
                    'args': vars(args)
                }
                torch.save(checkpoint, output_dir / f'prior_epoch_{epoch+1:03d}.pt')
                print(f"💾 Saved checkpoint: prior_epoch_{epoch+1:03d}.pt")
    
    # 保存最终模型
    if accelerator.is_main_process:
        final_checkpoint = {
            'epoch': args.epochs,
            'trainer_state_dict': trainer.state_dict(),
            'args': vars(args)
        }
        torch.save(final_checkpoint, output_dir / 'prior_final.pt')
        print(f"🎉 Prior training completed! Final model saved to {output_dir / 'prior_final.pt'}")
    
    return 0


if __name__ == '__main__':
    exit(main())
