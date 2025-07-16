"""
Kaggle专用训练脚本 - 微多普勒DALLE2解码器训练
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
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm

from dalle2_pytorch import Unet, Decoder, OpenClipAdapter
from dalle2_pytorch.trainer import DecoderTrainer
from dalle2_pytorch.dataloaders import create_micro_doppler_dataloader
from dalle2_pytorch.vqgan_vae import VQGanVAE, NullVQGanVAE


def parse_args():
    parser = argparse.ArgumentParser(description='Train Micro-Doppler DALLE2 Decoder on Kaggle')
    
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
    parser.add_argument('--dim', type=int, default=128,
                        help='Base dimension for U-Net')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Dimension multipliers for U-Net layers')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of image channels')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--use_vqgan', action='store_true',
                        help='Use VQ-GAN VAE for latent diffusion')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
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
    
    # EMA参数
    parser.add_argument('--ema_beta', type=float, default=0.99,
                        help='EMA decay rate')
    parser.add_argument('--ema_update_after_step', type=int, default=500,
                        help='Start EMA updates after N steps')
    parser.add_argument('--ema_update_every', type=int, default=10,
                        help='Update EMA every N steps')
    
    # CLIP参数
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        help='CLIP model to use')
    
    # 实验参数
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')
    
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
    for i in range(1, 32):  # ID_1 to ID_31
        folder = data_path / f"ID_{i}"
        if folder.exists():
            user_folders.append(folder)
            # 统计图像数量
            image_count = len(list(folder.glob('*.png'))) + len(list(folder.glob('*.jpg')))
            print(f"✅ Found ID_{i} with {image_count} images")
        else:
            print(f"⚠️  Missing ID_{i}")
    
    print(f"📊 Total user folders found: {len(user_folders)}/31")
    return len(user_folders) > 0


def create_model(args):
    """创建解码器模型"""
    print("🏗️  Creating decoder model...")
    
    # 创建CLIP适配器
    clip = OpenClipAdapter(args.clip_model)
    
    # 创建VQ-GAN VAE (如果指定)
    if args.use_vqgan:
        print("🎨 Using VQ-GAN VAE for latent diffusion")
        vae = VQGanVAE(
            dim=32,
            image_size=args.image_size,
            channels=args.channels,
            layers=3,
            vq_codebook_dim=256,
            vq_codebook_size=1024,
            vq_decay=0.8,
            use_vgg_and_gan=True
        )
    else:
        print("🖼️  Using pixel-space diffusion")
        vae = NullVQGanVAE(channels=args.channels)
    
    # 创建U-Net
    unet = Unet(
        dim=args.dim,
        image_embed_dim=512,  # CLIP embedding dimension
        cond_dim=128,
        channels=args.channels,
        dim_mults=tuple(args.dim_mults),
        cond_on_image_embeds=True,
        cond_on_text_encodings=False,  # 不使用文本条件
        self_attn=True,
        attn_heads=8,
        attn_dim_head=64,
        cosine_sim_cross_attn=True,
        cosine_sim_self_attn=True
    )
    
    # 创建解码器
    decoder = Decoder(
        unet=unet,
        clip=clip,
        vae=vae if args.use_vqgan else None,
        image_sizes=(args.image_size,),
        timesteps=args.timesteps,
        image_cond_drop_prob=0.1,
        text_cond_drop_prob=0.5
    )
    
    return decoder


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


def save_samples(decoder_trainer, epoch, output_dir, num_samples=8):
    """生成并保存样本图像"""
    print(f"🎨 Generating samples for epoch {epoch}...")
    
    # 创建随机图像embeddings用于采样
    device = next(decoder_trainer.decoder.parameters()).device
    image_embeds = torch.randn(num_samples, 512, device=device)
    
    # 生成样本
    with torch.no_grad():
        samples = decoder_trainer.sample(image_embed=image_embeds)
    
    # 保存样本
    samples_dir = Path(output_dir) / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    for i, sample in enumerate(samples):
        # 转换从[-1, 1]到[0, 1]
        sample = (sample + 1) / 2
        sample = torch.clamp(sample, 0, 1)
        
        # 保存图像
        from torchvision.utils import save_image
        save_image(sample, samples_dir / f'epoch_{epoch:03d}_sample_{i:02d}.png')
    
    print(f"✅ Saved {len(samples)} samples to {samples_dir}")


def main():
    args = parse_args()
    
    # 检查Kaggle环境
    if not check_kaggle_environment():
        print("❌ Environment check failed!")
        return 1
    
    # 设置输出目录
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("decoder_%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存参数
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"📂 Output directory: {output_dir}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🔧 GPU count: {gpu_count}")
        for i in range(gpu_count):
            print(f"🔧 GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    # Accelerator配置 - 添加DDP参数处理未使用的参数
    accelerator = Accelerator(
        mixed_precision='fp16',
        kwargs_handlers=[
            # 处理分布式训练中未使用的参数
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    )

    # 创建模型和数据加载器
    decoder = create_model(args)
    dataloader = create_dataloader(args)

    # 移动模型到GPU
    decoder = decoder.to(device)

    # 多GPU检测 (让Accelerator处理多GPU)
    if torch.cuda.device_count() > 1:
        print(f"🔥 Multi-GPU detected: {torch.cuda.device_count()} GPUs (will use Accelerator)")
        # 调整批次大小以利用多GPU
        if args.batch_size < torch.cuda.device_count() * 4:
            original_batch_size = args.batch_size
            args.batch_size = torch.cuda.device_count() * 8
            print(f"🔧 Adjusted batch size from {original_batch_size} to {args.batch_size} for multi-GPU")
            # 重新创建数据加载器
            dataloader = create_dataloader(args)
    else:
        print("⚠️  Using single GPU")
    
    print(f"📊 Dataset size: {len(dataloader.dataset)} images")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📈 Total batches per epoch: {len(dataloader)}")
    
    # 创建训练器
    decoder_trainer = DecoderTrainer(
        decoder=decoder,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=args.ema_beta,
        ema_update_after_step=args.ema_update_after_step,
        ema_update_every=args.ema_update_every,
        accelerator=accelerator
    )
    
    # 准备分布式训练
    decoder_trainer, dataloader = accelerator.prepare(decoder_trainer, dataloader)

    # 显示训练配置
    print(f"🚀 Starting training for {args.epochs} epochs")
    print(f"🔧 Accelerator device: {accelerator.device}")
    print(f"🔧 Number of processes: {accelerator.num_processes}")
    print(f"🔧 Distributed type: {accelerator.distributed_type}")
    if accelerator.num_processes > 1:
        print(f"🔥 Multi-GPU training enabled with {accelerator.num_processes} GPUs!")
    else:
        print(f"⚠️  Single GPU training (check if multi-GPU is available)")
    
    # 训练循环
    for epoch in range(args.epochs):
        decoder_trainer.train()
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch in progress_bar:
            images = batch['image']
            
            # 训练步骤
            loss = decoder_trainer(images, unet_number=1)

            # 处理分布式训练中的update调用
            if hasattr(decoder_trainer, 'module'):
                # 分布式训练：访问原始trainer
                decoder_trainer.module.update(unet_number=1)
            else:
                # 单GPU训练：直接调用
                decoder_trainer.update(unet_number=1)

            # loss已经是float，不需要.item()
            epoch_loss += loss
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        print(f'📊 Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}')
        
        # 生成样本
        if (epoch + 1) % args.sample_every == 0:
            if accelerator.is_main_process:
                save_samples(decoder_trainer, epoch + 1, output_dir)
        
        # 保存模型
        if (epoch + 1) % args.save_every == 0:
            if accelerator.is_main_process:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': decoder_trainer.state_dict(),
                    'args': vars(args)
                }
                torch.save(checkpoint, output_dir / f'decoder_epoch_{epoch+1:03d}.pt')
                print(f"💾 Saved checkpoint: epoch_{epoch+1:03d}.pt")
    
    # 保存最终模型
    if accelerator.is_main_process:
        final_checkpoint = {
            'epoch': args.epochs,
            'model_state_dict': decoder_trainer.state_dict(),
            'args': vars(args)
        }
        torch.save(final_checkpoint, output_dir / 'decoder_final.pt')
        print(f"🎉 Training completed! Final model saved to {output_dir / 'decoder_final.pt'}")
    
    return 0


if __name__ == '__main__':
    exit(main())
