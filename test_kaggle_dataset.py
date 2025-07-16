"""
Kaggle数据集测试脚本
验证数据加载和模型创建是否正常工作
适配数据集结构: /kaggle/input/dataset/ID_1 到 ID_31
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
import json
import argparse
from collections import defaultdict

from dalle2_pytorch import Unet, Decoder, OpenClipAdapter
from dalle2_pytorch.micro_doppler_dalle2 import (
    UserConditionedPriorNetwork, 
    UserConditionedDiffusionPrior, 
    MicroDopplerDALLE2
)
from dalle2_pytorch.dataloaders import create_micro_doppler_dataloader, MicroDopplerDataset


def analyze_kaggle_dataset(data_root='/kaggle/input/dataset'):
    """分析Kaggle数据集结构和统计信息"""
    
    print("🔍 Analyzing Kaggle dataset structure...")
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"❌ Dataset path {data_path} does not exist!")
        return False
    
    # 统计信息
    user_stats = {}
    total_images = 0
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # 检查每个用户文件夹
    for i in range(1, 32):  # ID_1 to ID_31
        folder_name = f"ID_{i}"
        folder_path = data_path / folder_name
        
        if folder_path.exists() and folder_path.is_dir():
            # 统计图像文件
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(folder_path.glob(f'*{ext}')))
                image_files.extend(list(folder_path.glob(f'*{ext.upper()}')))
            
            user_stats[i] = {
                'folder': folder_name,
                'path': str(folder_path),
                'image_count': len(image_files),
                'image_files': [f.name for f in image_files[:5]]  # 只显示前5个文件名
            }
            total_images += len(image_files)
            
            print(f"✅ {folder_name}: {len(image_files)} images")
            if len(image_files) > 0:
                print(f"   📁 Sample files: {', '.join(user_stats[i]['image_files'])}")
        else:
            print(f"❌ Missing: {folder_name}")
            user_stats[i] = {'folder': folder_name, 'image_count': 0}
    
    # 打印统计摘要
    print(f"\n📊 Dataset Summary:")
    print(f"   Total users: {len([u for u in user_stats.values() if u['image_count'] > 0])}/31")
    print(f"   Total images: {total_images}")
    print(f"   Average images per user: {total_images/31:.1f}")
    
    # 检查图像尺寸 (采样几张图像)
    print(f"\n🖼️  Checking image properties...")
    sample_images = []
    for i in range(1, min(6, 32)):  # 检查前5个用户
        folder_path = data_path / f"ID_{i}"
        if folder_path.exists():
            for ext in image_extensions:
                files = list(folder_path.glob(f'*{ext}'))
                if files:
                    sample_images.append(files[0])
                    break
    
    if sample_images:
        sizes = []
        for img_path in sample_images[:3]:  # 只检查前3张
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)
                    print(f"   📏 {img_path.parent.name}/{img_path.name}: {img.size}, mode: {img.mode}")
            except Exception as e:
                print(f"   ❌ Error reading {img_path}: {e}")
        
        if sizes:
            unique_sizes = list(set(sizes))
            print(f"   📐 Unique image sizes: {unique_sizes}")
    
    return user_stats


def test_dataloader(data_root='/kaggle/input/dataset', num_users=31):
    """测试微多普勒数据加载器"""
    
    print("\n🧪 Testing MicroDopplerDataset...")
    
    try:
        # 测试数据集创建
        dataset = MicroDopplerDataset(
            data_root=data_root,
            num_users=num_users,
            image_size=256
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("❌ No samples found in dataset!")
            return None
        
        # 测试数据加载
        sample = dataset[0]
        print(f"✅ Sample keys: {sample.keys()}")
        print(f"   📏 Image shape: {sample['image'].shape}")
        print(f"   👤 User ID: {sample['user_id']}")
        print(f"   📁 Original folder: {sample.get('original_folder', 'N/A')}")
        
        # 测试数据加载器
        dataloader = create_micro_doppler_dataloader(
            data_root=data_root,
            batch_size=4,
            num_workers=0,  # 在Kaggle中使用0避免多进程问题
            num_users=num_users,
            shuffle=True
        )
        
        batch = next(iter(dataloader))
        print(f"✅ Batch image shape: {batch['image'].shape}")
        print(f"   👥 Batch user_ids: {batch['user_id']}")
        
        # 测试用户分布
        user_dist = dataset.get_user_distribution()
        print(f"✅ User distribution: {dict(sorted(user_dist.items()))}")
        
        return dataloader
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_models(num_users=31):
    """测试模型创建和前向传播"""
    
    print("\n🧪 Testing model creation...")
    
    try:
        # 测试UserConditionedPriorNetwork
        print("🏗️  Creating UserConditionedPriorNetwork...")
        prior_network = UserConditionedPriorNetwork(
            dim=512,
            num_users=num_users,
            user_embed_dim=64,
            depth=2,  # 减小深度用于测试
            dim_head=64,
            heads=4,
            num_timesteps=1000,  # 标准扩散步数 (训练用1000步)
            rotary_emb=True
        )
        
        # 测试前向传播
        batch_size = 4
        image_embed = torch.randn(batch_size, 512)
        user_ids = torch.randint(0, num_users, (batch_size,))
        timesteps = torch.randint(0, 1000, (batch_size,)).long()  # 匹配模型的timesteps范围
        
        pred = prior_network(image_embed, timesteps, user_ids=user_ids)
        print(f"✅ Prior network output shape: {pred.shape}")
        
        # 测试UserConditionedDiffusionPrior
        print("🏗️  Creating UserConditionedDiffusionPrior...")
        clip = OpenClipAdapter('ViT-B/32')
        
        diffusion_prior = UserConditionedDiffusionPrior(
            net=prior_network,
            clip=clip,
            timesteps=1000,        # 训练时使用1000步
            sample_timesteps=64,   # 推理时使用64步 (DDIM加速)
            num_users=num_users
        )
        
        # 测试训练前向传播
        images = torch.randn(batch_size, 3, 256, 256)
        with torch.no_grad():
            image_embeds = clip.embed_image(images).image_embed
        
        loss = diffusion_prior(image_embeds, user_ids=user_ids)
        print(f"✅ Prior training loss: {loss.item():.4f}")
        
        # 测试采样
        with torch.no_grad():
            sampled_embeds = diffusion_prior.sample(
                user_ids=user_ids[:2],
                num_samples_per_batch=1,
                cond_scale=1.0
            )
        print(f"✅ Sampled embeddings shape: {sampled_embeds.shape}")
        
        # 测试Decoder
        print("🏗️  Creating Decoder...")
        unet = Unet(
            dim=64,  # 减小维度用于测试
            image_embed_dim=512,
            cond_dim=128,
            channels=3,
            dim_mults=(1, 2, 4),
            cond_on_image_embeds=True,
            cond_on_text_encodings=False
        )
        
        decoder = Decoder(
            unet=unet,
            clip=clip,
            image_sizes=(256,),
            timesteps=1000  # 标准扩散步数
        )
        
        # 测试解码器训练
        decoder_loss = decoder(images)
        print(f"✅ Decoder training loss: {decoder_loss.item():.4f}")
        
        # 测试解码器采样
        with torch.no_grad():
            generated_images = decoder.sample(image_embed=sampled_embeds)
        print(f"✅ Generated images shape: {generated_images.shape}")
        
        print("🎉 All model tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_compatibility(dataloader):
    """测试训练兼容性"""
    
    print("\n🧪 Testing training compatibility...")
    
    try:
        # 创建简化模型用于测试
        clip = OpenClipAdapter('ViT-B/32')
        
        prior_network = UserConditionedPriorNetwork(
            dim=256,  # 减小维度
            num_users=31,
            user_embed_dim=32,
            depth=2,
            dim_head=32,
            heads=4,
            num_timesteps=1000,  # 标准扩散步数
            rotary_emb=True
        )
        
        diffusion_prior = UserConditionedDiffusionPrior(
            net=prior_network,
            clip=clip,
            timesteps=1000,      # 训练时使用1000步
            sample_timesteps=64, # 推理时使用64步
            num_users=31
        )
        
        # 测试一个批次
        batch = next(iter(dataloader))
        images = batch['image']
        user_ids = batch['user_id']
        
        print(f"📊 Batch info:")
        print(f"   Images shape: {images.shape}")
        print(f"   User IDs: {user_ids}")
        print(f"   Unique users in batch: {torch.unique(user_ids).tolist()}")
        
        # 测试先验训练步骤
        with torch.no_grad():
            image_embeds = diffusion_prior.clip.embed_image(images).image_embed
        
        prior_loss = diffusion_prior(image_embeds, user_ids=user_ids)
        print(f"✅ Prior loss with real data: {prior_loss.item():.4f}")
        
        print("🎉 Training compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Kaggle Micro-Doppler Dataset')
    parser.add_argument('--data_root', type=str, default='/kaggle/input/dataset',
                        help='Path to Kaggle dataset')
    parser.add_argument('--num_users', type=int, default=31,
                        help='Number of users (ID_1 to ID_31)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Kaggle dataset tests...")
    print(f"📂 Data root: {args.data_root}")
    
    try:
        # 1. 分析数据集
        user_stats = analyze_kaggle_dataset(args.data_root)
        if not user_stats:
            return 1
        
        # 2. 测试数据加载器
        dataloader = test_dataloader(args.data_root, args.num_users)
        if dataloader is None:
            return 1
        
        # 3. 测试模型
        if not test_models(args.num_users):
            return 1
        
        # 4. 测试训练兼容性
        if not test_training_compatibility(dataloader):
            return 1
        
        print("\n🎉 All tests passed! Ready for training on Kaggle.")
        print("\n📋 Next steps:")
        print("1. Run decoder training: python train_kaggle_decoder.py --use_vqgan")
        print("2. Run prior training: python train_kaggle_prior.py")
        print("3. Generate images with trained models")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
