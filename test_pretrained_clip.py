#!/usr/bin/env python3
"""
测试使用预训练CLIP的简单DALLE2配置
"""

import torch
import torch.nn as nn
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter
import argparse
from pathlib import Path

def create_simple_dalle2():
    """创建使用预训练CLIP的简单DALLE2模型"""
    print("🔧 Creating DALLE2 with pretrained OpenAI CLIP...")
    
    # 使用预训练的OpenAI CLIP
    clip = OpenAIClipAdapter('ViT-B/32')
    print("✅ Pretrained CLIP loaded")
    
    # 创建简单的Prior网络
    prior_network = DiffusionPriorNetwork(
        dim=512,
        depth=2,  # 减少深度
        dim_head=64,
        heads=4   # 减少头数
    )
    
    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=clip,
        timesteps=100,  # 减少时间步
        cond_drop_prob=0.2
    )
    print("✅ Prior network created")
    
    # 创建简单的U-Net
    unet = Unet(
        dim=64,  # 减少维度
        image_embed_dim=512,
        cond_dim=128,
        channels=3,
        dim_mults=(1, 2),  # 只用两层
        cond_on_image_embeds=True,
        cond_on_text_encodings=False
    )
    
    decoder = Decoder(
        unet=unet,
        clip=clip,
        image_sizes=(224,),  # 单一尺寸
        timesteps=100,       # 减少时间步
        sample_timesteps=20, # 快速采样
        image_cond_drop_prob=0.1,
        text_cond_drop_prob=0.5
    )
    print("✅ Decoder created")
    
    # 组合成完整的DALLE2
    dalle2 = DALLE2(
        prior=diffusion_prior,
        decoder=decoder
    )
    print("✅ DALLE2 model created successfully")
    
    return dalle2

def test_memory_and_forward():
    """测试内存使用和前向传播"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
        torch.cuda.empty_cache()
        print(f"🔧 Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    try:
        # 创建模型
        dalle2 = create_simple_dalle2()
        dalle2 = dalle2.to(device)
        
        if torch.cuda.is_available():
            print(f"🔧 GPU memory after model: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 测试文本到图像生成
        print("🎨 Testing text-to-image generation...")
        
        with torch.no_grad():
            # 简单的文本输入
            text = ["a red car", "a blue house"]
            
            # 生成图像
            images = dalle2(
                text,
                cond_scale=2.0,
                return_pil_images=False
            )
            
            print(f"✅ Generated images shape: {images.shape}")
            print(f"✅ Images range: [{images.min().item():.3f}, {images.max().item():.3f}]")
            
            if torch.cuda.is_available():
                print(f"🔧 GPU memory after generation: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if torch.cuda.is_available():
            print(f"🔧 GPU memory at error: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        return False

def test_with_your_data():
    """测试使用您的微多普勒数据"""
    print("🔬 Testing with micro-Doppler-like data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        dalle2 = create_simple_dalle2()
        dalle2 = dalle2.to(device)
        
        # 模拟您的数据：用户ID到图像的映射
        # 这里我们用简单的文本描述代替用户ID
        user_descriptions = [
            "user walking pattern",
            "user running pattern", 
            "user standing pattern"
        ]
        
        print("🎯 Generating micro-Doppler patterns...")
        
        with torch.no_grad():
            images = dalle2(
                user_descriptions,
                cond_scale=1.5,
                return_pil_images=False
            )
            
            print(f"✅ Generated {len(user_descriptions)} patterns")
            print(f"✅ Pattern shape: {images.shape}")
            
            # 检查生成的图像是否有变化
            for i, desc in enumerate(user_descriptions):
                img = images[i]
                print(f"🔧 {desc}: range [{img.min().item():.3f}, {img.max().item():.3f}], std {img.std().item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in micro-Doppler test: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_basic', action='store_true', help='Test basic functionality')
    parser.add_argument('--test_micro_doppler', action='store_true', help='Test micro-Doppler generation')
    args = parser.parse_args()
    
    print("🚀 Testing DALLE2 with pretrained CLIP...")
    
    if args.test_basic or not any([args.test_basic, args.test_micro_doppler]):
        print("\n=== Basic Functionality Test ===")
        success = test_memory_and_forward()
        if not success:
            print("❌ Basic test failed")
            return
    
    if args.test_micro_doppler:
        print("\n=== Micro-Doppler Test ===")
        success = test_with_your_data()
        if not success:
            print("❌ Micro-Doppler test failed")
            return
    
    print("\n✅ All tests passed! You can now adapt this for your specific use case.")
    print("💡 Next steps:")
    print("   1. Replace text descriptions with your user ID embeddings")
    print("   2. Fine-tune on your micro-Doppler dataset")
    print("   3. Adjust model size based on your memory constraints")

if __name__ == "__main__":
    main()
