#!/usr/bin/env python3
"""
超轻量级DALLE2测试 - 极简配置避免内存问题
"""

import torch
import torch.nn as nn
from dalle2_pytorch import OpenAIClipAdapter, Unet, Decoder
import argparse

def create_ultra_lightweight_decoder():
    """创建超轻量级解码器"""
    print("🔧 Creating ultra-lightweight decoder...")
    
    # 使用预训练CLIP
    clip = OpenAIClipAdapter('ViT-B/32')
    print("✅ CLIP loaded")
    
    # 创建极简U-Net
    unet = Unet(
        dim=16,                    # 极小维度
        image_embed_dim=512,       # CLIP embedding维度
        cond_dim=32,               # 极小条件维度
        channels=3,
        dim_mults=(1,),            # 只有一层，不进行下采样
        cond_on_image_embeds=True,
        cond_on_text_encodings=False,
        # 禁用所有可能的内存消耗功能
        memory_efficient=True,
        init_dim=None,
        init_conv_kernel_size=3,   # 更小的卷积核
        resnet_groups=1,           # 最小组数
        attn_dim_head=8,           # 极小注意力头
        attn_heads=1,              # 单个注意力头
        ff_mult=1,                 # 最小前馈倍数
        layer_attns=False,         # 禁用层注意力
        layer_cross_attns=False,   # 禁用交叉注意力
        use_sparse_linear_attn=False,  # 禁用稀疏注意力
        block_kv_size=None,        # 禁用块注意力
        max_mem_len=0,             # 禁用记忆
    )
    
    print(f"🔧 U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # 创建解码器
    decoder = Decoder(
        unet=unet,
        clip=clip,
        vae=None,                  # 像素空间
        image_sizes=(224,),        # 单一尺寸
        timesteps=50,              # 极少时间步
        sample_timesteps=5,        # 极少采样步数
        image_cond_drop_prob=0.1,
        text_cond_drop_prob=0.0,   # 不使用文本条件
        beta_schedule='linear',    # 简单调度
        predict_x_start=True,
        predict_v=False,
        learned_variance=False
    )
    
    print("✅ Ultra-lightweight decoder created")
    return decoder

def test_step_by_step():
    """逐步测试，找出内存瓶颈"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if not torch.cuda.is_available():
        print("❌ No CUDA available")
        return False
    
    print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    torch.cuda.empty_cache()
    
    try:
        # 步骤1: 创建模型
        print("\n=== Step 1: Creating model ===")
        decoder = create_ultra_lightweight_decoder()
        print(f"🔧 Memory after model creation: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 步骤2: 移动到GPU
        print("\n=== Step 2: Moving to GPU ===")
        decoder = decoder.to(device)
        print(f"🔧 Memory after GPU transfer: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 步骤3: 创建小批量数据
        print("\n=== Step 3: Creating test data ===")
        batch_size = 1
        test_images = torch.randn(batch_size, 3, 224, 224, device=device)
        print(f"🔧 Memory after data creation: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 步骤4: 前向传播
        print("\n=== Step 4: Forward pass ===")
        with torch.no_grad():
            loss = decoder(test_images)
            print(f"✅ Forward pass successful, loss: {loss.item():.4f}")
            print(f"🔧 Memory after forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 步骤5: 采样测试
        print("\n=== Step 5: Sampling test ===")
        with torch.no_grad():
            # 获取图像embedding
            image_embed, _ = decoder.clip.embed_image(test_images)
            print(f"🔧 Image embed shape: {image_embed.shape}")
            print(f"🔧 Memory after embedding: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
            # 尝试采样
            samples = decoder.sample(image_embed=image_embed)
            print(f"✅ Sampling successful, shape: {samples.shape}")
            print(f"🔧 Memory after sampling: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        print(f"🔧 Memory at error: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        return False

def test_even_smaller():
    """测试更小的配置"""
    print("\n=== Testing even smaller configuration ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    try:
        # 只测试CLIP
        print("Testing CLIP only...")
        clip = OpenAIClipAdapter('ViT-B/32')
        clip = clip.to(device)
        
        test_images = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            image_embed, _ = clip.embed_image(test_images)
            print(f"✅ CLIP works, embed shape: {image_embed.shape}")
            print(f"🔧 Memory with CLIP only: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 测试最小U-Net
        print("Testing minimal U-Net...")
        unet = Unet(
            dim=8,                     # 更小
            image_embed_dim=512,
            cond_dim=16,               # 更小
            channels=3,
            dim_mults=(1,),
            cond_on_image_embeds=True,
            cond_on_text_encodings=False,
            memory_efficient=True,
            attn_dim_head=4,           # 更小
            attn_heads=1,
            ff_mult=1,
            layer_attns=False,
            layer_cross_attns=False,
        )
        
        print(f"🔧 Minimal U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Even smaller config failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_by_step', action='store_true', help='Test step by step')
    parser.add_argument('--even_smaller', action='store_true', help='Test even smaller config')
    args = parser.parse_args()
    
    print("🚀 Ultra-lightweight DALLE2 test...")
    
    if args.step_by_step or not any([args.step_by_step, args.even_smaller]):
        success = test_step_by_step()
        if not success and not args.even_smaller:
            print("\n🔧 Trying even smaller configuration...")
            test_even_smaller()
    
    if args.even_smaller:
        test_even_smaller()

if __name__ == "__main__":
    main()
