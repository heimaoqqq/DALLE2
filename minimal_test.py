#!/usr/bin/env python3
"""
极简扩散模型测试 - 用于诊断内存问题
"""

import torch
import torch.nn as nn
from dalle2_pytorch import OpenClipAdapter
from dalle2_pytorch.dalle2_pytorch import Unet, Decoder
import argparse

def create_minimal_unet():
    """创建最小的U-Net配置"""
    return Unet(
        dim=16,  # 极小的基础维度
        image_embed_dim=512,  # CLIP embedding维度
        cond_dim=64,  # 极小的条件维度
        channels=3,
        dim_mults=(1,),  # 只有一层
        cond_on_image_embeds=True,
        cond_on_text_encodings=False,
        # 禁用所有可能消耗内存的功能
        memory_efficient=True,
        init_dim=None,
        init_conv_kernel_size=7,
        resnet_groups=1,  # 最小组数
        attn_dim_head=8,  # 极小的注意力头
        attn_heads=1,  # 单个注意力头
        ff_mult=1,  # 最小的前馈倍数
        layer_attns=False,  # 禁用层注意力
        layer_cross_attns=False,  # 禁用交叉注意力
    )

def test_memory_usage():
    """测试内存使用"""
    print("🔧 Testing minimal configuration...")
    
    # 创建CLIP
    clip = OpenClipAdapter('ViT-B/32')
    print(f"✅ CLIP created successfully")
    
    # 创建最小U-Net
    try:
        unet = create_minimal_unet()
        print(f"✅ Minimal U-Net created successfully")
        print(f"🔧 U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
    except Exception as e:
        print(f"❌ U-Net creation failed: {e}")
        return False
    
    # 创建解码器
    try:
        decoder = Decoder(
            unet=unet,
            clip=clip,
            vae=None,  # 像素空间
            image_sizes=(224,),
            timesteps=100,  # 极少的时间步
            sample_timesteps=10,
            image_cond_drop_prob=0.1,
            text_cond_drop_prob=0.0,
            beta_schedule='cosine',
            predict_x_start=True,
            predict_v=False,
            learned_variance=False
        )
        print(f"✅ Decoder created successfully")
    except Exception as e:
        print(f"❌ Decoder creation failed: {e}")
        return False
    
    # 测试前向传播
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = decoder.to(device)
        
        # 创建测试数据
        batch_size = 1
        test_images = torch.randn(batch_size, 3, 224, 224, device=device)
        
        print(f"🔧 Testing forward pass with batch_size={batch_size}")
        print(f"🔧 GPU memory before: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 前向传播
        with torch.no_grad():
            loss = decoder(test_images)
            print(f"✅ Forward pass successful, loss: {loss.item():.4f}")
            print(f"🔧 GPU memory after: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        print(f"🔧 GPU memory at failure: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_forward', action='store_true', help='Test forward pass')
    args = parser.parse_args()
    
    print("🚀 Starting minimal memory test...")
    
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    else:
        print("❌ No GPU available")
        return
    
    success = test_memory_usage()
    
    if success:
        print("✅ Minimal configuration works!")
        if args.test_forward:
            print("🎯 You can try scaling up the configuration gradually")
    else:
        print("❌ Even minimal configuration failed")
        print("🔧 This suggests a fundamental issue with the environment or library")

if __name__ == "__main__":
    main()
