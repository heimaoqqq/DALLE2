#!/usr/bin/env python3
"""
多GPU训练启动脚本
确保正确使用所有可用的GPU
"""

import os
import sys
import subprocess
import torch

def setup_multi_gpu():
    """设置多GPU环境"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"🔧 Detected {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        print(f"🔧 GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
    
    if gpu_count > 1:
        # 设置环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))
        os.environ['NCCL_DEBUG'] = 'INFO'  # 调试NCCL
        print(f"🔥 Multi-GPU setup complete: {gpu_count} GPUs")
        return True
    else:
        print("⚠️  Only 1 GPU available")
        return False

def run_decoder_training():
    """运行解码器训练"""
    print("🚀 Starting Decoder Training with Multi-GPU")
    
    cmd = [
        sys.executable, "train_kaggle_decoder.py",
        "--use_vqgan",
        "--batch_size", "16",  # 增加批次大小利用多GPU
        "--epochs", "50",
        "--experiment_name", "decoder_vqgan_multi_gpu",
        "--lr", "3e-4",
        "--image_size", "256"
    ]
    
    subprocess.run(cmd)

def run_prior_training():
    """运行先验训练"""
    print("🚀 Starting Prior Training with Multi-GPU")
    
    cmd = [
        sys.executable, "train_kaggle_prior.py",
        "--batch_size", "32",  # 增加批次大小利用多GPU
        "--epochs", "50",
        "--experiment_name", "prior_multi_gpu",
        "--lr", "3e-4"
    ]
    
    subprocess.run(cmd)

def main():
    """主函数"""
    print("🔥 Multi-GPU Training Setup")
    print("=" * 50)
    
    # 设置多GPU
    multi_gpu_available = setup_multi_gpu()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python train_multi_gpu.py decoder  # 训练解码器")
        print("  python train_multi_gpu.py prior    # 训练先验")
        print("  python train_multi_gpu.py both     # 训练两者")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "decoder":
        run_decoder_training()
    elif mode == "prior":
        run_prior_training()
    elif mode == "both":
        print("🔄 Training both models sequentially...")
        run_decoder_training()
        print("\n" + "="*50)
        run_prior_training()
    else:
        print(f"❌ Unknown mode: {mode}")
        print("Available modes: decoder, prior, both")

if __name__ == "__main__":
    main()
