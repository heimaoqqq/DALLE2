#!/usr/bin/env python3
"""
Kaggle多GPU训练启动脚本
自动检测环境并使用正确的方式启动训练
"""

import os
import sys
import subprocess
import torch

def check_gpu_setup():
    """检查GPU设置"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"🔧 Detected {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        print(f"🔧 GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
    
    return gpu_count

def run_with_accelerate_launch(script_name, args):
    """使用accelerate launch启动训练"""
    print(f"🚀 Starting {script_name} with accelerate launch")
    
    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes=2",
        "--mixed_precision=fp16",
        script_name
    ] + args
    
    print(f"📝 Command: {' '.join(cmd)}")
    return subprocess.run(cmd)

def run_direct(script_name, args):
    """直接运行脚本"""
    print(f"🚀 Starting {script_name} directly")
    
    cmd = [sys.executable, script_name] + args
    
    print(f"📝 Command: {' '.join(cmd)}")
    return subprocess.run(cmd)

def main():
    """主函数"""
    print("🔥 Kaggle Multi-GPU Training Launcher")
    print("=" * 50)
    
    # 检查GPU
    gpu_count = check_gpu_setup()
    if gpu_count == 0:
        print("❌ No GPU available!")
        return 1
    
    # 解析参数
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python start_training.py decoder [args...]  # 训练解码器")
        print("  python start_training.py prior [args...]    # 训练先验")
        return 1
    
    mode = sys.argv[1].lower()
    remaining_args = sys.argv[2:]
    
    # 设置默认参数
    if mode == "decoder":
        script_name = "train_kaggle_decoder.py"
        default_args = [
            "--use_vqgan",
            "--batch_size", "8",
            "--epochs", "50",
            "--experiment_name", "decoder_vqgan_v1"
        ]
    elif mode == "prior":
        script_name = "train_kaggle_prior.py"
        default_args = [
            "--batch_size", "16",
            "--epochs", "50", 
            "--experiment_name", "prior_v1"
        ]
    else:
        print(f"❌ Unknown mode: {mode}")
        return 1
    
    # 合并参数
    final_args = default_args + remaining_args
    
    # 尝试多GPU启动
    if gpu_count > 1:
        print(f"🔥 Attempting multi-GPU training with {gpu_count} GPUs")
        try:
            result = run_with_accelerate_launch(script_name, final_args)
            if result.returncode == 0:
                print("✅ Multi-GPU training completed successfully!")
                return 0
            else:
                print("⚠️  Multi-GPU training failed, falling back to single GPU")
        except Exception as e:
            print(f"⚠️  accelerate launch failed: {e}")
            print("⚠️  Falling back to single GPU")
    
    # 回退到单GPU
    print("🔧 Running with single GPU")
    result = run_direct(script_name, final_args)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
    
    return result.returncode

if __name__ == "__main__":
    exit(main())
