#!/usr/bin/env python3
"""
检查diffusers库兼容性和Stable Diffusion可用性
"""

import sys
import torch
import subprocess
import importlib.util

def check_package_version(package_name):
    """检查包版本"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return None, "Not installed"
        
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown')
        return version, "OK"
    except Exception as e:
        return None, str(e)

def check_environment():
    """检查环境兼容性"""
    print("🔧 Environment Compatibility Check")
    print("=" * 50)
    
    # Python版本
    python_version = sys.version.split()[0]
    print(f"Python: {python_version}")
    
    # PyTorch
    torch_version, torch_status = check_package_version('torch')
    print(f"PyTorch: {torch_version} ({torch_status})")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    else:
        print("CUDA: Not available")
    
    # 关键包检查
    packages = [
        'diffusers',
        'transformers', 
        'accelerate',
        'safetensors',
        'PIL',
        'numpy'
    ]
    
    print("\n📦 Package Versions:")
    for pkg in packages:
        version, status = check_package_version(pkg)
        print(f"  {pkg}: {version} ({status})")
    
    return True

def test_stable_diffusion_basic():
    """测试Stable Diffusion基本功能"""
    print("\n🎨 Testing Stable Diffusion...")
    
    try:
        from diffusers import StableDiffusionPipeline
        print("✅ diffusers import successful")
        
        # 检查可用模型
        model_id = "runwayml/stable-diffusion-v1-5"
        print(f"🔧 Testing model: {model_id}")
        
        # 创建pipeline (不加载权重，只测试兼容性)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            safety_checker=None,  # 禁用安全检查节省内存
            requires_safety_checker=False
        )
        print("✅ Pipeline creation successful")
        
        if torch.cuda.is_available():
            print("🔧 Moving to GPU...")
            pipe = pipe.to("cuda")
            print(f"🔧 GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 测试生成
        print("🎨 Testing generation...")
        with torch.no_grad():
            prompt = "a simple test image"
            image = pipe(
                prompt, 
                num_inference_steps=10,  # 快速测试
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            
            print(f"✅ Generation successful: {image.size}")
            if torch.cuda.is_available():
                print(f"🔧 GPU memory after generation: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Stable Diffusion test failed: {e}")
        return False

def test_lightweight_sd():
    """测试轻量级Stable Diffusion配置"""
    print("\n🪶 Testing lightweight Stable Diffusion...")
    
    try:
        from diffusers import StableDiffusionPipeline
        
        # 使用更小的配置
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            
            # 启用内存优化
            pipe.enable_attention_slicing()
            pipe.enable_model_cpu_offload()
            print("✅ Memory optimizations enabled")
        
        # 测试小尺寸生成
        with torch.no_grad():
            image = pipe(
                "test pattern",
                num_inference_steps=5,
                guidance_scale=5.0,
                height=256,  # 更小尺寸
                width=256
            ).images[0]
            
            print(f"✅ Lightweight generation successful: {image.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lightweight test failed: {e}")
        return False

def recommend_setup():
    """推荐设置"""
    print("\n💡 Recommendations:")
    print("=" * 50)
    
    # 检查diffusers版本
    diffusers_version, _ = check_package_version('diffusers')
    
    if diffusers_version:
        major, minor = diffusers_version.split('.')[:2]
        if int(major) == 0 and int(minor) >= 30:
            print("✅ diffusers version is recent and compatible")
        else:
            print("⚠️  Consider upgrading diffusers:")
            print("   pip install -U diffusers")
    
    print("\n🎯 For your micro-Doppler project:")
    print("1. Use Stable Diffusion v1.5 (most stable)")
    print("2. Enable memory optimizations")
    print("3. Start with 256x256 images")
    print("4. Use float16 precision")
    print("5. Consider fine-tuning on your dataset")

def main():
    print("🚀 Diffusers Compatibility Check")
    print("=" * 50)
    
    # 基本环境检查
    check_environment()
    
    # 测试Stable Diffusion
    if torch.cuda.is_available():
        sd_basic = test_stable_diffusion_basic()
        if not sd_basic:
            print("\n🔧 Trying lightweight configuration...")
            test_lightweight_sd()
    else:
        print("\n⚠️  No CUDA available, skipping GPU tests")
    
    # 推荐设置
    recommend_setup()

if __name__ == "__main__":
    main()
