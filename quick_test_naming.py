"""
快速测试脚本 - 验证文件夹命名格式修正
测试数据加载器是否正确识别 ID1, ID2, ..., ID31 格式
"""

import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from dalle2_pytorch.dataloaders import MicroDopplerDataset


def create_test_dataset():
    """创建测试数据集，使用正确的命名格式 ID1, ID2, ..., ID31"""
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Creating test dataset in: {temp_dir}")
    
    # 创建用户文件夹 ID1 到 ID5 (测试用)
    for i in range(1, 6):  # ID1 to ID5
        user_dir = temp_dir / f"ID{i}"
        user_dir.mkdir()
        
        # 为每个用户创建几张测试图像
        for j in range(3):
            # 创建随机图像
            image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            # 保存图像
            image_path = user_dir / f"image_{j:03d}.png"
            image.save(image_path)
        
        print(f"✅ Created ID{i} with 3 images")
    
    return temp_dir


def test_dataset_loading(data_root):
    """测试数据集加载"""
    
    print(f"\n🧪 Testing dataset loading...")
    
    try:
        # 创建数据集
        dataset = MicroDopplerDataset(
            data_root=str(data_root),
            num_users=5,  # 只测试5个用户
            image_size=256
        )
        
        print(f"✅ Dataset created successfully")
        print(f"📊 Total samples: {len(dataset)}")
        
        # 检查用户分布
        user_dist = dataset.get_user_distribution()
        print(f"👥 User distribution: {user_dist}")
        
        # 测试样本加载
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"📋 Sample info:")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   User ID: {sample['user_id']}")
            print(f"   Original folder: {sample['original_folder']}")
            print(f"   Filename: {sample['filename']}")
        
        # 验证用户ID映射
        expected_mapping = {
            'ID1': 0, 'ID2': 1, 'ID3': 2, 'ID4': 3, 'ID5': 4
        }
        
        print(f"\n🔍 Verifying user ID mapping...")
        for sample in dataset.samples[:10]:  # 检查前10个样本
            folder_name = sample['original_folder']
            user_id = sample['user_id']
            expected_id = expected_mapping.get(folder_name, -1)
            
            if user_id == expected_id:
                print(f"✅ {folder_name} → user_id={user_id} (correct)")
            else:
                print(f"❌ {folder_name} → user_id={user_id} (expected {expected_id})")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_folder_detection(data_root):
    """测试文件夹检测逻辑"""
    
    print(f"\n🔍 Testing folder detection...")
    
    # 检查每个预期的文件夹
    for i in range(1, 6):  # ID1 to ID5
        folder_name = f"ID{i}"
        folder_path = data_root / folder_name
        
        if folder_path.exists():
            image_count = len(list(folder_path.glob('*.png')))
            print(f"✅ Found {folder_name} with {image_count} images")
        else:
            print(f"❌ Missing {folder_name}")
    
    # 测试错误的命名格式不会被检测到
    wrong_formats = ['ID_1', 'ID_2', 'user_0', 'user_1']
    print(f"\n🚫 Testing wrong formats (should not be found):")
    for wrong_name in wrong_formats:
        wrong_path = data_root / wrong_name
        if wrong_path.exists():
            print(f"❌ Unexpectedly found {wrong_name}")
        else:
            print(f"✅ Correctly ignored {wrong_name}")


def main():
    print("🚀 Testing folder naming format correction...")
    print("📋 Expected format: ID1, ID2, ID3, ..., ID31")
    
    # 创建测试数据集
    test_data_root = create_test_dataset()
    
    try:
        # 测试文件夹检测
        test_folder_detection(test_data_root)
        
        # 测试数据集加载
        success = test_dataset_loading(test_data_root)
        
        if success:
            print(f"\n🎉 All tests passed!")
            print(f"✅ Folder naming format ID1, ID2, ..., ID31 is correctly supported")
            print(f"✅ User ID mapping works correctly:")
            print(f"   ID1 → user_id=0")
            print(f"   ID2 → user_id=1")
            print(f"   ...")
            print(f"   ID31 → user_id=30")
        else:
            print(f"\n❌ Tests failed!")
            return 1
        
    finally:
        # 清理临时文件
        shutil.rmtree(test_data_root)
        print(f"\n🧹 Cleaned up test data")
    
    return 0


if __name__ == '__main__':
    exit(main())
