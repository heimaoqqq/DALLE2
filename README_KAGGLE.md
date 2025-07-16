# Kaggle微多普勒DALLE2训练指南

本指南专门针对您的Kaggle数据集结构进行了优化，数据集路径为 `/kaggle/input/dataset`，包含31个用户文件夹 `ID_1` 到 `ID_31`。

## 🗂️ 数据集结构

```
/kaggle/input/dataset/
├── ID1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── ID2/
│   ├── image1.png
│   └── ...
└── ID31/
    └── ...
```

## 🚀 快速开始

### 1. 测试环境和数据集

```bash
# 测试数据集和模型
python test_kaggle_dataset.py
```

这将验证：
- ✅ 数据集结构正确性
- ✅ 图像加载正常
- ✅ 用户ID映射正确 (ID1→user_id=0, ID2→user_id=1, ...)
- ✅ 模型创建和前向传播
- ✅ 训练兼容性

### 2. 训练解码器 (第一阶段)

```bash
# 使用VQ-GAN进行潜在空间扩散 (推荐)
python train_kaggle_decoder.py \
    --use_vqgan \
    --batch_size 8 \
    --epochs 50 \
    --experiment_name "decoder_vqgan_v1"

# 或者使用像素空间扩散
python train_kaggle_decoder.py \
    --batch_size 4 \
    --epochs 50 \
    --experiment_name "decoder_pixel_v1"
```

### 3. 训练先验网络 (第二阶段)

```bash
python train_kaggle_prior.py \
    --batch_size 16 \
    --epochs 50 \
    --experiment_name "prior_v1"
```

### 4. 生成新图像

```bash
python generate_micro_doppler.py \
    --decoder_path /kaggle/working/outputs/decoder_vqgan_v1/decoder_final.pt \
    --prior_path /kaggle/working/outputs/prior_v1/prior_final.pt \
    --user_ids 0 1 2 3 4 \
    --num_samples_per_user 4
```

## ⚙️ Kaggle优化配置

### 内存优化
- **批次大小**: 解码器8，先验16 (适配Kaggle GPU内存)
- **混合精度**: 自动启用FP16节省内存
- **工作进程**: 设为2避免内存溢出

### 训练优化
- **训练轮数**: 减少到50轮 (Kaggle时间限制)
- **保存频率**: 每10轮保存一次
- **采样频率**: 每5轮生成样本

### 存储优化
- **输出路径**: `/kaggle/working/outputs/`
- **自动清理**: 只保留最新检查点

## 📊 预期性能

### 训练时间 (Kaggle GPU)
- **解码器**: ~6-8小时 (50轮，VQ-GAN模式)
- **先验网络**: ~3-4小时 (50轮)

### 内存使用
- **VQ-GAN模式**: ~12GB GPU内存
- **像素模式**: ~15GB GPU内存

### 生成质量
- **分辨率**: 256×256像素
- **用户特异性**: 高 (能生成特定用户的步态模式)
- **多样性**: 中等 (通过调整cond_scale控制)

## 🔧 参数调优

### 提高生成质量
```bash
# 增加模型容量
python train_kaggle_decoder.py \
    --dim 256 \
    --dim_mults 1 2 4 8 16 \
    --use_vqgan

# 调整条件引导强度
python generate_micro_doppler.py \
    --cond_scale 3.0 \
    --prior_cond_scale 2.5
```

### 加速训练
```bash
# 减少模型大小
python train_kaggle_prior.py \
    --dim 256 \
    --depth 4 \
    --heads 6
```

## 📈 监控训练

### 检查训练进度
```bash
# 查看损失曲线
ls /kaggle/working/outputs/*/samples/

# 检查生成样本
ls /kaggle/working/outputs/*/prior_samples/
```

### 常见问题排查

**内存不足 (OOM)**:
```bash
# 减少批次大小
python train_kaggle_decoder.py --batch_size 4
python train_kaggle_prior.py --batch_size 8
```

**训练太慢**:
```bash
# 减少模型大小
python train_kaggle_decoder.py --dim 64 --dim_mults 1 2 4
```

**生成质量差**:
```bash
# 增加训练轮数
python train_kaggle_decoder.py --epochs 100
```

## 🎯 用户ID映射

数据集中的文件夹名称会自动映射到模型的user_id：

| 文件夹名 | 模型user_id | 生成时使用 |
|----------|-------------|------------|
| ID1      | 0           | --user_ids 0 |
| ID2      | 1           | --user_ids 1 |
| ...      | ...         | ... |
| ID31     | 30          | --user_ids 30 |

## 📁 输出文件结构

```
/kaggle/working/outputs/
├── decoder_vqgan_v1/
│   ├── args.json                 # 训练参数
│   ├── decoder_epoch_010.pt      # 检查点
│   ├── decoder_final.pt          # 最终模型
│   └── samples/                  # 生成样本
│       ├── epoch_005_sample_00.png
│       └── ...
└── prior_v1/
    ├── args.json
    ├── prior_final.pt
    └── prior_samples/
        └── prior_samples_epoch_005.pt
```

## 🔄 完整训练流程

```bash
# 1. 环境测试
python test_kaggle_dataset.py

# 2. 解码器训练 (6-8小时)
python train_kaggle_decoder.py --use_vqgan --batch_size 8 --epochs 50

# 3. 先验训练 (3-4小时)
python train_kaggle_prior.py --batch_size 16 --epochs 50

# 4. 生成测试
python generate_micro_doppler.py \
    --decoder_path /kaggle/working/outputs/decoder_*/decoder_final.pt \
    --prior_path /kaggle/working/outputs/prior_*/prior_final.pt \
    --user_ids 0 5 10 15 20 25 30 \
    --num_samples_per_user 2
```

## 💡 高级技巧

### 数据增强
训练脚本已内置适合时频图的数据增强：
- 水平翻转 (50%概率)
- 小幅旋转 (±5度)
- 轻微颜色抖动

### 用户间插值
```bash
python generate_micro_doppler.py \
    --interpolate \
    --interp_user_1 0 \
    --interp_user_2 10 \
    --interp_steps 8
```

### 批量生成
```bash
# 为所有用户生成样本
python generate_micro_doppler.py \
    --user_ids $(seq 0 30) \
    --num_samples_per_user 3
```

## 🎉 预期结果

训练完成后，您将获得：
1. **解码器模型**: 能从CLIP embedding生成256×256微多普勒图像
2. **先验模型**: 能从用户ID生成对应的CLIP embedding
3. **完整系统**: 输入用户ID，输出该用户的步态微多普勒图像
4. **数据增广**: 为每个用户生成任意数量的新样本

这个系统将有效解决您的微多普勒时频图数据量不足问题！
