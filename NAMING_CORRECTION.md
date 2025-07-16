# 文件夹命名格式修正说明

## 🔧 修正内容

根据您的反馈，数据集文件夹命名格式应该是：
- ✅ **正确格式**: `ID1`, `ID2`, `ID3`, ..., `ID31`
- ❌ **错误格式**: `ID_1`, `ID_2`, `ID_3`, ..., `ID_31`

## 📁 数据集结构

```
/kaggle/input/dataset/
├── ID1/           # user_id = 0
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── ID2/           # user_id = 1
│   ├── image1.png
│   └── ...
├── ID3/           # user_id = 2
│   └── ...
...
└── ID31/          # user_id = 30
    └── ...
```

## 🔄 用户ID映射

| 文件夹名 | 模型user_id | 说明 |
|----------|-------------|------|
| ID1      | 0           | 第1个用户 |
| ID2      | 1           | 第2个用户 |
| ID3      | 2           | 第3个用户 |
| ...      | ...         | ... |
| ID31     | 30          | 第31个用户 |

## 📝 修改的文件

### 1. 数据加载器 (`micro_doppler_loader.py`)
```python
# 修改前
self.data_root / f"ID_{user_id + 1}"  # ID_1, ID_2, ...

# 修改后  
self.data_root / f"ID{user_id + 1}"   # ID1, ID2, ...
```

### 2. Kaggle训练脚本
- `train_kaggle_decoder.py`
- `train_kaggle_prior.py`
- `test_kaggle_dataset.py`

```python
# 修改前
for i in range(1, 32):  # ID_1 to ID_31
    folder = data_path / f"ID_{i}"

# 修改后
for i in range(1, 32):  # ID1 to ID31
    folder = data_path / f"ID{i}"
```

### 3. 文档文件
- `README_KAGGLE.md`

## 🧪 验证修正

运行测试脚本验证修正是否正确：

```bash
# 快速测试命名格式
python quick_test_naming.py

# 完整测试 (需要真实数据集)
python test_kaggle_dataset.py
```

## 🔍 兼容性支持

数据加载器现在支持多种命名格式，按优先级顺序：

1. **主要格式**: `ID1`, `ID2`, ..., `ID31` ✅
2. **备用格式**: `ID_1`, `ID_2`, ..., `ID_31` 
3. **通用格式**: `user_0`, `user_1`, ..., `user_30`

这确保了向后兼容性，同时优先使用您的正确格式。

## 🚀 使用方法

修正后的使用方法完全相同：

```bash
# 1. 测试数据集
python test_kaggle_dataset.py

# 2. 训练解码器
python train_kaggle_decoder.py --use_vqgan --batch_size 8 --epochs 50

# 3. 训练先验
python train_kaggle_prior.py --batch_size 16 --epochs 50

# 4. 生成图像
python generate_micro_doppler.py \
    --decoder_path /kaggle/working/outputs/decoder_*/decoder_final.pt \
    --prior_path /kaggle/working/outputs/prior_*/prior_final.pt \
    --user_ids 0 1 2 3 4 \
    --num_samples_per_user 4
```

## ✅ 验证清单

- [x] 数据加载器支持 `ID1` 到 `ID31` 格式
- [x] 用户ID正确映射 (ID1→0, ID2→1, ..., ID31→30)
- [x] 训练脚本更新文件夹检测逻辑
- [x] 测试脚本验证命名格式
- [x] 文档更新正确的命名示例
- [x] 保持向后兼容性

## 🎯 预期结果

修正后，系统将：
1. 正确识别您的 `ID1` 到 `ID31` 文件夹
2. 将它们映射到 `user_id` 0到30
3. 在生成时使用 `--user_ids 0 1 2 ...` 来指定用户
4. 生成对应用户的微多普勒时频图像

所有功能保持不变，只是文件夹命名格式得到了正确支持！
