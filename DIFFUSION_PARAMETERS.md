# 扩散模型参数详解

## 🕐 **Timesteps参数说明**

### 基本概念
- **`timesteps`**: 扩散过程的总时间步数
- **前向过程**: 从原图像逐步加噪声，t=0→t=T
- **逆向过程**: 从纯噪声逐步去噪生成图像，t=T→t=0

### 数学表示
```
前向过程: x₀ → x₁ → x₂ → ... → x_T (纯噪声)
逆向过程: x_T → x_{T-1} → ... → x₁ → x₀ (生成图像)
```

## 📊 **常见Timesteps设置**

### 训练阶段
| 模型类型 | Timesteps | 说明 |
|----------|-----------|------|
| **DDPM (标准)** | 1000 | 原始论文设置，质量最好 |
| **快速训练** | 500 | 训练速度2倍，质量略降 |
| **实验测试** | 100-250 | 快速验证代码 |

### 推理阶段 (可以减少)
| 采样方法 | Timesteps | 加速比 | 质量 |
|----------|-----------|--------|------|
| **DDPM** | 1000 | 1x | 最高 |
| **DDIM** | 250 | 4x | 高 |
| **DDIM** | 100 | 10x | 中等 |
| **DDIM** | 50 | 20x | 可接受 |

## ⚙️ **项目中的Timesteps配置**

### 训练配置
```python
# 解码器 (U-Net)
decoder = Decoder(
    unet=unet,
    timesteps=1000,  # 标准设置，最佳质量
    # ...
)

# 先验网络
diffusion_prior = UserConditionedDiffusionPrior(
    net=prior_network,
    timesteps=1000,        # 训练时使用1000步
    sample_timesteps=64,   # 推理时使用64步 (DDIM加速)
    # ...
)
```

### 生成配置
```python
# 快速生成 (推荐)
sample_timesteps=64    # 16倍加速，质量良好

# 高质量生成
sample_timesteps=250   # 4倍加速，质量很好

# 最高质量生成
sample_timesteps=1000  # 无加速，最佳质量
```

## 🎯 **为什么选择1000步？**

### 理论依据
1. **DDPM论文标准**: 原始论文使用1000步
2. **噪声调度**: 1000步提供足够细粒度的噪声调度
3. **训练稳定性**: 更多步数使训练更稳定
4. **生成质量**: 更多步数通常产生更好的图像质量

### 实际考虑
- **训练时间**: 1000步 vs 100步，训练时间差异不大
- **内存占用**: 主要影响推理时间，不影响训练内存
- **质量提升**: 从100步到1000步，质量提升显著

## 🚀 **推理加速策略**

### DDIM采样
```python
# 训练时
timesteps=1000

# 推理时 (自动使用DDIM)
sample_timesteps=64  # 仅用64步生成，速度快16倍
```

### 质量 vs 速度权衡
```python
# 最快 (2-3秒/图)
sample_timesteps=25

# 平衡 (5-8秒/图) - 推荐
sample_timesteps=64

# 高质量 (15-20秒/图)
sample_timesteps=250

# 最高质量 (60秒/图)
sample_timesteps=1000
```

## 📈 **训练建议**

### 阶段性训练
```python
# 第一阶段: 快速验证 (可选)
timesteps=250
epochs=10

# 第二阶段: 完整训练
timesteps=1000
epochs=100
```

### 渐进式训练 (高级)
```python
# 前期: 较少步数，快速收敛
timesteps=500
epochs=30

# 后期: 完整步数，精细调优
timesteps=1000
epochs=70
```

## 🔧 **实际配置示例**

### Kaggle训练配置
```python
# 解码器训练
python train_kaggle_decoder.py \
    --timesteps 1000 \
    --epochs 50

# 先验训练  
python train_kaggle_prior.py \
    --timesteps 1000 \
    --sample_timesteps 64 \
    --epochs 50

# 生成图像
python generate_micro_doppler.py \
    --sample_timesteps 64  # 快速生成
```

### 高质量生成配置
```python
# 最高质量生成
python generate_micro_doppler.py \
    --sample_timesteps 250 \
    --cond_scale 2.0 \
    --prior_cond_scale 2.0
```

## 💡 **优化建议**

### 训练阶段
- ✅ **使用1000步**: 保证训练质量
- ✅ **标准噪声调度**: 使用cosine或linear调度
- ✅ **充足训练**: 至少50-100个epoch

### 推理阶段
- ✅ **使用DDIM**: 大幅加速推理
- ✅ **64步采样**: 质量和速度的最佳平衡
- ✅ **条件引导**: 使用2.0-3.0的引导强度

### 内存优化
- ✅ **梯度检查点**: 减少训练内存
- ✅ **混合精度**: 使用FP16训练
- ✅ **批次大小**: 根据GPU内存调整

## 🎨 **微多普勒特定建议**

### 时频图特性
- **时间维度**: 对应扩散的时间步
- **频率维度**: 需要保持频率结构
- **用户特征**: 通过条件引导保持用户特异性

### 推荐配置
```python
# 训练配置
timesteps=1000           # 充分的扩散步数
batch_size=8            # 适合Kaggle GPU
epochs=50               # 充足的训练轮数

# 生成配置  
sample_timesteps=64     # 平衡质量和速度
cond_scale=2.0          # 适中的条件引导
num_samples_per_user=4  # 每用户生成4张图
```

这样配置可以在保证生成质量的同时，获得合理的训练和推理速度！
