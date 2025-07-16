# Micro-Doppler DALLE2 Implementation

This is a modified version of DALLE2-pytorch specifically designed for generating micro-Doppler time-frequency images conditioned on user IDs. The implementation replaces text conditioning with user ID conditioning, making it suitable for gait recognition and user-specific micro-Doppler pattern generation.

## Features

- **User-Conditioned Generation**: Generate micro-Doppler images for specific users (31 users supported)
- **Two-Stage Training**: VQ-VAE decoder + diffusion prior architecture
- **Flexible Data Loading**: Support for both hierarchical and flat directory structures
- **Classifier-Free Guidance**: Enhanced generation quality through conditioning
- **Interpolation**: Generate smooth transitions between different users' patterns

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd DALLE2-pytorch

# Install dependencies
pip install -e .

# Additional dependencies for micro-Doppler
pip install accelerate torchmetrics
```

## Dataset Structure

### Option 1: Hierarchical Structure
```
data_root/
├── user_0/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── user_1/
│   ├── image_001.png
│   └── ...
└── user_30/
    └── ...
```

### Option 2: Flat Structure with Metadata
```
data_root/
├── images/
│   ├── user0_001.png
│   ├── user0_002.png
│   ├── user1_001.png
│   └── ...
└── metadata.json
```

Metadata format:
```json
[
  {"filename": "user0_001.png", "user_id": 0},
  {"filename": "user0_002.png", "user_id": 0},
  {"filename": "user1_001.png", "user_id": 1},
  ...
]
```

## Quick Start

### 1. Test the Implementation

```bash
# Test with dummy data
python test_micro_doppler.py --create_dummy --num_users 5

# Test with your data
python test_micro_doppler.py --data_root /path/to/your/data --num_users 31
```

### 2. Train the Decoder (Stage 1)

```bash
python train_micro_doppler_decoder.py \
    --data_root /path/to/your/data \
    --num_users 31 \
    --batch_size 16 \
    --epochs 100 \
    --output_dir ./outputs/decoder \
    --experiment_name decoder_v1
```

### 3. Train the Prior (Stage 2)

```bash
python train_micro_doppler_prior.py \
    --data_root /path/to/your/data \
    --num_users 31 \
    --batch_size 32 \
    --epochs 100 \
    --output_dir ./outputs/prior \
    --experiment_name prior_v1
```

### 4. Generate New Images

```bash
python generate_micro_doppler.py \
    --decoder_path ./outputs/decoder/decoder_v1/decoder_final.pt \
    --prior_path ./outputs/prior/prior_v1/prior_final.pt \
    --user_ids 0 1 2 3 4 \
    --num_samples_per_user 4 \
    --output_dir ./generated_images
```

## Training Parameters

### Decoder Training
- `--dim`: Base U-Net dimension (default: 128)
- `--dim_mults`: Dimension multipliers (default: [1, 2, 4, 8])
- `--timesteps`: Diffusion timesteps (default: 1000)
- `--use_vqgan`: Enable VQ-GAN VAE for latent diffusion
- `--lr`: Learning rate (default: 3e-4)
- `--batch_size`: Batch size (default: 16)

### Prior Training
- `--dim`: Prior network dimension (default: 512)
- `--depth`: Transformer depth (default: 6)
- `--heads`: Attention heads (default: 8)
- `--cond_drop_prob`: Conditioning dropout for classifier-free guidance (default: 0.2)
- `--lr`: Learning rate (default: 3e-4)
- `--batch_size`: Batch size (default: 32)

## Generation Options

### Basic Generation
```bash
python generate_micro_doppler.py \
    --decoder_path path/to/decoder.pt \
    --prior_path path/to/prior.pt \
    --user_ids 0 1 2 \
    --num_samples_per_user 4
```

### User Interpolation
```bash
python generate_micro_doppler.py \
    --decoder_path path/to/decoder.pt \
    --prior_path path/to/prior.pt \
    --interpolate \
    --interp_user_1 0 \
    --interp_user_2 5 \
    --interp_steps 8
```

## Model Architecture

### UserConditionedPriorNetwork
- Replaces text embeddings with user ID embeddings
- Uses causal transformer for sequence modeling
- Supports classifier-free guidance through conditioning dropout

### UserConditionedDiffusionPrior
- Diffusion model in CLIP embedding space
- Conditioned on user IDs instead of text
- Generates image embeddings for the decoder

### MicroDopplerDALLE2
- Complete end-to-end model
- Combines prior and decoder for user-conditioned generation
- Supports both training and inference

## Key Modifications from Original DALLE2

1. **Text → User ID Conditioning**: Replaced text embeddings with user ID embeddings
2. **Custom Data Loader**: Added `MicroDopplerDataset` for time-frequency images
3. **User-Specific Training**: Modified training loops for user-conditioned generation
4. **Simplified Architecture**: Removed text encoder dependencies

## Performance Tips

1. **GPU Memory**: Use smaller batch sizes if you encounter OOM errors
2. **Training Time**: Decoder training is typically faster than prior training
3. **Quality**: Higher `cond_scale` values improve conditioning but may reduce diversity
4. **Convergence**: Monitor loss curves and sample quality during training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Increase `num_workers` for data loading
3. **Poor Quality**: Increase training epochs or adjust conditioning scales
4. **User Imbalance**: Ensure balanced dataset across all users

### Debug Mode
```bash
# Test with small models
python test_micro_doppler.py --create_dummy --num_users 3

# Monitor training
tensorboard --logdir ./outputs
```

## Citation

If you use this implementation, please cite the original DALLE2 paper and this modification:

```bibtex
@misc{ramesh2022hierarchical,
    title={Hierarchical Text-Conditional Image Generation with CLIP Latents},
    author={Aditya Ramesh and Prafulla Dhariwal and Alex Nichol and Casey Chu and Mark Chen},
    year={2022},
    eprint={2204.06125},
    archivePrefix={arXiv}
}
```

## License

This project maintains the same MIT license as the original DALLE2-pytorch implementation.
