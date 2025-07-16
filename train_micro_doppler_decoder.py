"""
Training script for Micro-Doppler DALLE2 Decoder
Trains the decoder (VQ-VAE stage) for micro-Doppler time-frequency images
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm

from dalle2_pytorch import Unet, Decoder, OpenClipAdapter
from dalle2_pytorch.trainer import DecoderTrainer
from dalle2_pytorch.dataloaders import create_micro_doppler_dataloader
from dalle2_pytorch.vqgan_vae import VQGanVAE, NullVQGanVAE


def parse_args():
    parser = argparse.ArgumentParser(description='Train Micro-Doppler DALLE2 Decoder')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing micro-Doppler dataset')
    parser.add_argument('--num_users', type=int, default=31,
                        help='Number of users in the dataset')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (assumes square images)')
    parser.add_argument('--flat_structure', action='store_true',
                        help='Use flat directory structure with metadata file')
    parser.add_argument('--metadata_file', type=str, default='metadata.json',
                        help='Metadata file for flat structure')
    
    # Model arguments
    parser.add_argument('--dim', type=int, default=128,
                        help='Base dimension for U-Net')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Dimension multipliers for U-Net layers')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of image channels')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--use_vqgan', action='store_true',
                        help='Use VQ-GAN VAE for latent diffusion')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--sample_every', type=int, default=5,
                        help='Generate samples every N epochs')
    
    # EMA arguments
    parser.add_argument('--ema_beta', type=float, default=0.99,
                        help='EMA decay rate')
    parser.add_argument('--ema_update_after_step', type=int, default=1000,
                        help='Start EMA updates after N steps')
    parser.add_argument('--ema_update_every', type=int, default=10,
                        help='Update EMA every N steps')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for models and samples')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    
    # CLIP arguments
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        help='CLIP model to use')
    
    return parser.parse_args()


def create_model(args):
    """Create the decoder model"""
    
    # Create CLIP adapter
    clip = OpenClipAdapter(args.clip_model)
    
    # Create VQ-GAN VAE if specified
    if args.use_vqgan:
        vae = VQGanVAE(
            dim=32,                    # 基础维度
            image_size=args.image_size, # 256
            channels=args.channels,     # 3
            layers=3,                  # 下采样层数: 256->128->64->32
            vq_codebook_dim=256,       # VQ codebook维度
            vq_codebook_size=1024,     # codebook大小
            vq_decay=0.8,             # EMA衰减
            use_vgg_and_gan=True       # 使用VGG感知损失和GAN损失
        )
    else:
        vae = NullVQGanVAE(channels=args.channels)
    
    # Create U-Net
    unet = Unet(
        dim=args.dim,
        image_embed_dim=512,  # CLIP embedding dimension
        cond_dim=128,
        channels=args.channels,
        dim_mults=tuple(args.dim_mults),
        cond_on_image_embeds=True,
        cond_on_text_encodings=False  # We don't use text for micro-Doppler
    )
    
    # Create decoder
    decoder = Decoder(
        unet=unet,
        clip=clip,
        vae=vae if args.use_vqgan else None,
        image_sizes=(args.image_size,),
        timesteps=args.timesteps,
        image_cond_drop_prob=0.1,
        text_cond_drop_prob=0.5
    )
    
    return decoder


def create_dataloader(args):
    """Create data loader for micro-Doppler dataset"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataloader
    dataloader = create_micro_doppler_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        num_users=args.num_users,
        shuffle=True,
        transform=transform,
        metadata_file=args.metadata_file if args.flat_structure else None,
        flat_structure=args.flat_structure
    )
    
    return dataloader


def save_samples(decoder_trainer, epoch, output_dir, num_samples=8):
    """Generate and save sample images"""
    
    # Create random image embeddings for sampling
    device = next(decoder_trainer.decoder.parameters()).device
    image_embeds = torch.randn(num_samples, 512, device=device)
    
    # Generate samples
    with torch.no_grad():
        samples = decoder_trainer.sample(image_embed=image_embeds)
    
    # Save samples
    samples_dir = Path(output_dir) / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    for i, sample in enumerate(samples):
        # Convert from [-1, 1] to [0, 1]
        sample = (sample + 1) / 2
        sample = torch.clamp(sample, 0, 1)
        
        # Save image
        from torchvision.utils import save_image
        save_image(sample, samples_dir / f'epoch_{epoch:03d}_sample_{i:02d}.png')


def main():
    args = parse_args()
    
    # Setup output directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Create model and dataloader
    decoder = create_model(args)
    dataloader = create_dataloader(args)
    
    # Create trainer
    decoder_trainer = DecoderTrainer(
        decoder=decoder,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=args.ema_beta,
        ema_update_after_step=args.ema_update_after_step,
        ema_update_every=args.ema_update_every,
        accelerator=accelerator
    )
    
    # Prepare for distributed training
    decoder_trainer, dataloader = accelerator.prepare(decoder_trainer, dataloader)
    
    print(f"Starting training for {args.epochs} epochs")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}")
    
    # Training loop
    for epoch in range(args.epochs):
        decoder_trainer.train()
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch in progress_bar:
            images = batch['image']
            
            # Train step
            loss = decoder_trainer(images, unet_number=1)
            decoder_trainer.update(unet_number=1)
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}')
        
        # Generate samples
        if (epoch + 1) % args.sample_every == 0:
            if accelerator.is_main_process:
                save_samples(decoder_trainer, epoch + 1, output_dir)
        
        # Save model
        if (epoch + 1) % args.save_every == 0:
            if accelerator.is_main_process:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': decoder_trainer.state_dict(),
                    'args': vars(args)
                }
                torch.save(checkpoint, output_dir / f'decoder_epoch_{epoch+1:03d}.pt')
    
    # Save final model
    if accelerator.is_main_process:
        final_checkpoint = {
            'epoch': args.epochs,
            'model_state_dict': decoder_trainer.state_dict(),
            'args': vars(args)
        }
        torch.save(final_checkpoint, output_dir / 'decoder_final.pt')
        print(f"Training completed! Final model saved to {output_dir / 'decoder_final.pt'}")


if __name__ == '__main__':
    main()
