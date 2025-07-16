"""
Training script for Micro-Doppler DALLE2 Prior
Trains the user-conditioned diffusion prior for micro-Doppler time-frequency images
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

from dalle2_pytorch import OpenClipAdapter
from dalle2_pytorch.micro_doppler_dalle2 import (
    UserConditionedPriorNetwork, 
    UserConditionedDiffusionPrior
)
from dalle2_pytorch.dataloaders import create_micro_doppler_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train Micro-Doppler DALLE2 Prior')
    
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
    parser.add_argument('--dim', type=int, default=512,
                        help='Dimension for prior network')
    parser.add_argument('--depth', type=int, default=6,
                        help='Depth of causal transformer')
    parser.add_argument('--dim_head', type=int, default=64,
                        help='Dimension per attention head')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--sample_timesteps', type=int, default=64,
                        help='Number of timesteps for sampling')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
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
    
    # Conditioning arguments
    parser.add_argument('--cond_drop_prob', type=float, default=0.2,
                        help='Conditioning dropout probability for classifier-free guidance')
    
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
    """Create the user-conditioned diffusion prior model"""
    
    # Create CLIP adapter
    clip = OpenClipAdapter(args.clip_model)
    
    # Create prior network
    prior_network = UserConditionedPriorNetwork(
        dim=args.dim,
        num_users=args.num_users,
        depth=args.depth,
        dim_head=args.dim_head,
        heads=args.heads,
        timesteps=args.timesteps
    )
    
    # Create diffusion prior
    diffusion_prior = UserConditionedDiffusionPrior(
        net=prior_network,
        clip=clip,
        timesteps=args.timesteps,
        sample_timesteps=args.sample_timesteps,
        cond_drop_prob=args.cond_drop_prob
    )
    
    return diffusion_prior


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


def save_samples(diffusion_prior, epoch, output_dir, num_users=5, samples_per_user=2):
    """Generate and save sample image embeddings"""
    
    device = next(diffusion_prior.parameters()).device
    
    # Sample from different users
    user_ids = torch.arange(num_users, device=device)
    user_ids = user_ids.repeat_interleave(samples_per_user)
    
    # Generate image embeddings
    with torch.no_grad():
        image_embeds = diffusion_prior.sample(
            user_ids=user_ids,
            num_samples_per_batch=1,
            cond_scale=2.0
        )
    
    # Save embeddings
    samples_dir = Path(output_dir) / 'prior_samples'
    samples_dir.mkdir(exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'user_ids': user_ids.cpu(),
        'image_embeds': image_embeds.cpu()
    }, samples_dir / f'prior_samples_epoch_{epoch:03d}.pt')


class PriorTrainer:
    """Simple trainer for the diffusion prior"""
    
    def __init__(self, model, lr=3e-4, weight_decay=1e-2, accelerator=None):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.accelerator = accelerator
        
    def train_step(self, batch):
        """Single training step"""
        images = batch['image']
        user_ids = batch['user_id']
        
        # Get image embeddings from CLIP
        with torch.no_grad():
            image_embeds = self.model.clip.embed_image(images).image_embed
        
        # Forward pass
        loss = self.model(image_embeds, user_ids=user_ids)
        
        # Backward pass
        if self.accelerator:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])


def main():
    args = parse_args()
    
    # Setup output directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / f"prior_{args.experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Create model and dataloader
    diffusion_prior = create_model(args)
    dataloader = create_dataloader(args)
    
    # Create trainer
    trainer = PriorTrainer(
        model=diffusion_prior,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accelerator=accelerator
    )
    
    # Prepare for distributed training
    trainer.model, trainer.optimizer, dataloader = accelerator.prepare(
        trainer.model, trainer.optimizer, dataloader
    )
    
    print(f"Starting prior training for {args.epochs} epochs")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}")
    
    # Training loop
    for epoch in range(args.epochs):
        trainer.model.train()
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch in progress_bar:
            # Train step
            loss = trainer.train_step(batch)
            
            epoch_loss += loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}')
        
        # Generate samples
        if (epoch + 1) % args.sample_every == 0:
            if accelerator.is_main_process:
                save_samples(trainer.model, epoch + 1, output_dir)
        
        # Save model
        if (epoch + 1) % args.save_every == 0:
            if accelerator.is_main_process:
                checkpoint = {
                    'epoch': epoch + 1,
                    'trainer_state_dict': trainer.state_dict(),
                    'args': vars(args)
                }
                torch.save(checkpoint, output_dir / f'prior_epoch_{epoch+1:03d}.pt')
    
    # Save final model
    if accelerator.is_main_process:
        final_checkpoint = {
            'epoch': args.epochs,
            'trainer_state_dict': trainer.state_dict(),
            'args': vars(args)
        }
        torch.save(final_checkpoint, output_dir / 'prior_final.pt')
        print(f"Prior training completed! Final model saved to {output_dir / 'prior_final.pt'}")


if __name__ == '__main__':
    main()
