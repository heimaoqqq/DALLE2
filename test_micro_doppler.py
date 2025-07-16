"""
Test script for Micro-Doppler DALLE2 implementation
Validates data loading, model creation, and basic functionality
"""

import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
import json
import argparse

from dalle2_pytorch import Unet, Decoder, OpenClipAdapter
from dalle2_pytorch.micro_doppler_dalle2 import (
    UserConditionedPriorNetwork, 
    UserConditionedDiffusionPrior, 
    MicroDopplerDALLE2
)
from dalle2_pytorch.dataloaders import create_micro_doppler_dataloader, MicroDopplerDataset


def create_dummy_dataset(output_dir, num_users=5, images_per_user=10, image_size=256):
    """Create a dummy micro-Doppler dataset for testing"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy dataset in {output_path}")
    
    # Create hierarchical structure
    for user_id in range(num_users):
        user_dir = output_path / f"user_{user_id}"
        user_dir.mkdir(exist_ok=True)
        
        for img_id in range(images_per_user):
            # Create a random RGB image that looks like a time-frequency plot
            # Use different patterns for different users
            image = np.random.rand(image_size, image_size, 3) * 255
            
            # Add some user-specific patterns
            if user_id % 2 == 0:
                # Add horizontal stripes for even users
                for i in range(0, image_size, 20):
                    image[i:i+5, :, :] = 255
            else:
                # Add vertical stripes for odd users
                for i in range(0, image_size, 20):
                    image[:, i:i+5, :] = 255
            
            # Save image
            image = Image.fromarray(image.astype(np.uint8))
            image.save(user_dir / f"image_{img_id:03d}.png")
    
    print(f"Created dummy dataset with {num_users} users, {images_per_user} images per user")
    return output_path


def test_dataloader(data_root, num_users=5):
    """Test the micro-Doppler dataloader"""
    
    print("Testing MicroDopplerDataset...")
    
    # Test dataset creation
    dataset = MicroDopplerDataset(
        data_root=data_root,
        num_users=num_users,
        image_size=256
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Test data loading
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"User ID: {sample['user_id']}")
    print(f"User embed shape: {sample['user_embed'].shape}")
    
    # Test dataloader
    dataloader = create_micro_doppler_dataloader(
        data_root=data_root,
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        num_users=num_users,
        shuffle=True
    )
    
    batch = next(iter(dataloader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch user_ids: {batch['user_id']}")
    print(f"Batch user_embeds shape: {batch['user_embed'].shape}")
    
    # Test user distribution
    user_dist = dataset.get_user_distribution()
    print(f"User distribution: {user_dist}")
    
    print("✓ Dataloader test passed!")
    return dataloader


def test_models(num_users=5):
    """Test model creation and forward passes"""
    
    print("Testing model creation...")
    
    # Test UserConditionedPriorNetwork
    prior_network = UserConditionedPriorNetwork(
        dim=512,
        num_users=num_users,
        depth=2,  # Smaller for testing
        dim_head=64,
        heads=4,
        timesteps=100
    )
    
    # Test forward pass
    batch_size = 4
    image_embed = torch.randn(batch_size, 512)
    user_ids = torch.randint(0, num_users, (batch_size,))
    timesteps = torch.randint(0, 100, (batch_size,))
    
    pred = prior_network(image_embed, timesteps, user_ids=user_ids)
    print(f"Prior network output shape: {pred.shape}")
    
    # Test UserConditionedDiffusionPrior
    clip = OpenClipAdapter('ViT-B/32')
    
    diffusion_prior = UserConditionedDiffusionPrior(
        net=prior_network,
        clip=clip,
        timesteps=100,
        sample_timesteps=10,
        num_users=num_users
    )
    
    # Test training forward pass
    images = torch.randn(batch_size, 3, 256, 256)
    with torch.no_grad():
        image_embeds = clip.embed_image(images).image_embed
    
    loss = diffusion_prior(image_embeds, user_ids=user_ids)
    print(f"Prior training loss: {loss.item():.4f}")
    
    # Test sampling
    with torch.no_grad():
        sampled_embeds = diffusion_prior.sample(
            user_ids=user_ids[:2],
            num_samples_per_batch=1,
            cond_scale=1.0
        )
    print(f"Sampled embeddings shape: {sampled_embeds.shape}")
    
    # Test Decoder
    unet = Unet(
        dim=64,  # Smaller for testing
        image_embed_dim=512,
        cond_dim=128,
        channels=3,
        dim_mults=(1, 2, 4),
        cond_on_image_embeds=True,
        cond_on_text_encodings=False
    )
    
    decoder = Decoder(
        unet=unet,
        clip=clip,
        image_sizes=(256,),
        timesteps=100
    )
    
    # Test decoder training
    decoder_loss = decoder(images)
    print(f"Decoder training loss: {decoder_loss.item():.4f}")
    
    # Test decoder sampling
    with torch.no_grad():
        generated_images = decoder.sample(image_embed=sampled_embeds)
    print(f"Generated images shape: {generated_images.shape}")
    
    # Test complete MicroDopplerDALLE2
    micro_dalle2 = MicroDopplerDALLE2(
        prior=diffusion_prior,
        decoder=decoder,
        num_users=num_users
    )
    
    # Test end-to-end generation
    test_user_ids = [0, 1]
    with torch.no_grad():
        final_images = micro_dalle2(test_user_ids)
    print(f"Final generated images shape: {final_images.shape}")
    
    print("✓ Model tests passed!")
    return micro_dalle2


def test_training_compatibility(dataloader, model):
    """Test that models work with the dataloader"""
    
    print("Testing training compatibility...")
    
    batch = next(iter(dataloader))
    images = batch['image']
    user_ids = batch['user_id']
    
    # Test prior training step
    with torch.no_grad():
        image_embeds = model.prior.clip.embed_image(images).image_embed
    
    prior_loss = model.prior(image_embeds, user_ids=user_ids)
    print(f"Prior loss with real data: {prior_loss.item():.4f}")
    
    # Test decoder training step
    decoder_loss = model.decoder(images)
    print(f"Decoder loss with real data: {decoder_loss.item():.4f}")
    
    print("✓ Training compatibility test passed!")


def main():
    parser = argparse.ArgumentParser(description='Test Micro-Doppler DALLE2')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to existing dataset (if None, creates dummy dataset)')
    parser.add_argument('--num_users', type=int, default=5,
                        help='Number of users for testing')
    parser.add_argument('--create_dummy', action='store_true',
                        help='Force creation of dummy dataset')
    
    args = parser.parse_args()
    
    # Setup data
    if args.data_root is None or args.create_dummy:
        data_root = create_dummy_dataset('./test_data', num_users=args.num_users)
    else:
        data_root = Path(args.data_root)
        if not data_root.exists():
            print(f"Data root {data_root} does not exist, creating dummy dataset...")
            data_root = create_dummy_dataset('./test_data', num_users=args.num_users)
    
    print(f"Using data from: {data_root}")
    
    try:
        # Test dataloader
        dataloader = test_dataloader(data_root, num_users=args.num_users)
        
        # Test models
        model = test_models(num_users=args.num_users)
        
        # Test training compatibility
        test_training_compatibility(dataloader, model)
        
        print("\n🎉 All tests passed! The Micro-Doppler DALLE2 implementation is working correctly.")
        print("\nNext steps:")
        print("1. Prepare your real micro-Doppler dataset")
        print("2. Train the decoder using: python train_micro_doppler_decoder.py")
        print("3. Train the prior using: python train_micro_doppler_prior.py")
        print("4. Generate new images using the trained models")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
