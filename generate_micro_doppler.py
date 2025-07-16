"""
Generation script for Micro-Doppler DALLE2
Generate new micro-Doppler time-frequency images conditioned on user IDs
"""

import torch
import argparse
from pathlib import Path
import json
from torchvision.utils import save_image
from torchvision import transforms

from dalle2_pytorch import Unet, Decoder, OpenClipAdapter
from dalle2_pytorch.micro_doppler_dalle2 import (
    UserConditionedPriorNetwork, 
    UserConditionedDiffusionPrior, 
    MicroDopplerDALLE2
)


def load_trained_models(decoder_path, prior_path, num_users=31):
    """Load trained decoder and prior models"""
    
    print("Loading trained models...")
    
    # Load decoder checkpoint
    decoder_checkpoint = torch.load(decoder_path, map_location='cpu')
    decoder_args = decoder_checkpoint['args']
    
    # Recreate decoder
    clip = OpenClipAdapter(decoder_args.get('clip_model', 'ViT-B/32'))
    
    unet = Unet(
        dim=decoder_args.get('dim', 128),
        image_embed_dim=512,
        cond_dim=128,
        channels=decoder_args.get('channels', 3),
        dim_mults=tuple(decoder_args.get('dim_mults', [1, 2, 4, 8])),
        cond_on_image_embeds=True,
        cond_on_text_encodings=False
    )
    
    decoder = Decoder(
        unet=unet,
        clip=clip,
        image_sizes=(decoder_args.get('image_size', 256),),
        timesteps=decoder_args.get('timesteps', 1000)
    )
    
    # Load decoder weights
    decoder.load_state_dict(decoder_checkpoint['model_state_dict']['decoder'])
    
    # Load prior checkpoint
    prior_checkpoint = torch.load(prior_path, map_location='cpu')
    prior_args = prior_checkpoint['args']
    
    # Recreate prior
    prior_network = UserConditionedPriorNetwork(
        dim=prior_args.get('dim', 512),
        num_users=num_users,
        depth=prior_args.get('depth', 6),
        dim_head=prior_args.get('dim_head', 64),
        heads=prior_args.get('heads', 8),
        timesteps=prior_args.get('timesteps', 1000)
    )
    
    diffusion_prior = UserConditionedDiffusionPrior(
        net=prior_network,
        clip=clip,
        timesteps=prior_args.get('timesteps', 1000),
        sample_timesteps=prior_args.get('sample_timesteps', 64),
        num_users=num_users
    )
    
    # Load prior weights
    diffusion_prior.load_state_dict(prior_checkpoint['trainer_state_dict']['model'])
    
    # Create complete model
    micro_dalle2 = MicroDopplerDALLE2(
        prior=diffusion_prior,
        decoder=decoder,
        num_users=num_users
    )
    
    print("✓ Models loaded successfully!")
    return micro_dalle2


def generate_images(
    model, 
    user_ids, 
    num_samples_per_user=4, 
    cond_scale=2.0, 
    prior_cond_scale=2.0,
    device='cuda'
):
    """Generate images for specified user IDs"""
    
    model = model.to(device)
    model.eval()
    
    all_images = []
    all_user_ids = []
    
    with torch.no_grad():
        for user_id in user_ids:
            print(f"Generating {num_samples_per_user} images for user {user_id}...")
            
            # Create user ID tensor
            user_tensor = torch.tensor([user_id] * num_samples_per_user, device=device)
            
            # Generate images
            images = model(
                user_ids=user_tensor,
                cond_scale=cond_scale,
                prior_cond_scale=prior_cond_scale
            )
            
            all_images.append(images)
            all_user_ids.extend([user_id] * num_samples_per_user)
    
    # Concatenate all images
    all_images = torch.cat(all_images, dim=0)
    
    return all_images, all_user_ids


def save_generated_images(images, user_ids, output_dir, prefix="generated"):
    """Save generated images to disk"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert from [-1, 1] to [0, 1] range
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Save individual images
    for i, (image, user_id) in enumerate(zip(images, user_ids)):
        filename = f"{prefix}_user_{user_id:02d}_sample_{i:03d}.png"
        save_image(image, output_path / filename)
    
    # Save grid of all images
    grid_filename = f"{prefix}_grid.png"
    save_image(images, output_path / grid_filename, nrow=4, padding=2)
    
    print(f"✓ Saved {len(images)} images to {output_path}")
    
    # Save metadata
    metadata = {
        'num_images': len(images),
        'user_ids': user_ids,
        'image_files': [f"{prefix}_user_{uid:02d}_sample_{i:03d}.png" 
                       for i, uid in enumerate(user_ids)]
    }
    
    with open(output_path / f"{prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def interpolate_between_users(model, user_id_1, user_id_2, num_steps=8, device='cuda'):
    """Generate interpolation between two users"""
    
    model = model.to(device)
    model.eval()
    
    print(f"Generating interpolation between user {user_id_1} and user {user_id_2}...")
    
    with torch.no_grad():
        # Generate embeddings for both users
        user_1_tensor = torch.tensor([user_id_1], device=device)
        user_2_tensor = torch.tensor([user_id_2], device=device)
        
        embed_1 = model.prior.sample(user_ids=user_1_tensor, num_samples_per_batch=1)
        embed_2 = model.prior.sample(user_ids=user_2_tensor, num_samples_per_batch=1)
        
        # Create interpolation
        alphas = torch.linspace(0, 1, num_steps, device=device)
        interpolated_embeds = []
        
        for alpha in alphas:
            interp_embed = (1 - alpha) * embed_1 + alpha * embed_2
            interpolated_embeds.append(interp_embed)
        
        interpolated_embeds = torch.cat(interpolated_embeds, dim=0)
        
        # Generate images from interpolated embeddings
        images = model.decoder.sample(image_embed=interpolated_embeds)
    
    return images


def main():
    parser = argparse.ArgumentParser(description='Generate Micro-Doppler images')
    
    # Model paths
    parser.add_argument('--decoder_path', type=str, required=True,
                        help='Path to trained decoder checkpoint')
    parser.add_argument('--prior_path', type=str, required=True,
                        help='Path to trained prior checkpoint')
    
    # Generation parameters
    parser.add_argument('--user_ids', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='User IDs to generate images for')
    parser.add_argument('--num_samples_per_user', type=int, default=4,
                        help='Number of samples to generate per user')
    parser.add_argument('--cond_scale', type=float, default=2.0,
                        help='Conditioning scale for decoder')
    parser.add_argument('--prior_cond_scale', type=float, default=2.0,
                        help='Conditioning scale for prior')
    
    # Interpolation parameters
    parser.add_argument('--interpolate', action='store_true',
                        help='Generate interpolation between users')
    parser.add_argument('--interp_user_1', type=int, default=0,
                        help='First user for interpolation')
    parser.add_argument('--interp_user_2', type=int, default=1,
                        help='Second user for interpolation')
    parser.add_argument('--interp_steps', type=int, default=8,
                        help='Number of interpolation steps')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./generated_images',
                        help='Output directory for generated images')
    parser.add_argument('--num_users', type=int, default=31,
                        help='Total number of users in the model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for generation')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    try:
        # Load models
        model = load_trained_models(
            args.decoder_path, 
            args.prior_path, 
            num_users=args.num_users
        )
        
        # Generate regular images
        if args.user_ids:
            print(f"Generating images for users: {args.user_ids}")
            images, user_ids = generate_images(
                model=model,
                user_ids=args.user_ids,
                num_samples_per_user=args.num_samples_per_user,
                cond_scale=args.cond_scale,
                prior_cond_scale=args.prior_cond_scale,
                device=args.device
            )
            
            save_generated_images(
                images, 
                user_ids, 
                args.output_dir, 
                prefix="generated"
            )
        
        # Generate interpolation
        if args.interpolate:
            interp_images = interpolate_between_users(
                model=model,
                user_id_1=args.interp_user_1,
                user_id_2=args.interp_user_2,
                num_steps=args.interp_steps,
                device=args.device
            )
            
            interp_user_ids = [f"{args.interp_user_1}-{args.interp_user_2}"] * len(interp_images)
            save_generated_images(
                interp_images,
                interp_user_ids,
                args.output_dir,
                prefix="interpolation"
            )
        
        print("\n🎉 Generation completed successfully!")
        print(f"Check the output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Generation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
