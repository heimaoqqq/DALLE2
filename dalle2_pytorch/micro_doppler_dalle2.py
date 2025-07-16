"""
Micro-Doppler DALLE2 Implementation
Modified DALLE2 for user-conditioned micro-Doppler time-frequency image generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random

from dalle2_pytorch.dalle2_pytorch import (
    DiffusionPriorNetwork, DiffusionPrior, Decoder, 
    NoiseScheduler, exists, default, eval_decorator,
    module_device, UnetOutput
)
from dalle2_pytorch.vqgan_vae import NullVQGanVAE
import torchvision.transforms as T


class UserConditionedPriorNetwork(nn.Module):
    """
    User-conditioned diffusion prior network for micro-Doppler images
    Replaces text conditioning with user ID conditioning
    """
    
    def __init__(
        self,
        dim,
        num_users=31,
        user_embed_dim=64,  # 专门的用户embedding维度
        num_timesteps=None,
        num_time_embeds=1,
        num_image_embeds=1,
        num_user_embeds=1,
        self_cond=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_users = num_users
        self.user_embed_dim = user_embed_dim

        self.num_time_embeds = num_time_embeds
        self.num_image_embeds = num_image_embeds
        self.num_user_embeds = num_user_embeds

        # User embedding layer - 使用较小的embedding维度然后投影
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.user_proj = nn.Linear(user_embed_dim, dim)

        # User embeddings processing
        if num_user_embeds > 1:
            self.to_user_embeds = nn.Sequential(
                nn.Linear(dim, dim * num_user_embeds),
                Rearrange('b n (m d) -> b n m d', m = num_user_embeds)
            )
        else:
            self.to_user_embeds = nn.Identity()
        
        # Time embeddings
        self.continuous_embedded_time = not exists(num_timesteps)
        if exists(num_timesteps):
            self.to_time_embeds = nn.Sequential(
                nn.Embedding(num_timesteps, dim * num_time_embeds),
                Rearrange('b (n d) -> b n d', n = num_time_embeds)
            )
        else:
            from dalle2_pytorch.dalle2_pytorch import SinusoidalPosEmb, MLP
            self.to_time_embeds = nn.Sequential(
                SinusoidalPosEmb(dim), 
                MLP(dim, dim * num_time_embeds),
                Rearrange('b (n d) -> b n d', n = num_time_embeds)
            )
        
        # Image embeddings
        self.to_image_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_image_embeds) if num_image_embeds > 1 else nn.Identity(),
            Rearrange('b (n d) -> b n d', n = num_image_embeds)
        )
        
        # Learned query and transformer
        self.learned_query = nn.Parameter(torch.randn(dim))

        # Import CausalTransformer from the main module
        from dalle2_pytorch.dalle2_pytorch import CausalTransformer

        # Filter out parameters that CausalTransformer doesn't accept
        transformer_kwargs = {k: v for k, v in kwargs.items()
                            if k in ['depth', 'dim_head', 'heads', 'ff_mult', 'norm_in', 'norm_out',
                                   'attn_dropout', 'ff_dropout', 'final_proj', 'normformer', 'rotary_emb']}

        self.causal_transformer = CausalTransformer(dim=dim, **transformer_kwargs)
        
        # Null embeddings for classifier-free guidance
        self.null_user_embeds = nn.Parameter(torch.randn(1, num_user_embeds, dim))
        self.null_image_embed = nn.Parameter(torch.randn(1, dim))
        
        # Self conditioning
        self.self_cond = self_cond
        
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)
        
        if cond_scale == 1:
            return logits
            
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale
        
    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        user_ids=None,
        user_embeds=None,
        self_cond=None,
        cond_drop_prob=0.
    ):
        batch, device = image_embed.shape[0], image_embed.device
        
        # Handle user conditioning
        if exists(user_ids):
            user_embeds = self.user_embedding(user_ids)  # [batch, user_embed_dim]
            user_embeds = self.user_proj(user_embeds)     # [batch, dim]
            user_embeds = user_embeds.unsqueeze(1)        # [batch, 1, dim]
        elif exists(user_embeds):
            # Ensure user_embeds has the right shape [batch, 1, dim]
            if user_embeds.dim() == 2:
                user_embeds = user_embeds.unsqueeze(1)
        else:
            # Use null user embeddings
            user_embeds = repeat(self.null_user_embeds, '1 n d -> b n d', b=batch)

        # Apply conditioning dropout for classifier-free guidance
        if cond_drop_prob > 0:
            keep_mask = torch.rand(batch, device=device) > cond_drop_prob
            null_user_embeds = repeat(self.null_user_embeds, '1 n d -> b n d', b=batch)
            user_embeds = torch.where(
                rearrange(keep_mask, 'b -> b 1 1'),
                user_embeds,
                null_user_embeds
            )

        # Process embeddings - user_embeds is already [batch, 1, dim] or [batch, num_user_embeds, dim]
        user_embeds = self.to_user_embeds(user_embeds)
        time_embeds = self.to_time_embeds(diffusion_timesteps)
        image_embeds = self.to_image_embeds(image_embed)
        
        # Self conditioning
        if self.self_cond:
            if exists(self_cond):
                self_cond = self.to_image_embeds(self_cond)
            else:
                self_cond = torch.zeros_like(image_embeds)
            image_embeds = torch.cat((image_embeds, self_cond), dim=-1)
        
        # Prepare tokens for transformer
        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b=batch)
        tokens = torch.cat((
            time_embeds,
            user_embeds,
            image_embeds,
            learned_queries
        ), dim=-2)
        
        # Apply causal transformer
        tokens = self.causal_transformer(tokens)
        
        # Extract prediction
        pred_image_embed = tokens[..., -1, :]
        return pred_image_embed


class UserConditionedDiffusionPrior(DiffusionPrior):
    """
    User-conditioned diffusion prior for micro-Doppler images
    """
    
    def __init__(
        self,
        net,
        *,
        clip=None,
        image_embed_dim=None,
        num_users=31,
        timesteps=1000,
        sample_timesteps=None,
        cond_drop_prob=0.,
        user_cond_drop_prob=None,
        **kwargs
    ):
        # Initialize parent class but override some parameters
        super().__init__(
            net=net,
            clip=clip,
            image_embed_dim=image_embed_dim,
            timesteps=timesteps,
            sample_timesteps=sample_timesteps,
            cond_drop_prob=cond_drop_prob,
            text_cond_drop_prob=user_cond_drop_prob,  # Reuse text_cond_drop_prob for user conditioning
            condition_on_text_encodings=False,  # We don't use text encodings
            **kwargs
        )
        
        self.num_users = num_users
        self.user_cond_drop_prob = default(user_cond_drop_prob, cond_drop_prob)
        
    def forward(
        self,
        image_embed,
        *,
        user_ids=None,
        user_embeds=None,
        **kwargs
    ):
        """
        Forward pass with user conditioning instead of text conditioning
        """
        batch, device = image_embed.shape[0], image_embed.device
        
        # Sample random timesteps
        times = torch.randint(0, self.noise_scheduler.timesteps, (batch,), device=device, dtype=torch.long)
        
        # Add noise to image embeddings
        noise = torch.randn_like(image_embed)
        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)
        
        # Self conditioning
        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(
                    image_embed_noisy,
                    times,
                    user_ids=user_ids,
                    user_embeds=user_embeds,
                    cond_drop_prob=0.
                ).detach()
        
        # Predict noise
        pred = self.net(
            image_embed_noisy,
            times,
            user_ids=user_ids,
            user_embeds=user_embeds,
            self_cond=self_cond,
            cond_drop_prob=self.user_cond_drop_prob
        )
        
        # Calculate loss
        if self.predict_x_start:
            target = image_embed
        else:
            target = noise
            
        loss = F.mse_loss(pred, target)
        return loss
    
    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        user_ids=None,
        user_embeds=None,
        num_samples_per_batch=2,
        cond_scale=1.,
        timesteps=None
    ):
        """
        Sample image embeddings conditioned on user IDs
        """
        timesteps = default(timesteps, self.sample_timesteps)
        timesteps = default(timesteps, self.noise_scheduler.timesteps)
        
        if exists(user_ids):
            batch = user_ids.shape[0]
            device = user_ids.device
        elif exists(user_embeds):
            batch = user_embeds.shape[0]
            device = user_embeds.device
        else:
            raise ValueError("Either user_ids or user_embeds must be provided")
        
        # Expand for multiple samples per batch
        if num_samples_per_batch > 1:
            if exists(user_ids):
                user_ids = repeat(user_ids, 'b -> (b n)', n=num_samples_per_batch)
            if exists(user_embeds):
                user_embeds = repeat(user_embeds, 'b ... -> (b n) ...', n=num_samples_per_batch)
            batch = batch * num_samples_per_batch
        
        # Start with random noise
        image_embed = torch.randn(batch, self.image_embed_dim, device=device)
        
        # Denoising loop
        for i in reversed(range(timesteps)):
            times = torch.full((batch,), i, device=device, dtype=torch.long)
            
            # Self conditioning
            self_cond = None
            if self.net.self_cond:
                self_cond = self.net.forward_with_cond_scale(
                    image_embed,
                    times,
                    user_ids=user_ids,
                    user_embeds=user_embeds,
                    cond_scale=cond_scale
                )
            
            # Predict and denoise
            pred = self.net.forward_with_cond_scale(
                image_embed,
                times,
                user_ids=user_ids,
                user_embeds=user_embeds,
                self_cond=self_cond,
                cond_scale=cond_scale
            )
            
            image_embed = self.noise_scheduler.p_sample(image_embed, times, pred)
        
        return image_embed


class MicroDopplerDALLE2(nn.Module):
    """
    Complete DALLE2 model for micro-Doppler time-frequency image generation
    """
    
    def __init__(
        self,
        *,
        prior,
        decoder,
        num_users=31,
        prior_num_samples=2
    ):
        super().__init__()
        assert isinstance(prior, UserConditionedDiffusionPrior)
        assert isinstance(decoder, Decoder)
        
        self.prior = prior
        self.decoder = decoder
        self.num_users = num_users
        self.prior_num_samples = prior_num_samples
        
        self.to_pil = T.ToPILImage()
        
    @torch.no_grad()
    @eval_decorator
    def forward(
        self,
        user_ids,
        cond_scale=1.,
        prior_cond_scale=1.,
        return_pil_images=False
    ):
        """
        Generate micro-Doppler images conditioned on user IDs
        """
        device = module_device(self)
        
        if isinstance(user_ids, int):
            user_ids = torch.tensor([user_ids], device=device)
        elif isinstance(user_ids, (list, tuple)):
            user_ids = torch.tensor(user_ids, device=device)
        
        # Sample image embeddings from prior
        image_embed = self.prior.sample(
            user_ids=user_ids,
            num_samples_per_batch=self.prior_num_samples,
            cond_scale=prior_cond_scale
        )
        
        # Generate images from embeddings
        images = self.decoder.sample(image_embed=image_embed, cond_scale=cond_scale)
        
        if return_pil_images:
            images = list(map(self.to_pil, images.unbind(dim=0)))
        
        return images
