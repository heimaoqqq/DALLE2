"""
Micro-Doppler Time-Frequency Image Dataset Loader
Designed for 31-user gait micro-Doppler time-frequency images (256x256 RGB)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
from typing import Optional, Callable, List, Union


class MicroDopplerDataset(Dataset):
    """
    Dataset for micro-Doppler time-frequency images with user ID conditions
    
    Expected directory structure:
    data_root/
    ├── user_0/
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    ├── user_1/
    │   ├── image_001.png
    │   └── ...
    └── ...
    
    Or flat structure with metadata:
    data_root/
    ├── images/
    │   ├── user0_001.png
    │   ├── user0_002.png
    │   └── ...
    └── metadata.json  # Contains user_id mapping
    """
    
    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        num_users: int = 31,
        transform: Optional[Callable] = None,
        metadata_file: Optional[str] = None,
        image_extensions: List[str] = ['.png', '.jpg', '.jpeg'],
        flat_structure: bool = False
    ):
        """
        Args:
            data_root: Root directory containing the dataset
            image_size: Target image size (assumes square images)
            num_users: Total number of users (0 to num_users-1)
            transform: Optional transform to apply to images
            metadata_file: Path to metadata JSON file (for flat structure)
            image_extensions: Supported image file extensions
            flat_structure: If True, expects flat directory with metadata file
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.num_users = num_users
        self.image_extensions = image_extensions
        self.flat_structure = flat_structure
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
            
        # Load dataset
        self.samples = self._load_samples(metadata_file)
        
        print(f"Loaded {len(self.samples)} micro-Doppler images from {num_users} users")
        
    def _load_samples(self, metadata_file: Optional[str]) -> List[dict]:
        """Load all image paths and corresponding user IDs"""
        samples = []
        
        if self.flat_structure and metadata_file:
            # Load from flat structure with metadata
            samples = self._load_flat_structure(metadata_file)
        else:
            # Load from hierarchical structure
            samples = self._load_hierarchical_structure()
            
        return samples
    
    def _load_hierarchical_structure(self) -> List[dict]:
        """Load samples from hierarchical directory structure"""
        samples = []

        for user_id in range(self.num_users):
            # 支持多种命名格式: ID_1, ID_2, ... (主要格式) 或 user_0, user_1, ... 或 ID1, ID2, ... (备用格式)
            user_dir_formats = [
                self.data_root / f"ID_{user_id + 1}",     # ID_1 对应 user_id=0 (主要格式)
                self.data_root / f"user_{user_id}",       # user_0 对应 user_id=0 (备用格式)
                self.data_root / f"ID{user_id + 1}"       # ID1 对应 user_id=0 (备用格式)
            ]

            user_dir = None
            for dir_format in user_dir_formats:
                if dir_format.exists():
                    user_dir = dir_format
                    break

            if user_dir is None:
                print(f"Warning: User directory for user_id {user_id} not found, skipping...")
                continue

            # Find all image files in user directory
            image_count = 0
            for ext in self.image_extensions:
                image_files = list(user_dir.glob(f"*{ext}"))
                for image_path in image_files:
                    samples.append({
                        'image_path': str(image_path),
                        'user_id': user_id,  # 统一使用0-30的user_id
                        'filename': image_path.name,
                        'original_folder': user_dir.name
                    })
                    image_count += 1

            if image_count > 0:
                print(f"Loaded {image_count} images from {user_dir.name} (user_id: {user_id})")

        return samples
    
    def _load_flat_structure(self, metadata_file: str) -> List[dict]:
        """Load samples from flat structure with metadata file"""
        samples = []
        metadata_path = self.data_root / metadata_file
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file {metadata_path} not found")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        images_dir = self.data_root / "images"
        
        for item in metadata:
            image_path = images_dir / item['filename']
            if image_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'user_id': item['user_id'],
                    'filename': item['filename']
                })
            else:
                print(f"Warning: Image {image_path} not found, skipping...")
                
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict containing:
                - 'image': Transformed image tensor [3, H, W]
                - 'user_id': User ID as integer tensor
                - 'filename': Original filename
        """
        sample = self.samples[idx]

        # Load and transform image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Return user_id as tensor (will be converted to embedding in model)
        user_id = torch.tensor(sample['user_id'], dtype=torch.long)

        return {
            'image': image,
            'user_id': user_id,
            'filename': sample['filename']
        }
    
    def get_user_distribution(self) -> dict:
        """Get distribution of samples per user"""
        user_counts = {}
        for sample in self.samples:
            user_id = sample['user_id']
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        return user_counts


def create_micro_doppler_dataloader(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 256,
    num_users: int = 31,
    shuffle: bool = True,
    pin_memory: bool = True,
    transform: Optional[Callable] = None,
    metadata_file: Optional[str] = None,
    flat_structure: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for micro-Doppler time-frequency images
    
    Args:
        data_root: Root directory containing the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        image_size: Target image size (assumes square images)
        num_users: Total number of users
        shuffle: Whether to shuffle the dataset
        pin_memory: Whether to pin memory for faster GPU transfer
        transform: Optional transform to apply to images
        metadata_file: Path to metadata JSON file (for flat structure)
        flat_structure: If True, expects flat directory with metadata file
        **kwargs: Additional arguments passed to DataLoader
        
    Returns:
        DataLoader instance
    """
    dataset = MicroDopplerDataset(
        data_root=data_root,
        image_size=image_size,
        num_users=num_users,
        transform=transform,
        metadata_file=metadata_file,
        flat_structure=flat_structure
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Important for consistent batch sizes in training
        **kwargs
    )


def create_sample_metadata_file(data_root: str, output_file: str = "metadata.json"):
    """
    Create a sample metadata file for flat directory structure
    
    Args:
        data_root: Root directory containing images
        output_file: Output metadata file name
    """
    data_path = Path(data_root)
    images_dir = data_path / "images"
    
    if not images_dir.exists():
        print(f"Images directory {images_dir} not found")
        return
        
    metadata = []
    image_extensions = ['.png', '.jpg', '.jpeg']
    
    for ext in image_extensions:
        for image_path in images_dir.glob(f"*{ext}"):
            # Extract user_id from filename (assuming format: userX_XXX.ext)
            filename = image_path.name
            try:
                user_part = filename.split('_')[0]
                if user_part.startswith('user'):
                    user_id = int(user_part.replace('user', ''))
                else:
                    user_id = int(user_part)
                    
                metadata.append({
                    'filename': filename,
                    'user_id': user_id
                })
            except (ValueError, IndexError):
                print(f"Warning: Could not extract user_id from {filename}")
                
    # Save metadata
    output_path = data_path / output_file
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Created metadata file: {output_path}")
    print(f"Total samples: {len(metadata)}")
