#!/usr/bin/env python3
"""
HDF5 Training DataLoader for Anime Image Tagger
Efficiently loads preprocessed images and teacher features from HDF5 files
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import h5py
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tag_vocabulary import TagVocabulary, load_vocabulary_for_training

logger = logging.getLogger(__name__)


@dataclass
class HDF5DataConfig:
    """Configuration for HDF5 dataset"""
    hdf5_dir: Path
    vocab_dir: Path
    
    # Normalization settings
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Memory settings
    cache_size_gb: float = 4.0  # Amount of data to cache in memory
    preload_files: int = 2  # Number of HDF5 files to keep open
    
    # Training settings
    use_teacher_features: bool = True
    teacher_feature_dropout: float = 0.0  # Randomly drop teacher features
    
    # Augmentation settings (for later)
    augmentation_enabled: bool = False
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    random_flip_prob: float = 0.5
    color_jitter: bool = False
    
    # Multi-GPU settings
    distributed: bool = False
    rank: int = 0
    world_size: int = 1


class HDF5FileManager:
    """Manages multiple HDF5 files efficiently"""
    
    def __init__(self, hdf5_files: List[Path], preload_files: int = 2):
        self.hdf5_files = sorted(hdf5_files)
        self.preload_files = preload_files
        self.file_handles: Dict[str, h5py.File] = {}
        self.file_info: Dict[str, Dict] = {}
        
        # Load index information
        self._load_file_info()
        
        # Calculate total samples
        self.total_samples = sum(info['num_samples'] for info in self.file_info.values())
        
        # Create global index mapping
        self._create_index_mapping()
        
    def _load_file_info(self):
        """Load metadata about each HDF5 file"""
        for file_path in self.hdf5_files:
            with h5py.File(file_path, 'r') as f:
                # Get number of samples
                num_samples = f['images/student_640'].shape[0]
                
                # Get available features
                features = {
                    'anime_teacher': list(f['anime_teacher'].keys()) if 'anime_teacher' in f else [],
                    'clip_teacher': list(f['clip_teacher'].keys()) if 'clip_teacher' in f else []
                }
                
                self.file_info[str(file_path)] = {
                    'num_samples': num_samples,
                    'features': features,
                    'path': file_path
                }
                
        logger.info(f"Loaded info for {len(self.hdf5_files)} HDF5 files, "
                   f"total samples: {sum(info['num_samples'] for info in self.file_info.values())}")
    
    def _create_index_mapping(self):
        """Create mapping from global index to (file, local_index)"""
        self.index_to_file = []
        cumulative = 0
        
        for file_path in self.hdf5_files:
            num_samples = self.file_info[str(file_path)]['num_samples']
            for local_idx in range(num_samples):
                self.index_to_file.append((str(file_path), local_idx))
            cumulative += num_samples
    
    def get_file_handle(self, file_path: str) -> h5py.File:
        """Get file handle, opening if necessary"""
        if file_path not in self.file_handles:
            # Close oldest file if we have too many open
            if len(self.file_handles) >= self.preload_files:
                oldest = next(iter(self.file_handles))
                self.file_handles[oldest].close()
                del self.file_handles[oldest]
            
            # Open new file
            self.file_handles[file_path] = h5py.File(file_path, 'r')
            
        return self.file_handles[file_path]
    
    def get_sample(self, global_idx: int) -> Tuple[str, int, h5py.File]:
        """Get file handle and local index for global index"""
        file_path, local_idx = self.index_to_file[global_idx]
        file_handle = self.get_file_handle(file_path)
        return file_path, local_idx, file_handle
    
    def close_all(self):
        """Close all open file handles"""
        for f in self.file_handles.values():
            f.close()
        self.file_handles.clear()
    
    def __del__(self):
        """Cleanup file handles"""
        self.close_all()


class HDF5Dataset(Dataset):
    """PyTorch Dataset for loading from HDF5 files"""
    
    def __init__(self, 
                 config: HDF5DataConfig,
                 tag_files: Optional[List[Path]] = None,
                 split: str = 'train'):
        
        self.config = config
        self.split = split
        
        # Load vocabulary
        self.vocab = load_vocabulary_for_training(config.vocab_dir)
        logger.info(f"Loaded vocabulary with {len(self.vocab.tag_to_index)} tags")
        
        # Discover HDF5 files
        hdf5_files = list(Path(config.hdf5_dir).glob("features_part_*.h5"))
        if not hdf5_files:
            raise ValueError(f"No HDF5 files found in {config.hdf5_dir}")
        
        # Initialize file manager
        self.file_manager = HDF5FileManager(hdf5_files, config.preload_files)
        
        # Load tag annotations if provided
        self.tag_annotations = {}
        if tag_files:
            self._load_tag_annotations(tag_files)
        else:
            logger.warning("No tag files provided, will use teacher predictions only")
        
        # Setup normalization
        self.normalize = T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        
        # Setup augmentation (if enabled)
        self.augmentation = self._setup_augmentation() if config.augmentation_enabled else None
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_size = int(config.cache_size_gb * 1024 * 1024 * 1024 / (3 * 640 * 640))  # Approximate
        
    def _load_tag_annotations(self, tag_files: List[Path]):
        """Load tag annotations from JSON files"""
        logger.info(f"Loading tag annotations from {len(tag_files)} files")
        
        for tag_file in tag_files:
            try:
                with open(tag_file, 'r') as f:
                    data = json.load(f)
                
                # Expected format: {"image_path": ["tag1", "tag2", ...]}
                for image_path, tags in data.items():
                    # Convert tags to indices
                    tag_indices = self.vocab.get_tag_indices(tags)
                    # Convert to hierarchical format
                    hierarchical_tags = self.vocab.encode_tags_hierarchical(tags)
                    
                    self.tag_annotations[image_path] = {
                        'tags': tags,
                        'indices': tag_indices,
                        'hierarchical': hierarchical_tags
                    }
                    
            except Exception as e:
                logger.error(f"Error loading {tag_file}: {e}")
        
        logger.info(f"Loaded annotations for {len(self.tag_annotations)} images")
    
    def _setup_augmentation(self):
        """Setup augmentation transforms"""
        transforms = []
        
        if self.config.random_flip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(p=self.config.random_flip_prob))
        
        if self.config.color_jitter:
            transforms.append(T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ))
        
        return T.Compose(transforms) if transforms else None
    
    def __len__(self):
        return self.file_manager.total_samples
    
    def _load_from_cache_or_hdf5(self, idx: int) -> Dict[str, Any]:
        """Load data from cache or HDF5"""
        if idx in self.cache:
            return self.cache[idx]
        
        # Get file info
        file_path, local_idx, h5_file = self.file_manager.get_sample(idx)
        
        # Load image (stored as uint8)
        image = h5_file['images/student_640'][local_idx]  # (3, 640, 640)
        
        # Load metadata
        padding_info = h5_file['images/padding_info'][local_idx]  # (4,) - x1,y1,x2,y2
        original_size = h5_file['images/original_sizes'][local_idx]  # (2,) - w,h
        image_path = h5_file['metadata/paths'][local_idx]
        
        # Load teacher features if available and not dropped
        teacher_features = {}
        if self.config.use_teacher_features and random.random() > self.config.teacher_feature_dropout:
            # Anime teacher features
            if 'anime_teacher' in h5_file:
                anime_features = {}
                for key in ['tag_logits', 'tag_probs', 'top_k_indices', 'top_k_probs']:
                    if key in h5_file['anime_teacher']:
                        anime_features[key] = h5_file[f'anime_teacher/{key}'][local_idx]
                teacher_features['anime'] = anime_features
            
            # CLIP features
            if 'clip_teacher' in h5_file:
                clip_features = {}
                for key in ['cls_token', 'patch_tokens_avg', 'top_patches']:
                    if key in h5_file['clip_teacher']:
                        clip_features[key] = h5_file[f'clip_teacher/{key}'][local_idx]
                teacher_features['clip'] = clip_features
        
        data = {
            'image': image,
            'padding_info': padding_info,
            'original_size': original_size,
            'image_path': image_path,
            'teacher_features': teacher_features
        }
        
        # Add to cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[idx] = data
        
        return data
    
    def _create_tag_labels(self, image_path: str, teacher_features: Dict) -> Dict[str, torch.Tensor]:
        """Create tag labels from annotations or teacher predictions"""
        
        # Initialize label tensors
        num_tags = len(self.vocab.tag_to_index)
        binary_labels = torch.zeros(num_tags, dtype=torch.float32)
        hierarchical_labels = {}
        
        # Use ground truth annotations if available
        if image_path in self.tag_annotations:
            anno = self.tag_annotations[image_path]
            
            # Set binary labels
            for idx in anno['indices']:
                if idx != self.vocab.unk_index:  # Skip UNK tokens
                    binary_labels[idx] = 1.0
            
            # Set hierarchical labels
            hierarchical_labels = anno['hierarchical']
        
        # Otherwise use teacher predictions
        elif 'anime' in teacher_features and 'tag_probs' in teacher_features['anime']:
            # Use anime teacher predictions
            tag_probs = teacher_features['anime']['tag_probs']
            
            # Map from 70k teacher tags to our 200k vocabulary
            # This requires a mapping that should be precomputed
            # For now, simplified version:
            
            if 'top_k_indices' in teacher_features['anime']:
                top_k_indices = teacher_features['anime']['top_k_indices']
                top_k_probs = teacher_features['anime']['top_k_probs']
                
                # Threshold predictions
                threshold = 0.5
                for idx, prob in zip(top_k_indices, top_k_probs):
                    if prob > threshold:
                        # Map teacher index to our vocabulary (placeholder)
                        # In practice, you'd have a precomputed mapping
                        our_idx = int(idx) % num_tags  # Simplified mapping
                        binary_labels[our_idx] = float(prob)
        
        # Convert to hierarchical format if needed
        if not hierarchical_labels:
            # Convert binary labels to hierarchical
            active_indices = torch.where(binary_labels > 0)[0].tolist()
            active_tags = [self.vocab.get_tag_from_index(idx) for idx in active_indices]
            hierarchical_labels = self.vocab.encode_tags_hierarchical(active_tags)
        
        # Create hierarchical tensor (num_groups, tags_per_group)
        hierarchical_tensor = torch.zeros(
            self.vocab.num_groups, 
            self.vocab.tags_per_group, 
            dtype=torch.float32
        )
        
        for group_id, group_indices in hierarchical_labels.items():
            for group_idx in group_indices:
                hierarchical_tensor[group_id, group_idx] = 1.0
        
        return {
            'binary': binary_labels,
            'hierarchical': hierarchical_tensor,
            'num_tags': len([idx for idx in binary_labels if idx > 0])
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample"""
        
        # Load data from HDF5
        data = self._load_from_cache_or_hdf5(idx)
        
        # Convert image to float and normalize
        image = data['image'].astype(np.float32) / 255.0  # Convert uint8 to float
        image = torch.from_numpy(image)
        
        # Apply normalization
        image = self.normalize(image)
        
        # Apply augmentation if enabled
        if self.augmentation is not None and self.split == 'train':
            image = self.augmentation(image)
        
        # Create tag labels
        labels = self._create_tag_labels(data['image_path'], data['teacher_features'])
        
        # Prepare output
        sample = {
            'image': image,
            'labels': labels,
            'metadata': {
                'index': idx,
                'path': data['image_path'],
                'padding_info': data['padding_info'],
                'original_size': data['original_size']
            }
        }
        
        # Add teacher features if available
        if data['teacher_features']:
            # Convert numpy arrays to tensors
            teacher_tensors = {}
            
            if 'anime' in data['teacher_features']:
                anime_tensors = {}
                for key, value in data['teacher_features']['anime'].items():
                    if isinstance(value, np.ndarray):
                        anime_tensors[key] = torch.from_numpy(value)
                    else:
                        anime_tensors[key] = value
                teacher_tensors['anime'] = anime_tensors
            
            if 'clip' in data['teacher_features']:
                clip_tensors = {}
                for key, value in data['teacher_features']['clip'].items():
                    if isinstance(value, np.ndarray):
                        clip_tensors[key] = torch.from_numpy(value)
                    else:
                        clip_tensors[key] = value
                teacher_tensors['clip'] = clip_tensors
            
            sample['teacher_features'] = teacher_tensors
        
        return sample


class HDF5DataLoader:
    """Factory for creating DataLoaders with proper configuration"""
    
    @staticmethod
    def create_train_loader(
        config: HDF5DataConfig,
        tag_files: Optional[List[Path]] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True
    ) -> DataLoader:
        """Create training dataloader"""
        
        # Create dataset
        dataset = HDF5Dataset(config, tag_files, split='train')
        
        # Create sampler for distributed training
        sampler = None
        if config.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                drop_last=drop_last
            )
            shuffle = False
        else:
            shuffle = True
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        return dataloader
    
    @staticmethod
    def create_val_loader(
        config: HDF5DataConfig,
        tag_files: Optional[List[Path]] = None,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create validation dataloader"""
        
        # Disable augmentation for validation
        val_config = HDF5DataConfig(**config.__dict__)
        val_config.augmentation_enabled = False
        
        # Create dataset
        dataset = HDF5Dataset(val_config, tag_files, split='val')
        
        # Create dataloader (no shuffling for validation)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return dataloader


class RegionCropDataset(HDF5Dataset):
    """Extended dataset that supports region-based cropping"""
    
    def __init__(self, 
                 config: HDF5DataConfig,
                 tag_files: Optional[List[Path]] = None,
                 split: str = 'train',
                 region_crop_prob: float = 0.5,
                 min_crop_size: int = 384):
        
        super().__init__(config, tag_files, split)
        self.region_crop_prob = region_crop_prob
        self.min_crop_size = min_crop_size
    
    def _extract_region_crop(self, image: torch.Tensor, padding_info: np.ndarray) -> torch.Tensor:
        """Extract a random crop from the actual image region"""
        x1, y1, x2, y2 = padding_info
        
        # Calculate actual image size within padding
        img_width = x2 - x1
        img_height = y2 - y1
        
        # Determine crop size
        min_dim = min(img_width, img_height)
        crop_size = random.randint(self.min_crop_size, min(min_dim, 640))
        
        # Random position within actual image area
        if img_width > crop_size:
            crop_x = random.randint(x1, x2 - crop_size)
        else:
            crop_x = x1
            
        if img_height > crop_size:
            crop_y = random.randint(y1, y2 - crop_size)
        else:
            crop_y = y1
        
        # Crop and resize to 640x640
        cropped = TF.crop(image, crop_y, crop_x, crop_size, crop_size)
        resized = TF.resize(cropped, (640, 640), interpolation=T.InterpolationMode.LANCZOS)
        
        return resized
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with optional region cropping"""
        sample = super().__getitem__(idx)
        
        # Apply region cropping in training
        if (self.split == 'train' and 
            random.random() < self.region_crop_prob and 
            'padding_info' in sample['metadata']):
            
            # Extract region crop
            sample['image'] = self._extract_region_crop(
                sample['image'], 
                sample['metadata']['padding_info']
            )
            
            # Mark as cropped
            sample['metadata']['is_cropped'] = True
        
        return sample


# Utility functions for creating dataloaders
def create_dataloaders(
    hdf5_dir: Path,
    vocab_dir: Path,
    tag_files: Optional[List[Path]] = None,
    batch_size: int = 32,
    num_workers: int = 8,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    use_region_crops: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Create config
    config = HDF5DataConfig(
        hdf5_dir=hdf5_dir,
        vocab_dir=vocab_dir,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )
    
    # Create loaders
    if use_region_crops:
        # Use region crop dataset for training
        train_dataset = RegionCropDataset(config, tag_files, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not distributed,
            sampler=DistributedSampler(train_dataset) if distributed else None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
    else:
        train_loader = HDF5DataLoader.create_train_loader(
            config, tag_files, batch_size, num_workers
        )
    
    val_loader = HDF5DataLoader.create_val_loader(
        config, tag_files, batch_size * 2, num_workers
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataloader
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_dir', type=str, required=True)
    parser.add_argument('--vocab_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Create config
    config = HDF5DataConfig(
        hdf5_dir=Path(args.hdf5_dir),
        vocab_dir=Path(args.vocab_dir)
    )
    
    # Create dataloader
    train_loader = HDF5DataLoader.create_train_loader(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Test loading
    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"Number of batches: {len(train_loader)}")
    
    # Load a few batches
    for i, batch in enumerate(tqdm(train_loader, desc="Testing dataloader")):
        print(f"\nBatch {i}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Binary labels shape: {batch['labels']['binary'].shape}")
        print(f"  Hierarchical labels shape: {batch['labels']['hierarchical'].shape}")
        print(f"  Avg tags per image: {batch['labels']['num_tags'].float().mean():.1f}")
        
        if 'teacher_features' in batch:
            print(f"  Teacher features included: {list(batch['teacher_features'].keys())}")
        
        if i >= 2:
            break
    
    print("\nDataloader test complete!")
