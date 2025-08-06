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
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import h5py
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# These imports are assumed to exist in your project
from tag_vocabulary import TagVocabulary, load_vocabulary_for_training
from model_architecture import create_model

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
    
    # Augmentation settings
    augmentation_enabled: bool = False
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    random_flip_prob: float = 0.5
    color_jitter: bool = False
    
    # Multi-GPU settings
    distributed: bool = False
    rank: int = 0
    world_size: int = 1
    
    # Teacher vocabulary mapping
    teacher_vocab_path: Optional[Path] = None  # Path to teacher vocabulary mapping


class HDF5FileManager:
    """Manages multiple HDF5 files efficiently"""
    
    def __init__(self, hdf5_files: List[Path], preload_files: int = 2):
        self.hdf5_files = sorted(hdf5_files)
        self.preload_files = max(1, preload_files)  # Ensure at least 1 file
        self.file_handles: Dict[str, h5py.File] = {}
        self.file_info: Dict[str, Dict] = {}
        self.access_order = []  # Track access order for LRU
        
        # Load index information
        self._load_file_info()
        
        # Calculate total samples
        self.total_samples = sum(info['num_samples'] for info in self.file_info.values())
        
        if self.total_samples == 0:
            raise ValueError("No samples found in HDF5 files")
        
        # Create global index mapping
        self._create_index_mapping()
        
    def _load_file_info(self):
        """Load metadata about each HDF5 file"""
        for file_path in self.hdf5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    # Validate required datasets exist
                    if 'images/student_640' not in f:
                        logger.warning(f"Skipping {file_path}: missing images/student_640")
                        continue
                    
                    # Get number of samples
                    num_samples = f['images/student_640'].shape[0]
                    
                    if num_samples == 0:
                        logger.warning(f"Skipping {file_path}: no samples")
                        continue
                    
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
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
                
        if not self.file_info:
            raise ValueError("No valid HDF5 files found")
            
        logger.info(f"Loaded info for {len(self.file_info)} HDF5 files, "
                   f"total samples: {sum(info['num_samples'] for info in self.file_info.values())}")
    
    def _create_index_mapping(self):
        """Create mapping from global index to (file, local_index)"""
        self.index_to_file = []
        
        for file_path in sorted(self.file_info.keys()):  # Ensure consistent ordering
            num_samples = self.file_info[file_path]['num_samples']
            for local_idx in range(num_samples):
                self.index_to_file.append((file_path, local_idx))
    
    def get_file_handle(self, file_path: str) -> h5py.File:
        """Get file handle, opening if necessary with LRU eviction"""
        if file_path not in self.file_handles:
            # Close oldest file if we have too many open
            if len(self.file_handles) >= self.preload_files:
                # Remove least recently used
                lru_file = None
                for path in self.access_order:
                    if path != file_path and path in self.file_handles:
                        lru_file = path
                        break
                
                if lru_file:
                    try:
                        self.file_handles[lru_file].close()
                    except:
                        pass
                    del self.file_handles[lru_file]
                    self.access_order.remove(lru_file)
            
            # Open new file
            try:
                self.file_handles[file_path] = h5py.File(file_path, 'r')
            except Exception as e:
                logger.error(f"Failed to open {file_path}: {e}")
                raise
        
        # Update access order
        if file_path in self.access_order:
            self.access_order.remove(file_path)
        self.access_order.append(file_path)
        
        return self.file_handles[file_path]
    
    def get_sample(self, global_idx: int) -> Tuple[str, int, h5py.File]:
        """Get file handle and local index for global index"""
        if global_idx >= len(self.index_to_file):
            raise IndexError(f"Index {global_idx} out of range (max: {len(self.index_to_file)-1})")
        
        file_path, local_idx = self.index_to_file[global_idx]
        file_handle = self.get_file_handle(file_path)
        return file_path, local_idx, file_handle
    
    def close_all(self):
        """Close all open file handles"""
        for f in self.file_handles.values():
            try:
                f.close()
            except:
                pass
        self.file_handles.clear()
        self.access_order.clear()
    
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
        
        # Load teacher vocabulary mapping if provided
        self.teacher_to_vocab_mapping = {}
        if config.teacher_vocab_path and config.teacher_vocab_path.exists():
            self._load_teacher_mapping(config.teacher_vocab_path)
        
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
        # More accurate cache size calculation
        bytes_per_sample = 3 * 640 * 640 * 4  # float32
        self.cache_size = int(config.cache_size_gb * 1024**3 / bytes_per_sample)
        
    def _load_teacher_mapping(self, mapping_path: Path):
        """Load mapping from teacher vocabulary to our vocabulary"""
        try:
            with open(mapping_path, 'r') as f:
                mapping_data = json.load(f)
            
            # Expected format: {"teacher_tag": "our_tag"} or {"teacher_idx": our_idx}
            for teacher_key, our_key in mapping_data.items():
                if isinstance(teacher_key, str):
                    # Tag name mapping
                    teacher_idx = int(teacher_key) if teacher_key.isdigit() else None
                    if teacher_idx is not None:
                        our_idx = self.vocab.tag_to_index.get(our_key, self.vocab.unk_index)
                        self.teacher_to_vocab_mapping[teacher_idx] = our_idx
                else:
                    # Direct index mapping
                    self.teacher_to_vocab_mapping[int(teacher_key)] = int(our_key)
            
            logger.info(f"Loaded teacher mapping with {len(self.teacher_to_vocab_mapping)} entries")
        except Exception as e:
            logger.error(f"Failed to load teacher mapping: {e}")
    
    def _load_tag_annotations(self, tag_files: List[Path]):
        """Load tag annotations from JSON files"""
        logger.info(f"Loading tag annotations from {len(tag_files)} files")
        
        for tag_file in tag_files:
            try:
                with open(tag_file, 'r') as f:
                    data = json.load(f)
                
                # Expected format: {"image_path": ["tag1", "tag2", ...]}
                for image_path, tags in data.items():
                    # Ensure tags is a list
                    if not isinstance(tags, list):
                        logger.warning(f"Invalid tag format for {image_path}")
                        continue
                    
                    # Convert tags to indices safely
                    tag_indices = []
                    for tag in tags:
                        idx = self.vocab.tag_to_index.get(tag, self.vocab.unk_index)
                        if idx != self.vocab.unk_index:
                            tag_indices.append(idx)
                    
                    # Convert to hierarchical format if the method exists
                    hierarchical_tags = {}
                    if hasattr(self.vocab, 'encode_tags_hierarchical'):
                        hierarchical_tags = self.vocab.encode_tags_hierarchical(tags)
                    
                    self.tag_annotations[image_path] = {
                        'tags': tags,
                        'indices': tag_indices,
                        'hierarchical': hierarchical_tags
                    }
                    
            except Exception as e:
                logger.error(f"Error loading {tag_file}: {e}")
                traceback.print_exc()
        
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
    
    def _decode_string(self, data):
        """Safely decode bytes to string if needed"""
        if isinstance(data, bytes):
            return data.decode('utf-8', errors='ignore')
        elif isinstance(data, np.ndarray):
            # Handle numpy string arrays
            return str(data)
        return str(data)
    
    def _load_from_cache_or_hdf5(self, idx: int) -> Dict[str, Any]:
        """Load data from cache or HDF5"""
        if idx in self.cache:
            return self.cache[idx]
        
        try:
            # Get file info
            file_path, local_idx, h5_file = self.file_manager.get_sample(idx)
            
            # Load image (stored as uint8)
            image = h5_file['images/student_640'][local_idx]  # (3, 640, 640)
            
            # Load metadata with proper error handling
            padding_info = h5_file['images/padding_info'][local_idx] if 'images/padding_info' in h5_file else np.array([0, 0, 640, 640])
            original_size = h5_file['images/original_sizes'][local_idx] if 'images/original_sizes' in h5_file else np.array([640, 640])
            
            # Handle image path - might be bytes
            if 'metadata/paths' in h5_file:
                image_path = self._decode_string(h5_file['metadata/paths'][local_idx])
            else:
                image_path = f"image_{idx}"
            
            # Load teacher features if available and not dropped
            teacher_features = {}
            if self.config.use_teacher_features and random.random() > self.config.teacher_feature_dropout:
                # Anime teacher features
                if 'anime_teacher' in h5_file:
                    anime_features = {}
                    for key in ['tag_logits', 'tag_probs', 'top_k_indices', 'top_k_probs']:
                        full_key = f'anime_teacher/{key}'
                        if full_key in h5_file:
                            anime_features[key] = np.array(h5_file[full_key][local_idx])
                    if anime_features:
                        teacher_features['anime'] = anime_features
                
                # CLIP features
                if 'clip_teacher' in h5_file:
                    clip_features = {}
                    for key in ['cls_token', 'patch_tokens_avg', 'top_patches']:
                        full_key = f'clip_teacher/{key}'
                        if full_key in h5_file:
                            clip_features[key] = np.array(h5_file[full_key][local_idx])
                    if clip_features:
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
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample to avoid crashing
            return {
                'image': np.zeros((3, 640, 640), dtype=np.uint8),
                'padding_info': np.array([0, 0, 640, 640]),
                'original_size': np.array([640, 640]),
                'image_path': f"error_{idx}",
                'teacher_features': {}
            }
    
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
                if 0 <= idx < num_tags:  # Bounds check
                    binary_labels[idx] = 1.0
            
            # Set hierarchical labels
            hierarchical_labels = anno.get('hierarchical', {})
        
        # Otherwise use teacher predictions
        elif 'anime' in teacher_features and teacher_features['anime']:
            # Use anime teacher predictions
            if 'top_k_indices' in teacher_features['anime'] and 'top_k_probs' in teacher_features['anime']:
                top_k_indices = teacher_features['anime']['top_k_indices']
                top_k_probs = teacher_features['anime']['top_k_probs']
                
                # Threshold predictions
                threshold = 0.5
                for teacher_idx, prob in zip(top_k_indices, top_k_probs):
                    if prob > threshold:
                        # Map teacher index to our vocabulary
                        if self.teacher_to_vocab_mapping:
                            our_idx = self.teacher_to_vocab_mapping.get(int(teacher_idx), -1)
                        else:
                            # Fallback: assume direct mapping if within range
                            our_idx = int(teacher_idx) if int(teacher_idx) < num_tags else -1
                        
                        if 0 <= our_idx < num_tags:
                            binary_labels[our_idx] = float(prob)
        
        # Convert to hierarchical format if needed and method exists
        if not hierarchical_labels and hasattr(self.vocab, 'encode_tags_hierarchical'):
            # Convert binary labels to hierarchical
            active_indices = torch.where(binary_labels > 0.5)[0].tolist()
            active_tags = []
            for idx in active_indices:
                if hasattr(self.vocab, 'index_to_tag') and idx in self.vocab.index_to_tag:
                    active_tags.append(self.vocab.index_to_tag[idx])
            
            if active_tags:
                hierarchical_labels = self.vocab.encode_tags_hierarchical(active_tags)
        
        # Create hierarchical tensor if we have the vocab structure
        if hasattr(self.vocab, 'num_groups') and hasattr(self.vocab, 'tags_per_group'):
            hierarchical_tensor = torch.zeros(
                self.vocab.num_groups, 
                self.vocab.tags_per_group, 
                dtype=torch.float32
            )
            
            for group_id, group_indices in hierarchical_labels.items():
                if isinstance(group_indices, (list, tuple)):
                    for group_idx in group_indices:
                        if 0 <= group_id < self.vocab.num_groups and 0 <= group_idx < self.vocab.tags_per_group:
                            hierarchical_tensor[group_id, group_idx] = 1.0
        else:
            # Fallback: create dummy hierarchical tensor
            hierarchical_tensor = torch.zeros(1, num_tags, dtype=torch.float32)
            hierarchical_tensor[0] = binary_labels
        
        return {
            'binary': binary_labels,
            'hierarchical': hierarchical_tensor,
            'num_tags': (binary_labels > 0.5).sum().item()
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample"""
        try:
            # Load data from HDF5
            data = self._load_from_cache_or_hdf5(idx)
            
            # Convert image to float and normalize
            image = data['image'].astype(np.float32) / 255.0  # Convert uint8 to float
            image = torch.from_numpy(image.copy())  # Use copy to ensure contiguous memory
            
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
                    'padding_info': torch.from_numpy(data['padding_info'].astype(np.float32)),
                    'original_size': torch.from_numpy(data['original_size'].astype(np.float32))
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
                            anime_tensors[key] = torch.from_numpy(value.astype(np.float32))
                        else:
                            anime_tensors[key] = value
                    teacher_tensors['anime'] = anime_tensors
                
                if 'clip' in data['teacher_features']:
                    clip_tensors = {}
                    for key, value in data['teacher_features']['clip'].items():
                        if isinstance(value, np.ndarray):
                            clip_tensors[key] = torch.from_numpy(value.astype(np.float32))
                        else:
                            clip_tensors[key] = value
                    teacher_tensors['clip'] = clip_tensors
                
                sample['teacher_features'] = teacher_tensors
            
            return sample
            
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {e}")
            # Return a valid dummy sample to avoid crashing the dataloader
            return {
                'image': torch.zeros(3, 640, 640, dtype=torch.float32),
                'labels': {
                    'binary': torch.zeros(len(self.vocab.tag_to_index), dtype=torch.float32),
                    'hierarchical': torch.zeros(1, len(self.vocab.tag_to_index), dtype=torch.float32),
                    'num_tags': 0
                },
                'metadata': {
                    'index': idx,
                    'path': f"error_{idx}",
                    'padding_info': torch.zeros(4, dtype=torch.float32),
                    'original_size': torch.tensor([640.0, 640.0], dtype=torch.float32)
                }
            }


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
        
        # Create dataloader with proper worker settings
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=min(num_workers, os.cpu_count() or 1),  # Don't exceed CPU count
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=lambda w: np.random.seed(np.random.get_state()[1][0] + w)  # Proper worker seeding
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
        val_config = HDF5DataConfig(**{k: v for k, v in config.__dict__.items()})
        val_config.augmentation_enabled = False
        val_config.teacher_feature_dropout = 0.0  # No dropout during validation
        
        # Create dataset
        dataset = HDF5Dataset(val_config, tag_files, split='val')
        
        # Create dataloader (no shuffling for validation)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(num_workers, os.cpu_count() or 1),
            pin_memory=pin_memory and torch.cuda.is_available(),
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
    
    def _extract_region_crop(self, image: torch.Tensor, padding_info: torch.Tensor) -> torch.Tensor:
        """Extract a random crop from the actual image region"""
        # Convert padding_info to integers
        x1, y1, x2, y2 = padding_info.int().tolist()
        
        # Ensure valid bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(640, x2), min(640, y2)
        
        # Calculate actual image size within padding
        img_width = x2 - x1
        img_height = y2 - y1
        
        if img_width <= 0 or img_height <= 0:
            return image  # Return original if invalid padding
        
        # Determine crop size
        min_dim = min(img_width, img_height)
        if min_dim < self.min_crop_size:
            return image  # Return original if too small to crop
        
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
        try:
            cropped = TF.crop(image, crop_y, crop_x, crop_size, crop_size)
            resized = TF.resize(cropped, (640, 640), interpolation=T.InterpolationMode.LANCZOS)
            return resized
        except Exception as e:
            logger.warning(f"Crop failed: {e}, returning original")
            return image
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with optional region cropping"""
        sample = super().__getitem__(idx)
        
        # Apply region cropping in training
        if (self.split == 'train' and 
            random.random() < self.region_crop_prob and 
            'padding_info' in sample['metadata']):
            
            # Extract region crop
            cropped_image = self._extract_region_crop(
                sample['image'], 
                sample['metadata']['padding_info']
            )
            sample['image'] = cropped_image
            
            # Mark as cropped
            sample['metadata']['is_cropped'] = True
        else:
            sample['metadata']['is_cropped'] = False
        
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
    use_region_crops: bool = False,
    teacher_vocab_path: Optional[Path] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Create config
    config = HDF5DataConfig(
        hdf5_dir=hdf5_dir,
        vocab_dir=vocab_dir,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        teacher_vocab_path=teacher_vocab_path
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
            num_workers=min(num_workers, os.cpu_count() or 1),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        train_loader = HDF5DataLoader.create_train_loader(
            config, tag_files, batch_size, num_workers
        )
    
    val_loader = HDF5DataLoader.create_val_loader(
        config, tag_files, batch_size * 2, num_workers
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function for handling variable-sized data"""
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Stack labels
    binary_labels = torch.stack([item['labels']['binary'] for item in batch])
    hierarchical_labels = torch.stack([item['labels']['hierarchical'] for item in batch])
    num_tags = torch.tensor([item['labels']['num_tags'] for item in batch])
    
    # Collect metadata
    metadata = {
        'indices': [item['metadata']['index'] for item in batch],
        'paths': [item['metadata']['path'] for item in batch],
        'padding_info': torch.stack([item['metadata']['padding_info'] for item in batch]),
        'original_sizes': torch.stack([item['metadata']['original_size'] for item in batch])
    }
    
    # Handle optional cropping flag
    if 'is_cropped' in batch[0]['metadata']:
        metadata['is_cropped'] = torch.tensor([item['metadata'].get('is_cropped', False) for item in batch])
    
    # Collect teacher features if present
    teacher_features = {}
    if 'teacher_features' in batch[0] and batch[0]['teacher_features']:
        # Organize by teacher type
        for teacher_type in batch[0]['teacher_features'].keys():
            teacher_features[teacher_type] = {}
            for key in batch[0]['teacher_features'][teacher_type].keys():
                features = []
                for item in batch:
                    if 'teacher_features' in item and teacher_type in item['teacher_features']:
                        features.append(item['teacher_features'][teacher_type][key])
                if features:
                    teacher_features[teacher_type][key] = torch.stack(features)
    
    result = {
        'image': images,
        'labels': {
            'binary': binary_labels,
            'hierarchical': hierarchical_labels,
            'num_tags': num_tags
        },
        'metadata': metadata
    }
    
    if teacher_features:
        result['teacher_features'] = teacher_features
    
    return result


if __name__ == "__main__":
    # Test the dataloader
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_dir', type=str, required=True, help='Directory containing HDF5 files')
    parser.add_argument('--vocab_dir', type=str, required=True, help='Directory containing vocabulary files')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_region_crops', action='store_true', help='Use region-based cropping')
    parser.add_argument('--teacher_vocab_path', type=str, help='Path to teacher vocabulary mapping')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create config
    config = HDF5DataConfig(
        hdf5_dir=Path(args.hdf5_dir),
        vocab_dir=Path(args.vocab_dir),
        teacher_vocab_path=Path(args.teacher_vocab_path) if args.teacher_vocab_path else None
    )
    
    # Create dataloader with custom collate function
    if args.use_region_crops:
        dataset = RegionCropDataset(config, split='train')
    else:
        dataset = HDF5Dataset(config, split='train')
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # Test loading
    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"Number of batches: {len(train_loader)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load a few batches
    for i, batch in enumerate(tqdm(train_loader, desc="Testing dataloader")):
        print(f"\nBatch {i}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Image dtype: {batch['image'].dtype}")
        print(f"  Image range: [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")
        print(f"  Binary labels shape: {batch['labels']['binary'].shape}")
        print(f"  Hierarchical labels shape: {batch['labels']['hierarchical'].shape}")
        print(f"  Avg tags per image: {batch['labels']['num_tags'].float().mean():.1f}")
        print(f"  Paths sample: {batch['metadata']['paths'][:2]}")
        
        if 'teacher_features' in batch:
            print(f"  Teacher features included: {list(batch['teacher_features'].keys())}")
            for teacher_type, features in batch['teacher_features'].items():
                print(f"    {teacher_type}: {list(features.keys())}")
        
        if 'is_cropped' in batch['metadata']:
            print(f"  Cropped samples: {batch['metadata']['is_cropped'].sum().item()}/{args.batch_size}")
        
        if i >= 2:
            break
    
    print("\n" + "="*50)
    print("Dataloader test complete!")
    print("="*50)