#!/usr/bin/env python3
"""
Simplified DataLoader for Direct Training (No Teacher Distillation)
Loads images and tags directly from JSON annotations
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import random
from collections import defaultdict, Counter
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, WeightedRandomSampler
import h5py
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedDataConfig:
    """Configuration for simplified dataset"""
    data_dir: Path  # Directory with images
    json_dir: Path  # Directory with JSON annotations
    vocab_path: Path  # Path to vocabulary file
    
    # Image settings
    image_size: int = 640
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Training settings
    top_k_tags: int = 100000  # Use top 100k most frequent tags
    min_tag_frequency: int = 100  # Minimum occurrences to include tag
    
    # Augmentation settings
    augmentation_enabled: bool = True
    random_flip_prob: float = 0.5
    color_jitter: bool = True
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    
    # Sampling settings
    frequency_weighted_sampling: bool = True
    sample_weight_power: float = 0.5  # Power for frequency weighting
    
    # Multi-GPU settings
    distributed: bool = False
    rank: int = 0
    world_size: int = 1
    
    # Cache settings
    cache_size_gb: float = 8.0  # Cache frequently accessed data
    preload_metadata: bool = True  # Preload all JSON metadata


class TagVocabulary:
    """Simplified tag vocabulary manager"""
    
    def __init__(self, vocab_path: Optional[Path] = None, min_frequency: int = 100):
        self.tag_to_index = {}
        self.index_to_tag = {}
        self.tag_frequencies = {}
        self.min_frequency = min_frequency
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        
        # Rating classes
        self.rating_to_index = {
            "general": 0,
            "sensitive": 1,
            "questionable": 2,
            "explicit": 3,
            "unknown": 4
        }
        self.index_to_rating = {v: k for k, v in self.rating_to_index.items()}
        
        if vocab_path and vocab_path.exists():
            self.load_vocabulary(vocab_path)
    
    def build_from_annotations(self, json_files: List[Path], top_k: int = 100000):
        """Build vocabulary from JSON annotation files"""
        logger.info(f"Building vocabulary from {len(json_files)} files")
        
        # Count tag frequencies
        tag_counter = Counter()
        rating_counter = Counter()
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for entry in data:
                    if 'tags' in entry:
                        tags = entry['tags'].split() if isinstance(entry['tags'], str) else entry['tags']
                        tag_counter.update(tags)
                    
                    if 'rating' in entry:
                        rating_counter[entry['rating']] += 1
                        
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        # Select top-k most frequent tags
        most_common = tag_counter.most_common(top_k)
        
        # Build vocabulary
        self.tag_to_index = {self.pad_token: 0, self.unk_token: 1}
        current_index = 2
        
        for tag, freq in most_common:
            if freq >= self.min_frequency:
                self.tag_to_index[tag] = current_index
                self.tag_frequencies[tag] = freq
                current_index += 1
        
        self.index_to_tag = {v: k for k, v in self.tag_to_index.items()}
        
        logger.info(f"Built vocabulary with {len(self.tag_to_index)} tags")
        logger.info(f"Rating distribution: {dict(rating_counter)}")
    
    def save_vocabulary(self, path: Path):
        """Save vocabulary to file"""
        vocab_data = {
            'tag_to_index': self.tag_to_index,
            'tag_frequencies': self.tag_frequencies,
            'rating_to_index': self.rating_to_index,
            'min_frequency': self.min_frequency
        }
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Saved vocabulary to {path}")
    
    def load_vocabulary(self, path: Path):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.tag_to_index = vocab_data['tag_to_index']
        self.tag_frequencies = vocab_data.get('tag_frequencies', {})
        self.rating_to_index = vocab_data.get('rating_to_index', self.rating_to_index)
        self.min_frequency = vocab_data.get('min_frequency', 100)
        
        self.index_to_tag = {int(v): k for k, v in self.tag_to_index.items()}
        self.index_to_rating = {v: k for k, v in self.rating_to_index.items()}
        
        logger.info(f"Loaded vocabulary with {len(self.tag_to_index)} tags")
    
    def encode_tags(self, tags: List[str]) -> torch.Tensor:
        """Encode tags to binary vector"""
        num_tags = len(self.tag_to_index)
        encoded = torch.zeros(num_tags, dtype=torch.float32)
        
        for tag in tags:
            idx = self.tag_to_index.get(tag, self.tag_to_index[self.unk_token])
            if idx != self.tag_to_index[self.unk_token]:
                encoded[idx] = 1.0
        
        return encoded
    
    def decode_tags(self, encoded: torch.Tensor, threshold: float = 0.5) -> List[str]:
        """Decode binary vector to tags"""
        indices = torch.where(encoded > threshold)[0].tolist()
        tags = []
        
        for idx in indices:
            if idx in self.index_to_tag and idx != self.tag_to_index[self.unk_token]:
                tags.append(self.index_to_tag[idx])
        
        return tags


class SimplifiedDataset(Dataset):
    """Simplified dataset for direct training"""
    
    def __init__(self, 
                 config: SimplifiedDataConfig,
                 json_files: List[Path],
                 split: str = 'train',
                 vocab: Optional[TagVocabulary] = None):
        
        self.config = config
        self.split = split
        
        # Load or build vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = TagVocabulary(config.vocab_path, config.min_tag_frequency)
            if not config.vocab_path.exists():
                self.vocab.build_from_annotations(json_files, config.top_k_tags)
                self.vocab.save_vocabulary(config.vocab_path)
        
        logger.info(f"Using vocabulary with {len(self.vocab.tag_to_index)} tags")
        
        # Load annotations
        self.annotations = []
        self._load_annotations(json_files)
        
        # Setup transformations
        self.normalize = T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        self.augmentation = self._setup_augmentation() if config.augmentation_enabled and split == 'train' else None
        
        # Create sample weights for frequency-based sampling
        self.sample_weights = None
        if config.frequency_weighted_sampling and split == 'train':
            self._calculate_sample_weights()
        
        # Cache for frequently accessed images
        self.cache = {}
        bytes_per_image = 3 * config.image_size * config.image_size * 4  # float32
        self.max_cache_size = int(config.cache_size_gb * 1024**3 / bytes_per_image)
        
        logger.info(f"Dataset initialized with {len(self.annotations)} samples")
    
    def _load_annotations(self, json_files: List[Path]):
        """Load and filter annotations"""
        logger.info(f"Loading annotations from {len(json_files)} files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for entry in data:
                    # Skip if missing required fields
                    if 'filename' not in entry or 'tags' not in entry:
                        continue
                    
                    # Parse tags
                    tags = entry['tags'].split() if isinstance(entry['tags'], str) else entry['tags']
                    
                    # Filter tags by vocabulary
                    valid_tags = [t for t in tags if t in self.vocab.tag_to_index]
                    
                    # Skip if no valid tags
                    if not valid_tags:
                        continue
                    
                    # Get rating
                    rating = entry.get('rating', 'unknown')
                    if rating not in self.vocab.rating_to_index:
                        rating = 'unknown'
                    
                    # Build image path
                    image_path = self.config.data_dir / entry['filename']
                    
                    # Skip if image doesn't exist
                    if not image_path.exists():
                        continue
                    
                    self.annotations.append({
                        'image_path': str(image_path),
                        'tags': valid_tags,
                        'rating': rating,
                        'num_tags': len(valid_tags)
                    })
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.annotations)} valid annotations")
    
    def _calculate_sample_weights(self):
        """Calculate sample weights based on tag frequencies"""
        logger.info("Calculating sample weights for frequency-based sampling")
        
        weights = []
        for anno in self.annotations:
            # Calculate weight based on inverse frequency of tags
            weight = 0.0
            for tag in anno['tags']:
                if tag in self.vocab.tag_frequencies:
                    # Inverse frequency with power scaling
                    freq = self.vocab.tag_frequencies[tag]
                    weight += (1.0 / freq) ** self.config.sample_weight_power
            
            # Average weight across tags
            weight = weight / max(1, len(anno['tags']))
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        self.sample_weights = weights
        logger.info(f"Sample weights calculated, range: [{weights.min():.6f}, {weights.max():.6f}]")
    
    def _setup_augmentation(self):
        """Setup augmentation transforms"""
        transforms = []
        
        # Random horizontal flip
        if self.config.random_flip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(p=self.config.random_flip_prob))
        
        # Random resized crop
        if self.config.random_crop_scale != (1.0, 1.0):
            transforms.append(T.RandomResizedCrop(
                self.config.image_size,
                scale=self.config.random_crop_scale,
                ratio=(0.9, 1.1),
                interpolation=T.InterpolationMode.LANCZOS
            ))
        
        # Color jitter
        if self.config.color_jitter:
            transforms.append(T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02
            ))
        
        return T.Compose(transforms) if transforms else None
    
    def __len__(self):
        return len(self.annotations)
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        # Check cache
        if image_path in self.cache:
            return self.cache[image_path].clone()
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize to target size
            image = TF.resize(image, (self.config.image_size, self.config.image_size), 
                            interpolation=T.InterpolationMode.LANCZOS)
            
            # Convert to tensor
            image = TF.to_tensor(image)
            
            # Cache if space available
            if len(self.cache) < self.max_cache_size:
                self.cache[image_path] = image.clone()
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return black image on error
            return torch.zeros(3, self.config.image_size, self.config.image_size)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample"""
        try:
            # Get annotation
            anno = self.annotations[idx]
            
            # Load image
            image = self._load_image(anno['image_path'])
            
            # Apply augmentation
            if self.augmentation is not None:
                image = self.augmentation(image)
            
            # Normalize
            image = self.normalize(image)
            
            # Encode tags
            tag_labels = self.vocab.encode_tags(anno['tags'])
            
            # Encode rating
            rating_label = self.vocab.rating_to_index[anno['rating']]
            
            return {
                'image': image,
                'tag_labels': tag_labels,
                'rating_label': torch.tensor(rating_label, dtype=torch.long),
                'metadata': {
                    'index': idx,
                    'path': anno['image_path'],
                    'num_tags': anno['num_tags'],
                    'tags': anno['tags'],
                    'rating': anno['rating']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {e}")
            # Return dummy sample
            return {
                'image': torch.zeros(3, self.config.image_size, self.config.image_size),
                'tag_labels': torch.zeros(len(self.vocab.tag_to_index)),
                'rating_label': torch.tensor(4, dtype=torch.long),  # unknown
                'metadata': {
                    'index': idx,
                    'path': f"error_{idx}",
                    'num_tags': 0,
                    'tags': [],
                    'rating': 'unknown'
                }
            }


class FrequencyBasedSampler:
    """Sampler that balances common and rare tags"""
    
    def __init__(self, dataset: SimplifiedDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Group samples by tag frequency buckets
        self.frequency_buckets = self._create_frequency_buckets()
    
    def _create_frequency_buckets(self):
        """Group samples into frequency buckets"""
        buckets = {
            'rare': [],      # < 1k occurrences
            'uncommon': [],  # 1k - 10k
            'common': [],    # 10k - 100k
            'very_common': [] # > 100k
        }
        
        for idx, anno in enumerate(self.dataset.annotations):
            avg_freq = 0
            for tag in anno['tags']:
                if tag in self.dataset.vocab.tag_frequencies:
                    avg_freq += self.dataset.vocab.tag_frequencies[tag]
            
            avg_freq = avg_freq / max(1, len(anno['tags']))
            
            if avg_freq < 1000:
                buckets['rare'].append(idx)
            elif avg_freq < 10000:
                buckets['uncommon'].append(idx)
            elif avg_freq < 100000:
                buckets['common'].append(idx)
            else:
                buckets['very_common'].append(idx)
        
        # Log distribution
        for bucket, indices in buckets.items():
            logger.info(f"Frequency bucket '{bucket}': {len(indices)} samples")
        
        return buckets
    
    def get_balanced_batch_sampler(self):
        """Create a batch sampler that balances frequencies"""
        # Sample proportionally from each bucket
        proportions = {
            'rare': 0.3,
            'uncommon': 0.3,
            'common': 0.25,
            'very_common': 0.15
        }
        
        batch_indices = []
        
        for bucket, prop in proportions.items():
            if self.frequency_buckets[bucket]:
                n_samples = int(self.batch_size * prop)
                indices = random.sample(self.frequency_buckets[bucket], 
                                      min(n_samples, len(self.frequency_buckets[bucket])))
                batch_indices.extend(indices)
        
        return batch_indices


def create_dataloaders(
    data_dir: Path,
    json_dir: Path,
    vocab_path: Path,
    batch_size: int = 96,
    num_workers: int = 8,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    frequency_sampling: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Discover JSON files
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    # Split into train/val (90/10 split)
    random.shuffle(json_files)
    split_idx = int(len(json_files) * 0.9)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]
    
    # Create config
    config = SimplifiedDataConfig(
        data_dir=data_dir,
        json_dir=json_dir,
        vocab_path=vocab_path,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        frequency_weighted_sampling=frequency_sampling
    )
    
    # Build vocabulary once
    vocab = TagVocabulary(vocab_path)
    if not vocab_path.exists():
        vocab.build_from_annotations(json_files, config.top_k_tags)
        vocab.save_vocabulary(vocab_path)
    
    # Create datasets
    train_dataset = SimplifiedDataset(config, train_files, split='train', vocab=vocab)
    val_dataset = SimplifiedDataset(config, val_files, split='val', vocab=vocab)
    
    # Create samplers
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    elif frequency_sampling and train_dataset.sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function"""
    images = torch.stack([item['image'] for item in batch])
    tag_labels = torch.stack([item['tag_labels'] for item in batch])
    rating_labels = torch.stack([item['rating_label'] for item in batch])
    
    metadata = {
        'indices': [item['metadata']['index'] for item in batch],
        'paths': [item['metadata']['path'] for item in batch],
        'num_tags': torch.tensor([item['metadata']['num_tags'] for item in batch]),
        'tags': [item['metadata']['tags'] for item in batch],
        'ratings': [item['metadata']['rating'] for item in batch]
    }
    
    return {
        'images': images,
        'tag_labels': tag_labels,
        'rating_labels': rating_labels,
        'metadata': metadata
    }


if __name__ == "__main__":
    # Test the simplified dataloader
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing JSON annotations')
    parser.add_argument('--vocab_path', type=str, default='vocabulary.json', help='Path to vocabulary file')
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--frequency_sampling', action='store_true', help='Use frequency-based sampling')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        json_dir=Path(args.json_dir),
        vocab_path=Path(args.vocab_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frequency_sampling=args.frequency_sampling
    )
    
    # Test loading
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Number of train batches: {len(train_loader)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load a few batches
    for i, batch in enumerate(tqdm(train_loader, desc="Testing dataloader")):
        batch = collate_fn([train_loader.dataset[j] for j in range(min(args.batch_size, len(train_loader.dataset)))])
        
        print(f"\nBatch {i}:")
        print(f"  Image shape: {batch['images'].shape}")
        print(f"  Tag labels shape: {batch['tag_labels'].shape}")
        print(f"  Rating labels shape: {batch['rating_labels'].shape}")
        print(f"  Avg tags per image: {batch['metadata']['num_tags'].float().mean():.1f}")
        print(f"  Tag sparsity: {(batch['tag_labels'] > 0).float().mean():.4f}")
        print(f"  Sample paths: {batch['metadata']['paths'][:2]}")
        print(f"  Sample ratings: {batch['metadata']['ratings'][:4]}")
        
        if i >= 2:
            break
    
    print("\n" + "="*50)
    print("Simplified dataloader test complete!")
    print("="*50)