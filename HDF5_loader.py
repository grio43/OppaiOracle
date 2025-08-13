#!/usr/bin/env python3
"""
Simplified DataLoader for Direct Training (No Teacher Distillation)
Loads images and tags directly from JSON annotations
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

@dataclass
class SimplifiedDataConfig:
    data_dir: Path
    json_dir: Path
    vocab_path: Path

    # Image settings
    image_size: int = 640
    # Updated mean/std for anime style images
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    # Training settings
    top_k_tags: int = 100000
    min_tag_frequency: int = 100

    # Augmentation settings
    augmentation_enabled: bool = True
    random_flip_prob: float = 0.5
    color_jitter: bool = True
    # Less aggressive cropping
    random_crop_scale: Tuple[float, float] = (0.9, 1.0)

    # Sampling settings
    frequency_weighted_sampling: bool = True
    sample_weight_power: float = 0.5

    # Multi-GPU settings
    distributed: bool = False
    rank: int = 0
    world_size: int = 1

    # Cache settings
    cache_size_gb: float = 8.0
    preload_metadata: bool = True


class TagVocabulary:
    """Simplified tag vocabulary manager with distinct pad/unk tokens"""
    def __init__(self, vocab_path: Optional[Path] = None, min_frequency: int = 100):
        self.tag_to_index: Dict[str, int] = {}
        self.index_to_tag: Dict[int, str] = {}
        self.tag_frequencies: Dict[str, int] = {}
        self.min_frequency = min_frequency
        # distinct tokens rather than blank spaces
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

        most_common = tag_counter.most_common(top_k)

        # Build vocabulary with pad and unk at positions 0 and 1
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
        """Encode tags to binary vector.
        Unknown tags will activate the <UNK> token rather than being discarded.
        """
        num_tags = len(self.tag_to_index)
        encoded = torch.zeros(num_tags, dtype=torch.float32)
        for tag in tags:
            idx = self.tag_to_index.get(tag)
            if idx is None:
                # Set unknown index if tag not in vocabulary
                encoded[self.tag_to_index[self.unk_token]] = 1.0
            else:
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
    """Simplified dataset for direct training with orientation-aware augmentation"""

    def __init__(
        self,
        config: SimplifiedDataConfig,
        json_files: List[Path],
        split: str = 'train',
        vocab: Optional[TagVocabulary] = None
    ):
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
        self.annotations: List[Dict[str, Any]] = []
        self._load_annotations(json_files)

        # Normalization transform
        self.normalize = T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        # Setup augmentation pipeline (excludes horizontal flip)
        self.augmentation = self._setup_augmentation() if config.augmentation_enabled and split == 'train' else None

        # Sample weights for frequency-based sampling
        self.sample_weights: Optional[np.ndarray] = None
        if config.frequency_weighted_sampling and split == 'train':
            self._calculate_sample_weights()

        # Cache for frequently accessed images
        self.cache: Dict[str, torch.Tensor] = {}
        bytes_per_image = 3 * config.image_size * config.image_size * 4  # float32
        self.max_cache_size = int(config.cache_size_gb * 1024**3 / bytes_per_image)

        # Orientation mapping for left/right tags
        self.orientation_tag_map: Dict[str, str] = {
            'hair_over_left_eye': 'hair_over_right_eye',
            'hair_over_right_eye': 'hair_over_left_eye',
            'hand_on_left_hip': 'hand_on_right_hip',
            'hand_on_right_hip': 'hand_on_left_hip',
            'looking_to_the_left': 'looking_to_the_right',
            'looking_to_the_right': 'looking_to_the_left',
        }

        logger.info(f"Dataset initialized with {len(self.annotations)} samples")

    def _load_annotations(self, json_files: List[Path]):
        """Load and filter annotations"""
        logger.info(f"Loading annotations from {len(json_files)} files")
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for entry in data:
                    if 'filename' not in entry or 'tags' not in entry:
                        continue
                    tags = entry['tags'].split() if isinstance(entry['tags'], str) else entry['tags']
                    # Filter tags by vocabulary
                    valid_tags = [t for t in tags if t in self.vocab.tag_to_index]
                    if not valid_tags:
                        continue
                    rating = entry.get('rating', 'unknown')
                    if rating not in self.vocab.rating_to_index:
                        rating = 'unknown'
                    image_path = self.config.data_dir / entry['filename']
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
            weight = 0.0
            for tag in anno['tags']:
                if tag in self.vocab.tag_frequencies:
                    freq = self.vocab.tag_frequencies[tag]
                    weight += (1.0 / freq) ** self.config.sample_weight_power
            weight = weight / max(1, len(anno['tags']))
            weights.append(weight)
        weights = np.array(weights)
        weights = weights / weights.sum()
        self.sample_weights = weights
        logger.info(f"Sample weights calculated, range: [{weights.min():.6f}, {weights.max():.6f}]")

    def _setup_augmentation(self) -> Optional[T.Compose]:
        """Setup augmentation transforms (excluding horizontal flip)"""
        transforms: List[Any] = []
        # Random resized crop with configured scale
        if self.config.random_crop_scale != (1.0, 1.0):
            transforms.append(T.RandomResizedCrop(
                self.config.image_size,
                scale=self.config.random_crop_scale,
                ratio=(0.9, 1.1),
                interpolation=T.InterpolationMode.LANCZOS
            ))
        # Color jitter with stronger parameters
        if self.config.color_jitter:
            transforms.append(T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ))
        # Random gamma correction
        def gamma_transform(img: torch.Tensor) -> torch.Tensor:
            gamma = random.uniform(0.8, 1.2)
            return TF.adjust_gamma(img, gamma=gamma)
        transforms.append(T.Lambda(gamma_transform))
        return T.Compose(transforms) if transforms else None

    def __len__(self) -> int:
        return len(self.annotations)

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image, with caching"""
        if image_path in self.cache:
            return self.cache[image_path].clone()
        try:
            image = Image.open(image_path).convert('RGB')
            image = TF.resize(image, (self.config.image_size, self.config.image_size),
                              interpolation=T.InterpolationMode.LANCZOS)
            image = TF.to_tensor(image)
            if len(self.cache) < self.max_cache_size:
                self.cache[image_path] = image.clone()
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, self.config.image_size, self.config.image_size)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample with orientation-aware flipping"""
        try:
            anno = self.annotations[idx]
            # Load image
            image = self._load_image(anno['image_path'])
            # Copy tags so we can modify without altering the annotation
            tags = list(anno['tags'])
            # Random horizontal flip with orientation-specific tag remapping
            if (self.config.random_flip_prob > 0 and self.split == 'train' and
                    random.random() < self.config.random_flip_prob):
                image = TF.hflip(image)
                tags = [self.orientation_tag_map.get(t, t) for t in tags]
            # Apply further augmentation (crop, jitter, gamma)
            if self.augmentation is not None:
                image = self.augmentation(image)
            # Normalize
            image = self.normalize(image)
            # Encode tags
            tag_labels = self.vocab.encode_tags(tags)
            # Encode rating
            rating_label = self.vocab.rating_to_index[anno['rating']]
            return {
                'image': image,
                'labels': {
                    'tags': tag_labels,
                    'rating': rating_label,
                    'binary': tag_labels
                },
                'tag_labels': tag_labels,
                'rating_label': rating_label,
                'metadata': {
                    'index': idx,
                    'path': anno['image_path'],
                    'num_tags': anno['num_tags'],
                    'tags': tags,
                    'rating': anno['rating']
                }
            }
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {e}")
            return {
                'image': torch.zeros(3, self.config.image_size, self.config.image_size),
                'tag_labels': torch.zeros(len(self.vocab.tag_to_index)),
                'rating_label': torch.tensor(self.vocab.rating_to_index.get('unknown', 4), dtype=torch.long),
                'metadata': {
                    'index': idx,
                    'path': f"error_{idx}",
                    'num_tags': 0,
                    'tags': [],
                    'rating': 'unknown'
                }
            }


def create_dataloaders(
    data_dir: Path,
    json_dir: Path,
    vocab_path: Path,
    batch_size: int = 32,
    num_workers: int = 8,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    frequency_sampling: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    Splits the dataset at the JSON-file level as before.
    """
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    random.shuffle(json_files)
    split_idx = int(len(json_files) * 0.9)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    config = SimplifiedDataConfig(
        data_dir=data_dir,
        json_dir=json_dir,
        vocab_path=vocab_path,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        frequency_weighted_sampling=frequency_sampling
    )

    vocab = TagVocabulary(vocab_path)
    if not vocab_path.exists():
        vocab.build_from_annotations(json_files, config.top_k_tags)
        vocab.save_vocabulary(vocab_path)

    train_dataset = SimplifiedDataset(config, train_files, split='train', vocab=vocab)
    val_dataset = SimplifiedDataset(config, val_files, split='val', vocab=vocab)

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
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to assemble batch"""
    images = torch.stack([item['image'] for item in batch])
    tag_labels = torch.stack([item['tag_labels'] for item in batch])
    rating_labels = torch.stack([torch.tensor(item['rating_label']) for item in batch])
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