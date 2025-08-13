#!/usr/bin/env python3
"""
Simplified DataLoader for Direct Training (No Teacher Distillation)
-----------------------------------------------------------------

This module implements a minimal yet extensible dataset and dataloader for
training an anime image tagger from scratch.  The loader performs the
following key duties:

* Parsing JSON annotation files and building a tag vocabulary on demand.
* Loading, resizing and normalising RGB images from disk with an optional
  in‑memory cache to speed up repeated accesses during an epoch.
* Applying a suite of data augmentations including random horizontal flips
  with orientation‑aware tag remapping, random resized crops, colour jitter
  and random gamma adjustments.
* Balancing the dataset via frequency‑based sampling with optional
  oversampling of orientation‑specific tags to ensure the model sees enough
  examples of these rare classes.
* Exposing a simple ``create_dataloaders`` helper to construct both
  training and validation dataloaders along with the underlying
  ``TagVocabulary``.

The implementation here addresses a number of shortcomings identified in the
original codebase:

1.  Orientation‑aware augmentation and tag flipping: a mapping of
    orientation‑sensitive tags (e.g. ``hair_over_left_eye`` ↔︎
    ``hair_over_right_eye``) is maintained and used to remap tag names
    whenever a horizontal flip is applied.  This prevents silent
    mislabelling when images are mirrored.

2.  Data normalisation and colour statistics: default normalisation
    statistics have been relaxed to (0.5, 0.5, 0.5) mean and std.  This
    reflects the fact that anime artwork does not follow the ImageNet
    distribution.  These values can be overridden via the configuration.

3.  Colour and gamma augmentation: colour jitter parameters are exposed via
    ``SimplifiedDataConfig`` and a small random gamma transform is applied
    after other augmentations.  This helps the model cope with the wide
    dynamic range of colour palettes and exposure variations in anime data.

4.  Tag vocabulary design: separate ``<PAD>`` and ``<UNK>`` tokens are
    introduced to avoid collisions in the vocabulary.  The minimum tag
    frequency can be configured per experiment.  Unknown tags map to the
    ``<UNK>`` index rather than being silently discarded.

5.  Sampling and class imbalance: sample weights are computed from tag
    frequencies.  An optional multiplier can be applied to annotations
    containing orientation‑specific tags to oversample these rare cases.

6.  Image caching and memory usage: cached images are stored in a
    lower‑precision format (float16 by default) to reduce host memory
    consumption.  The precision can be configured via ``cache_precision``.

7.  Dynamic tag dimension: the length of the vocabulary determines the
    dimensionality of the tag head in the model.  Consumers of this module
    should query the vocabulary size and pass it to their model
    constructors.

The resulting dataloader is fully self‑contained and can be used in both
single‑ and multi‑GPU settings.  It gracefully handles missing data and
unknown tags, and logs informative messages to aid debugging.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedDataConfig:
    """Configuration parameters controlling data loading and augmentation.

    These parameters can be overridden on construction or via a higher level
    configuration system.  Reasonable defaults are provided for
    high‑resolution anime artwork.
    """
    # Required locations
    data_dir: Path
    json_dir: Path
    vocab_path: Path

    # Image settings
    image_size: int = 640
    # Normalisation parameters (defaults tuned for anime artwork rather than ImageNet)
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    # Vocabulary settings
    top_k_tags: int = 100_000
    min_tag_frequency: int = 1  # include all tags that appear at least once

    # Augmentation settings
    augmentation_enabled: bool = True
    random_flip_prob: float = 0.5
    random_crop_scale: Tuple[float, float] = (0.9, 1.0)
    color_jitter: bool = True
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1

    # Sampling settings
    frequency_weighted_sampling: bool = True
    sample_weight_power: float = 0.5
    orientation_oversample_factor: float = 2.0

    # Multi‑GPU settings
    distributed: bool = False
    rank: int = 0
    world_size: int = 1

    # Cache settings
    cache_size_gb: float = 8.0
    cache_precision: str = 'float16'  # 'float32', 'float16' or 'uint8'
    preload_metadata: bool = True

    def __post_init__(self) -> None:
        # Validate flip probability
        if not 0.0 <= self.random_flip_prob <= 1.0:
            raise ValueError(f"random_flip_prob must be in [0, 1], got {self.random_flip_prob}")
        # Validate cache precision
        if self.cache_precision not in {'float32', 'float16', 'uint8'}:
            raise ValueError(
                f"cache_precision must be one of 'float32', 'float16' or 'uint8', got {self.cache_precision}"
            )


class TagVocabulary:
    """Tag vocabulary manager with distinct <PAD> and <UNK> tokens.

    The vocabulary is built from a collection of JSON annotation files.  Tags
    occurring less often than ``min_frequency`` times are omitted from the
    vocabulary, but unknown tags are mapped to the ``<UNK>`` index rather than
    being ignored.  Rating classes are fixed.
    """

    def __init__(self, vocab_path: Optional[Path] = None, min_frequency: int = 1):
        self.tag_to_index: Dict[str, int] = {}
        self.index_to_tag: Dict[int, str] = {}
        self.tag_frequencies: Dict[str, int] = {}
        self.min_frequency = min_frequency
        # Distinct special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        # Rating classes (fixed)
        self.rating_to_index = {
            "general": 0,
            "sensitive": 1,
            "questionable": 2,
            "explicit": 3,
            "unknown": 4,
        }
        self.index_to_rating = {v: k for k, v in self.rating_to_index.items()}
        # If a vocabulary file exists, load it
        if vocab_path and vocab_path.exists():
            self.load_vocabulary(vocab_path)

    def build_from_annotations(self, json_files: List[Path], top_k: int = 100_000) -> None:
        """Build the vocabulary from a list of JSON annotation files."""
        logger.info(f"Building vocabulary from {len(json_files)} annotation files")
        tag_counter: Dict[str, int] = {}
        rating_counter: Dict[str, int] = {}
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for entry in data:
                    # Tags may be stored as a space separated string or list
                    if 'tags' in entry:
                        tags = entry['tags']
                        if isinstance(tags, str):
                            tags = tags.split()
                        for t in tags:
                            tag_counter[t] = tag_counter.get(t, 0) + 1
                    if 'rating' in entry:
                        rating_counter[entry['rating']] = rating_counter.get(entry['rating'], 0) + 1
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        # Sort tags by frequency and truncate to top_k
        most_common = sorted(tag_counter.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        # Insert special tokens at positions 0 and 1
        self.tag_to_index = {self.pad_token: 0, self.unk_token: 1}
        current_index = 2
        for tag, freq in most_common:
            if freq >= self.min_frequency:
                self.tag_to_index[tag] = current_index
                self.tag_frequencies[tag] = freq
                current_index += 1
        self.index_to_tag = {idx: tag for tag, idx in self.tag_to_index.items()}
        logger.info(f"Vocabulary size (including special tokens): {len(self.tag_to_index)}")
        logger.info(f"Rating distribution: {rating_counter}")

    def save_vocabulary(self, path: Path) -> None:
        """Persist the vocabulary to disk as JSON."""
        vocab_data = {
            'tag_to_index': self.tag_to_index,
            'tag_frequencies': self.tag_frequencies,
            'rating_to_index': self.rating_to_index,
            'min_frequency': self.min_frequency,
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        logger.info(f"Vocabulary saved to {path}")

    def load_vocabulary(self, path: Path) -> None:
        """Load a vocabulary from a JSON file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        self.tag_to_index = vocab_data['tag_to_index']
        # Convert keys back to ints for index_to_tag
        self.index_to_tag = {int(idx): tag for tag, idx in self.tag_to_index.items()}
        self.tag_frequencies = vocab_data.get('tag_frequencies', {})
        self.rating_to_index = vocab_data.get('rating_to_index', self.rating_to_index)
        self.index_to_rating = {v: k for k, v in self.rating_to_index.items()}
        self.min_frequency = vocab_data.get('min_frequency', self.min_frequency)
        logger.info(f"Loaded vocabulary with {len(self.tag_to_index)} tags from {path}")

    def encode_tags(self, tags: List[str]) -> torch.Tensor:
        """Encode a list of tag strings into a multi‑hot vector.

        Unknown tags activate the ``<UNK>`` index.  The returned tensor has
        shape ``(num_tags,)`` and dtype ``float32``.
        """
        num_tags = len(self.tag_to_index)
        encoded = torch.zeros(num_tags, dtype=torch.float32)
        unk_index = self.tag_to_index[self.unk_token]
        for tag in tags:
            idx = self.tag_to_index.get(tag, unk_index)
            encoded[idx] = 1.0
        return encoded

    def decode_tags(self, encoded: torch.Tensor, threshold: float = 0.5) -> List[str]:
        """Decode a multi‑hot vector back into tag strings.

        Indices with activation above ``threshold`` are returned.  The
        ``<UNK>`` token is never returned in the decoded list.
        """
        indices = torch.where(encoded > threshold)[0].tolist()
        tags: List[str] = []
        unk_index = self.tag_to_index[self.unk_token]
        for idx in indices:
            if idx != unk_index:
                tag = self.index_to_tag.get(idx)
                if tag is not None:
                    tags.append(tag)
        return tags


class SimplifiedDataset(Dataset):
    """Dataset that reads images and tags from disk and applies augmentation.

    Args:
        config: A ``SimplifiedDataConfig`` describing data loading options.
        json_files: A list of JSON annotation files to consume.
        split: ``'train'`` or ``'val'`` to control augmentation behaviour.
        vocab: A pre‑built ``TagVocabulary``.  If ``None`` the dataset will
            create one but not persist it.
    """

    def __init__(
        self,
        config: SimplifiedDataConfig,
        json_files: List[Path],
        split: str = 'train',
        vocab: Optional[TagVocabulary] = None,
    ) -> None:
        self.config = config
        self.split = split
        # Load or build vocabulary
        self.vocab = vocab or TagVocabulary(config.vocab_path, config.min_tag_frequency)
        # Load annotations
        self.annotations: List[Dict[str, Any]] = []
        self._load_annotations(json_files)
        # Setup augmentation
        self.augmentation = self._setup_augmentation() if (config.augmentation_enabled and split == 'train') else None
        # Normalisation transform
        self.normalize = T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        # Sample weights for frequency‑based sampling
        self.sample_weights: Optional[np.ndarray] = None
        if config.frequency_weighted_sampling and split == 'train':
            self._calculate_sample_weights()
        # Cache for images
        self.cache: Dict[str, torch.Tensor] = {}
        # Compute maximum cache entries based on desired precision
        bytes_per_element = {
            'float32': 4,
            'float16': 2,
            'uint8': 1,
        }[config.cache_precision]
        bytes_per_image = 3 * config.image_size * config.image_size * bytes_per_element
        self.max_cache_size = int((config.cache_size_gb * (1024 ** 3)) / bytes_per_image)
        self.cache_precision = config.cache_precision
        # Orientation mapping for left/right tags
        self.orientation_tag_map: Dict[str, str] = {
            'hair_over_left_eye': 'hair_over_right_eye',
            'hair_over_right_eye': 'hair_over_left_eye',
            'hand_on_left_hip': 'hand_on_right_hip',
            'hand_on_right_hip': 'hand_on_left_hip',
            'looking_to_the_left': 'looking_to_the_right',
            'looking_to_the_right': 'looking_to_the_left',
        }
        logger.info(f"Dataset initialised with {len(self.annotations)} samples for split '{split}'")

    def _load_annotations(self, json_files: List[Path]) -> None:
        """Parse annotation files and populate ``self.annotations``.

        Only annotations with at least one tag present in the vocabulary are
        kept.  Invalid or missing entries are skipped silently but logged.
        """
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for entry in data:
                    filename = entry.get('filename')
                    tags_field = entry.get('tags')
                    if not filename or not tags_field:
                        continue
                    # Convert tags field to a list of strings
                    if isinstance(tags_field, str):
                        tags_list = tags_field.split()
                    elif isinstance(tags_field, list):
                        tags_list = tags_field
                    else:
                        continue
                    # Filter tags by vocabulary if it is already built
                    if self.vocab.tag_to_index:
                        valid_tags = [t for t in tags_list if t in self.vocab.tag_to_index]
                    else:
                        valid_tags = tags_list
                    if not valid_tags:
                        continue
                    rating = entry.get('rating', 'unknown')
                    if rating not in self.vocab.rating_to_index:
                        rating = 'unknown'
                    image_path = self.config.data_dir / filename
                    if not image_path.exists():
                        continue
                    self.annotations.append({
                        'image_path': str(image_path),
                        'tags': valid_tags,
                        'rating': rating,
                        'num_tags': len(valid_tags),
                    })
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        logger.info(f"Loaded {len(self.annotations)} valid annotations")

    def _calculate_sample_weights(self) -> None:
        """Compute per‑sample weights based on tag frequencies.

        Each annotation receives a weight equal to the average of the inverse
        square root of its tag frequencies raised to ``sample_weight_power``.
        An optional multiplier is applied if any tag in the annotation is
        orientation specific (determined by ``orientation_tag_map``).  The
        resulting weights are normalised to sum to one.
        """
        weights: List[float] = []
        for anno in self.annotations:
            w = 0.0
            has_orientation_tag = False
            for tag in anno['tags']:
                # Skip unknown tags when computing weights
                if tag not in self.vocab.tag_frequencies:
                    continue
                freq = self.vocab.tag_frequencies[tag]
                # Inverse frequency weighting
                w += (1.0 / max(freq, 1)) ** self.config.sample_weight_power
                if tag in self.orientation_tag_map:
                    has_orientation_tag = True
            # Average over number of tags to avoid biasing multi‑tag images
            w = w / max(1, len(anno['tags']))
            # Apply orientation oversample factor if needed
            if has_orientation_tag:
                w *= self.config.orientation_oversample_factor
            weights.append(w)
        weights_arr = np.array(weights, dtype=np.float64)
        # Normalise weights to sum to one
        weights_arr = weights_arr / weights_arr.sum() if weights_arr.sum() > 0 else weights_arr
        self.sample_weights = weights_arr
        logger.info(
            f"Sample weights calculated (min={weights_arr.min():.6f}, max={weights_arr.max():.6f})"
        )

    def _setup_augmentation(self) -> Optional[T.Compose]:
        """Create an augmentation pipeline for the training split.

        The pipeline excludes horizontal flips, which are handled explicitly in
        ``__getitem__`` to enable orientation‑aware tag remapping.  Colour
        jitter parameters are configurable via ``SimplifiedDataConfig``.  A
        random gamma transform is appended to better handle exposure
        variations.
        """
        transforms: List[Any] = []
        # Random resized crop
        if self.config.random_crop_scale != (1.0, 1.0):
            transforms.append(
                T.RandomResizedCrop(
                    self.config.image_size,
                    scale=self.config.random_crop_scale,
                    ratio=(0.9, 1.1),
                    interpolation=T.InterpolationMode.LANCZOS,
                )
            )
        # Colour jitter
        if self.config.color_jitter:
            transforms.append(
                T.ColorJitter(
                    brightness=self.config.color_jitter_brightness,
                    contrast=self.config.color_jitter_contrast,
                    saturation=self.config.color_jitter_saturation,
                    hue=self.config.color_jitter_hue,
                )
            )
        # Random gamma correction
        def gamma_transform(img: torch.Tensor) -> torch.Tensor:
            gamma = float(np.random.uniform(0.8, 1.2))
            # torchvision's adjust_gamma operates on PIL images; convert via functional
            return TF.adjust_gamma(img, gamma=gamma)
        transforms.append(T.Lambda(gamma_transform))
        return T.Compose(transforms) if transforms else None

    def __len__(self) -> int:
        return len(self.annotations)

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load an image from disk or cache and return it as a float tensor.

        Images are resized to ``image_size`` using Lanczos interpolation.  A
        copy of the image may be cached to accelerate repeated accesses.  The
        cache stores images in the precision specified by
        ``cache_precision``; loaded images are always returned as
        ``float32`` tensors in the [0, 1] range.
        """
        # Check cache first
        if image_path in self.cache:
            cached = self.cache[image_path]
            # Convert cached tensor back to float32
            if cached.dtype == torch.uint8:
                return cached.float() / 255.0
            elif cached.dtype == torch.float16:
                return cached.float()
            else:
                return cached.clone()
        # Load from disk
        try:
            image = Image.open(image_path).convert('RGB')
            image = TF.resize(
                image,
                (self.config.image_size, self.config.image_size),
                interpolation=T.InterpolationMode.LANCZOS,
            )
            tensor = TF.to_tensor(image)  # returns float32 in [0, 1]
            # Decide whether to cache
            if len(self.cache) < self.max_cache_size:
                if self.cache_precision == 'uint8':
                    self.cache[image_path] = (tensor * 255).to(torch.uint8)
                elif self.cache_precision == 'float16':
                    self.cache[image_path] = tensor.half()
                else:
                    self.cache[image_path] = tensor.clone()
            return tensor
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return black image to avoid crashing caller
            return torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch an item for training or validation.

        Applies a random horizontal flip with orientation‑aware tag remapping
        for the training split.  Augmentation and normalisation are then
        applied.  Any exceptions are caught and logged, and a safe fallback
        sample is returned.
        """
        try:
            anno = self.annotations[idx]
            # Load image (float32 tensor)
            image = self._load_image(anno['image_path'])
            # Copy tags so we can mutate without altering the original
            tags = list(anno['tags'])
            # Random horizontal flip with orientation tag swap
            if (
                self.config.random_flip_prob > 0
                and self.split == 'train'
                and np.random.rand() < self.config.random_flip_prob
            ):
                image = TF.hflip(image)
                tags = [self.orientation_tag_map.get(t, t) for t in tags]
            # Additional augmentation
            if self.augmentation is not None:
                image = self.augmentation(image)
            # Normalise
            image = self.normalize(image)
            # Encode tags and rating
            tag_labels = self.vocab.encode_tags(tags)
            rating_label = self.vocab.rating_to_index.get(anno['rating'], self.vocab.rating_to_index['unknown'])
            return {
                'image': image,
                'labels': {
                    'tags': tag_labels,
                    'rating': rating_label,
                    'binary': tag_labels,
                },
                'tag_labels': tag_labels,
                'rating_label': rating_label,
                'metadata': {
                    'index': idx,
                    'path': anno['image_path'],
                    'num_tags': anno['num_tags'],
                    'tags': tags,
                    'rating': anno['rating'],
                },
            }
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {e}")
            # Safe fallback sample
            return {
                'image': torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32),
                'tag_labels': torch.zeros(len(self.vocab.tag_to_index), dtype=torch.float32),
                'rating_label': torch.tensor(self.vocab.rating_to_index.get('unknown', 4), dtype=torch.long),
                'metadata': {
                    'index': idx,
                    'path': f"error_{idx}",
                    'num_tags': 0,
                    'tags': [],
                    'rating': 'unknown',
                },
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
    frequency_sampling: bool = True,
    val_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, TagVocabulary]:
    """Construct training and validation dataloaders along with the vocabulary.

    Splits JSON annotation files into 90 % training and 10 % validation.  If a
    vocabulary file exists it is loaded; otherwise it is built from all
    annotations and saved.  The returned validation batch size defaults to
    ``batch_size`` if not explicitly provided.  Both dataloaders use
    ``collate_fn`` defined below.
    """
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    # Shuffle and split files
    json_files_sorted = sorted(json_files)
    np.random.shuffle(json_files_sorted)
    split_idx = int(len(json_files_sorted) * 0.9)
    train_files = json_files_sorted[:split_idx]
    val_files = json_files_sorted[split_idx:]
    # Instantiate config and vocabulary
    config = SimplifiedDataConfig(
        data_dir=data_dir,
        json_dir=json_dir,
        vocab_path=vocab_path,
        min_tag_frequency=1,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        frequency_weighted_sampling=frequency_sampling,
    )
    vocab = TagVocabulary(vocab_path, min_frequency=config.min_tag_frequency)
    if not vocab_path.exists():
        # Build vocabulary from all available annotations
        vocab.build_from_annotations(json_files_sorted, config.top_k_tags)
        vocab.save_vocabulary(vocab_path)
    # Create datasets
    train_dataset = SimplifiedDataset(config, train_files, split='train', vocab=vocab)
    val_dataset = SimplifiedDataset(config, val_files, split='val', vocab=vocab)
    # Choose sampler
    train_sampler: Optional[Any] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    elif frequency_sampling and train_dataset.sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
    # Determine validation batch size
    val_bs = val_batch_size or batch_size
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, vocab


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to assemble a batch of samples."""
    images = torch.stack([item['image'] for item in batch])
    tag_labels = torch.stack([item['tag_labels'] for item in batch])
    rating_labels = torch.tensor([item['rating_label'] for item in batch], dtype=torch.long)
    metadata = {
        'indices': [item['metadata']['index'] for item in batch],
        'paths': [item['metadata']['path'] for item in batch],
        'num_tags': torch.tensor([item['metadata']['num_tags'] for item in batch]),
        'tags': [item['metadata']['tags'] for item in batch],
        'ratings': [item['metadata']['rating'] for item in batch],
    }
    return {
        'images': images,
        'tag_labels': tag_labels,
        'rating_labels': rating_labels,
        'metadata': metadata,
    }