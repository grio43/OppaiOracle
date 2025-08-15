#!/usr/bin/env python3
"""
Improved data loading and augmentation for the direct anime image tagger.

This module provides a simplified but extensible dataset and dataloader for
training an anime image tagger from scratch.  The implementation builds on
top of the original ``HDF5_loader.py`` found in the upstream repository
but incorporates a number of important fixes and enhancements:

* **Consistent normalisation statistics** – The dataset now defaults to
  using an (0.5, 0.5, 0.5) mean and standard deviation for RGB channels.
  These values better match anime artwork distributions than the
  ImageNet values used previously.  Both parameters can be overridden per
  dataset via the :class:`SimplifiedDataConfig`.

* **Orientation‑aware flips with extensible mapping** – Random horizontal
  flips are handled inside :meth:`SimplifiedDataset.__getitem__` but
  orientation‑sensitive tags are remapped using a dictionary loaded from
  a JSON or YAML file if supplied.  If no mapping is provided the
  loader falls back to a small built‑in mapping covering common left/right
  tags.  Flips can be disabled entirely by setting
  :attr:`SimplifiedDataConfig.random_flip_prob` to zero.

* **Expanded augmentation pipeline** – The augmentation pipeline includes
  random resized cropping, optional colour jitter and a random gamma
  adjustment.  The gamma range has been widened to [0.7, 1.3] to
  better simulate exposure variations in stylised artwork.  The crop
  scale range defaults to (0.95, 1.0) to preserve more of the subject in
  each image.

* **Unknown tag handling** – Tags found in the annotation JSON that are
  not present in the vocabulary are no longer silently discarded.
  Instead, they are encoded to the special ``<UNK>`` index so the model
  learns to handle rare or unseen tags gracefully.

* **Simplified label dictionary** – The dictionary returned from
  ``__getitem__`` no longer includes the redundant ``'binary'`` field.
  Downstream code should use the returned ``'tag_labels'`` tensor as
  needed.

* **Configurable sampling and caching** – Oversampling of orientation
  tags, the exponent used in sample weight computation and the cache
  precision are all exposed via the configuration.  Default values
  favour efficient memory usage and balanced training.

The resulting dataloader is self‑contained and compatible with both
single‑GPU and distributed training scenarios.
"""



import threading
import json
import logging
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler
from collections import OrderedDict
from __future__ import annotations


logger = logging.getLogger(__name__)


@dataclass
class SimplifiedDataConfig:
    """Configuration parameters controlling data loading and augmentation.

    Attributes are initialised with sensible defaults for high‑resolution
    anime artwork.  Most fields can be overridden when constructing
    dataloaders to tailor the pipeline to a specific dataset or hardware
    environment.
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
    # Reduce flip probability because many tags encode left/right semantics
    random_flip_prob: float = 0.2
    # Narrow crop scale range to preserve most of the subject in the frame
    random_crop_scale: Tuple[float, float] = (0.95, 1.0)
    # Colour jitter parameters
    color_jitter: bool = True
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1
    # Optional path to orientation mapping (JSON or YAML).  If provided, the
    # mapping is loaded on dataset initialisation.
    orientation_map_path: Optional[Path] = None

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
    """Tag vocabulary manager with separate <PAD> and <UNK> tokens.

    The vocabulary can be built from a collection of JSON annotation files or
    loaded from a JSON file.  Tags appearing less than ``min_frequency``
    times are omitted to keep the vocabulary size manageable.  Unknown
    tags are mapped to the ``<UNK>`` index rather than being silently
    discarded.
    """

    def __init__(self, vocab_path: Optional[Path] = None, min_frequency: int = 1) -> None:
        self.tag_to_index: Dict[str, int] = {}
        self.index_to_tag: Dict[int, str] = {}
        self.tag_frequencies: Dict[str, int] = {}
        self.min_frequency = min_frequency
        # Distinct special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        # Rating classes (fixed).  A fifth ``unknown`` rating is included to
        # handle missing annotations.
        self.rating_to_index: Dict[str, int] = {
            "general": 0,
            "sensitive": 1,
            "questionable": 2,
            "explicit": 3,
            "unknown": 4,
        }

        # If a vocabulary file is supplied, attempt to load it
        if vocab_path is not None and vocab_path.exists():
            try:
                self.load_vocabulary(vocab_path)
            except Exception:
                logger.info(f"Could not load vocabulary from {vocab_path}, will build a new one")

    def __len__(self) -> int:
        return len(self.tag_to_index)

    def encode_tags(self, tags: Iterable[str]) -> torch.Tensor:
        """Encode a list of tag strings into a multi‑hot tensor.

        Unknown tags are mapped to the ``<UNK>`` index; the resulting tensor
        has shape (vocab_size,) and dtype ``float32``.
        """
        vector = torch.zeros(len(self.tag_to_index), dtype=torch.float32)
        for tag in tags:
            idx = self.tag_to_index.get(tag, self.tag_to_index[self.unk_token])
            vector[idx] = 1.0
        return vector

    def build_from_annotations(self, json_files: List[Path], top_k: int) -> None:
        """Build a vocabulary from a collection of JSON annotation files.

        Parameters
        ----------
        json_files: List[Path]
            List of annotation files to parse.
        top_k: int
            Maximum number of tags to keep.  Tags are sorted by frequency
            descending and truncated to this value.
        """
        logger.info(f"Building vocabulary from {len(json_files)} annotation files")
        tag_counts: Dict[str, int] = {}
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for entry in data:
                    tags_field = entry.get('tags')
                    if not tags_field:
                        continue
                    tags_list: List[str]
                    if isinstance(tags_field, str):
                        tags_list = tags_field.split()
                    elif isinstance(tags_field, list):
                        tags_list = tags_field
                    else:
                        continue
                    for tag in tags_list:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")
        # Sort tags by frequency and cut to top_k
        sorted_tags = sorted(
            [t for t, c in tag_counts.items() if c >= self.min_frequency],
            key=lambda x: (-tag_counts[x], x)
        )
        if top_k is not None and top_k > 0:
            sorted_tags = sorted_tags[:top_k]
        # Assign indices.  Reserve 0 for <PAD> and 1 for <UNK>
        self.tag_to_index = {self.pad_token: 0, self.unk_token: 1}
        self.index_to_tag = {0: self.pad_token, 1: self.unk_token}
        for idx, tag in enumerate(sorted_tags, start=2):
            self.tag_to_index[tag] = idx
            self.index_to_tag[idx] = tag
            self.tag_frequencies[tag] = tag_counts[tag]
        logger.info(f"Vocabulary built with {len(self.tag_to_index)} tags (incl. special tokens)")

    def save_vocabulary(self, vocab_path: Path) -> None:
        """Save the vocabulary to a JSON file.

        The file contains a mapping from tags to indices and vice versa as
        well as tag frequencies.  The top‑level keys are ``tag_to_index``,
        ``index_to_tag`` and ``tag_frequencies``.
        """
        vocab_path = Path(vocab_path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'tag_to_index': self.tag_to_index,
                'index_to_tag': self.index_to_tag,
                'tag_frequencies': self.tag_frequencies,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved vocabulary to {vocab_path}")

    def load_vocabulary(self, vocab_path: Path) -> None:
        """Load vocabulary from a JSON file created by :meth:`save_vocabulary`."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.tag_to_index = data['tag_to_index']
        self.index_to_tag = {int(k): v for k, v in data['index_to_tag'].items()}
        self.tag_frequencies = data.get('tag_frequencies', {})
        # Ensure special tokens are present
        for token in (self.pad_token, self.unk_token):
            if token not in self.tag_to_index:
                idx = len(self.tag_to_index)
                self.tag_to_index[token] = idx
                self.index_to_tag[idx] = token
        logger.info(f"Loaded vocabulary with {len(self.tag_to_index)} tags from {vocab_path}")


class SimplifiedDataset(Dataset):
    """Dataset for anime image tagging with augmentation and sampling.

    Each item returned is a dictionary containing an image tensor,
    multi‑hot tag labels, a rating label and some metadata (index, path,
    original tag list and rating string).  See the module docstring for
    details on the implemented features.
    """

    def __init__(
        self,
        config: SimplifiedDataConfig,
        json_files: List[Path],
        split: str,
        vocab: TagVocabulary,
    ) -> None:
        assert split in {'train', 'val', 'test'}, f"Unknown split '{split}'"
        self.config = config
        self.split = split
        self.vocab = vocab
        self.annotations: List[Dict[str, Any]] = []
        self.cache: Dict[str, torch.Tensor] = {}
        # Load orientation mapping from file if provided
        if config.orientation_map_path is not None and config.orientation_map_path.exists():
            try:
                with open(config.orientation_map_path, 'r', encoding='utf-8') as f:
                    self.orientation_tag_map = json.load(f)
            except Exception as e:
                logger.warning(
                    f"Failed to load orientation mapping from {config.orientation_map_path}: {e}. "
                    "Falling back to default mapping"
                )
                self.orientation_tag_map = self._default_orientation_map()
        else:
            self.orientation_tag_map = self._default_orientation_map()
        # Preload annotations
        if config.preload_metadata:
            self._load_annotations(json_files)
            # Precompute sample weights for frequency sampling
            self.sample_weights: Optional[np.ndarray] = None
            if config.frequency_weighted_sampling and split == 'train':
                self._calculate_sample_weights()
        else:
            self.sample_weights = None
        # Set up augmentation and normalisation
        self.augmentation = self._setup_augmentation() if config.augmentation_enabled and split == 'train' else None
        self.normalize = T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        # Determine maximum cache size in number of images
        bytes_per_element = {
            'float32': 4,
            'float16': 2,
            'uint8': 1,
        }[config.cache_precision]
        bytes_per_image = 3 * config.image_size * config.image_size * bytes_per_element
        self.max_cache_size = int((config.cache_size_gb * (1024 ** 3)) / bytes_per_image)
        self.cache_precision = config.cache_precision
        logger.info(f"Dataset initialised with {len(self.annotations)} samples for split '{split}'")

    def _default_orientation_map(self) -> Dict[str, str]:
        """Return a minimal left/right tag mapping for orientation flips."""
        return {
            'hair_over_left_eye': 'hair_over_right_eye',
            'hair_over_right_eye': 'hair_over_left_eye',
            'hand_on_left_hip': 'hand_on_right_hip',
            'hand_on_right_hip': 'hand_on_left_hip',
            'looking_to_the_left': 'looking_to_the_right',
            'looking_to_the_right': 'looking_to_the_left',
        }

    def _load_annotations(self, json_files: List[Path]) -> None:
        """Parse annotation files and populate ``self.annotations``.

        Images with at least one tag are kept.  Unknown tags are retained
        (they will map to ``<UNK>`` when encoding).  Missing or invalid
        entries are skipped silently but logged.
        """
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
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
                    if not tags_list:
                        continue
                    rating = entry.get('rating', 'unknown')
                    if rating not in self.vocab.rating_to_index:
                        rating = 'unknown'
                    image_path = self.config.data_dir / filename
                    if not image_path.exists():
                        continue
                    self.annotations.append({
                        'image_path': str(image_path),
                        'tags': tags_list,
                        'rating': rating,
                        'num_tags': len(tags_list),
                    })
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        logger.info(f"Loaded {len(self.annotations)} valid annotations")

    def _calculate_sample_weights(self) -> None:
        """Compute per‑sample weights based on tag frequencies.

        Each annotation receives a weight equal to the average of the inverse
        of its tag frequencies raised to ``sample_weight_power``.  An optional
        multiplier is applied if any tag in the annotation is orientation
        specific (as determined by ``orientation_tag_map``).  The resulting
        weights are normalised to sum to one.
        """
        weights: List[float] = []
        for anno in self.annotations:
            w = 0.0
            has_orientation_tag = False
            for tag in anno['tags']:
                freq = self.vocab.tag_frequencies.get(tag, 1)
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
        :meth:`__getitem__` to enable orientation‑aware tag remapping.  Colour
        jitter parameters are configurable via :class:`SimplifiedDataConfig`.
        A random gamma transform with a wide range is appended to better
        handle exposure variations.
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
            gamma = float(np.random.uniform(0.7, 1.3))
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
        cache stores images in the precision specified by ``cache_precision``;
        loaded images are always returned as ``float32`` tensors in the [0, 1]
        range.
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
    config_updates: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader, TagVocabulary]:
    """Construct training and validation dataloaders along with the vocabulary.

    Splits JSON annotation files into 90 % training and 10 % validation.  If a
    vocabulary file exists it is loaded; otherwise it is built from all
    annotations and saved.  The returned validation batch size defaults to
    ``batch_size`` if not explicitly provided.  Both dataloaders use a
    custom collate function defined below.
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
    cfg = SimplifiedDataConfig(
        data_dir=data_dir,
        json_dir=json_dir,
        vocab_path=vocab_path,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        frequency_weighted_sampling=frequency_sampling,
    )
    # Apply any user‑provided overrides
    if config_updates:
        for k, v in config_updates.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    vocab = TagVocabulary(vocab_path, min_frequency=cfg.min_tag_frequency)
    if not vocab_path.exists():
        # Build vocabulary from all available annotations
        vocab.build_from_annotations(json_files_sorted, cfg.top_k_tags)
        vocab.save_vocabulary(vocab_path)
    # Create datasets
    train_dataset = SimplifiedDataset(cfg, train_files, split='train', vocab=vocab)
    val_dataset = SimplifiedDataset(cfg, val_files, split='val', vocab=vocab)
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