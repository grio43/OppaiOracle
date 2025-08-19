#!/usr/bin/env python3

from __future__ import annotations
from orientation_handler import OrientationHandler, OrientationMonitor
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
from vocabulary import TagVocabulary

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
            # List of annotation dictionaries loaded from JSON files
            self.annotations: List[Dict[str, Any]] = []
            # LRU cache for loaded images
            self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
            # Initialize orientation handler for flip augmentation
            # Only needed for training split when flips are enabled
            if split == 'train' and config.random_flip_prob > 0:
                self.orientation_handler = OrientationHandler(
                    mapping_file=config.orientation_map_path,
                    random_flip_prob=config.random_flip_prob,
                    strict_mode=False,  # Don't fail if mapping is incomplete
                    skip_unmapped=True   # Skip flipping images with unmapped orientation tags
                )
                
                # Pre-compute mappings if vocabulary is available for better performance
                if vocab and hasattr(vocab, 'tag_to_index'):
                    all_tags = set(vocab.tag_to_index.keys())
                    self.precomputed_mappings = self.orientation_handler.precompute_all_mappings(all_tags)
                    
                    # Validate mappings and log any issues
                    validation_issues = self.orientation_handler.validate_dataset_tags(all_tags)
                    if validation_issues:
                        logger.warning(f"Orientation mapping validation issues: {validation_issues}")
                        
                        # Save validation report for review
                        validation_report_path = Path("orientation_validation_report.json")
                        with open(validation_report_path, 'w') as f:
                            json.dump(validation_issues, f, indent=2)
                        logger.info(f"Saved validation report to {validation_report_path}")
                        
                        # Optionally fail if strict validation is enabled
                        if hasattr(config, 'strict_orientation_validation') and config.strict_orientation_validation:
                            raise ValueError(f"Critical orientation mapping issues found. Check {validation_report_path}")
                else:
                    self.precomputed_mappings = None
                
                # Initialize monitor for training
                self.orientation_monitor = OrientationMonitor(threshold_unmapped=20)
            else:
                self.orientation_handler = None
                self.precomputed_mappings = None
                self.orientation_monitor = None
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
            self.cache_lock = threading.Lock()  # Add thread lock for cache operations

            # Initialise locks and counters for error handling.  `_error_counts` is
            # used to track temporary I/O failures per image and must be
            # accessed under `_error_counts_lock` to ensure thread safety.
            self._error_counts_lock = threading.Lock()
            self._error_counts: Dict[str, int] = {}

            # Load annotation metadata from the provided JSON files.  This
            # populates ``self.annotations`` with valid entries.
            self._load_annotations(json_files)
            # Compute sampling weights if frequency‑weighted sampling is
            # enabled and this is the training split; otherwise, assign
            # ``None`` so that standard shuffling is used.
            if self.config.frequency_weighted_sampling and split == 'train':
                self._calculate_sample_weights()
            else:
                self.sample_weights = None

            logger.info(f"Dataset initialised with {len(self.annotations)} samples for split '{split}'")

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
            specific (as determined by the orientation handler).  The resulting
            weights are normalised to sum to one.
            """
            weights: List[float] = []
            # Get orientation tags from handler if available
            orientation_tags = set()
            if self.orientation_handler:
                # Collect all orientation-aware tags (explicit mappings + reverse mappings)
                orientation_tags.update(self.orientation_handler.explicit_mappings.keys())
                orientation_tags.update(self.orientation_handler.reverse_mappings.keys())
            
            for anno in self.annotations:
                w = 0.0
                has_orientation_tag = False
                for tag in anno['tags']:
                    freq = self.vocab.tag_frequencies.get(tag, 1)
                    # Inverse frequency weighting
                    w += (1.0 / max(freq, 1)) ** self.config.sample_weight_power
                    if tag in orientation_tags:
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
        # Convert random_crop_scale to a tuple for comparison; YAML may deserialize it as a list.
        try:
            scale_tuple = tuple(self.config.random_crop_scale)
        except TypeError:
            # If a single number is provided, replicate it to both elements
            scale_tuple = (self.config.random_crop_scale, self.config.random_crop_scale)
        if scale_tuple != (1.0, 1.0):
            transforms.append(
                T.RandomResizedCrop(
                    self.config.image_size,
                    scale=scale_tuple,
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
        loaded images are always returned as ``float32`` tensors in the [0, 1]
        range.
        """
        # Check cache first
        with self.cache_lock:
            if image_path in self.cache:
                cached = self.cache[image_path]
                # Move to end to mark as recently used (LRU behavior)
                self.cache.move_to_end(image_path)
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
            # Add to cache with LRU eviction if needed
            if self.cache_precision == 'uint8':
                cached_tensor = (tensor * 255).to(torch.uint8)
            elif self.cache_precision == 'float16':
                cached_tensor = tensor.half()
            else:
                cached_tensor = tensor.clone()
            
            # Evict least recently used item if cache is at capacity
            with self.cache_lock:
                if len(self.cache) >= self.max_cache_size:
                    # Remove the least recently used item (first item in OrderedDict)
                    self.cache.popitem(last=False)
                
                # Add new item to cache (automatically becomes most recently used)
                self.cache[image_path] = cached_tensor
            
            return tensor
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return black image to avoid crashing caller
            return torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32)

    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch an item for training or validation.

        Applies a random horizontal flip with orientation‑aware tag remapping
        for the training split.  Augmentation and normalisation are then
        applied.
        
        Args:
            idx: Index of item to fetch
            
        Returns:
            Dictionary containing image, labels, and metadata
            
        Raises:
            IndexError: If index is out of bounds
            RuntimeError: If critical error occurs during processing
        """
        if idx < 0 or idx >= len(self.annotations):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.annotations)} samples")
        
        anno = self.annotations[idx]
        image_path = anno['image_path']
        
        # Track error count for this specific image (thread‑safe).  The
        # ``_error_counts`` dictionary is protected by
        # ``_error_counts_lock`` so that concurrent dataloader workers
        # correctly update the retry counters.
        with self._error_counts_lock:
            error_count = self._error_counts.get(image_path, 0)
        max_retries = 3
        
        try:
            # Load image (float32 tensor)
            # Load image (float32 tensor)
            image = self._load_image(image_path)
            
            # Copy tags so we can mutate without altering the original
            tags = list(anno['tags'])
            

            # Random horizontal flip with orientation-aware tag swapping
            # Only attempt flip if handler is available and random check passes
            if (
                self.orientation_handler is not None
                and self.split == 'train'
                and np.random.rand() < self.config.random_flip_prob
            ):
                # Use orientation handler to handle complex tags and determine if flip should proceed
                swapped_tags, should_flip = self.orientation_handler.handle_complex_tags(tags)
                
                if should_flip:
                    # Apply the horizontal flip to the image
                    image = TF.hflip(image)
                    # Use the swapped tags
                    tags = swapped_tags
                    
                    # Update monitoring statistics
                    if self.orientation_monitor:
                        self.orientation_monitor.check_health(self.orientation_handler)
                # If should_flip is False, the handler determined this image should not be flipped
                # (e.g., contains text, signature, or unmapped orientation tags)
            
            # Additional augmentation
            if self.augmentation is not None:
                image = self.augmentation(image)
            
            # Normalise
            image = self.normalize(image)
            
            # Encode tags and rating
            tag_labels = self.vocab.encode_tags(tags)
            rating_label = self.vocab.rating_to_index.get(anno['rating'], self.vocab.rating_to_index['unknown'])
            
            # Reset error count on successful load.  We use a lock to
            # ensure that concurrent workers do not race when updating
            # ``_error_counts``.
            with self._error_counts_lock:
                if image_path in self._error_counts:
                    del self._error_counts[image_path]

            # Package the sample as a dictionary.  Labels are nested
            # under a single ``labels`` key to avoid duplication.  Tag
            # labels are returned as a multi‑hot vector and rating as an
            # integer index.
            return {
                'image': image,
                'labels': {
                    'tags': tag_labels,
                    'rating': rating_label,
                },
                'metadata': {
                    'index': idx,
                    'path': anno['image_path'],
                    'num_tags': anno['num_tags'],
                    'tags': tags,
                    'rating': anno['rating'],
                },
            }
            
        except (IOError, OSError) as e:
            # File I/O errors – may be temporary.  Increment the retry count
            # under the lock.  If the maximum number of retries is
            # exceeded, propagate a runtime error to abort training.
            with self._error_counts_lock:
                self._error_counts[image_path] = error_count + 1
            
            if error_count >= max_retries:
                logger.error(f"Failed to load {image_path} after {max_retries} attempts: {e}")
                # After multiple failures, raise to stop training
                raise RuntimeError(f"Persistent failure loading {image_path}: {e}") from e
            else:
                logger.warning(f"Error loading {image_path} (attempt {error_count + 1}/{max_retries}): {e}")
                # Return an error sample for this attempt
                return self._create_error_sample(idx, image_path, "io_error")
                
        except Exception as e:
            # Unexpected errors should not be silenced
            logger.error(f"Unexpected error in __getitem__ for index {idx}, path {image_path}: {e}")
            raise RuntimeError(f"Unexpected error processing sample {idx}") from e

    def _create_error_sample(self, idx: int, image_path: str, error_type: str) -> Dict[str, Any]:
        """Create an error sample with appropriate defaults.
        
        This should only be used for recoverable errors like temporary I/O issues.
        """
        logger.debug(f"Creating error sample for {image_path}, type: {error_type}")
        return {
            'image': torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32),
            'labels': {
                'tags': torch.zeros(len(self.vocab.tag_to_index), dtype=torch.float32),
                'rating': torch.tensor(self.vocab.rating_to_index.get('unknown', 4), dtype=torch.long),
            },
            'metadata': {
                'index': idx,
                'path': image_path,
                'num_tags': 0,
                'tags': [],
                'rating': 'unknown',
                'error_type': error_type,
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
    # Extract nested labels.  Tag labels are stacked into a 2D tensor and
    # rating labels are collected into a 1D tensor.
    tag_labels = torch.stack([item['labels']['tags'] for item in batch])
    rating_labels = torch.tensor([item['labels']['rating'] for item in batch], dtype=torch.long)
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