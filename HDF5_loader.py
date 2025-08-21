#!/usr/bin/env python3
"""
Data loading and augmentation utilities for the anime tagger.

This module provides a simplified HDF5/JSON loader with support for
letterbox-style resizing.  Images are resized to fit within a square
canvas while preserving aspect ratio and padded with a neutral colour.
Padding information is returned so downstream modules can mask out
padded regions (e.g. during vision transformer patchification).

The default pad colour is a mid‑grey (114,114,114) as commonly used by
YOLO models.  The pad colour and patch size are configurable via
``SimplifiedDataConfig``.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy
from pathlib import Path
import json
import threading
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from collections import OrderedDict
from logging.handlers import QueueHandler

from orientation_handler import OrientationHandler, OrientationMonitor  # type: ignore
# Import TagVocabulary from the vocabulary module rather than a relative package path.
# The vocabulary module should reside on the Python path for this import to succeed.
from vocabulary import TagVocabulary

logger = logging.getLogger(__name__)


def letterbox_resize(
    image: torch.Tensor,
    target_size: int,
    pad_color: Iterable[int] = (114, 114, 114),
    patch_size: int = 16,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Resize an image tensor to a square canvas while preserving aspect ratio.

    The input image is first scaled by the minimal factor required to fit
    within the ``target_size``.  Any remaining space is padded equally on
    each side using the provided ``pad_color``.  If the resulting
    dimensions are not divisible by ``patch_size`` then additional padding
    is applied to the bottom and right edges so that the final width and
    height are multiples of ``patch_size``.  This ensures that when the
    image is partitioned into non‑overlapping patches (e.g. for a vision
    transformer) no partial patches are dropped.

    Args:
        image: Input image tensor of shape (C, H, W) with values in [0, 1].
        target_size: Desired square size (both height and width) of the output.
        pad_color: RGB colour used to fill padded regions.  Values should be
            integers in the 0‑255 range.
        patch_size: Patch size used by the downstream model.  The output
            dimensions will be rounded up to the next multiple of this value.

    Returns:
        A tuple containing:
          * The padded image tensor of shape (C, H_out, W_out).
          * A dictionary with keys ``scale`` and ``pad`` describing the
            applied scaling factor and padding on each side (left, top,
            right, bottom).  These values can be used to derive padding
            masks during patchification.
    """
    c, h, w = image.shape
    # Compute scaling factor to fit the longer side into target_size.
    r = min(target_size / float(h), target_size / float(w))
    # Avoid degenerate new sizes.
    new_h = int(round(h * r))
    new_w = int(round(w * r))
    # Resize using bilinear interpolation on tensors.
    resized = TF.resize(image, [new_h, new_w], interpolation=T.InterpolationMode.BILINEAR)
    # Compute padding needed to reach target_size.
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # After initial letterbox, ensure divisibility by patch_size.
    final_h = target_size
    final_w = target_size
    extra_pad_bottom = 0
    extra_pad_right = 0
    # Only apply divisibility padding when patch_size > 1; None is treated as no divisibility requirement
    if patch_size > 1:
        if final_h % patch_size != 0:
            extra_pad_bottom = patch_size - (final_h % patch_size)
        if final_w % patch_size != 0:
            extra_pad_right = patch_size - (final_w % patch_size)
    final_h += extra_pad_bottom
    final_w += extra_pad_right
    # Create canvas and fill with pad colour.
    canvas = torch.zeros((c, final_h, final_w), dtype=resized.dtype)
    # Normalise pad colour to [0,1] range.
    pad_vals = torch.tensor(pad_color, dtype=resized.dtype) / 255.0
    for ch in range(c):
        canvas[ch, :, :] = pad_vals[ch]
    # Paste resized image into the centre region.
    start_y = pad_top
    end_y = pad_top + new_h
    start_x = pad_left
    end_x = pad_left + new_w
    canvas[:, start_y:end_y, start_x:end_x] = resized
    info = {
        "scale": r,
        "pad": (pad_left, pad_top, pad_right + extra_pad_right, pad_bottom + extra_pad_bottom),
        "out_size": (final_h, final_w),
        "in_size": (h, w),
    }
    return canvas, info


@dataclass
class SimplifiedDataConfig:
    """
    Configuration parameters controlling data loading and augmentation.

    These settings are tailored for high‑resolution anime artwork.  The
    ``pad_color`` is used during letterbox resizing to fill any empty
    regions and should match the neutral colour used during inference.
    ``patch_size`` defines the patch size expected by the vision
    transformer; ``image_size`` should be divisible by ``patch_size``.
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
    # Padding colour used for letterbox resizing (RGB in 0‑255 range).
    pad_color: Tuple[int, int, int] = (114, 114, 114)
    # Patch size for downstream model.  ``image_size`` should be divisible by this.
    patch_size: int = 16

    # Vocabulary settings
    top_k_tags: int = 100_000
    min_tag_frequency: int = 1  # include all tags that appear at least once

    # Augmentation settings
    augmentation_enabled: bool = True
    # Reduce flip probability because many tags encode left/right semantics
    random_flip_prob: float = 0.2
    # Safer default: fail if flips requested but mapping invalid
    strict_orientation_validation: bool = True
    skip_unmapped: bool = True
    # Narrow crop scale range to preserve most of the subject in the frame.
    # When (1.0, 1.0) no random cropping is performed.
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

    # Transparency handling
    update_transparency_tags: bool = True  # Whether to update tags when compositing transparent images
    composite_background_tag: str = "gray_background"  # Tag to add when compositing

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

    # Error handling settings
    skip_error_samples: bool = True 
    validate_on_init: bool = True  

    def __post_init__(self) -> None:
        # Validate flip probability
        if not 0.0 <= self.random_flip_prob <= 1.0:
            raise ValueError(f"random_flip_prob must be in [0, 1], got {self.random_flip_prob}")
        # Validate cache precision
        if self.cache_precision not in {'float32', 'float16', 'uint8'}:
            raise ValueError(
                f"cache_precision must be one of 'float32', 'float16' or 'uint8', got {self.cache_precision}"
            )
        # Ensure image_size divisible by patch_size
        if self.image_size % self.patch_size != 0:
            logger.warning(
                f"image_size ({self.image_size}) is not divisible by patch_size ({self.patch_size}); "
                f"letterbox_resize will pad further to the next multiple."
            )


def _make_worker_init_fn(base_seed: int, log_queue: Optional[object]):
    """Seed torch/numpy/random per worker and disable file logging in workers.
    If a logging queue is provided, attach a QueueHandler so logs go to main.
    """
    def _init_fn(worker_id: int):
        seed = (base_seed + worker_id) % (2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Remove any FileHandlers in workers (avoid cross-process file writes)
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                from logging import FileHandler
                if isinstance(h, FileHandler):
                    root.removeHandler(h)
            except Exception:
                pass

        # Attach QueueHandler if queue supplied
        if log_queue is not None:
            try:
                qh = QueueHandler(log_queue)
                qh.setLevel(logging.INFO)
                root.addHandler(qh)
            except Exception:
                pass
    return _init_fn


class SimplifiedDataset(Dataset):
    """Dataset for anime image tagging with augmentation and sampling.

    Each item returned is a dictionary containing an image tensor,
    multi‑hot tag labels, a rating label and some metadata (index, path,
    original tag list and rating string).  Letterbox resizing is applied
    on the fly to preserve the original aspect ratio and avoid dropping
    edge pixels when splitting into patches.  Padding information is
    returned in the metadata for use by downstream modules.
    """

    def __init__(
            self,
            config: SimplifiedDataConfig,
            json_files: List[Path],
            split: str,
            vocab: TagVocabulary,
        ) -> None:
            # Create a deep copy of config to avoid mutation by downstream components
            config = copy.deepcopy(config)

            assert split in {'train', 'val', 'test'}, f"Unknown split '{split}'"
            self.config = config
            self.split = split
            self.vocab = vocab
            # List of annotation dictionaries loaded from JSON files
            self.annotations: List[Dict[str, Any]] = []
            # LRU cache for loaded images
            self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

            # Ensure orientation_map_path is a Path object if provided
            self._orientation_stats = {'flips': 0, 'skipped': 0, 'processed': 0}           
            if config.orientation_map_path and isinstance(config.orientation_map_path, str):
                config.orientation_map_path = Path(config.orientation_map_path)
            # Initialize orientation handler for flip augmentation
            # Only needed for training split when flips are enabled
            if split == 'train' and config.random_flip_prob > 0:
                self.orientation_handler = OrientationHandler(
                    mapping_file=config.orientation_map_path,
                    random_flip_prob=config.random_flip_prob,
                    strict_mode=config.strict_orientation_validation,
                    skip_unmapped=config.skip_unmapped
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
                        from pathlib import Path as _Path  # avoid namespace confusion in compiled docs
                        validation_report_path = _Path("orientation_validation_report.json")
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
            # Approximate bytes per cached image by assuming they will be stored at config.image_size
            bytes_per_image = 3 * config.image_size * config.image_size * bytes_per_element
            self.max_cache_size = int((config.cache_size_gb * (1024 ** 3)) / bytes_per_image)
            self.cache_precision = config.cache_precision
            self.cache_lock = threading.Lock()  # Add thread lock for cache operations

            # Initialise locks and counters for error handling.  `_error_counts` is
            # used to track temporary I/O failures per image and must be
            # accessed under `_error_counts_lock` to ensure thread safety.
            self._error_counts_lock = threading.Lock()
            self._error_counts: Dict[str, int] = {}
            # Track permanently failed images
            self._failed_images: set[str] = set()

            # Optionally validate all images are readable at initialization
            if config.validate_on_init and split == 'train':
                self._validate_dataset_images()

            # Filter out known bad images if skip_error_samples is enabled
            self._filter_failed_images()            

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

                # Normalize to list format: single dict → [dict], invalid → skip
                if isinstance(data, dict):
                    # Single annotation object - wrap in list
                    data = [data]
                elif not isinstance(data, list):
                    # Neither dict nor list - invalid format
                    logger.warning(
                        f"Skipping {json_file}: expected list or dict, got {type(data).__name__}"
                    )
                    continue

                for entry in data:
                    # Validate entry is a dict before accessing keys
                    if not isinstance(entry, dict):
                        logger.warning(f"Skipping non-dict entry in {json_file}: {type(entry).__name__}")
                        continue

                    filename = entry.get('filename')
                    tags_field = entry.get('tags')
                    if not filename or not tags_field:
                        continue
                    tags_list: List[str]
                    if isinstance(tags_field, str):
                        tags_list = tags_field.split()
                    elif isinstance(tags_field, list):
                        tags_list = tags_field
                    else:
                        continue
                    # Deduplicate tags while preserving order
                    seen = set()
                    deduplicated_tags = []
                    for tag in tags_list:
                        if tag and tag not in seen:  # Also filter out empty strings
                            seen.add(tag)
                            deduplicated_tags.append(tag)

                    record: Dict[str, Any] = {
                        'image_path': str(self.config.data_dir / filename),
                        'tags': deduplicated_tags,
                        'rating': entry.get('rating', 'unknown'),
                        'num_tags': len(deduplicated_tags)
                    }
                    self.annotations.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")

    def _validate_dataset_images(self) -> None:
        """Validate that all images in the dataset are readable.
        
        This runs at initialization to identify problematic images early
        rather than discovering them during training.
        """
        logger.info("Validating dataset images...")
        failed_count = 0
        total = len(self.annotations)
        
        for i, anno in enumerate(self.annotations):
            if i % 1000 == 0:
                logger.info(f"Validated {i}/{total} images...")
            
            image_path = anno['image_path']
            try:
                # Quick validation - just try to open and get size
                with Image.open(image_path) as img:
                    _ = img.size
            except Exception as e:
                logger.warning(f"Image validation failed for {image_path}: {e}")
                self._failed_images.add(image_path)
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(
                f"Found {failed_count}/{total} ({failed_count/total*100:.1f}%) unreadable images. "
                f"These will be {'skipped' if self.config.skip_error_samples else 'replaced with blank samples'}."
            )
            # Save report of failed images
            report_path = Path("failed_images_report.txt")
            with open(report_path, 'w') as f:
                for path in sorted(self._failed_images):
                    f.write(f"{path}\n")
            logger.info(f"Saved list of failed images to {report_path}")

    def _filter_failed_images(self) -> None:
        """Remove known failed images from annotations if skip_error_samples is enabled."""
        if self.config.skip_error_samples and self._failed_images:
            original_count = len(self.annotations)
            self.annotations = [a for a in self.annotations if a['image_path'] not in self._failed_images]
            filtered_count = original_count - len(self.annotations)
            logger.info(f"Filtered out {filtered_count} samples with known bad images")

    def _calculate_sample_weights(self) -> None:
        """Compute sampling weights for frequency‑weighted sampling."""
        weights: List[float] = []
        # Build a set of tags that influence orientation oversampling
        orientation_tags = set()
        if self.orientation_handler is not None and self.precomputed_mappings:
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
            return TF.adjust_gamma(img, gamma=gamma)
        transforms.append(T.Lambda(gamma_transform))
        return T.Compose(transforms) if transforms else None

    def __len__(self) -> int:
        return len(self.annotations)

    def _load_image(self, image_path: str) -> Tuple[torch.Tensor, bool]:
        """Load an image from disk or cache and return it as a float tensor.
        """
        # Check cache first
        with self.cache_lock:
            if image_path in self.cache:
                cached_data = self.cache[image_path]
                # Move to end to mark as recently used (LRU behavior)
                self.cache.move_to_end(image_path)

                # Handle both old cache format (just tensor) and new (tuple with flag)
                if isinstance(cached_data, tuple):
                    cached, was_composited = cached_data
                else:
                    cached = cached_data
                    was_composited = False

                # Convert cached tensor back to float32
                if cached.dtype == torch.uint8:
                    result = cached.float() / 255.0
                elif cached.dtype == torch.float16:
                    result = cached.float()
                else:
                    result = cached.clone()
                return result, was_composited
        # Load from disk
        try:
            # Open without forcing RGB so we can properly handle alpha first
            pil_img = Image.open(image_path)
            was_composited = False

            # Resolve a safe, neutral gray pad/composite color from config (works with dict or object)
            def _resolve_pad_color(cfg, default=(128, 128, 128)):
                # cfg may be an object with attribute or a dict with key
                val = getattr(cfg, "pad_color", None)
                if val is None and isinstance(cfg, dict):
                    val = cfg.get("pad_color", None)
                if val is None:
                    return default
                # Accept [r,g,b] as ints 0..255 or floats 0..1
                if isinstance(val, (list, tuple)):
                    if len(val) >= 3:
                        # If all floats in [0,1], scale to [0,255]
                        if all(isinstance(v, float) for v in val) and all(0.0 <= v <= 1.0 for v in val[:3]):
                            return tuple(int(round(255.0 * v)) for v in val[:3])
                        return tuple(int(v) for v in val[:3])
                if isinstance(val, int):
                    return (val, val, val)
                return default

            pad_color = _resolve_pad_color(self.config)

            # Composite transparent images with error handling
            try:
                if pil_img.mode in ('RGBA', 'LA') or ('transparency' in pil_img.info):
                    was_composited = True  # Mark that we're compositing
                    rgba = pil_img.convert('RGBA')
                    bg = Image.new('RGB', rgba.size, pad_color)
                    bg.paste(rgba, mask=rgba.split()[3])  # use alpha as mask
                    pil_img = bg  # now RGB on neutral gray background
                else:
                    pil_img = pil_img.convert('RGB')
            except Exception as e:
                logger.warning(f"Alpha compositing failed for {image_path}, converting to RGB directly: {e}")
                pil_img = pil_img.convert('RGB')
                was_composited = False

            tensor = TF.to_tensor(pil_img)  # float32 in [0, 1]

            # Add to cache with LRU eviction if needed (atomic operation)
            with self.cache_lock:
                if self.cache_precision == 'uint8':
                    cached_tensor = (tensor * 255).to(torch.uint8)
                elif self.cache_precision == 'float16':
                    cached_tensor = tensor.half()
                else:
                    cached_tensor = tensor.clone()

                if len(self.cache) >= self.max_cache_size:
                    # Remove the least recently used item (first item in OrderedDict)
                    self.cache.popitem(last=False)
                # Store as tuple with composite flag
                self.cache[image_path] = (cached_tensor, was_composited)

            return tensor, was_composited
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return black image to avoid crashing caller
            return torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32), False

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch an item for training or validation.

        Applies a random horizontal flip with orientation‑aware tag remapping
        for the training split.  Letterbox resizing is performed to
        preserve aspect ratio.  Augmentation and normalisation are then
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

        # Skip if this is a known failed image and skip_error_samples is enabled
        if self.config.skip_error_samples and image_path in self._failed_images:
            # Return next valid sample (with wraparound)
            next_idx = (idx + 1) % len(self.annotations)
            logger.debug(f"Skipping known failed image at index {idx}, using index {next_idx}")
            return self.__getitem__(next_idx)

        # Track error count for this specific image (thread‑safe).
        with self._error_counts_lock:
            error_count = self._error_counts.get(image_path, 0)
        max_retries = 3
        try:
            # Load image tensor and get composite flag
            image, was_composited = self._load_image(image_path)
            # Copy tags so we can mutate without altering the original
            tags = list(anno['tags'])

            # Update tags if image was composited from transparency
            if was_composited:
                # Remove transparent_background if present
                tags = [t for t in tags if t != 'transparent_background']
                # Add gray_background if not already present
                if 'gray_background' not in tags:
                    tags.append('gray_background')

            # Random horizontal flip with orientation-aware tag swapping
            if (
                self.orientation_handler is not None
                and self.split == 'train'
                and np.random.rand() < self.config.random_flip_prob
            ):
                swapped_tags, should_flip = self.orientation_handler.handle_complex_tags(tags)
                if should_flip:
                    image = TF.hflip(image)
                    self._orientation_stats['flips'] += 1
                else:
                    self._orientation_stats['skipped'] += 1
                self._orientation_stats['processed'] += 1
                tags = swapped_tags
                if self.orientation_monitor:
                    self.orientation_monitor.check_health(self.orientation_handler)
            # Perform letterbox resize to preserve aspect ratio
            image, lb_info = letterbox_resize(
                image,
                target_size=self.config.image_size,
                pad_color=self.config.pad_color,
                patch_size=self.config.patch_size,
            )
            # Apply additional augmentation (colour jitter, gamma) if enabled
            if self.augmentation is not None:
                image = self.augmentation(image)
            # Normalise
            image = self.normalize(image)
            # Encode tags and rating
            tag_labels = self.vocab.encode_tags(tags)
            rating_label = self.vocab.rating_to_index.get(anno['rating'], self.vocab.rating_to_index['unknown'])
            # Reset error count on successful load.
            with self._error_counts_lock:
                if image_path in self._error_counts:
                    del self._error_counts[image_path]
            # Package the sample as a dictionary.  Labels are nested
            # under a single ``labels`` key to avoid duplication.  Tag
            # labels are returned as a multi‑hot vector and rating as an
            # integer index.  Include scaling and padding information in
            # metadata for downstream use.
            return {
                'image': image,
                'labels': {
                    'tags': tag_labels,
                    'rating': rating_label,
                },
                'metadata': {
                    'index': idx,
                    'path': anno['image_path'],
                    'num_tags': len(tags),  # Updated count
                    'tags': tags,  # Updated tags
                    'rating': anno['rating'],
                    'scale': lb_info['scale'],
                    'pad': lb_info['pad'],
                    'was_composited': was_composited,  # Add to metadata for debugging
                },
            }
        except (IOError, OSError) as e:
            # File I/O errors – may be temporary.  Increment the retry count
            # under the lock.  If the maximum number of retries is
            # exceeded, propagate a runtime error to abort training.
            with self._error_counts_lock:
                self._error_counts[image_path] = error_count + 1               
                if error_count >= max_retries:
                    self._failed_images.add(image_path)

            if error_count >= max_retries:
                logger.error(f"Failed to load {image_path} after {max_retries} attempts: {e}")
                if self.config.skip_error_samples:
                    # Skip to next sample instead of crashing
                    next_idx = (idx + 1) % len(self.annotations)
                    logger.info(f"Skipping permanently failed image, moving to index {next_idx}")
                    return self.__getitem__(next_idx)
                else:
                    raise RuntimeError(f"Persistent failure loading {image_path}: {e}") from e
            else:
                logger.warning(f"Error loading {image_path} (attempt {error_count + 1}/{max_retries}): {e}")
                if self.config.skip_error_samples:
                    return self.__getitem__((idx + 1) % len(self.annotations))                
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
        # Use letterbox padding on a zero image so downstream patchify has correct shape.
        blank = torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32)
        padded, lb_info = letterbox_resize(
            blank,
            target_size=self.config.image_size,
            pad_color=self.config.pad_color,
            patch_size=self.config.patch_size,
        )
        # Ensure consistent tensor shapes and dtypes
        tag_labels = torch.zeros(len(self.vocab.tag_to_index), dtype=torch.float32)
        rating_label = torch.tensor(self.vocab.rating_to_index.get('unknown', 4), dtype=torch.long)

        return {
            'image': padded,
            'labels': {
                'tags': tag_labels,
                'rating': rating_label,
            },
            'metadata': {
                'index': idx,
                'path': image_path,
                'num_tags': 0,
                'tags': [],
                'rating': 'unknown',
                'error_type': error_type,
                'scale': lb_info['scale'],
                'pad': lb_info['pad'],
                'was_composited': False,
            },
        }


    def get_orientation_stats(self) -> Dict[str, Any]:
        """Get orientation statistics from this dataset instance.
        
        Note: In multi-worker settings, these are local to this worker process.
        Statistics are not aggregated across workers automatically.
        
        Returns:
            Dictionary with orientation statistics
        """
        if not hasattr(self, '_orientation_stats'):
            return {
                'total_flips': 0,
                'skipped_flips': 0,
                'processed_samples': 0,
                'flip_rate': 0.0,
                'skip_rate': 0.0,
                'has_handler': False,
                'worker_local': True
            }
        
        stats = self._orientation_stats.copy()
        processed = stats.get('processed', 0)
        
        return {
            'total_flips': stats.get('flips', 0),
            'skipped_flips': stats.get('skipped', 0),
            'processed_samples': processed,
            'flip_rate': stats['flips'] / max(1, processed),
            'skip_rate': stats['skipped'] / max(1, processed),
            'has_handler': self.orientation_handler is not None,
            'worker_local': True  # Flag to indicate these are worker-local stats
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
    seed: Optional[int] = None,
    log_queue: Optional[object] = None,
    force_val_persistent_workers: bool = False,
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

    # Deterministic generator and per-worker init
    base_seed = int(seed if seed is not None else torch.initial_seed() % (2**31 - 1))
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    worker_init_fn = _make_worker_init_fn(base_seed, log_queue)

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
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0 and force_val_persistent_workers),
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return train_loader, val_loader, vocab


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to assemble a batch of samples."""
    images = torch.stack([item['image'] for item in batch])
    # Extract nested labels.  Tag labels are stacked into a 2D tensor and
    # rating labels are collected into a 1D tensor.
    tag_labels = torch.stack([item['labels']['tags'] for item in batch])
    rating_labels = torch.tensor([item['labels']['rating'] for item in batch], dtype=torch.long)
    # Collate metadata lists and keep padding info for downstream usage.
    metadata = {
        'indices': [item['metadata']['index'] for item in batch],
        'paths': [item['metadata']['path'] for item in batch],
        'num_tags': torch.tensor([item['metadata']['num_tags'] for item in batch]),
        'tags': [item['metadata']['tags'] for item in batch],
        'ratings': [item['metadata']['rating'] for item in batch],
        'scales': [item['metadata'].get('scale') for item in batch],
        'pads': [item['metadata'].get('pad') for item in batch],
    }
    # Derive a per-pixel padding mask (True=content, False=padding) so downstream
    # modules (e.g., ViT attention) can ignore padded regions.
    B, C, H, W = images.shape
    padding_mask = torch.ones((B, H, W), dtype=torch.bool)
    for i, pad in enumerate(metadata['pads']):
        if pad is None:
            continue
        left, top, right, bottom = pad
        if top > 0:
            padding_mask[i, :top, :] = False
        if bottom > 0:
            padding_mask[i, H - bottom:, :] = False
        if left > 0:
            padding_mask[i, :, :left] = False
        if right > 0:
            padding_mask[i, :, W - right:] = False
    return {
        'images': images,
        'tag_labels': tag_labels,
        'rating_labels': rating_labels,
        'padding_mask': padding_mask,  # (B, H, W), bool
        'metadata': metadata,
    }