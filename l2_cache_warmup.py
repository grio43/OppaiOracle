#!/usr/bin/env python3
"""
Sidecar Cache Warmup Script for OppaiOracle Training Pipeline

Pre-populates the sidecar cache by processing a configurable percentage of the
dataset through the existing data loading pipeline. This reduces first-epoch
overhead and makes subsequent training runs start fast immediately.

IMPORTANT: Only unflipped images are cached (flipped versions are computed
on-the-fly during training). This is by design to save disk space and allow
epoch-varying flips.

PERFORMANCE RECOMMENDATION:
    GPU mode is 3-5x faster than CPU mode and recommended for large datasets
    (>10K images) or when available. GPU mode also eliminates any potential
    duplicate processing by using single-threaded sequential processing.

Usage:
    # GPU-accelerated mode (RECOMMENDED - 3-5x faster):
    python l2_cache_warmup.py --config configs/unified_config.yaml --use-gpu --target-vram-gb 31.5

    # CPU multi-process mode (fallback if GPU unavailable):
    python l2_cache_warmup.py --config configs/unified_config.yaml --percentage 50 --workers 16

See GPU_WARMUP_GUIDE.md for detailed GPU configuration and performance tuning.
"""

from __future__ import annotations
import os
import sys
import time
import signal
import atexit
import random
import logging
import argparse
import json
from pathlib import Path

# Try to use orjson for faster JSON parsing (3-5x faster than stdlib json)
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False
from typing import Optional, Tuple, List, Union

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import from existing codebase
from Configuration_System import load_config, FullConfig
from dataset_loader import SidecarJsonDataset, load_vocabulary_for_training, SharedVocabularyManager
from cache_codec import get_sidecar_path, save_sidecar
from utils.cache_monitor import monitor
from utils.cache_keys import compute_cache_config_hash

# GPU batch processing (optional)
try:
    from gpu_batch_processor import GPUBatchProcessor
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    GPUBatchProcessor = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CacheWarmupConfig:
    """Configuration for cache warmup script."""

    def __init__(self, args: argparse.Namespace, full_config: FullConfig):
        self.config_path = args.config
        self.percentage = args.percentage
        self.num_workers = args.workers
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.check_existing = args.check_existing
        self.force = args.force if hasattr(args, 'force') else False
        self.force_rebuild_metadata_cache = args.force_rebuild_metadata_cache if hasattr(args, 'force_rebuild_metadata_cache') else False

        # GPU acceleration settings
        self.use_gpu = args.use_gpu if hasattr(args, 'use_gpu') else False
        self.target_vram_gb = args.target_vram_gb if hasattr(args, 'target_vram_gb') else 31.5
        self.target_vram_util = args.target_vram_util if hasattr(args, 'target_vram_util') else 0.9
        self.gpu_device_id = args.gpu_device if hasattr(args, 'gpu_device') else 0
        self.max_batch_size = args.max_batch_size if hasattr(args, 'max_batch_size') else 2048

        # Async preload settings
        self.enable_preload = not (args.no_preload if hasattr(args, 'no_preload') else False)
        self.preload_workers = args.preload_workers if hasattr(args, 'preload_workers') else 4
        self.preload_queue_depth = args.preload_queue_depth if hasattr(args, 'preload_queue_depth') else 2
        self.preload_ram_headroom_gb = args.preload_ram_headroom_gb if hasattr(args, 'preload_ram_headroom_gb') else 8.0

        # Extract from unified config
        self.full_config = full_config
        data_cfg = full_config.data

        # Dataset paths
        storage_locations = data_cfg.storage_locations
        active_location = next((loc for loc in storage_locations if loc.get('enabled', False)), None)
        if not active_location:
            raise ValueError("No enabled storage location found in config")
        self.data_path = Path(active_location['path'])

        # Cache configuration
        self.sidecar_storage_dtype = getattr(data_cfg, 'sidecar_storage_dtype', 'bfloat16')

        # Preprocessing parameters (must match training exactly!)
        self.image_size = data_cfg.image_size
        self.pad_color = tuple(data_cfg.pad_color)
        self.normalize_mean = tuple(data_cfg.normalize_mean)
        self.normalize_std = tuple(data_cfg.normalize_std)

        # Vocabulary path
        self.vocab_path = Path(full_config.vocab_path)

        # Load vocab size early for cache key consistency (matches dataset_loader behavior)
        self._vocab_size: Optional[int] = None
        if self.vocab_path.exists():
            try:
                vocab = load_vocabulary_for_training(self.vocab_path)
                self._vocab_size = len(vocab.tag_to_index)
            except Exception as e:
                logger.warning(f"Could not load vocabulary for cache hash: {e}")

        # Compute config hash once (must match dataset_loader computation exactly!)
        # Warmup does NOT support joint transforms, so has_joint_transforms=False
        self.config_hash = compute_cache_config_hash(
            image_size=self.image_size,
            pad_color=self.pad_color,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            storage_dtype=self.sidecar_storage_dtype,
            vocab_size=self._vocab_size,
            has_joint_transforms=False,  # Warmup never uses joint transforms
        )

        # Disable flipping (only cache unflipped versions)
        self.random_flip_prob = 0.0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file does not exist: {self.vocab_path}")

        if not (1 <= self.percentage <= 100):
            raise ValueError(f"Percentage must be 1-100, got {self.percentage}")

        if self.num_workers < 1:
            raise ValueError(f"Workers must be >= 1, got {self.num_workers}")

        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.batch_size}")

        # Check GPU requirements
        if self.use_gpu:
            if not GPU_AVAILABLE:
                raise RuntimeError(
                    "GPU mode requested but CUDA is not available. "
                    "Either install CUDA-enabled PyTorch or run without --use-gpu flag."
                )
            if GPUBatchProcessor is None:
                raise ImportError(
                    "GPU mode requested but gpu_batch_processor module not found. "
                    "Ensure gpu_batch_processor.py is in the same directory."
                )
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("bfloat16 GPU warmup requested but CUDA device does not support bf16.")
            logger.info(f"GPU acceleration enabled (device {self.gpu_device_id})")
            logger.info(f"  Target VRAM: {self.target_vram_gb:.1f} GB")
            logger.info(f"  Target utilization: {self.target_vram_util * 100:.1f}%")

        # Enforce bfloat16-only cache to avoid fp32/fp16 VRAM usage.
        if self.sidecar_storage_dtype != 'bfloat16':
            raise ValueError(f"Only bfloat16 sidecar storage is supported, got '{self.sidecar_storage_dtype}'.")
        if not hasattr(torch, 'bfloat16'):
            raise RuntimeError("bfloat16 sidecar storage requested but PyTorch does not support bfloat16.")

        # IMPORTANT: Warmup does NOT support joint transforms (geometry augmentations).
        # If your training config uses joint_transforms, the cache will be invalidated
        # at training time due to config hash mismatch (has_joint_transforms=False vs True).
        # This is by design: joint transforms should be applied at training time, not cached.
        logger.warning(
            "IMPORTANT: Cache warmup does NOT support joint transforms (geometry augmentations). "
            "If your training config uses joint_transforms, ALL cached images will be "
            "automatically invalidated and reprocessed during training (config hash mismatch). "
            "This is expected behavior - joint transforms must be applied fresh each epoch."
        )

        logger.info(f"Cache warmup configuration validated successfully")
        logger.info(f"  Mode: {'GPU-accelerated' if self.use_gpu else 'CPU multi-process'}")
        if not self.use_gpu:
            logger.info(f"  TIP: Add --use-gpu for 3-5x faster warmup (see GPU_WARMUP_GUIDE.md)")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Image size: {self.image_size}x{self.image_size}")
        logger.info(f"  Storage dtype: {self.sidecar_storage_dtype}")
        logger.info(f"  Warmup: {self.percentage}% of dataset")
        logger.info(f"  Workers: {self.num_workers if not self.use_gpu else 'N/A (GPU mode)'}")
        logger.info(f"  Batch size: {self.batch_size if not self.use_gpu else 'Dynamic (GPU mode)'}")

def discover_json_files(data_path: Path) -> List[Path]:
    """
    Discover all JSON sidecar files in the dataset.

    Mirrors the logic in dataset_loader.py lines 2367-2390.
    """
    logger.info(f"Discovering JSON files in {data_path}...")

    json_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                json_path = Path(root) / file
                json_files.append(json_path)

    logger.info(f"Found {len(json_files)} JSON files")
    return json_files


class SidecarCacheWorkerInitializer:
    """Picklable worker initialization for sidecar cache warmup DataLoader."""

    def __init__(self, shared_vocab_info: Optional[Tuple[str, int]] = None):
        """
        Args:
            shared_vocab_info: Tuple of (shm_name, vocab_size) for shared vocabulary
        """
        self.shared_vocab_info = shared_vocab_info

    def __call__(self, worker_id: int):
        """Initialize worker with random seed and vocabulary.

        Args:
            worker_id: Worker process ID
        """
        # Set random seed for reproducibility
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)

        # Load vocabulary from shared memory if provided
        # Currently a no-op as vocabulary loading happens in dataset's __getitem__
        if self.shared_vocab_info is not None:
            shm_name, vocab_size = self.shared_vocab_info
            # Vocabulary loading happens in dataset's __getitem__ via lazy init
            pass


class CacheWarmup:
    """Main cache warmup orchestrator for sidecar cache."""

    def __init__(self, config: CacheWarmupConfig):
        self.config = config
        self.vocab = None
        self.shared_vocab_manager: Optional[SharedVocabularyManager] = None
        self.interrupted = False
        self.failed_images: List[Path] = []  # Track paths of images that failed to process

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.warning("\nReceived interrupt signal, shutting down gracefully...")
        self.interrupted = True

    def load_vocabulary(self) -> None:
        """Load vocabulary and create shared memory version for workers."""
        logger.info(f"Loading vocabulary from {self.config.vocab_path}...")

        # Load vocabulary
        self.vocab = load_vocabulary_for_training(self.config.vocab_path)
        logger.info(f"Vocabulary loaded: {len(self.vocab.tag_to_index)} tags")

        # Create shared memory vocabulary for workers
        self.shared_vocab_manager = SharedVocabularyManager()
        shm_name = self.shared_vocab_manager.create_from_vocab(self.vocab)
        self.shared_vocab_info = (shm_name, self.shared_vocab_manager.vocab_size)

        # Register cleanup
        atexit.register(self.shared_vocab_manager.cleanup)

        logger.info("Shared vocabulary created for worker processes")

    def create_dataset(self, json_files: List[Path]) -> SidecarJsonDataset:
        """
        Create SidecarJsonDataset with warmup-specific configuration.

        CRITICAL: Must match training preprocessing exactly!
        """
        logger.info("Creating dataset...")

        # Create dataset with warmup overrides
        dataset = SidecarJsonDataset(
            root_dir=self.config.data_path,
            json_files=json_files,
            vocab=self.vocab,
            transform=None,  # No extra transforms

            # Image preprocessing (must match training!)
            image_size=self.config.image_size,
            pad_color=self.config.pad_color,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,

            # Sidecar cache configuration
            sidecar_cache_enabled=True,
            sidecar_storage_dtype=self.config.sidecar_storage_dtype,
            cpu_bf16_cache_pipeline=True,

            # Flip configuration (disabled - only cache unflipped)
            random_flip_prob=self.config.random_flip_prob,
            orientation_handler=None,  # No flip logic needed
            flip_overrides_path=None,

            # Stats (optional)
            stats_queue=None,
        )

        # Set epoch to 0 for consistency
        dataset.set_epoch(0)

        logger.info(f"Dataset created: {len(dataset)} samples")
        return dataset

    def create_subset(self, dataset: SidecarJsonDataset) -> Union[SidecarJsonDataset, Subset]:
        """Create subset based on percentage."""
        total_samples = len(dataset)
        num_samples = int(total_samples * self.config.percentage / 100)

        if num_samples == total_samples:
            logger.info(f"Warming 100% of dataset ({total_samples} samples)")
            return dataset

        # Random sampling with seed for reproducibility
        random.seed(self.config.seed)
        indices = random.sample(range(total_samples), num_samples)
        indices.sort()  # Sort for better cache locality

        subset = Subset(dataset, indices)
        logger.info(f"Warming {self.config.percentage}% of dataset ({num_samples}/{total_samples} samples)")

        return subset

    def create_dataloader(self, dataset) -> DataLoader:
        """Create DataLoader with worker initialization."""
        worker_init_fn = SidecarCacheWorkerInitializer(self.shared_vocab_info)

        # Use spawn context to avoid fork issues
        mp_context = mp.get_context('spawn')

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Sequential access for better cache locality
            num_workers=self.config.num_workers,
            pin_memory=False,  # Not training, no GPU transfer
            drop_last=False,
            persistent_workers=False,  # One-shot warmup
            prefetch_factor=2,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=mp_context,
        )

        logger.info(f"DataLoader created: {self.config.num_workers} workers, batch size {self.config.batch_size}")
        return dataloader

    def run_warmup(self, dataloader: DataLoader) -> None:
        """
        Run cache warmup loop.

        Simply iterating through the DataLoader triggers cache writes in
        SidecarJsonDataset.__getitem__() for cache misses.
        """
        logger.info("Starting cache warmup...")
        logger.info("Note: Only unflipped images are cached (flipped versions computed on-the-fly)")
        logger.info("Note: Failed/corrupted images will be automatically tracked and excluded from training")

        start_time = time.time()
        total_items = 0

        # Progress bar
        with tqdm(
            dataloader,
            desc="Warming cache",
            unit="batch",
            total=len(dataloader),
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Check for interrupt
                if self.interrupted:
                    logger.warning("Warmup interrupted by user")
                    break

                # Count items processed
                batch_size = batch['image'].shape[0]
                total_items += batch_size

                # Update progress bar with stats
                elapsed = time.time() - start_time
                items_per_sec = total_items / elapsed if elapsed > 0 else 0

                pbar.set_postfix({
                    'items/s': f'{items_per_sec:.1f}',
                    'total': total_items,
                })

                # Periodic logging
                if (batch_idx + 1) % 100 == 0:
                    logger.debug(f"Processed {total_items} items")

        elapsed = time.time() - start_time
        logger.info(f"Cache warmup completed: {total_items} items in {elapsed:.1f}s ({total_items/elapsed:.1f} items/s)")

        # Write exclusion file for failed images from dataset
        # The dataset tracks failed samples in its failed_samples set
        try:
            dataset = dataloader.dataset
            # Handle Subset wrapper
            if hasattr(dataset, 'dataset'):
                dataset = dataset.dataset
            if hasattr(dataset, 'failed_samples') and dataset.failed_samples:
                # Get image_ids for failed samples
                failed_ids = []
                for idx in dataset.failed_samples:
                    if idx < len(dataset.items):
                        failed_ids.append(dataset.items[idx]['image_id'])

                if failed_ids:
                    exclusion_path = self.config.data_path / 'cache_exclusions.txt'
                    with open(exclusion_path, 'a', encoding='utf-8') as f:
                        for image_id in failed_ids:
                            f.write(image_id + '\n')
                    logger.warning(f"Wrote {len(failed_ids)} failed image IDs to {exclusion_path}")
                    logger.warning("These images will be automatically excluded from training.")
        except Exception as e:
            logger.warning(f"Could not write exclusion list from CPU warmup: {e}")

    def run_gpu_warmup(self, json_files: List[Path], indices: List[int]) -> None:
        """
        Run GPU-accelerated cache warmup with dynamic batch sizing and async preloading.

        Args:
            json_files: All dataset JSON files
            indices: Indices to process (from subset)
        """
        logger.info("Starting GPU-accelerated cache warmup...")
        logger.info("Note: Batch size will dynamically adjust to target VRAM usage")
        logger.info("Note: Failed/corrupted images will be automatically tracked and excluded from training")

        # Initialize GPU batch processor with preload settings
        gpu_processor = GPUBatchProcessor(
            image_size=self.config.image_size,
            pad_color=self.config.pad_color,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
            device_id=self.config.gpu_device_id,
            target_vram_gb=self.config.target_vram_gb,
            target_vram_util=self.config.target_vram_util,
            initial_batch_size=self.config.batch_size,
            max_batch_size=self.config.max_batch_size,
            cpu_bf16_cache_pipeline=(self.config.sidecar_storage_dtype == 'bfloat16'),
            # Async preload settings
            enable_preload=self.config.enable_preload,
            preload_workers=self.config.preload_workers,
            preload_queue_depth=self.config.preload_queue_depth,
            preload_ram_headroom_gb=self.config.preload_ram_headroom_gb,
        )

        try:
            if gpu_processor.has_preloader():
                logger.info("Async image preloading: ENABLED")
            else:
                logger.info("Async image preloading: DISABLED")

            start_time = time.time()
            total_items = 0
            batch_accumulator = []
            batch_metadata = []  # Store (json_path, image_path, sidecar_path, source_mtime) for each item

            # Pending batches for async preloading: (batch_id, paths, metadata)
            pending_batches: List[Tuple[int, List[str], List[Tuple]]] = []
            batch_id = 0

            def process_pending_batch(proc_id: int, proc_paths: List[str], proc_meta: List[Tuple]) -> int:
                """Process a preloaded batch and write sidecars. Returns items written."""
                try:
                    images, masks, failed_indices = gpu_processor.process_batch_preloaded(proc_id)
                    failed_set = set(failed_indices)

                    # Validate batch alignment
                    if len(images) != len(proc_meta):
                        logger.error(f"Batch mismatch: {len(images)} images vs {len(proc_meta)} metadata, skipping")
                        return 0

                    # Track failed images for exclusion file
                    for fail_idx in failed_indices:
                        if fail_idx < len(proc_meta):
                            self.failed_images.append(proc_meta[fail_idx][1])  # image_path

                    # Write sidecar files (skip failed images)
                    items_written = 0
                    for i, (_, _, sidecar_path, source_mtime) in enumerate(proc_meta):
                        if i in failed_set:
                            continue
                        monitor.l2_miss()
                        if save_sidecar(
                            sidecar_path,
                            images[i],
                            masks[i],
                            self.config.config_hash,
                            self.config.image_size,
                            source_mtime=source_mtime,
                        ):
                            items_written += 1
                            monitor.l2_put_enqueued(images[i].numel() * images[i].element_size())

                    return items_written
                except Exception as e:
                    logger.error(f"Failed to process preloaded batch {proc_id}: {e}", exc_info=True)
                    return 0

            def process_sync_batch(paths: List[str], meta: List[Tuple]) -> int:
                """Process a batch synchronously (no preload). Returns items written."""
                try:
                    images, masks, failed_indices = gpu_processor.process_batch(paths)
                    failed_set = set(failed_indices)

                    if len(images) != len(meta):
                        logger.error(f"Batch mismatch: {len(images)} images vs {len(meta)} metadata, skipping")
                        return 0

                    for fail_idx in failed_indices:
                        if fail_idx < len(meta):
                            self.failed_images.append(meta[fail_idx][1])

                    items_written = 0
                    for i, (_, _, sidecar_path, source_mtime) in enumerate(meta):
                        if i in failed_set:
                            continue
                        monitor.l2_miss()
                        if save_sidecar(
                            sidecar_path,
                            images[i],
                            masks[i],
                            self.config.config_hash,
                            self.config.image_size,
                            source_mtime=source_mtime,
                        ):
                            items_written += 1
                            monitor.l2_put_enqueued(images[i].numel() * images[i].element_size())

                    return items_written
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}", exc_info=True)
                    gpu_processor.clear_cache()
                    return 0

            # Progress bar
            with tqdm(
                total=len(indices),
                desc="GPU warmup",
                unit="img",
            ) as pbar:
                for idx in indices:
                    # Check for interrupt
                    if self.interrupted:
                        logger.warning("Warmup interrupted by user")
                        break

                    # Get JSON file for this index
                    json_path = json_files[idx]

                    # Load JSON to get image path
                    try:
                        # Use orjson if available (3-5x faster for cache warmup)
                        if HAS_ORJSON:
                            annotation = orjson.loads(json_path.read_bytes())
                        else:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                annotation = json.load(f)

                        # Get image path (same logic as dataset_loader.py)
                        image_filename = annotation.get('filename')
                        if not image_filename:
                            logger.warning(f"No filename in {json_path}, skipping")
                            continue

                        image_path = json_path.parent / image_filename
                        if not image_path.exists():
                            logger.warning(f"Image not found: {image_path}, skipping")
                            continue

                        # Get source file mtime for cache invalidation
                        try:
                            source_mtime = os.path.getmtime(image_path)
                        except OSError:
                            source_mtime = None

                        # Check if sidecar already exists (if --check-existing enabled)
                        sidecar_path = get_sidecar_path(str(image_path))
                        if self.config.check_existing and os.path.exists(sidecar_path):
                            # Already cached, skip
                            pbar.update(1)
                            continue

                        # Add to batch
                        batch_accumulator.append(str(image_path))
                        batch_metadata.append((json_path, str(image_path), sidecar_path, source_mtime))

                    except Exception as e:
                        logger.warning(f"Failed to process {json_path}: {e}")
                        continue

                    # Process batch when it reaches current batch size
                    current_batch_size = gpu_processor.get_current_batch_size()
                    if len(batch_accumulator) >= current_batch_size:
                        items_attempted = len(batch_metadata)

                        if gpu_processor.has_preloader():
                            # Async path: submit for preload and process pending batches
                            if gpu_processor.submit_preload(batch_id, batch_accumulator):
                                pending_batches.append((batch_id, batch_accumulator, batch_metadata))
                                batch_id += 1
                                batch_accumulator = []
                                batch_metadata = []

                                # Process oldest pending batch if we have more than 1
                                # (allows overlap: batch N preloads while batch N-1 processes)
                                if len(pending_batches) > 1:
                                    proc_id, proc_paths, proc_meta = pending_batches.pop(0)
                                    items_written = process_pending_batch(proc_id, proc_paths, proc_meta)
                                    total_items += items_written

                                    # Update progress bar
                                    elapsed = time.time() - start_time
                                    items_per_sec = total_items / elapsed if elapsed > 0 else 0
                                    vram_stats = gpu_processor.get_vram_stats()

                                    pbar.update(len(proc_meta))
                                    pbar.set_postfix({
                                        'next_batch': current_batch_size,
                                        'written': items_written,
                                        'preload': gpu_processor.get_preload_pending_count(),
                                        'imgs/s': f'{items_per_sec:.1f}',
                                        'VRAM': f'{vram_stats["utilization"]*100:.1f}%',
                                    })
                            else:
                                # Preloader at capacity - fall back to sync processing
                                items_written = process_sync_batch(batch_accumulator, batch_metadata)
                                total_items += items_written
                                batch_accumulator = []
                                batch_metadata = []

                                elapsed = time.time() - start_time
                                items_per_sec = total_items / elapsed if elapsed > 0 else 0
                                vram_stats = gpu_processor.get_vram_stats()

                                pbar.update(items_attempted)
                                pbar.set_postfix({
                                    'batch': current_batch_size,
                                    'written': items_written,
                                    'mode': 'sync',
                                    'imgs/s': f'{items_per_sec:.1f}',
                                    'VRAM': f'{vram_stats["utilization"]*100:.1f}%',
                                })
                        else:
                            # Sync path (no preloader)
                            items_written = process_sync_batch(batch_accumulator, batch_metadata)
                            total_items += items_written
                            batch_accumulator = []
                            batch_metadata = []

                            elapsed = time.time() - start_time
                            items_per_sec = total_items / elapsed if elapsed > 0 else 0
                            vram_stats = gpu_processor.get_vram_stats()

                            pbar.update(items_attempted)
                            pbar.set_postfix({
                                'batch': current_batch_size,
                                'written': items_written,
                                'imgs/s': f'{items_per_sec:.1f}',
                                'VRAM': f'{vram_stats["utilization"]*100:.1f}%',
                            })

                        # Adjust batch size based on VRAM usage
                        new_batch_size = gpu_processor.adjust_batch_size()
                        if new_batch_size != current_batch_size:
                            logger.info(f"Batch size adjusted: {current_batch_size} -> {new_batch_size}")

                # Drain remaining pending batches
                while pending_batches and not self.interrupted:
                    proc_id, proc_paths, proc_meta = pending_batches.pop(0)
                    items_written = process_pending_batch(proc_id, proc_paths, proc_meta)
                    total_items += items_written
                    pbar.update(len(proc_meta))

                # Process remaining items in accumulator (sync - not worth preloading)
                if batch_accumulator and not self.interrupted:
                    items_written = process_sync_batch(batch_accumulator, batch_metadata)
                    total_items += items_written
                    pbar.update(len(batch_metadata))

            elapsed = time.time() - start_time
            items_per_sec = total_items / elapsed if elapsed > 0 else 0
            logger.info(f"GPU warmup completed: {total_items} items in {elapsed:.1f}s ({items_per_sec:.1f} items/s)")

            # Write exclusion file for failed images (append mode for resumable runs)
            # Store just the image_id (stem) for format-agnostic exclusion
            if self.failed_images:
                exclusion_path = self.config.data_path / 'cache_exclusions.txt'
                with open(exclusion_path, 'a', encoding='utf-8') as f:
                    for path in self.failed_images:
                        # Write just the stem (image_id) - works regardless of file extension
                        image_id = Path(path).stem
                        f.write(image_id + '\n')
                logger.warning(f"Wrote {len(self.failed_images)} failed image IDs to {exclusion_path}")
                logger.warning("These images will be automatically excluded from training.")

            # Log cache monitor summary for debugging
            if monitor.enabled:
                logger.info(monitor.format_summary())

        finally:
            # Always cleanup GPU resources, even on exception or interrupt
            gpu_processor.clear_cache()

    def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")

        # Cleanup shared vocabulary
        if self.shared_vocab_manager is not None:
            self.shared_vocab_manager.cleanup()

        logger.info("Shutdown complete")

    def run(self) -> None:
        """Main execution flow."""
        try:
            # Discover JSON files first (required for metadata cache validation)
            json_files = discover_json_files(self.config.data_path)

            # Try to use metadata cache for consistency with training pipeline
            try:
                from utils.metadata_cache import try_load_metadata_cache

                logger.info("Loading metadata cache (same source as training)...")
                force_rebuild = getattr(self.config, 'force_rebuild_metadata_cache', False)

                cached_items = try_load_metadata_cache(
                    root_dir=self.config.data_path,
                    json_files=json_files,  # Pass actual list for validation
                    force_rebuild=force_rebuild,
                    num_workers=16,
                    logger=logger
                )

                if cached_items:
                    json_files = [Path(item['dir']) / f"{item['image_id']}.json" for item in cached_items]
                    logger.info(f"✓ Using {len(json_files)} files from metadata cache")
                else:
                    logger.info(f"Using {len(json_files)} files from filesystem discovery")
            except ImportError:
                logger.warning("Metadata cache module not found, using filesystem discovery")
            except Exception as e:
                logger.warning(f"Error loading metadata cache: {e}, using filesystem discovery")

            if not json_files:
                raise ValueError(f"No JSON files found in {self.config.data_path}")

            # Determine subset indices
            total_samples = len(json_files)
            num_samples = int(total_samples * self.config.percentage / 100)

            if num_samples == total_samples:
                logger.info(f"Warming 100% of dataset ({total_samples} samples)")
                indices = list(range(total_samples))
            else:
                # Random sampling with seed for reproducibility
                random.seed(self.config.seed)
                indices = random.sample(range(total_samples), num_samples)
                indices.sort()  # Sort for better cache locality
                logger.info(f"Warming {self.config.percentage}% of dataset ({num_samples}/{total_samples} samples)")

            # Branch based on GPU/CPU mode
            if self.config.use_gpu:
                # GPU-accelerated path (no dataset/dataloader needed)
                logger.info("Using GPU-accelerated warmup path")
                self.run_gpu_warmup(json_files, indices)
            else:
                # CPU multiprocessing path (original)
                logger.info("Using CPU multi-process warmup path")

                # Load vocabulary
                self.load_vocabulary()

                # Create dataset
                dataset = self.create_dataset(json_files)

                # Filter out already-cached items if --check-existing enabled
                if self.config.check_existing:
                    logger.info("Filtering out already-cached images...")

                    filtered_indices = []
                    for idx in indices:
                        # Get image_id for this index
                        json_path = json_files[idx]
                        try:
                            # Use orjson if available (3-5x faster)
                            if HAS_ORJSON:
                                annotation = orjson.loads(json_path.read_bytes())
                            else:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    annotation = json.load(f)
                            image_filename = annotation.get('filename')
                            if image_filename:
                                image_path = json_path.parent / image_filename
                                # Check if sidecar file exists
                                sidecar_path = get_sidecar_path(str(image_path))
                                if not os.path.exists(sidecar_path):
                                    # Not in cache, keep it
                                    filtered_indices.append(idx)
                            else:
                                # No filename, keep for processing (will be skipped later)
                                filtered_indices.append(idx)
                        except Exception:
                            # Error reading JSON, keep for processing
                            filtered_indices.append(idx)

                    logger.info(f"  Found {len(indices) - len(filtered_indices)} cached, "
                                f"{len(filtered_indices)} to process")
                    indices = filtered_indices
                    num_samples = len(indices)

                # Create subset
                subset = Subset(dataset, indices) if num_samples < total_samples else dataset

                # Create dataloader
                dataloader = self.create_dataloader(subset)

                # Run warmup
                self.run_warmup(dataloader)

            # Graceful shutdown
            self.shutdown()

            logger.info("Cache warmup finished successfully!")

        except Exception as e:
            logger.error(f"Cache warmup failed: {e}", exc_info=True)
            raise
        finally:
            # Ensure cleanup happens
            if self.shared_vocab_manager is not None:
                try:
                    self.shared_vocab_manager.cleanup()
                except Exception:
                    pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pre-populate sidecar cache for OppaiOracle training pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/unified_config.yaml',
        help='Path to unified configuration file'
    )

    parser.add_argument(
        '--percentage',
        type=int,
        default=100,
        help='Percentage of dataset to warm (1-100)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of parallel worker processes'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Number of items to process per batch'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling'
    )

    parser.add_argument(
        '--check-existing',
        action='store_true',
        help='Skip items already in cache (slower but resumable)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force warmup even if config hash mismatch detected (dangerous)'
    )

    parser.add_argument(
        '--force-rebuild-metadata-cache',
        action='store_true',
        help='Force rebuild of metadata cache (for consistency with training)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity level'
    )

    # GPU acceleration arguments
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Enable GPU-accelerated batch processing with dynamic VRAM management'
    )

    parser.add_argument(
        '--target-vram-gb',
        type=float,
        default=31.5,
        help='Target VRAM capacity in GB (default: 31.5 for RTX 5090)'
    )

    parser.add_argument(
        '--target-vram-util',
        type=float,
        default=0.9,
        help='Target VRAM utilization ratio (default: 0.9 for 90%% usage)'
    )

    parser.add_argument(
        '--gpu-device',
        type=int,
        default=0,
        help='CUDA device ID to use for GPU acceleration (default: 0)'
    )

    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=2048,
        help='Maximum batch size for GPU mode (default: 2048 for RTX 5090)'
    )

    # Async preload arguments
    parser.add_argument(
        '--no-preload',
        action='store_true',
        help='Disable async image preloading'
    )

    parser.add_argument(
        '--preload-workers',
        type=int,
        default=4,
        help='Number of threads for async image preloading (default: 4)'
    )

    parser.add_argument(
        '--preload-queue-depth',
        type=int,
        default=2,
        help='Maximum batches to preload ahead (default: 2)'
    )

    parser.add_argument(
        '--preload-ram-headroom-gb',
        type=float,
        default=8.0,
        help='Minimum free RAM to maintain during preloading in GB (default: 8.0)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 80)
    logger.info("Sidecar Cache Warmup Script for OppaiOracle")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}...")
    full_config = load_config(args.config)

    # Create warmup config
    warmup_config = CacheWarmupConfig(args, full_config)
    warmup_config.validate()

    # Run warmup
    warmup = CacheWarmup(warmup_config)
    warmup.run()

    logger.info("All done! Cache is ready for training.")


if __name__ == '__main__':
    main()
