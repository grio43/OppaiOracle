#!/usr/bin/env python3
"""
L2 LMDB Cache Warmup Script for OppaiOracle Training Pipeline

Pre-populates the L2 LMDB cache by processing a configurable percentage of the
dataset through the existing data loading pipeline. This reduces first-epoch
overhead and makes subsequent training runs start fast immediately.

IMPORTANT: Only unflipped images are cached in L2 (flipped versions are computed
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
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import from existing codebase
from Configuration_System import load_config, FullConfig
from dataset_loader import SidecarJsonDataset, load_vocabulary_for_training, SharedVocabularyManager
from l2_cache import start_l2_writer
from utils.cache_monitor import monitor

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

        # GPU acceleration settings
        self.use_gpu = args.use_gpu if hasattr(args, 'use_gpu') else False
        self.target_vram_gb = args.target_vram_gb if hasattr(args, 'target_vram_gb') else 31.5
        self.target_vram_util = args.target_vram_util if hasattr(args, 'target_vram_util') else 0.9
        self.gpu_device_id = args.gpu_device if hasattr(args, 'gpu_device') else 0

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
        self.l2_cache_path = data_cfg.l2_cache_path
        self.l2_max_size_gb = data_cfg.l2_max_size_gb
        self.l2_map_size_bytes = int(self.l2_max_size_gb * (1024 ** 3))
        self.l2_max_readers = getattr(data_cfg, 'l2_max_readers', 4096)
        self.l2_storage_dtype = getattr(data_cfg, 'l2_storage_dtype', 'bfloat16')

        # Preprocessing parameters (must match training exactly!)
        self.image_size = data_cfg.image_size
        self.pad_color = tuple(data_cfg.pad_color)
        self.normalize_mean = tuple(data_cfg.normalize_mean)
        self.normalize_std = tuple(data_cfg.normalize_std)

        # Vocabulary path
        self.vocab_path = Path(full_config.vocab_path)

        # Disable L1 cache during warmup (per-worker, not shared)
        self.l1_enabled = False
        self.l1_per_worker_mb = 0

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
            logger.info(f"GPU acceleration enabled (device {self.gpu_device_id})")
            logger.info(f"  Target VRAM: {self.target_vram_gb:.1f} GB")
            logger.info(f"  Target utilization: {self.target_vram_util * 100:.1f}%")

        # Check bfloat16 support if using that dtype
        if self.l2_storage_dtype == 'bfloat16':
            if not hasattr(torch, 'bfloat16'):
                logger.warning("PyTorch version does not support bfloat16, falling back to float16")
                self.l2_storage_dtype = 'float16'

        logger.info(f"Cache warmup configuration validated successfully")
        logger.info(f"  Mode: {'GPU-accelerated' if self.use_gpu else 'CPU multi-process'}")
        if not self.use_gpu:
            logger.info(f"  TIP: Add --use-gpu for 3-5x faster warmup (see GPU_WARMUP_GUIDE.md)")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Cache path: {self.l2_cache_path}")
        logger.info(f"  Cache size: {self.l2_max_size_gb:.1f} GB")
        logger.info(f"  Image size: {self.image_size}x{self.image_size}")
        logger.info(f"  Storage dtype: {self.l2_storage_dtype}")
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


def create_worker_init_fn(shared_vocab_info: Optional[Tuple[str, int]]):
    """
    Create worker initialization function for DataLoader.

    Populates vocabulary from shared memory in each worker process.
    """
    def worker_init_fn(worker_id: int):
        # Set random seed for reproducibility
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)

        # Load vocabulary from shared memory if provided
        if shared_vocab_info is not None:
            shm_name, vocab_size = shared_vocab_info
            # Vocabulary loading happens in dataset's __getitem__ via lazy init
            pass

    return worker_init_fn


def wait_for_queue_drain(queue: mp.Queue, timeout: float = 30.0, poll_interval: float = 0.5) -> None:
    """
    Wait for L2 writer queue to drain before exiting.

    Args:
        queue: L2 writer queue
        timeout: Maximum time to wait in seconds
        poll_interval: How often to check queue size
    """
    logger.info("Waiting for L2 writer queue to drain...")
    start_time = time.time()
    last_size = queue.qsize()

    with tqdm(desc="Draining queue", unit=" items", total=last_size) as pbar:
        while not queue.empty():
            current_size = queue.qsize()
            if current_size != last_size:
                pbar.total = current_size
                pbar.update(last_size - current_size)
                last_size = current_size

            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Queue drain timeout after {timeout}s, {current_size} items remaining")
                break

            time.sleep(poll_interval)

    # Final flush buffer
    time.sleep(2.0)
    logger.info("Queue drained successfully")


class CacheWarmup:
    """Main cache warmup orchestrator."""

    def __init__(self, config: CacheWarmupConfig):
        self.config = config
        self.l2_writer_queue: Optional[mp.Queue] = None
        self.l2_writer_process: Optional[mp.Process] = None
        self.vocab = None
        self.shared_vocab_manager: Optional[SharedVocabularyManager] = None
        self.interrupted = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.warning("\nReceived interrupt signal, shutting down gracefully...")
        self.interrupted = True

    def setup_l2_writer(self) -> None:
        """Start L2 writer process."""
        logger.info("Starting L2 writer process...")

        # Create cache directory if needed
        cache_path = Path(self.config.l2_cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Start writer process
        self.l2_writer_queue, self.l2_writer_process = start_l2_writer(
            path=str(cache_path),
            map_size_bytes=self.config.l2_map_size_bytes,
            max_map_size_multiplier=2  # Can grow to 2x initial size
        )

        logger.info(f"L2 writer process started (PID: {self.l2_writer_process.pid})")

    def load_vocabulary(self) -> None:
        """Load vocabulary and create shared memory version for workers."""
        logger.info(f"Loading vocabulary from {self.config.vocab_path}...")

        # Load vocabulary
        self.vocab = load_vocabulary_for_training(self.config.vocab_path)
        logger.info(f"Vocabulary loaded: {len(self.vocab)} tags")

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

            # L2 Cache configuration
            l2_enabled=True,
            l2_cache_path=self.config.l2_cache_path,
            l2_map_size_bytes=self.config.l2_map_size_bytes,
            l2_max_readers=self.config.l2_max_readers,
            l2_writer_queue=self.l2_writer_queue,
            l2_storage_dtype=self.config.l2_storage_dtype,
            cpu_bf16_cache_pipeline=True,

            # L1 Cache (disabled for warmup)
            l1_enabled=self.config.l1_enabled,
            l1_per_worker_mb=self.config.l1_per_worker_mb,

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

    def create_subset(self, dataset: SidecarJsonDataset) -> Subset:
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
        worker_init_fn = create_worker_init_fn(self.shared_vocab_info)

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
                queue_size = self.l2_writer_queue.qsize() if self.l2_writer_queue else 0

                pbar.set_postfix({
                    'items/s': f'{items_per_sec:.1f}',
                    'queue': queue_size,
                    'total': total_items,
                })

                # Periodic logging
                if (batch_idx + 1) % 100 == 0:
                    logger.debug(f"Processed {total_items} items, queue depth: {queue_size}")

        elapsed = time.time() - start_time
        logger.info(f"Cache warmup completed: {total_items} items in {elapsed:.1f}s ({total_items/elapsed:.1f} items/s)")

    def run_gpu_warmup(self, json_files: List[Path], indices: List[int]) -> None:
        """
        Run GPU-accelerated cache warmup with dynamic batch sizing.

        Args:
            json_files: All dataset JSON files
            indices: Indices to process (from subset)
        """
        logger.info("Starting GPU-accelerated cache warmup...")
        logger.info("Note: Batch size will dynamically adjust to target VRAM usage")

        # Initialize GPU batch processor
        gpu_processor = GPUBatchProcessor(
            image_size=self.config.image_size,
            pad_color=self.config.pad_color,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
            device_id=self.config.gpu_device_id,
            target_vram_gb=self.config.target_vram_gb,
            target_vram_util=self.config.target_vram_util,
            initial_batch_size=self.config.batch_size,
            cpu_bf16_cache_pipeline=(self.config.l2_storage_dtype == 'bfloat16')
        )

        # Helper function to convert tensor to bytes for L2 cache
        def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
            """Convert tensor to bytes for L2 cache storage."""
            return tensor.cpu().numpy().tobytes()

        # Helper function to compute L2 cache key
        def compute_l2_key(image_id: str) -> bytes:
            """Compute L2 cache key (unflipped only)."""
            config_hash = hashlib.md5(
                f"{self.config.image_size}|{self.config.pad_color}|"
                f"{self.config.normalize_mean}|{self.config.normalize_std}|"
                f"{self.config.l2_storage_dtype}".encode()
            ).hexdigest()[:8]
            return f"{image_id}|cfg{config_hash}".encode("utf-8")

        start_time = time.time()
        total_items = 0
        batch_accumulator = []
        batch_metadata = []  # Store (json_path, image_path, image_id) for each item

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
                    import json
                    with open(json_path, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)

                    # Get image path (same logic as dataset_loader.py)
                    image_filename = annotation.get('file_name')
                    if not image_filename:
                        logger.warning(f"No file_name in {json_path}, skipping")
                        continue

                    image_path = json_path.parent / image_filename
                    if not image_path.exists():
                        logger.warning(f"Image not found: {image_path}, skipping")
                        continue

                    # Generate image_id (relative path from data root)
                    try:
                        image_id = str(image_path.relative_to(self.config.data_path))
                    except ValueError:
                        image_id = str(image_path)

                    # Add to batch
                    batch_accumulator.append(str(image_path))
                    batch_metadata.append((json_path, image_path, image_id))

                except Exception as e:
                    logger.warning(f"Failed to process {json_path}: {e}")
                    continue

                # Process batch when it reaches current batch size
                current_batch_size = gpu_processor.get_current_batch_size()
                if len(batch_accumulator) >= current_batch_size:
                    # Process batch on GPU
                    try:
                        images, masks = gpu_processor.process_batch(batch_accumulator)

                        # Write to L2 cache
                        for i, (_, _, image_id) in enumerate(batch_metadata):
                            img_key = compute_l2_key(image_id)
                            mask_key = compute_l2_key(image_id) + b"|mask"

                            img_bytes = tensor_to_bytes(images[i])
                            mask_bytes = tensor_to_bytes(masks[i].to(torch.uint8))

                            self.l2_writer_queue.put_nowait((img_key, img_bytes))
                            self.l2_writer_queue.put_nowait((mask_key, mask_bytes))

                        total_items += len(batch_accumulator)

                        # Update progress bar
                        elapsed = time.time() - start_time
                        items_per_sec = total_items / elapsed if elapsed > 0 else 0
                        queue_size = self.l2_writer_queue.qsize()
                        vram_stats = gpu_processor.get_vram_stats()

                        pbar.update(len(batch_accumulator))
                        pbar.set_postfix({
                            'batch': current_batch_size,
                            'imgs/s': f'{items_per_sec:.1f}',
                            'VRAM': f'{vram_stats["utilization"]*100:.1f}%',
                            'queue': queue_size,
                        })

                        # Clear batch
                        batch_accumulator = []
                        batch_metadata = []

                        # Adjust batch size based on VRAM usage
                        new_batch_size = gpu_processor.adjust_batch_size()
                        if new_batch_size != current_batch_size:
                            logger.info(f"Batch size adjusted: {current_batch_size} -> {new_batch_size}")

                    except Exception as e:
                        logger.error(f"Failed to process batch: {e}", exc_info=True)
                        # Clear batch and continue
                        batch_accumulator = []
                        batch_metadata = []
                        gpu_processor.clear_cache()

            # Process remaining items
            if batch_accumulator and not self.interrupted:
                try:
                    images, masks = gpu_processor.process_batch(batch_accumulator)

                    for i, (_, _, image_id) in enumerate(batch_metadata):
                        img_key = compute_l2_key(image_id)
                        mask_key = compute_l2_key(image_id) + b"|mask"

                        img_bytes = tensor_to_bytes(images[i])
                        mask_bytes = tensor_to_bytes(masks[i].to(torch.uint8))

                        self.l2_writer_queue.put_nowait((img_key, img_bytes))
                        self.l2_writer_queue.put_nowait((mask_key, mask_bytes))

                    total_items += len(batch_accumulator)
                    pbar.update(len(batch_accumulator))

                except Exception as e:
                    logger.error(f"Failed to process final batch: {e}", exc_info=True)

        elapsed = time.time() - start_time
        logger.info(f"GPU warmup completed: {total_items} items in {elapsed:.1f}s ({total_items/elapsed:.1f} items/s)")

        # Clear GPU cache
        gpu_processor.clear_cache()

    def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")

        # Wait for queue to drain
        if self.l2_writer_queue is not None:
            wait_for_queue_drain(self.l2_writer_queue, timeout=60.0)

        # Cleanup shared vocabulary
        if self.shared_vocab_manager is not None:
            self.shared_vocab_manager.cleanup()

        logger.info("Shutdown complete")

    def run(self) -> None:
        """Main execution flow."""
        try:
            # Step 1: Setup L2 writer
            self.setup_l2_writer()

            # Step 2: Discover dataset files
            json_files = discover_json_files(self.config.data_path)
            if not json_files:
                raise ValueError(f"No JSON files found in {self.config.data_path}")

            # Step 3: Determine subset indices
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
                except:
                    pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pre-populate L2 LMDB cache for OppaiOracle training pipeline',
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

    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 80)
    logger.info("L2 LMDB Cache Warmup Script for OppaiOracle")
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
