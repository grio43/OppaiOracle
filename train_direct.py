#!/usr/bin/env python3
"""
Enhanced training script with comprehensive orientation handling for anime image tagger.
Demonstrates integration of the orientation handler with fail-fast behavior and statistics tracking.
"""

import gc
import logging
import os

# Set CUDA allocator config to reduce memory fragmentation
# Must be set BEFORE any torch/CUDA imports
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import multiprocessing as mp
import sys
import platform
import random
import queue
from datetime import datetime
import torch.distributed as dist
from dataclasses import dataclass
from contextlib import nullcontext
import signal
import threading

import shutil
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision
import numpy as np
from Monitor_log import MonitorConfig, TrainingMonitor
from utils.cache_monitor import monitor as cache_monitor
from evaluation_metrics import MetricComputer

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

from Configuration_System import load_config, create_config_parser, FullConfig
from utils.logging_setup import setup_logging

# Paths will be loaded from the unified config in the main function.
logger = logging.getLogger(__name__)

# Import the orientation handler
from orientation_handler import OrientationHandler, OrientationMonitor

# Cache monitor interval (steps). Configurable via CACHE_MONITOR_INTERVAL_STEPS env var.
# Default: 3000 steps (~50 minutes at 1 step/sec, ~96k images with batch_size=32)
CACHE_MONITOR_EVERY_STEPS = int(os.getenv('CACHE_MONITOR_INTERVAL_STEPS', '3000'))

# Import base modules with error handling
try:
    from dataset_loader import create_dataloaders
except ImportError as e:
    error_msg = (
        f"""MISSING REQUIRED FILE: dataset_loader.py
Please ensure dataset_loader.py exists in the current directory with create_dataloaders function.
Import error: {e}"""
    )
    raise ImportError(error_msg)

try:
    from model_architecture import create_model
except ImportError as e:
    error_msg = (
        f"""MISSING REQUIRED FILE: model_architecture.py
Please ensure model_architecture.py exists in the current directory with create_model function.
Import error: {e}"""
    )
    raise ImportError(error_msg)

# Import training utilities for checkpointing
from training_utils import (
    CheckpointManager,
    TrainingState,
    setup_seed,
    log_sample_order_hash,
    CosineAnnealingWarmupRestarts,
    validate_config_compatibility,
)
from training_utils import VOCAB_PATH as DEFAULT_VOCAB_PATH
from vocabulary import create_vocabulary_from_datasets  # NEW: rebuild vocab each run
from dataset_loader import AugmentationStats, validate_dataset
from utils.logging_sanitize import ensure_finite_tensor

# Add after other imports

# Alias to match usage below; avoids NameError and keeps intent clear.
amp_autocast = autocast

try:
    from loss_functions import MultiTaskLoss, AsymmetricFocalLoss
except ImportError as e:
    error_msg = (
        f"""MISSING REQUIRED FILE: loss_functions.py
Please ensure loss_functions.py exists in the current directory with MultiTaskLoss and AsymmetricFocalLoss classes.
Import error: {e}"""
    )
    raise ImportError(error_msg)


class RatingValidator:
    """Validator for rating labels with error tracking and configurable error handling.

    This class handles validation of rating labels during training and provides
    flexible error handling strategies to prevent training crashes from data quality issues.
    """

    def __init__(self, num_ratings: int, action: str = 'warn'):
        """
        Initialize the rating validator.

        Args:
            num_ratings: Number of valid rating classes (labels must be in [0, num_ratings))
            action: Error handling strategy:
                - 'error': Crash like before (raises RuntimeError)
                - 'warn': Log warning and skip batch (recommended for production)
                - 'clamp': Fix labels by clamping to valid range
                - 'ignore': Continue with invalid data (not recommended)
        """
        self.num_ratings = num_ratings
        self.action = action
        self.stats = {
            'total_batches': 0,
            'invalid_batches': 0,
            'invalid_samples': 0,
        }
        self.logger = logging.getLogger(__name__)

    def validate_and_handle(
        self,
        rating_labels: torch.Tensor,
        batch: dict,
        global_step: int
    ) -> Tuple[torch.Tensor, bool]:
        """
        Validate rating labels and handle errors based on configured action.

        Args:
            rating_labels: The rating labels tensor to validate
            batch: The full batch dict (may contain image_ids for logging)
            global_step: Current training step number

        Returns:
            (labels, is_valid): Tuple of (potentially fixed labels, whether batch is valid)
        """
        self.stats['total_batches'] += 1

        # Skip validation for float labels (used in some loss functions)
        if rating_labels.dtype not in (torch.long, torch.int64):
            return rating_labels, True

        # Check for out-of-range labels
        mask_valid = (rating_labels >= 0) & (rating_labels < self.num_ratings)

        if mask_valid.all():
            return rating_labels, True  # All valid

        # Found invalid labels - handle based on action
        num_invalid = (~mask_valid).sum().item()
        self.stats['invalid_batches'] += 1
        self.stats['invalid_samples'] += num_invalid

        min_val = rating_labels.min().item()
        max_val = rating_labels.max().item()

        error_msg = (
            f"Invalid rating labels at step {global_step}: "
            f"{num_invalid}/{len(rating_labels)} samples out of range. "
            f"Found min={min_val}, max={max_val}, expected [0, {self.num_ratings})."
        )

        # Log which samples are invalid (if batch has identifiers)
        if 'image_ids' in batch:
            invalid_indices = (~mask_valid).nonzero(as_tuple=True)[0].tolist()
            invalid_ids = [batch['image_ids'][i] for i in invalid_indices[:10]]  # Log first 10
            if len(invalid_indices) > 10:
                self.logger.warning(f"{error_msg} First 10 invalid samples: {invalid_ids}")
            else:
                self.logger.warning(f"{error_msg} Invalid samples: {invalid_ids}")

        if self.action == 'error':
            # Crash like before
            raise RuntimeError(error_msg)

        elif self.action == 'warn':
            # Log and skip this batch
            self.logger.warning(f"{error_msg} Skipping batch.")
            return rating_labels, False  # Signal to skip batch

        elif self.action == 'clamp':
            # Fix the labels by clamping
            rating_labels = rating_labels.clamp(0, self.num_ratings - 1)
            self.logger.warning(f"{error_msg} Clamped to valid range.")
            return rating_labels, True

        elif self.action == 'ignore':
            # Continue with invalid data (not recommended)
            self.logger.debug(f"{error_msg} Ignoring.")
            return rating_labels, True

        else:
            raise ValueError(f"Unknown rating validation action: {self.action}")

    def get_stats(self) -> dict:
        """Return validation statistics for monitoring."""
        stats = self.stats.copy()
        if stats['total_batches'] > 0:
            stats['invalid_rate'] = stats['invalid_batches'] / stats['total_batches']
        else:
            stats['invalid_rate'] = 0.0
        return stats


def assert_finite(*tensors, names=None, batch=None, outputs=None, config=None):
    """Assert that all tensors are finite, with optional debugging hooks."""
    if names is None:
        names = [f"Tensor {i}" for i in range(len(tensors))]

    for name, t in zip(names, tensors):
        if t is not None and hasattr(t, 'dtype') and t.is_floating_point():
            if not torch.isfinite(t).all():
                # Non-finite value detected, attempt to perform debug actions
                if config and hasattr(config, 'debug') and config.debug.enabled:
                    logger.error(f"Non-finite detected in '{name}'. Debug mode enabled, attempting to save context.")

                    # Log batch info if available and enabled
                    if config.debug.log_batch_info_on_error and batch:
                        # Log available metadata, avoiding large tensors
                        batch_info = {k: v for k, v in batch.items() if not isinstance(v, torch.Tensor) or v.numel() < 10}
                        logger.error(f"Problematic batch info: {batch_info}")

                    # Dump tensors to file if available and enabled
                    if config.debug.dump_tensors_on_error and batch and outputs:
                        dump_dir = Path(config.output_root) / config.experiment_name / "debug_dumps"
                        dump_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        dump_path = dump_dir / f"non_finite_dump_{name}_{timestamp}.pt"

                        dump_data = {
                            'failed_tensor_name': name,
                            'batch': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                            'outputs': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                        }

                        try:
                            torch.save(dump_data, dump_path)
                            logger.error(f"Saved debug tensors for failed tensor '{name}' to: {dump_path}")
                        except Exception as e:
                            logger.error(f"Failed to save debug tensor dump: {e}")

                # Always raise the error to halt training
                raise RuntimeError(f"Non-finite detected in {name}")


def setup_orientation_aware_training(
    data_dir: Path,
    json_dir: Path,
    vocab_path: Path,
    orientation_map_path: Optional[Path] = None,
    random_flip_prob: float = 0.35,
    strict_orientation: bool = True,
    safety_mode: str = "conservative",
    skip_unmapped: bool = False
) -> Dict[str, Any]:
    """
    Setup training with enhanced orientation handling.
    
    Args:
        data_dir: Directory containing images
        json_dir: Directory containing annotation JSONs
        vocab_path: Path to vocabulary file
        orientation_map_path: Path to orientation mapping JSON
        random_flip_prob: Probability of horizontal flips
        strict_orientation: If True, fail if flips enabled without proper mapping
        skip_unmapped: If True, skip flipping images with unmapped orientation tags
    
    Returns:
        Configuration dictionary with validated orientation settings
    """
    
    # Validate orientation setup
    if random_flip_prob > 0:
        if orientation_map_path is None:
            # Try to find default orientation map
            default_paths = [
                Path("configs/orientation_map.json"),
                Path("orientation_map.json"),
                Path("config/orientation_map.json")
            ]
            
            for path in default_paths:
                if path.exists():
                    orientation_map_path = path
                    logging.info(f"Found orientation map at: {path}")
                    break
            
            if orientation_map_path is None:
                if strict_orientation:
                    raise FileNotFoundError(
                        f"Horizontal flips enabled (prob={random_flip_prob}) but no orientation_map.json found. "
                        f"Searched in: {[str(p) for p in default_paths]}. "
                        f"Please provide orientation mapping or set random_flip_prob=0"
                    )
                else:
                    logging.warning(
                        f"Horizontal flips enabled (prob={random_flip_prob}) but no orientation map found. "
                        f"Using minimal defaults. This may cause incorrect orientation labels!"
                    )
        
        # Validate orientation map if provided
        if orientation_map_path and orientation_map_path.exists():
            handler = OrientationHandler(
                mapping_file=orientation_map_path,
                random_flip_prob=random_flip_prob,
                strict_mode=strict_orientation,
                safety_mode=safety_mode,
                skip_unmapped=skip_unmapped
            )
            
            # Validate mappings
            issues = handler.validate_mappings()
            if issues:
                logging.warning(f"Orientation mapping validation issues: {issues}")
                if strict_orientation and any(issues.values()):
                    raise ValueError(f"Critical orientation mapping issues found: {issues}")
    
    # Return only validated orientation settings; loader reads from config.data directly.
    return {
        "orientation": {
            "safety_mode": safety_mode,
            "skip_unmapped": skip_unmapped,
        }
    }


def train_with_orientation_tracking(config: FullConfig):
    """Training loop with orientation handling and statistics tracking."""

    import tempfile
    from utils.memory_monitor import MemoryMonitor

    logger = logging.getLogger(__name__)
    
    # --- Soft stop support (signals + sentinel files) -----------------------
    # Save a checkpoint at the next safe point (optimizer step) and exit.
    # Use threading.Event for signal-safe flag (CR-033)
    soft_stop_event = threading.Event()

    def _soft_stop_handler(signum, frame):
        """Signal-safe handler - only sets atomic event.

        IMPORTANT: Do NOT use logging or any non-reentrant functions here.
        Signal handlers can deadlock if they try to acquire locks held by
        the interrupted code. Only atomic operations are safe.
        """
        soft_stop_event.set()
        # Write to stderr is relatively safe (no locks in Python's signal handling)
        # but even this should be minimal. The actual message will be logged
        # when the training loop checks soft_stop_event.

    try:
        signal.signal(signal.SIGINT, _soft_stop_handler)
        signal.signal(signal.SIGTERM, _soft_stop_handler)
    except Exception as _e:
        logger.debug("Signal handler install skipped: %s", _e)

    # --- Manual TensorBoard image logging hotkey (press 'i' to log images) ---
    def _keyboard_listener(log_dir: Path, stop_event: threading.Event):
        """Background thread that listens for hotkey presses.

        Press 'i' to trigger immediate TensorBoard image logging.
        This creates a sentinel file that the training loop checks.
        Non-blocking and won't interrupt training.

        Alternative: manually create LOG_IMAGES_NOW file in log_dir.
        """
        image_sentinel = log_dir / "LOG_IMAGES_NOW"

        # Try Windows-specific keyboard handling
        try:
            import msvcrt  # Windows-only
            while not stop_event.is_set():
                try:
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                        if key == 'i':
                            image_sentinel.touch()
                            print(f"\n[Hotkey] 'i' pressed - will log images to TensorBoard at next step...")
                except Exception:
                    pass  # Ignore individual keypress errors
                stop_event.wait(0.1)  # Check every 100ms
            return
        except ImportError:
            pass  # Not on Windows, try Unix approach

        # Unix/Linux: use select-based approach
        # Note: requires terminal in raw mode for immediate response
        try:
            import sys
            import select
            import tty
            import termios

            # Check if stdin is a real terminal
            if not sys.stdin.isatty():
                return  # stdin redirected, can't read keys

            # Save terminal settings and set raw mode for immediate keypress
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())  # cbreak mode: immediate input, no echo
                while not stop_event.is_set():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
                        if key == 'i':
                            image_sentinel.touch()
                            print(f"\n[Hotkey] 'i' pressed - will log images to TensorBoard at next step...")
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            pass  # Keyboard monitoring not available (e.g., no terminal)

    # Start keyboard listener as daemon thread (won't block exit)
    _kb_thread = threading.Thread(
        target=_keyboard_listener,
        args=(Path(config.log_dir), soft_stop_event),
        daemon=True,
        name="KeyboardListener"
    )
    _kb_thread.start()
    logger.info("Keyboard hotkey listener started: press 'i' to log images to TensorBoard")

    # Seeding & determinism
    seed, deterministic_mode = setup_seed(config.training.seed, config.training.deterministic)

    use_anomaly = (
        getattr(config.training, "enable_anomaly_detection", False)
        or getattr(config.debug, "detect_anomaly", False)
    )
    anomaly_ctx = torch.autograd.detect_anomaly(check_nan=True) if use_anomaly else nullcontext()

    try:
        torch.use_deterministic_algorithms(deterministic_mode)
    except Exception:
        pass

    # Allow cuDNN to pick the fastest kernels when not in strict-deterministic mode
    torch.backends.cudnn.benchmark = bool(getattr(config.training, "benchmark", True))
    
    # Enable TensorFloat-32 (TF32) for massive speedup on Ampere+ GPUs
    # This uses Tensor Cores for FP32 matmuls (Linear layers) and convolutions
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TensorFloat-32 (TF32) enabled for matmul and cuDNN")

    # Ensure log_dir exists early for sentinel files and diagnostics
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Find the active data path from storage_locations
    active_location = next((loc for loc in config.data.storage_locations if loc.get('enabled')), None)

    if not active_location:
        error_msg = (
            "No enabled storage location found in your configuration.\n"
            "Please ensure you are providing a configuration file using the --config argument, for example:\n"
            "  python train_direct.py --config configs/unified_config.yaml\n"
            "And that your configuration file has an enabled entry under data.storage_locations."
        )
        raise ValueError(error_msg)

    active_data_path = Path(active_location['path'])

    # Validate path exists and is accessible (CR-036)
    if not active_data_path.exists():
        raise FileNotFoundError(
            f"Active data path does not exist: {active_data_path}\n"
            f"Please ensure the path in your configuration is correct and the storage is mounted."
        )

    if not active_data_path.is_dir():
        raise NotADirectoryError(
            f"Active data path is not a directory: {active_data_path}\n"
            f"Expected a directory containing training images and annotations."
        )

    if not os.access(active_data_path, os.R_OK):
        raise PermissionError(
            f"Active data path is not readable: {active_data_path}\n"
            f"Please check file permissions."
        )

    logger.info(f"Using active data path: {active_data_path} (validated)")

    # --- Prompt to (re)build vocabulary at startup ---------------------------------
    # Decide where the vocabulary should live and whether we already have one
    vocab_dest = Path(getattr(config, "vocab_path", str(DEFAULT_VOCAB_PATH)))
    check_path = vocab_dest / "vocabulary.json" if vocab_dest.is_dir() else vocab_dest
    has_vocab = check_path.exists()

    def _ask_yes_no(prompt: str, default: Optional[bool]) -> bool:
        """
        Simple Y/N prompt. If not attached to a TTY, fall back to the default.
        default=True  -> [Y/n]
        default=False -> [y/N]
        default=None  -> [y/n]

        Environment variables (CR-001):
        - OO_AUTO_REBUILD_VOCAB: "1", "true", "yes" for auto-rebuild
        - OO_NON_INTERACTIVE: "1" to force non-interactive mode
        """
        # Check environment variable override (CR-001 fix)
        env_override = os.environ.get("OO_AUTO_REBUILD_VOCAB")
        if env_override:
            result = env_override.lower() in ("1", "true", "yes")
            logger.info(f"Using environment variable OO_AUTO_REBUILD_VOCAB={env_override} -> {result}")
            return result

        # Force non-interactive mode via environment variable
        if os.environ.get("OO_NON_INTERACTIVE", "").lower() in ("1", "true", "yes"):
            result = bool(default) if default is not None else False
            logger.info(f"Non-interactive mode via OO_NON_INTERACTIVE -> using default: {result}")
            return result

        # Non-interactive (e.g., piped/cron) -> use default
        if not sys.stdin or not sys.stdin.isatty():
            result = bool(default) if default is not None else False
            logger.info(f"Non-TTY detected -> using default: {result}")
            return result

        choices = " [Y/n] " if default is True else (" [y/N] " if default is False else " [y/n] ")
        ans = input(prompt + choices).strip().lower()
        if ans in ("y", "yes"): return True
        if ans in ("n", "no"):  return False
        return bool(default) if default is not None else False

    # Default choice: build if missing, otherwise skip
    rebuild = _ask_yes_no(
        "Build a new tag vocabulary from dataset JSONs?",
        default=(not has_vocab)
    )

    # If the user declines but there is no vocabulary, build anyway to avoid crash
    if not has_vocab and not rebuild:
        logger.warning(
            "No vocabulary found at %s but rebuild was declined; "
            "building now to avoid a startup failure.", check_path
        )
        rebuild = True

    if rebuild:
        try:
            logger.info("Rebuilding tag vocabulary from dataset at %s", active_data_path)
            # Scans recursively for *.json sidecars
            rebuilt_vocab = create_vocabulary_from_datasets([active_data_path])
            vocab_file = (vocab_dest / "vocabulary.json") if vocab_dest.is_dir() else vocab_dest
            rebuilt_vocab.save_vocabulary(vocab_file)
            logger.info("Vocabulary rebuilt with %d tags -> %s",
                        len(rebuilt_vocab.tag_to_index), vocab_file)
        except Exception as e:
            logger.error("Failed to (re)build vocabulary: %s", e)
            raise
    else:
        logger.info("Using existing vocabulary at %s", check_path)
    # -------------------------------------------------------------------------------

    # Setup orientation handling
    orientation_config = setup_orientation_aware_training(
        data_dir=active_data_path,
        json_dir=active_data_path,
        vocab_path=Path(config.vocab_path),
        orientation_map_path=Path(config.data.orientation_map_path) if config.data.orientation_map_path else None,
        random_flip_prob=config.data.random_flip_prob,
        strict_orientation=config.data.strict_orientation_validation,
        safety_mode=config.data.orientation_safety_mode,
        skip_unmapped=config.data.skip_unmapped
    )

    stats_queue = mp.Queue(maxsize=1000) if config.training.use_tensorboard else None
    device = torch.device(config.training.device)
    device_type = device.type

    # Expose stats queue to dataloaders for optional telemetry
    config.data.stats_queue = stats_queue

    # Note: Orientation mapping validation is now done in setup_orientation_aware_training()
    # which is called earlier. That function handles strict mode and raises if needed.
    # No duplicate validation here to avoid inconsistent error handling.

    train_loader, val_loader, vocab = create_dataloaders(
        data_config=config.data,
        validation_config=config.validation,
        vocab_path=Path(config.vocab_path),
        active_data_path=active_data_path,
        distributed=config.training.distributed,
        rank=config.training.local_rank,
        world_size=config.training.world_size,
        seed=seed,
        debug_config=config.debug,
    )

    # Apply validation max_samples subsampling if configured
    # This significantly speeds up validation by using a random subset
    val_max_samples = getattr(config.validation, 'max_samples', None)
    if val_loader is not None and val_max_samples and val_max_samples < len(val_loader.dataset):
        original_val_size = len(val_loader.dataset)
        indices = np.random.choice(original_val_size, val_max_samples, replace=False)
        val_subset = Subset(val_loader.dataset, indices.tolist())

        # Extract validation dataloader config (fall back to training config)
        val_dl_cfg = getattr(config.validation, "dataloader", None)
        val_batch = getattr(val_dl_cfg, "batch_size", None) or config.data.batch_size
        val_num_workers = getattr(val_dl_cfg, "num_workers", None) or config.data.num_workers
        val_prefetch = getattr(val_dl_cfg, "prefetch_factor", None) or getattr(config.data, "prefetch_factor", 2)
        val_pin_memory = getattr(val_dl_cfg, "pin_memory", None)
        if val_pin_memory is None:
            val_pin_memory = getattr(config.data, "pin_memory", True)
        val_persistent = getattr(val_dl_cfg, "persistent_workers", None)
        if val_persistent is None:
            val_persistent = getattr(config.data, "persistent_workers", False)

        # Rebuild val_loader with the subset, preserving ALL config settings
        val_loader_kwargs = dict(
            batch_size=val_batch,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=val_pin_memory,
        )
        # Only add multiprocessing-specific kwargs when workers > 0
        if val_num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = val_prefetch
            val_loader_kwargs["persistent_workers"] = val_persistent

        val_loader = DataLoader(val_subset, **val_loader_kwargs)
        logger.info(
            f"Validation subsampled: {val_max_samples:,} of {original_val_size:,} samples "
            f"({100 * val_max_samples / original_val_size:.1f}%) "
            f"[workers={val_num_workers}, prefetch={val_prefetch}]"
        )

    # Set initial epoch before any DataLoader access (prevents worker spawn warnings)
    if hasattr(train_loader.dataset, 'set_epoch'):
        train_loader.dataset.set_epoch(0)
    if val_loader is not None and hasattr(val_loader.dataset, 'set_epoch'):
        val_loader.dataset.set_epoch(0)

    # --- Background Worker Warmup (Windows optimization) ---
    # On Windows, DataLoader workers spawn sequentially (~3.3s each) due to 'spawn' context.
    # Start worker spawning in background while model setup continues (parallel with torch.compile).
    # This hides the ~50s worker startup time behind model compilation/setup.
    _warmup_result = {"iter": None, "batch": None, "error": None, "complete": False, "duration": None}
    _warmup_start_time = time.perf_counter()

    def _warmup_workers():
        try:
            warmup_iter = iter(train_loader)
            warmup_batch = next(warmup_iter)  # Forces all workers to fully initialize
            _warmup_result["iter"] = warmup_iter
            _warmup_result["batch"] = warmup_batch
            _warmup_result["complete"] = True
            _warmup_result["duration"] = time.perf_counter() - _warmup_start_time
            logger.info(f"Worker warmup complete in {_warmup_result['duration']:.1f}s - all workers spawned in background")
        except Exception as e:
            _warmup_result["error"] = e
            _warmup_result["complete"] = True
            _warmup_result["duration"] = time.perf_counter() - _warmup_start_time
            logger.warning(f"Worker warmup failed after {_warmup_result['duration']:.1f}s: {e}")

    warmup_thread = threading.Thread(target=_warmup_workers, daemon=True, name="WorkerWarmup")
    warmup_thread.start()
    logger.info("Started background worker warmup (parallel with model setup)")

    # --- Orientation diagnostics (enabled by default) -----------------------
    # Create an OrientationMonitor to write/update unmapped_orientation_tags.txt
    # in the configured log directory and to surface suggestions.
    orientation_monitor = None
    oh = None
    try:
        oh = getattr(getattr(train_loader, "dataset", None), "orientation_handler", None)
        orientation_monitor = OrientationMonitor(out_dir=Path(config.log_dir))
    except Exception as e:
        logger.debug(f"OrientationMonitor init skipped: {e}")

    # Pre-training validation
    if getattr(config.debug, 'validate_input_data', False):
        logger.info("Starting pre-training input validation...")
        # Limiting validation to a few batches to avoid long startup times
        validate_dataset(train_loader, vocab, config, num_batches_to_check=10)
        validate_dataset(val_loader, vocab, config, num_batches_to_check=5)
        logger.info("Pre-training input validation complete.")

    num_tags = len(vocab.tag_to_index)
    num_ratings = len(vocab.rating_to_index)

    # Sync config.model.num_labels with actual vocabulary size
    # This is deferred until after vocabulary is loaded (config validation allows 0)
    if config.model.num_labels == 0:
        config.model.num_labels = num_tags
        logger.debug(f"Set config.model.num_labels to vocabulary size: {num_tags}")
    elif config.model.num_labels != num_tags:
        # This is a critical mismatch that will cause model loading failures
        # or incorrect predictions. Fail fast to prevent silent corruption.
        raise ValueError(
            f"CRITICAL: config.model.num_labels ({config.model.num_labels}) does not match "
            f"vocabulary size ({num_tags}). This mismatch will cause model architecture errors. "
            f"Please update your config or use a compatible vocabulary/checkpoint."
        )

    logger.info(f"Creating model with {num_tags} tags and {num_ratings} ratings")

    metric_computer = MetricComputer(
        num_labels=num_tags,
        skip_indices=[0, 1],  # Skip <PAD> (0) and <UNK> (1) in metric computation
    )

    rating_validation_action = getattr(config.training, 'rating_validation_action', 'warn')
    rating_validator = RatingValidator(
        num_ratings=num_ratings,
        action=rating_validation_action
    )
    logger.info(f"Rating validator created with action='{rating_validation_action}'")

    model_config = config.model.to_dict()
    model_config["num_tags"] = num_tags
    model_config["num_ratings"] = num_ratings

    # Filter out config keys that are in unified_config.yaml but not used by VisionTransformerConfig
    # These are legacy/alternate config fields that don't map to the current model architecture
    _unused_config_keys = {
        'architecture_type', 'attention_probs_dropout_prob',
        'hidden_dropout_prob', 'initializer_range', 'num_groups', 'num_labels',
        'num_special_tokens', 'tags_per_group', 'use_cls_token', 'use_color_token',
        'use_line_token', 'use_style_token'
    }
    model_config = {k: v for k, v in model_config.items() if k not in _unused_config_keys}

    model = create_model(**model_config)
    # Move model to device first, then apply dtype conversion
    # NOTE: channels_last memory format is applied LATER (after checkpoint loading and dtype conversion)
    # to ensure it's not lost during transformations. See the channels_last application block below.
    model.to(device)

    # Convert model to bfloat16 when AMP is enabled with bf16 dtype
    # This saves ~3.8 GB VRAM by storing parameters and gradients in bf16 instead of fp32
    # AMP autocast still handles mixed precision during forward/backward passes
    amp_dtype_cfg = str(getattr(config.training, "amp_dtype", "bfloat16")).lower()
    if getattr(config.training, "use_amp", True) and amp_dtype_cfg in ("bfloat16", "bf16"):
        model = model.bfloat16()
        logger.info("Model converted to bfloat16 for memory efficiency (~3.8 GB VRAM savings)")

    # Update monitor config with values from other parts of the config for backward compatibility
    if not hasattr(config, 'monitor'):
        # In case the config file is old and doesn't have a monitor section
        config.monitor = MonitorConfig()

    config.monitor.log_dir = config.log_dir
    config.monitor.use_tensorboard = config.training.use_tensorboard
    # Only set a default if not provided in config
    if not getattr(config.monitor, "tensorboard_dir", None):
        config.monitor.tensorboard_dir = str(Path(config.output_root) / config.experiment_name)
    config.monitor.use_wandb = config.training.use_wandb

    monitor = TrainingMonitor(config.monitor)

    # --- TensorBoard: initial hparams snapshot ---
    try:
        to_dict = getattr(config, "to_dict", None)
        hparams = to_dict() if callable(to_dict) else (
            vars(config) if hasattr(config, "__dict__") else {}
        )
        monitor.log_hyperparameters(hparams, {"init/placeholder": 0})
    except Exception:
        pass

    # Log loss hyperparameters
    tag_loss_cfg = config.training.tag_loss
    rating_loss_cfg = config.training.rating_loss
    loss_hparams = {
        "tag_loss": tag_loss_cfg.to_dict() if hasattr(tag_loss_cfg, "to_dict") else vars(tag_loss_cfg),
        "rating_loss": rating_loss_cfg.to_dict() if hasattr(rating_loss_cfg, "to_dict") else vars(rating_loss_cfg),
    }
    logger.info(f"Loss hyperparameters: {loss_hparams}")
    try:
        monitor.log_hyperparameters(loss_hparams, {"loss/init": 0})
    except Exception:
        pass

    # NOTE: TensorBoard model graph logging is deferred until AFTER torch.compile()
    # to avoid stride mismatch issues. See the model graph logging block below.

    # Detect and log Flex Attention configuration
    if config.model.use_flex_attention:
        logger.info("=" * 70)
        logger.info("Flex Attention Configuration:")
        logger.info(f"  PyTorch version: {torch.__version__}")

        # Check Flex Attention availability
        if hasattr(torch.nn.attention, 'flex_attention'):
            from torch.nn.attention.flex_attention import flex_attention
            logger.info("  Status: Flex Attention - AVAILABLE")
            logger.info(f"  Block size: {getattr(config.model, 'flex_block_size', 128)}")

            # Quick test of Flex Attention kernel
            try:
                with torch.no_grad():
                    test_q = torch.randn(1, 1, 16, 64, device=device, dtype=torch.bfloat16)
                    _ = flex_attention(test_q, test_q, test_q)
                    logger.info("  Test: Flex Attention kernel - WORKING")
            except Exception as e:
                logger.error(f"  Test: Flex Attention - FAILED: {e}")
        else:
            logger.error("  Flex Attention not available - requires PyTorch 2.5+")
            raise RuntimeError("Flex Attention requires PyTorch 2.5 or newer")

        logger.info("  Note: Flex Attention benefits from torch.compile() - kernel fusion enabled")
        logger.info("=" * 70)

    # torch.compile() optimization (PyTorch 2.0+)
    # Provides 15-35% speedup through graph optimization and kernel fusion
    # NOTE: Actual compilation is DEFERRED until after checkpoint loading to ensure
    # inductor kernels are compiled for the correct weight strides (channels_last).
    # Loading a checkpoint can reset tensor strides, causing stride mismatch errors.
    use_compile = getattr(config.training, "use_compile", False)
    if use_compile:
        # Check if Triton is available (required for torch.compile with inductor backend)
        try:
            import triton
            logger.info(f"Triton {triton.__version__} available for torch.compile")
        except ImportError:
            logger.warning("torch.compile() requires Triton but it's not installed")
            logger.warning("Install with: pip install triton-windows (Windows) or pip install triton (Linux)")
            logger.warning("Training will proceed without compilation - expect ~15-35% slower training")
            use_compile = False

    # Save compile settings - actual compilation happens after checkpoint loading
    compile_mode = getattr(config.training, "compile_mode", "default") if use_compile else None
    compile_fullgraph = getattr(config.training, "compile_fullgraph", False) if use_compile else None
    compile_dynamic = getattr(config.training, "compile_dynamic", True) if use_compile else None

    if not use_compile:
        logger.info("torch.compile() disabled (use_compile=false in config)")

    criterion = MultiTaskLoss(
        tag_loss_weight=0.9,
        rating_loss_weight=0.1,
        tag_loss_fn=AsymmetricFocalLoss(
            alpha=tag_loss_cfg.alpha,
            clip=tag_loss_cfg.clip,
            gamma_neg=tag_loss_cfg.gamma_neg,
            gamma_pos=tag_loss_cfg.gamma_pos,
            label_smoothing=tag_loss_cfg.label_smoothing,
            ignore_indices=[0, 1],  # Ignore <PAD> (0) and <UNK> (1) for tags
        ),
        rating_loss_fn=AsymmetricFocalLoss(
            alpha=rating_loss_cfg.alpha,
            clip=rating_loss_cfg.clip,
            gamma_neg=rating_loss_cfg.gamma_neg,
            gamma_pos=rating_loss_cfg.gamma_pos,
            label_smoothing=rating_loss_cfg.label_smoothing,
            ignore_indices=[4],  # Ignore "unknown" rating (index 4)
        ),
    )
    from training_utils import TrainingUtils
    # Construct betas tuple based on the selected optimizer
    betas = (config.training.adam_beta1, config.training.adam_beta2)
    if config.training.optimizer == 'adan':
        beta3 = getattr(config.training, 'adan_beta3', 0.99)
        betas = betas + (beta3,)

    optimizer = TrainingUtils.get_optimizer(
        model,
        optimizer_type=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=betas,
        eps=config.training.adam_epsilon
    )

    # ---- LR scheduler: STEP-BASED semantics ----
    # Interpret warmup / cycle lengths in optimizer updates (not epochs).

    # Validate gradient accumulation steps (CR-038)
    try:
        accum_raw = getattr(config.training, "gradient_accumulation_steps", 1)
        accum = int(accum_raw)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid gradient_accumulation_steps in config: {accum_raw!r}. "
            f"Must be a positive integer. Error: {e}"
        )

    if accum < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, got {accum}. "
            f"Use 1 to disable gradient accumulation."
        )

    # Warn if accumulation is suspiciously high
    batch_size = config.data.batch_size
    if accum > batch_size:
        logger.warning(
            f"gradient_accumulation_steps ({accum}) > batch_size ({batch_size}). "
            f"This is unusual and may indicate a configuration error."
        )

    # Validate num_epochs
    try:
        num_epochs = int(getattr(config.training, "num_epochs", 1))
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid num_epochs in config: must be a positive integer. Error: {e}"
        )

    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

    steps_per_epoch = max(1, len(train_loader))
    updates_per_epoch = (steps_per_epoch + accum - 1) // accum  # ceil division
    total_updates = num_epochs * updates_per_epoch

    logger.info(
        f"Scheduler setup: {num_epochs} epochs, {steps_per_epoch} steps/epoch, "
        f"{accum}x gradient accumulation = {updates_per_epoch} optimizer updates/epoch "
        f"({total_updates} total updates)"
    )
    warmup_steps = int(getattr(config.training, "warmup_steps", 10_000))
    num_cycles = int(getattr(config.training, "num_cycles", 1))
    cycle_decay = float(getattr(config.training, "cycle_decay", 0.9))

    # For multiple cycles, first_cycle_steps = total_updates / num_cycles
    # This ensures each cycle is roughly equal in length
    first_cycle_steps = total_updates // num_cycles if num_cycles > 1 else total_updates

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=1.0,  # Equal cycle lengths
        max_lr=config.training.learning_rate,
        min_lr=getattr(config.training, "lr_end", 1e-6),
        warmup_steps=warmup_steps,
        gamma=cycle_decay,  # Decay max_lr by this factor after each restart
    )

    if num_cycles > 1:
        logger.info(
            f"SGDR scheduler: {num_cycles} cycles of ~{first_cycle_steps} steps each, "
            f"gamma={cycle_decay} (LR decays by {(1-cycle_decay)*100:.0f}% per restart)"
        )

    amp_enabled = bool(config.training.use_amp) and device_type == 'cuda'
    amp_dtype_name = str(getattr(config.training, "amp_dtype", "bfloat16")).lower()
    if config.training.use_amp:
        if device_type != 'cuda':
            raise RuntimeError("bfloat16 AMP requested but CUDA device is not available.")
        if amp_dtype_name not in {"bfloat16", "bf16"}:
            raise ValueError(f"Only bfloat16 AMP is supported, got '{amp_dtype_name}'.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 AMP requested but CUDA device does not support bf16.")
    amp_dtype = torch.bfloat16

    # Provide an autocast wrapper compatible with both torch.amp and torch.cuda.amp
    # Older PyTorch versions do not accept the 'device_type' argument.
    from contextlib import contextmanager
    try:
        # Probe signature (do not enter context)
        _probe_ctx = autocast(device_type=device_type, enabled=False, dtype=amp_dtype)
        def amp_autocast():
            return autocast(device_type=device_type, enabled=amp_enabled, dtype=amp_dtype)
    except TypeError:  # Older API without device_type
        try:
            from torch.cuda.amp import autocast as cuda_autocast  # type: ignore
            def amp_autocast():
                return cuda_autocast(enabled=amp_enabled, dtype=amp_dtype)
        except Exception:
            @contextmanager
            def amp_autocast():
                yield

    # GradScaler is only needed for float16 AMP on Volta+ GPUs (compute capability >= 7)
    use_scaler = False
    if amp_enabled and amp_dtype == torch.float16:
        if torch.cuda.is_available():
            try:
                capability = torch.cuda.get_device_capability()
                use_scaler = capability[0] >= 7
                if not use_scaler:
                    logger.info(f"CUDA device capability {capability[0]}.{capability[1]} < 7.0. GradScaler disabled.")
            except Exception as e:
                logger.warning(f"Could not determine CUDA capability: {e}. GradScaler disabled.")
        else:
            logger.warning("AMP enabled but CUDA not available. GradScaler disabled.")
    # Create GradScaler - only specify device when using CUDA for AMP
    # GradScaler is only meaningful for CUDA; CPU always uses disabled scaler
    scaler_device = 'cuda' if (use_scaler and torch.cuda.is_available()) else None
    try:
        # PyTorch >= 2.x: torch.amp.GradScaler accepts optional 'device' kwarg
        if scaler_device:
            scaler = GradScaler(device=scaler_device, enabled=use_scaler)
        else:
            scaler = GradScaler(enabled=use_scaler)
    except TypeError:
        # Older torch.amp.GradScaler without 'device' kwarg
        try:
            scaler = GradScaler(enabled=use_scaler)
        except Exception:
            # Very old versions: use legacy CUDA GradScaler (may emit deprecation on newer torch)
            from torch.cuda.amp import GradScaler as CudaGradScaler  # type: ignore
            scaler = CudaGradScaler(enabled=use_scaler)
    if amp_enabled:
        logger.info(f"AMP enabled with dtype={amp_dtype} and GradScaler={'enabled' if use_scaler else 'disabled'}.")
    else:
        logger.info("AMP disabled.")

    checkpoint_dir = Path(config.output_root) / config.experiment_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=config.training.save_total_limit,
        keep_best=config.training.save_best_only
    )

    training_state = TrainingState()
    patience = getattr(config.training, "early_stopping_patience", None)
    es_threshold = getattr(config.training, "early_stopping_threshold", 0.0)
    # Early-stopping burn-in to avoid first-epoch outlier triggering patience
    burn_in_epochs = int(getattr(config.training, "early_stopping_burn_in_epochs", 0) or 0)
    burn_in_strategy = str(getattr(config.training, "early_stopping_burn_in_strategy", "median")).lower()
    _burn_in_vals = []  # collect val metric during burn-in window
    global_step = 0
    start_epoch = 0
    # Track mid-epoch resume info (for resuming from exact batch position)
    resume_batch_idx = 0
    is_mid_epoch = False
    # Soft-stop sentinel files (located in log_dir)
    stop_sentinel = Path(config.log_dir) / "STOP_TRAINING"
    save_sentinel = Path(config.log_dir) / "SAVE_CHECKPOINT"
    image_log_sentinel = Path(config.log_dir) / "LOG_IMAGES_NOW"
    early_exit = False

    # --- Resume logic controlled by config.training.resume_from ---
    resume_opt = str(getattr(config.training, "resume_from", "latest")).strip().lower()
    ckpt_path = None

    if resume_opt in ("", "none", "false", "off"):
        logger.info("Resume disabled by config (training.resume_from=%r). Starting fresh.", resume_opt)
    elif resume_opt == "latest":
        ckpt_path = checkpoint_manager.get_latest_checkpoint()
        if ckpt_path is None:
            logger.info("No latest checkpoint found. Starting fresh.")
    elif resume_opt == "best":
        ckpt_path = checkpoint_manager.get_best_checkpoint()
        if ckpt_path is None:
            # Fallback to latest if user asked for best but it's missing
            logger.warning("Requested resume_from='best' but best_model.pt not found; trying latest instead.")
            ckpt_path = checkpoint_manager.get_latest_checkpoint()
            if ckpt_path is None:
                logger.info("No checkpoint available. Starting fresh.")
    else:
        # Treat as explicit path
        try_path = Path(getattr(config.training, "resume_from"))
        if try_path.exists():
            ckpt_path = try_path
        else:
            logger.warning("Requested resume_from path does not exist: %s; starting fresh.", try_path)

    if ckpt_path:
        try:
            ckpt = checkpoint_manager.load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            if not ckpt:
                raise RuntimeError(f"Checkpoint returned empty data from {ckpt_path}")
            training_state = TrainingState.from_dict(ckpt.get('training_state', {}))
            start_epoch = ckpt.get('epoch', 0)
            global_step = ckpt.get('step', 0)
            # Preserve historical best; only reconcile when explicitly marked as best
            if ckpt.get('is_best', False):
                try:
                    loaded_best = float(ckpt.get('metrics', {}).get('val_f1_macro', training_state.best_metric))
                    training_state.best_metric = max(training_state.best_metric, loaded_best)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not parse best metric from checkpoint: {e}")
            # Extract mid-epoch resume info if available
            resume_batch_idx = getattr(training_state, 'batch_in_epoch', 0)
            is_mid_epoch = not getattr(training_state, 'is_epoch_boundary', True)

            # Fix: Convert 1-based checkpoint epoch back to 0-based for mid-epoch resume
            # Checkpoints store epoch+1, but the training loop uses 0-based indexing
            if is_mid_epoch and start_epoch > 0:
                start_epoch = start_epoch - 1
                logger.info(f"Mid-epoch resume: adjusted start_epoch to {start_epoch} (0-based)")

            if is_mid_epoch and resume_batch_idx > 0:
                logger.info("Resumed from %s (epoch=%s, step=%s, batch_in_epoch=%s) - mid-epoch resume",
                           ckpt_path, start_epoch, global_step, resume_batch_idx)
                # Warn about persistent_workers limitation with mid-epoch resume
                # Workers maintain RNG state that we can't restore - data order may differ slightly
                if getattr(config.data, 'persistent_workers', False):
                    logger.warning(
                        "persistent_workers=true with mid-epoch resume: worker RNG states cannot be restored. "
                        "Data order will be correct (via ResumableSampler), but per-worker augmentation RNG "
                        "may differ from original run. This is usually acceptable for training."
                    )
            else:
                logger.info("Resumed from %s (epoch=%s, step=%s)", ckpt_path, start_epoch, global_step)

            # Validate config compatibility between checkpoint and current config
            ckpt_config = ckpt.get('config', {})
            if ckpt_config:
                is_compatible, messages = validate_config_compatibility(
                    checkpoint_config=ckpt_config,
                    current_config=config,
                    strict=True  # Fail on critical mismatches
                )
                if messages:
                    logger.info("Config validation completed with %d messages", len(messages))
        except Exception as e:
            # CRITICAL: Don't silently continue with uninitialized state
            # This could overwrite existing checkpoints with bad data
            logger.exception("Failed to load checkpoint from %s. Error: %s", ckpt_path, e)
            raise RuntimeError(
                f"Checkpoint loading failed for {ckpt_path}. "
                f"To start fresh, set training.resume_from='none' in config. Error: {e}"
            ) from e

    def _ensure_conv2d_channels_last(model: torch.nn.Module, logger) -> int:
        """Force all Conv2d weights to channels_last format.

        model.to(memory_format=torch.channels_last) can silently fail to convert
        weights after checkpoint loading or dtype conversions. This explicitly
        verifies and forces the conversion for torch.compile compatibility.

        Returns number of tensors that needed fixing.
        """
        fixed_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.weight.ndim == 4:
                weight = module.weight
                if not weight.is_contiguous(memory_format=torch.channels_last):
                    old_stride = weight.stride()
                    module.weight.data = weight.contiguous(memory_format=torch.channels_last)
                    new_stride = module.weight.stride()
                    logger.warning(
                        f"Conv2d '{name}' had wrong strides {old_stride}, "
                        f"forced to channels_last: {new_stride}"
                    )
                    fixed_count += 1
        return fixed_count

    # Apply channels_last memory format AFTER all model transformations (device, dtype, checkpoint)
    # CRITICAL: This is the ONLY place channels_last should be applied. Earlier applications
    # (e.g., before bfloat16 conversion) are lost because dtype conversion creates new tensors.
    # torch.load() also restores tensors in contiguous (channels_first) format.
    # This must happen BEFORE torch.compile to ensure correct kernel generation.
    # Cache this for use throughout training (avoids repeated getattr calls)
    use_channels_last = getattr(config.training, "memory_format", "contiguous") == "channels_last"

    # CONFLICT RESOLUTION: channels_last is incompatible with torch.compile's inductor backend
    # due to stride mismatch issues in compiled kernels. Auto-disable channels_last when compile
    # is enabled since compile provides greater performance benefits (15-35% vs 5-15%).
    if use_channels_last and use_compile:
        logger.warning(
            "channels_last memory format is incompatible with torch.compile() due to stride "
            "mismatch issues in the inductor backend. Automatically disabling channels_last. "
            "torch.compile provides 15-35% speedup vs channels_last's 5-15%, so this is the "
            "optimal configuration. To use channels_last instead, set training.compile=false."
        )
        use_channels_last = False

    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        logger.info("Applied channels_last memory format to model")
        # Verify and force Conv2d weights - model.to() can fail silently after
        # checkpoint loading due to dtype mismatches or PyTorch bugs
        fixed = _ensure_conv2d_channels_last(model, logger)
        if fixed > 0:
            logger.warning(f"Fixed {fixed} Conv2d layer(s) with incorrect memory format")
        else:
            logger.info("All Conv2d layers verified as channels_last")

    # Now apply torch.compile() - AFTER checkpoint loading and memory format conversion
    # This ensures inductor kernels are compiled for the actual weights with correct strides
    if use_compile:
        logger.info("=" * 70)
        logger.info("Compiling model with torch.compile()...")
        logger.info("This will take 2-5 minutes on first forward pass but provides")
        logger.info("15-35% speedup for transformer training workloads.")

        logger.info(f"  compile_mode: {compile_mode}")
        logger.info(f"  fullgraph: {compile_fullgraph}")
        logger.info(f"  dynamic: {compile_dynamic}")

        # Clear inductor cache to avoid stale compiled kernels from previous runs
        # with different memory formats. This forces recompilation with current settings.
        try:
            torch._inductor.codecache.FxGraphCache.clear()
            logger.info("Cleared inductor FxGraphCache")
        except (AttributeError, Exception):
            pass  # Cache clearing API may not exist in all PyTorch versions

        try:
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic
            )
            logger.info("Model compiled successfully!")
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}")
            logger.warning("Continuing with eager mode (uncompiled)...")

        logger.info("=" * 70)

        # --- Force compilation during worker warmup (Windows optimization) ---
        # torch.compile() only wraps the model - actual compilation happens on first forward pass.
        # By triggering compilation NOW with a dummy input, we overlap it with worker warmup
        # (which is spawning workers in background thread), reducing total startup time.
        # Without this, worker spawn (~50s) and compilation (2-5min) would be sequential.
        try:
            logger.info("Triggering torch.compile graph compilation (overlapping with worker warmup)...")
            compile_start = time.time()

            # Create dummy input matching expected batch shape
            dummy_batch_size = config.data.batch_size
            dummy_images = torch.randn(
                dummy_batch_size, 3, config.data.image_size, config.data.image_size,
                device=device, dtype=dtype
            )
            if use_channels_last:
                dummy_images = dummy_images.contiguous(memory_format=torch.channels_last)

            # Trigger actual compilation (this takes 2-5 minutes on first run)
            with torch.no_grad():
                _ = model(dummy_images)

            compile_time = time.time() - compile_start
            logger.info(f"Graph compilation complete in {compile_time:.1f}s (workers spawned in parallel)")

            # Clear dummy tensors
            del dummy_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Forced compilation failed: {e}")
            logger.warning("Compilation will happen on first real training batch instead")

    # Log TensorBoard model graph - skip when model is compiled because FX trace
    # is incompatible with torch.compile (causes "FX to torch.jit.trace a dynamo-optimized function" error)
    if config.training.use_tensorboard and not use_compile:
        try:
            # Reuse warmup batch if available (avoids redundant worker spawn)
            warmup_thread.join(timeout=120)  # Wait for warmup to complete
            if _warmup_result["batch"] is not None:
                sample_batch = _warmup_result["batch"]
                logger.debug("Reusing warmup batch for TensorBoard graph logging")
            else:
                sample_batch = next(iter(train_loader))
            images = sample_batch['images'].to(device)
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            padding_mask = sample_batch.get('padding_mask', None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            monitor.log_model_graph(model, images, padding_mask)
        except Exception as e:
            logger.warning(f"Could not log model graph: {e}")
    elif config.training.use_tensorboard and use_compile:
        logger.info("Skipping TensorBoard model graph logging (incompatible with torch.compile)")

    # Track optimizer updates (optimizer steps), distinct from micro-steps (batches)
    # Maintain in training_state for resume compatibility - ensure all required fields exist
    _state_defaults = {
        'optimizer_updates': 0,
        'batch_in_epoch': 0,
        'is_epoch_boundary': True,
        'best_metric': float('-inf'),
    }
    for attr, default_val in _state_defaults.items():
        if not hasattr(training_state, attr):
            setattr(training_state, attr, default_val)

    # Create validation metrics once before training loop (CR-040 fix)
    # These will be reset each epoch instead of being recreated
    tc = getattr(config, "threshold_calibration", {})
    threshold = tc.get("default_threshold", 0.5) if isinstance(tc, dict) else getattr(tc, "default_threshold", 0.5)
    val_metrics = {
        'f1_macro': MultilabelF1Score(num_labels=num_tags, average="macro", threshold=threshold).to(device),
        'f1_micro': MultilabelF1Score(num_labels=num_tags, average="micro", threshold=threshold).to(device),
        'map_macro': MultilabelAveragePrecision(num_labels=num_tags, average="macro").to(device)
    }
    logger.info(f"Validation metrics initialized with {num_tags} tags, threshold={threshold}")

    # Initialize memory monitor to track RAM usage and prevent OOM
    mem_monitor = MemoryMonitor(warn_threshold_gb=115.0, critical_threshold_gb=125.0)
    logger.info("Memory monitor initialized (warn: 115 GB, critical: 125 GB)")

    # Track last validation step for step-based validation frequency
    last_validation_step = 0
    eval_steps = getattr(config.training, 'eval_steps', 0) or 0  # 0 means validate every epoch

    # NOTE: use_channels_last is defined earlier (before torch.compile) and cached for use here

    # Create dedicated CUDA stream for H2D transfers to enable pipelining
    # This allows H2D transfers to overlap with compute from the previous batch
    h2d_stream = torch.cuda.Stream() if device.type == 'cuda' else None
    if h2d_stream is not None:
        logger.info("H2D transfer stream created for async CPU→GPU pipelining")

    for epoch in range(start_epoch, config.training.num_epochs):
        # Ensure distinct shuffles across epochs in distributed mode
        # CRITICAL: This must succeed in distributed training or gradients will be corrupted
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
            logger.debug(f"Set distributed sampler epoch to {epoch}")

        # Set epoch on datasets for epoch-varying flip decisions
        # This ensures augmentation diversity across epochs while maintaining determinism
        try:
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)
                logger.debug(f"Train dataset epoch set to {epoch}")
            if val_loader is not None and hasattr(val_loader.dataset, 'set_epoch'):
                val_loader.dataset.set_epoch(epoch)
                logger.debug(f"Val dataset epoch set to {epoch}")
        except Exception as e:
            logger.debug(f"Dataset set_epoch skipped: {e}")

        model.train()
        running_loss = torch.tensor(0.0, device=device)  # Keep on GPU to avoid per-batch sync
        optimizer.zero_grad(set_to_none=True)  # Use set_to_none for memory efficiency
        accum_count = 0  # Tracks accumulated batches (handles skipped batches)
        processed_batches = 0  # Excludes skipped batches for accurate loss averaging
        total_train_samples = 0  # Track total samples for proper per-sample loss averaging
        skipped_batches = 0

        with anomaly_ctx:
            # Mid-epoch resume setup (before creating iterator to avoid double-init)
            start_step = 0
            if epoch == start_epoch and is_mid_epoch and resume_batch_idx > 0:
                # Try instant resume via ResumableSampler (O(1) instead of O(n) batch iteration)
                sampler = getattr(train_loader, 'sampler', None)
                if hasattr(sampler, 'set_start_index'):
                    # Set sampler offset BEFORE creating iterator
                    # Note: set_start_index expects SAMPLE index, not batch index
                    sample_offset = resume_batch_idx * train_loader.batch_size
                    sampler.set_start_index(sample_offset)
                    start_step = resume_batch_idx
                    logger.info(f"Resuming mid-epoch at batch {resume_batch_idx} (sample offset {sample_offset}, instant via sampler)")

            # Create iterator (sampler start_index already set if mid-epoch resume)
            # Reuse warmup iterator for first epoch to avoid redundant worker spawn
            if epoch == start_epoch and not is_mid_epoch:
                join_start = time.perf_counter()
                warmup_thread.join(timeout=120)  # Ensure warmup complete
                join_wait = time.perf_counter() - join_start
                warmup_duration = _warmup_result.get("duration", "N/A")
                if isinstance(warmup_duration, (int, float)):
                    logger.info(f"Warmup join waited {join_wait:.1f}s (warmup total: {warmup_duration:.1f}s)")
                else:
                    logger.info(f"Warmup join waited {join_wait:.1f}s (warmup duration unknown)")
                if _warmup_result["error"]:
                    logger.warning(f"Worker warmup failed, creating fresh iterator: {_warmup_result['error']}")
                    train_iter = iter(train_loader)
                elif _warmup_result["iter"] is not None:
                    train_iter = _warmup_result["iter"]
                    _warmup_result["iter"] = None  # Clear to prevent reuse
                    logger.info("Reusing warmed-up iterator (workers already spawned)")
                else:
                    train_iter = iter(train_loader)
            else:
                # Subsequent epochs or mid-epoch resume: create fresh iterator
                # (persistent_workers=True keeps workers alive, so spawn is instant)
                train_iter = iter(train_loader)

            # Fallback path for mid-epoch resume without ResumableSampler
            if epoch == start_epoch and is_mid_epoch and resume_batch_idx > 0 and start_step == 0:
                # No ResumableSampler - must iterate through batches (slow fallback)
                logger.info(f"Resuming mid-epoch: skipping {resume_batch_idx} batches (fallback mode, no ResumableSampler)...")
                skip_start = time.time()

                for i in range(resume_batch_idx):
                    next(train_iter)  # Consume and discard - batch still gets loaded
                    if (i + 1) % 500 == 0:
                        elapsed = time.time() - skip_start
                        rate = (i + 1) / elapsed
                        remaining = (resume_batch_idx - i - 1) / rate
                        logger.info(f"  Skip progress: {i + 1}/{resume_batch_idx} batches (~{remaining:.0f}s remaining)")

                skip_elapsed = time.time() - skip_start
                logger.info(f"Skip complete: {resume_batch_idx} batches in {skip_elapsed:.1f}s")
                start_step = resume_batch_idx

            for step, batch in enumerate(train_iter, start=start_step):
                # Periodic memory cleanup to prevent fragmentation (every 5000 steps)
                if global_step > 0 and global_step % 5000 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Filter out error samples that failed to load (zero-valued samples corrupt gradients)
                error_flags = batch.get('error')
                if error_flags is not None and isinstance(error_flags, torch.Tensor) and error_flags.any():
                    valid_mask = ~error_flags
                    num_errors = error_flags.sum().item()
                    if valid_mask.sum() == 0:
                        logger.warning(f"Skipping batch {global_step}: all {num_errors} samples failed to load")
                        skipped_batches += 1
                        continue
                    # Filter batch to only valid samples
                    logger.debug(f"Filtering {num_errors} error samples from batch {global_step}")
                    batch = {
                        k: v[valid_mask] if isinstance(v, torch.Tensor) and v.size(0) == len(error_flags) else v
                        for k, v in batch.items()
                    }

                # Validate rating labels on CPU BEFORE GPU transfer (avoids GPU sync per batch)
                cpu_rating_labels, is_valid = rating_validator.validate_and_handle(
                    batch['rating_labels'], batch, global_step
                )
                if not is_valid:
                    logger.warning(f"Skipping batch {global_step} due to invalid rating labels")
                    if accum_count > 0:
                        logger.warning(
                            f"Discarding {accum_count} accumulated gradient steps due to invalid batch. "
                            f"Consider reducing gradient_accumulation_steps if this happens frequently."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        accum_count = 0
                    skipped_batches += 1
                    continue

                # Transfer tensors to GPU (after CPU validation to avoid unnecessary GPU syncs)
                # Use dedicated H2D stream to overlap transfers with compute from previous batch
                pmask = batch.get('padding_mask', None)
                h2d_ctx = torch.cuda.stream(h2d_stream) if h2d_stream is not None else nullcontext()
                with h2d_ctx:
                    images = batch['images'].to(device, non_blocking=True)
                    if use_channels_last:
                        images = images.contiguous(memory_format=torch.channels_last)
                    tag_labels = batch['tag_labels'].to(device, non_blocking=True)
                    rating_labels = cpu_rating_labels.to(device, non_blocking=True)
                    if pmask is not None:
                        pmask = pmask.to(device=device, dtype=torch.bool, non_blocking=True)

                # Sync H2D stream before compute (ensures all transfers complete before model forward)
                if h2d_stream is not None:
                    torch.cuda.current_stream().wait_stream(h2d_stream)

                # Assert that input data is finite and labels are in range (only when debug enabled to avoid GPU sync)
                if config.debug.enabled:
                    assert_finite(images, tag_labels, names=['images', 'tag_labels'], batch=batch, config=config)

                if getattr(config.debug, 'log_input_stats', False) and (global_step % config.training.logging_steps == 0):
                    # OPTIMIZED: Single GPU sync via .tolist() instead of multiple .item() calls
                    with torch.no_grad():
                        img_stats = torch.stack([images.min(), images.max(), images.mean()]).cpu().tolist()
                    img_min, img_max, img_mean = img_stats
                    monitor.log_scalar('train/image_min', img_min, global_step)
                    monitor.log_scalar('train/image_max', img_max, global_step)
                    monitor.log_scalar('train/image_mean', img_mean, global_step)
                    logger.debug(f"Input stats - min: {img_min:.6f}, mean: {img_mean:.6f}, max: {img_max:.6f}")

                with amp_autocast():
                    outputs = model(images, padding_mask=pmask)

                    if getattr(config.debug, 'log_activation_stats', False) and (global_step % config.training.logging_steps == 0):
                        # OPTIMIZED: Single GPU sync via .tolist() instead of multiple .item() calls
                        tag_logits = outputs.get('tag_logits')
                        rating_logits = outputs.get('rating_logits')
                        with torch.no_grad():
                            if tag_logits is not None:
                                t_min, t_max, t_mean = torch.stack([tag_logits.min(), tag_logits.max(), tag_logits.mean()]).cpu().tolist()
                                monitor.log_scalar('train/tag_logits_min', t_min, global_step)
                                monitor.log_scalar('train/tag_logits_max', t_max, global_step)
                                monitor.log_scalar('train/tag_logits_mean', t_mean, global_step)
                                logger.debug(f"Tag logits stats - min: {t_min:.6f}, mean: {t_mean:.6f}, max: {t_max:.6f}")
                            if rating_logits is not None:
                                r_min, r_max, r_mean = torch.stack([rating_logits.min(), rating_logits.max(), rating_logits.mean()]).cpu().tolist()
                                monitor.log_scalar('train/rating_logits_min', r_min, global_step)
                                monitor.log_scalar('train/rating_logits_max', r_max, global_step)
                                monitor.log_scalar('train/rating_logits_mean', r_mean, global_step)
                                logger.debug(f"Rating logits stats - min: {r_min:.6f}, mean: {r_mean:.6f}, max: {r_max:.6f}")

                    # Assert that model outputs are finite before loss calculation (only when debug enabled to avoid GPU sync)
                    if config.debug.enabled:
                        assert_finite(
                            outputs['tag_logits'],
                            outputs['rating_logits'],
                            names=['tag_logits', 'rating_logits'],
                            batch=batch,
                            outputs=outputs,
                            config=config
                        )

                    loss, losses = criterion(outputs['tag_logits'], outputs['rating_logits'], tag_labels, rating_labels)

                # PERF: Skipped isfinite(loss) check to avoid per-step CPU-GPU sync.
                # GradScaler will handle NaN/Inf gradients by skipping the optimizer step.
                # if not torch.isfinite(loss):
                #     # Avoid calling .item() on non-finite tensor (can crash on some PyTorch versions)
                #     logger.warning(f"Found non-finite loss at step {global_step}; skipping step")
                #     overflow_cfg = getattr(config.training, 'overflow_backoff_on_nan', None)
                #     if overflow_cfg and getattr(overflow_cfg, "enabled", False):
                #         factor = getattr(overflow_cfg, "factor", 0.1)
                #         MIN_LR = 1e-8  # Prevent learning rate from going to zero
                #         for g in optimizer.param_groups:
                #             g["lr"] = max(g["lr"] * factor, MIN_LR)
                #             if g["lr"] == MIN_LR:
                #                 logger.warning(f"Learning rate hit minimum bound {MIN_LR}")
                #     if accum_count > 0:
                #         logger.warning(
                #             f"Discarding {accum_count} accumulated gradient steps due to non-finite loss. "
                #             f"Consider reducing learning rate or enabling gradient clipping."
                #         )
                #     optimizer.zero_grad(set_to_none=True)
                #     # CRITICAL: Update scaler state when skipping batch to maintain consistency
                #     # Without this, the scaler's internal loss scale may become stale
                #     scaler.update()
                #     accum_count = 0
                #     skipped_batches += 1
                #     continue

                # Average the loss over accumulation micro-steps before backward
                scaler.scale(loss / accum).backward()
                accum_count += 1

                if accum_count >= accum:
                    # --- TensorBoard: param/grad histograms (throttled) ---
                    try:
                        interval = getattr(config, "param_hist_interval_steps", None)
                        if interval is None:
                            training_cfg = getattr(config, "training", None)
                            interval = getattr(training_cfg, "param_hist_interval_steps", None) if training_cfg else None
                        if interval and (global_step % int(interval) == 0):
                            monitor.log_param_and_grad_histograms(model, global_step)
                    except Exception:
                        pass
                    scaler.unscale_(optimizer)

                    if getattr(config.debug, 'log_gradient_norm', False) and (global_step % config.training.logging_steps == 0):
                        # Compute gradient norm using foreach operations (avoids memory spike from concatenation)
                        grads = [p.grad for p in model.parameters() if p.grad is not None]
                        if grads:
                            # Use _foreach_norm for efficient per-tensor norms, then combine
                            norms = torch._foreach_norm(grads, ord=2)
                            total_norm = torch.stack(norms).norm(2).item()
                        else:
                            total_norm = 0.0
                        monitor.log_scalar('train/grad_norm', total_norm, global_step)

                    grad_clip_cfg = getattr(config.training, 'gradient_clipping', None)
                    if grad_clip_cfg and getattr(grad_clip_cfg, 'enabled', True):
                        max_norm = getattr(grad_clip_cfg, 'max_norm', 1.0)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    # Check all gradients for finiteness without concatenation
                    # Uses foreach operations to avoid GPU memory spike from concatenation
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    if grads:
                        # Use foreach to check all gradients in parallel without concat
                        # torch._foreach_norm returns a list of norms - check if any is inf/nan
                        norms = torch._foreach_norm(grads, ord=2)
                        stacked = torch.stack(norms)
                        grads_finite = torch.isfinite(stacked).all().item()
                    else:
                        grads_finite = True
                    if not grads_finite:
                        logger.warning("Skipping optimizer step due to non-finite gradients")
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        accum_count = 0
                        continue

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # Use set_to_none for memory efficiency
                    accum_count = 0
                    global_step += 1
                    try:
                        scheduler.step()
                    except Exception:
                        # keep training even if a rare scheduler state issue occurs
                        pass

                    # Count optimizer updates and handle periodic checkpointing
                    try:
                        training_state.optimizer_updates += 1
                        save_every = int(getattr(getattr(config, 'training', {}), 'save_steps', 0) or 0)
                    except Exception:
                        save_every = 0

                    if save_every > 0 and (training_state.optimizer_updates % save_every == 0):
                        try:
                            # Include current batch in sample count for accurate averaging
                            current_train_loss = (running_loss + loss.detach() * batch_size).item() / max(1, total_train_samples + batch_size)
                        except Exception:
                            current_train_loss = float('nan')

                        training_state.epoch = epoch + 1
                        training_state.global_step = global_step
                        training_state.train_loss = float(current_train_loss) if np.isfinite(current_train_loss) else current_train_loss
                        # Track mid-epoch position for resume
                        training_state.batch_in_epoch = step
                        training_state.is_epoch_boundary = False

                        try:
                            checkpoint_manager.save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch + 1,
                                step=global_step,
                                metrics={'train_loss': float(current_train_loss) if np.isfinite(current_train_loss) else current_train_loss},
                                training_state=training_state,
                                is_best=False,
                                config=config.to_dict(),
                                train_loader=train_loader
                            )
                            logger.info(
                                "Periodic save: optimizer_update=%s, global_step=%s",
                                training_state.optimizer_updates,
                                global_step,
                            )
                        except Exception as e:
                            logger.warning("Periodic save failed: %s", e)

                # Accumulate loss weighted by batch size for proper per-sample averaging
                batch_size = images.size(0)
                running_loss = running_loss + loss.detach() * batch_size
                total_train_samples += batch_size
                processed_batches += 1

                # Early soft stop check - handles step 0 and mid-accumulation
                # Throttled: event check is cheap (atomic), but sentinel.exists() is a syscall
                # Check sentinel every 10 steps to balance responsiveness vs filesystem overhead
                stop_requested = soft_stop_event.is_set() or (step % 10 == 0 and stop_sentinel.exists())
                if stop_requested:
                    logger.info("Soft stop requested - saving checkpoint...")

                    # Calculate correct batch position for resume
                    if accum_count > 0:
                        # Discard partial gradients - they're incomplete
                        logger.info(f"Discarding {accum_count} incomplete accumulation steps")
                        optimizer.zero_grad(set_to_none=True)
                        # Restart from first batch of incomplete cycle (ensure non-negative)
                        save_batch_position = max(0, step - accum_count + 1)
                    else:
                        # Next batch to process
                        save_batch_position = step + 1

                    try:
                        current_train_loss = running_loss.item() / max(1, total_train_samples)
                    except Exception:
                        current_train_loss = float('nan')

                    # Update training state for checkpoint
                    training_state.epoch = epoch + 1
                    training_state.global_step = global_step
                    training_state.train_loss = float(current_train_loss) if np.isfinite(current_train_loss) else current_train_loss
                    training_state.batch_in_epoch = save_batch_position
                    training_state.is_epoch_boundary = False

                    try:
                        checkpoint_manager.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch + 1,
                            step=global_step,
                            metrics={'train_loss': float(current_train_loss) if np.isfinite(current_train_loss) else current_train_loss},
                            training_state=training_state,
                            is_best=False,
                            config=config.to_dict(),
                            train_loader=train_loader
                        )
                        logger.info(
                            "Soft stop checkpoint saved at global_step=%s, batch_in_epoch=%s (accum_count was %s)",
                            global_step, save_batch_position, accum_count
                        )
                    except Exception as e:
                        logger.error("Soft stop: failed to save checkpoint: %s", e)

                    early_exit = True
                    break

                # One-shot save handling (without stopping) - only at safe points after optimizer step
                if accum_count == 0 and global_step > 0:
                    save_now = save_sentinel.exists()
                    if save_now:
                        state_snapshot = {
                            'epoch': epoch + 1,
                            'global_step': global_step,
                            'step': step + 1,
                            'running_loss': running_loss,
                            'processed_batches': processed_batches,
                            'total_train_samples': total_train_samples
                        }

                        try:
                            current_train_loss = state_snapshot['running_loss'] / max(1, state_snapshot['total_train_samples'])
                        except Exception:
                            current_train_loss = float('nan')

                        # Update training state using frozen snapshot
                        training_state.epoch = state_snapshot['epoch']
                        training_state.global_step = state_snapshot['global_step']
                        training_state.train_loss = float(current_train_loss) if np.isfinite(current_train_loss) else current_train_loss
                        # Track mid-epoch position for resume
                        training_state.batch_in_epoch = step + 1  # Next batch to process (consistent with soft stop)
                        training_state.is_epoch_boundary = False

                        # Save checkpoint (updates last.pt atomically)
                        try:
                            checkpoint_manager.save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=state_snapshot['epoch'],
                                step=state_snapshot['global_step'],
                                metrics={'train_loss': float(current_train_loss) if np.isfinite(current_train_loss) else current_train_loss},
                                training_state=training_state,
                                is_best=False,
                                config=config.to_dict(),
                                train_loader=train_loader
                            )
                            logger.info("One-shot save: checkpoint written at step %s.", state_snapshot['global_step'])
                        except Exception as e:
                            logger.warning("One-shot save: failed to write checkpoint: %s", e)

                        # Clear the one-shot save sentinel
                        try:
                            save_sentinel.unlink()
                        except Exception:
                            pass

                # Log every N steps (throttled) and ensure first-step write
                if global_step == 1 or (global_step % config.training.logging_steps == 0):
                    monitor.log_step(
                        global_step,
                        loss.item(),
                        losses,
                        optimizer.param_groups[0]['lr'],
                        images.size(0),
                    )
                    # NOTE: Histogram logging moved to optimizer step block (lines 1166-1175)
                    # using param_hist_interval_steps for proper throttling

                # Periodic cache summary (hardcoded interval)
                if cache_monitor.enabled and (global_step % CACHE_MONITOR_EVERY_STEPS == 0):
                    try:
                        logging.getLogger('cache_monitor').info(cache_monitor.format_summary())
                    except Exception:
                        pass

                # Log rating validation statistics periodically
                if global_step % 1000 == 0:
                    stats = rating_validator.get_stats()
                    if stats.get('invalid_batches', 0) > 0:
                        logger.warning(
                            f"Rating validation stats at step {global_step}: "
                            f"{stats['invalid_batches']} invalid batches "
                            f"({stats['invalid_samples']} samples, "
                            f"{stats['invalid_rate']:.2%} rate)"
                        )
                        # Log to monitor for tracking
                        monitor.log_scalar('validation/invalid_rating_batches', stats['invalid_batches'], global_step)
                        monitor.log_scalar('validation/invalid_rating_rate', stats['invalid_rate'], global_step)

                # Memory monitoring (check every 2000 steps to reduce psutil overhead)
                # psutil calls are ~1-5ms each; at scale this adds up
                if global_step % 2000 == 0:
                    try:
                        mem_stats = mem_monitor.check_memory()
                        # Log to TensorBoard for tracking trends
                        monitor.log_scalar('memory/system_used_gb', mem_stats['system_used_gb'], global_step)
                        monitor.log_scalar('memory/system_percent', mem_stats['system_percent'], global_step)
                        monitor.log_scalar('memory/process_total_gb', mem_stats['total_process_gb'], global_step)
                        monitor.log_scalar('memory/workers_gb', mem_stats['workers_gb'], global_step)
                    except Exception as e:
                        logger.debug(f"Memory monitoring failed: {e}")

                # Orientation health check (throttled to every 1000 steps)
                if global_step % 1000 == 0:
                    try:
                        if orientation_monitor is not None and oh is not None:
                            orientation_monitor.check_health(oh)
                    except Exception:
                        pass

                # Step-based training image logging for TensorBoard
                # Also supports manual trigger via 'i' hotkey (creates LOG_IMAGES_NOW sentinel)
                # Sentinel check throttled to every 10 steps to reduce syscall overhead
                image_log_steps = getattr(config.monitor.tb_image_logging, 'image_log_steps', 0)
                manual_image_trigger = (global_step % 10 == 0) and image_log_sentinel.exists()
                should_log_images = (
                    config.training.use_tensorboard
                    and (
                        manual_image_trigger
                        or (image_log_steps > 0 and global_step % image_log_steps == 0 and global_step > 0)
                    )
                )
                if should_log_images:
                    # Clear manual trigger sentinel if it was used
                    if manual_image_trigger:
                        try:
                            image_log_sentinel.unlink()
                        except FileNotFoundError:
                            pass
                        logger.info(f"Manual image logging triggered at step {global_step}")
                    try:
                        with torch.no_grad():
                            probs = torch.sigmoid(outputs['tag_logits'])
                            tag_names = [vocab.index_to_tag[i] for i in range(len(vocab.index_to_tag))]
                            monitor.log_predictions(
                                step=global_step,
                                images=images,
                                predictions=probs,
                                targets=tag_labels,
                                tag_names=tag_names,
                                prefix="train",
                                max_images=config.monitor.tb_image_logging.max_samples,
                                topk=config.monitor.tb_image_logging.topk,
                                rating_logits=outputs.get('rating_logits'),
                                rating_labels=rating_labels,
                            )
                            logger.info(f"Logged {config.monitor.tb_image_logging.max_samples} training images to TensorBoard at step {global_step}")
                    except Exception as e:
                        logger.warning(f"Failed to log training images: {e}")

            if stats_queue:
                # Non-blocking drain of augmentation stats (accept both tuple and bare dict)
                while True:
                    try:
                        item = stats_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.warning(f"Error reading stats queue: {e}")
                        break

                    if isinstance(item, tuple) and len(item) == 2:
                        stat_type, stats_data = item
                    elif isinstance(item, dict):
                        # Back-compat: bare payload treated as aug_stats
                        stat_type, stats_data = 'aug_stats', item
                    else:
                        continue

                    if stat_type == 'aug_stats':
                        # Normalize keys to monitor schema and de-dupe semantics
                        sd = dict(stats_data)
                        if 'flip_total' not in sd and 'total_flips' in sd:
                            sd['flip_total'] = sd['total_flips']
                        if 'flip_safe' not in sd and 'safe_flips' in sd:
                            sd['flip_safe'] = sd['safe_flips']
                        if 'flip_skipped_text' not in sd and 'blocked_by_text' in sd:
                            sd['flip_skipped_text'] = sd['blocked_by_text']
                        if 'blocked_by_safety' in sd:
                            sd.setdefault('flip_skipped_unmapped', sd['blocked_by_safety'])
                            sd.setdefault('flip_blocked_safety', sd['blocked_by_safety'])
                        monitor.log_augmentations(global_step, sd)

            # Logging moved into inner loop (above) to avoid missing epoch-boundary steps.

        # If a soft stop was requested, exit training before validation
        if early_exit:
            logger.info("Soft stop engaged. Exiting training loop before validation.")
            break

        # Clear mid-epoch resume flag after completing the first resumed epoch
        if epoch == start_epoch and is_mid_epoch:
            logger.info(f"Completed resumed epoch {epoch + 1} - cleared mid-epoch flag")
            is_mid_epoch = False

        avg_train_loss = running_loss.item() / max(1, total_train_samples)  # Per-sample average, sync only at epoch end

        # Log skipped batch statistics for monitoring
        if skipped_batches > 0:
            skip_rate = skipped_batches / (processed_batches + skipped_batches)
            logger.info(f"Epoch {epoch+1}: Skipped {skipped_batches} batches ({skip_rate:.2%} of total)")
            monitor.log_scalar('train/skipped_batches', skipped_batches, global_step)
            monitor.log_scalar('train/skip_rate', skip_rate, global_step)

        # Log attention mask cache statistics
        try:
            from model_architecture import TransformerBlock
            cache_stats = TransformerBlock.get_cache_stats()
            if cache_stats['hits'] + cache_stats['misses'] > 0:
                logger.info(
                    f"Attention mask cache stats - "
                    f"Hit rate: {cache_stats['hit_rate']:.2%} "
                    f"({cache_stats['hits']}/{cache_stats['hits'] + cache_stats['misses']}), "
                    f"Entries: {cache_stats['entries']}, "
                    f"Size: {cache_stats['size_mb']:.2f} MB"
                )
                monitor.log_scalar('cache/mask_hit_rate', cache_stats['hit_rate'], global_step)
                monitor.log_scalar('cache/mask_entries', cache_stats['entries'], global_step)
        except Exception as e:
            logger.debug(f"Failed to log cache stats: {e}")

        # Check if we should run validation this epoch (based on eval_steps)
        # eval_steps=0 means validate every epoch; otherwise validate every N steps
        should_validate = (
            eval_steps == 0  # 0 means always validate at epoch end
            or epoch == start_epoch  # Always validate first epoch
            or global_step - last_validation_step >= eval_steps
        )

        if not should_validate:
            # Skip validation, use cached values from training state
            avg_val_loss = getattr(training_state, 'val_loss', 0.0) or 0.0
            val_f1_macro = getattr(training_state, 'val_f1_macro', 0.0) or 0.0
            val_f1_micro = val_f1_macro  # Approximation when skipping
            val_mAP = getattr(training_state, 'val_mAP', 0.0) or 0.0
            logger.info(
                f"Epoch {epoch+1}: Skipping validation (last at step {last_validation_step}, "
                f"next at step {last_validation_step + eval_steps}). "
                f"Using cached: val_loss={avg_val_loss:.4f}, F1={val_f1_macro:.4f}"
            )
        else:
            last_validation_step = global_step
            # Validation loop
            model.eval()
            val_loss = torch.tensor(0.0, device=device)  # Keep on GPU to avoid per-batch sync
            # Reset validation metrics for this epoch (CR-040 fix: reuse instead of recreate)
            try:
                for metric in val_metrics.values():
                    metric.reset()
            except Exception as e:
                logger.warning(f"Failed to reset validation metrics, recreating: {e}")
                # CRITICAL: Clean up old metrics before recreating to prevent GPU memory leak
                # Move old metrics to CPU and delete to free GPU memory
                # Note: Avoid torch.cuda.empty_cache() here as it causes expensive global GPU sync.
                # Moving to CPU + del is sufficient; the garbage collector handles cleanup.
                for name, metric in list(val_metrics.items()):
                    try:
                        metric.cpu()  # Move to CPU to free GPU memory
                    except Exception:
                        pass
                del val_metrics
                # Recreate metrics on failure to ensure clean state
                val_metrics = {
                    'f1_macro': MultilabelF1Score(num_labels=num_tags, average="macro", threshold=threshold).to(device),
                    'f1_micro': MultilabelF1Score(num_labels=num_tags, average="micro", threshold=threshold).to(device),
                    'map_macro': MultilabelAveragePrecision(num_labels=num_tags, average="macro").to(device)
                }
            total_val_samples = 0  # Track samples for proper loss averaging
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader):
                    # Filter out error samples that failed to load
                    error_flags = batch.get('error')
                    if error_flags is not None and isinstance(error_flags, torch.Tensor) and error_flags.any():
                        valid_mask = ~error_flags
                        if valid_mask.sum() == 0:
                            continue  # Skip entirely failed batches
                        batch = {
                            k: v[valid_mask] if isinstance(v, torch.Tensor) and v.size(0) == len(error_flags) else v
                            for k, v in batch.items()
                        }

                    images = batch['images'].to(device, non_blocking=True)
                    if use_channels_last:
                        images = images.contiguous(memory_format=torch.channels_last)
                    tag_labels = batch['tag_labels'].to(device, non_blocking=True)
                    rating_labels = batch['rating_labels'].to(device, non_blocking=True)
                    total_val_samples += images.size(0)  # Count actual samples processed

                    with amp_autocast():
                        pmask = batch.get('padding_mask', None)
                        if pmask is not None:
                            pmask = pmask.to(device=device, dtype=torch.bool, non_blocking=True)
                        outputs = model(images, padding_mask=pmask)
                        loss, _ = criterion(outputs['tag_logits'], outputs['rating_logits'], tag_labels, rating_labels)
                    # Accumulate loss weighted by batch size for proper per-sample averaging
                    val_loss = val_loss + loss.detach() * images.size(0)

                    # Update streaming metrics
                    probs = torch.sigmoid(outputs['tag_logits'])
                    # Keep targets as float for consistency with lightning_module.py and proper metric computation
                    targs = tag_labels.to(device=probs.device, dtype=probs.dtype)
                    val_metrics['f1_macro'].update(probs, targs)
                    val_metrics['f1_micro'].update(probs, targs)
                    val_metrics['map_macro'].update(probs, targs)

                    if val_step == 0 and config.training.use_tensorboard:
                        tag_names = [vocab.index_to_tag[i] for i in range(len(vocab.index_to_tag))]
                        monitor.log_predictions(
                            step=global_step,
                            images=images,
                            predictions=probs,
                            targets=tag_labels,
                            tag_names=tag_names,
                            prefix="val",
                            max_images=config.monitor.tb_image_logging.max_samples,
                            topk=config.monitor.tb_image_logging.topk,
                            rating_logits=outputs.get('rating_logits'),
                            rating_labels=rating_labels,
                        )

            # Batch all metric computations into single GPU->CPU sync
            # (avoids 4 sequential syncs by transferring all at once)
            val_loss_avg = val_loss / max(1, total_val_samples)  # Per-sample average
            f1_macro_tensor = val_metrics['f1_macro'].compute()
            f1_micro_tensor = val_metrics['f1_micro'].compute()
            mAP_tensor = val_metrics['map_macro'].compute()
            metrics_batch = torch.stack([val_loss_avg, f1_macro_tensor, f1_micro_tensor, mAP_tensor]).cpu()
            avg_val_loss, val_f1_macro, val_f1_micro, val_mAP = metrics_batch.tolist()
            logger.info(
                f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                f"Val F1(macro): {val_f1_macro:.4f}, Val F1(micro): {val_f1_micro:.4f}, Val mAP: {val_mAP:.4f}"
            )
            monitor.log_validation(global_step, {'loss': avg_val_loss, 'f1_macro': val_f1_macro, 'f1_micro': val_f1_micro, 'mAP': val_mAP})

        # Scheduler already stepped per optimizer update; just read the last LR here.
        try:
            current_lr = scheduler.get_last_lr()[0]
        except Exception:
            current_lr = optimizer.param_groups[0]['lr']
        # Note: Learning rate is already logged in monitor.log_step() during training

        training_state.epoch = epoch + 1
        training_state.global_step = global_step
        training_state.train_loss = avg_train_loss
        training_state.val_loss = avg_val_loss
        training_state.val_f1_macro = val_f1_macro
        training_state.val_mAP = val_mAP
        training_state.learning_rates.append(current_lr)

        # --- TensorBoard: periodic flush ---
        try:
            monitor.flush()
        except Exception:
            pass

        # Checkpointing and early stopping based on macro F1
        is_best = False
        # Handle burn-in (ignore early-stopping decisions for first N epochs)
        if burn_in_epochs > 0 and (epoch + 1) <= burn_in_epochs:
            _burn_in_vals.append(val_f1_macro)
            prev_best_for_log = training_state.best_metric  # Capture before any modifications
            # Track best during burn-in to avoid losing a great model
            if val_f1_macro > training_state.best_metric + es_threshold:
                training_state.best_metric = val_f1_macro
                training_state.best_epoch = epoch + 1
                is_best = True  # Save checkpoint for this best model
            # On the last burn-in epoch, reset baseline to a robust summary
            if (epoch + 1) == burn_in_epochs:
                try:
                    if burn_in_strategy == "last":
                        baseline = float(_burn_in_vals[-1])
                    elif burn_in_strategy == "mean":
                        baseline = float(np.mean(_burn_in_vals))
                    elif burn_in_strategy == "max":
                        baseline = float(np.max(_burn_in_vals))
                    else:  # default: median
                        baseline = float(np.median(_burn_in_vals))
                except Exception:
                    baseline = float(np.median(_burn_in_vals))
                # Keep the better of baseline or actual best achieved during burn-in
                best_during_burnin = float(np.max(_burn_in_vals))
                prev_best = prev_best_for_log  # Use value captured before any modifications this epoch
                training_state.best_metric = max(baseline, best_during_burnin)
                training_state.patience_counter = 0
                logger.info(
                    "Early-stopping burn-in complete (epochs=%d, strategy=%s). "
                    "Baseline set to %.4f (best during burn-in %.4f, prev best %.4f).",
                    burn_in_epochs, burn_in_strategy, baseline, best_during_burnin, prev_best,
                )
            # During burn-in: patience not updated, but best is still tracked
        else:
            # LR-aware early stopping: only count patience when LR has dropped
            # significantly within a cycle (in the "fine-tuning" phase)
            current_lr = scheduler.get_last_lr()[0]
            cycle_max_lr = scheduler.max_lr  # Already accounts for gamma decay
            lr_ratio = current_lr / cycle_max_lr if cycle_max_lr > 0 else 1.0

            if val_f1_macro > training_state.best_metric + es_threshold:
                training_state.best_metric = val_f1_macro
                training_state.patience_counter = 0
                training_state.best_epoch = epoch + 1
                is_best = True
            elif lr_ratio < 0.5:
                # Only count patience when LR < 50% of cycle max (fine-tuning phase)
                training_state.patience_counter += 1
            # else: During warmup/early-cycle phase, don't increment patience
            # This prevents false early stops during cosine-induced plateaus

        # Respect "save_best_only": skip cadence saves unless this is a new best.
        # Only handle best-at-epoch saves here; periodic saves happen in-loop
        if is_best:
            # Mark as epoch boundary since this is saved at end of epoch
            training_state.is_epoch_boundary = True
            training_state.batch_in_epoch = 0
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=global_step,
                metrics={'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_f1_macro': val_f1_macro, 'val_mAP': val_mAP},
                training_state=training_state,
                is_best=True,
                config=config.to_dict(),
                train_loader=train_loader
            )

        if patience and training_state.patience_counter >= patience:
            logger.info("Early stopping triggered: no improvement in val_f1_macro for %s epochs", patience)
            break

    # --- TensorBoard: final hparams snapshot ---
    try:
        to_dict = getattr(config, "to_dict", None)
        hparams = to_dict() if callable(to_dict) else (vars(config) if hasattr(config, "__dict__") else {})
        final_metrics = {}
        if 'avg_val_loss' in locals():
            final_metrics["final/val_loss"] = float(avg_val_loss)
        if 'avg_train_loss' in locals():
            final_metrics["final/train_loss"] = float(avg_train_loss)
        final_metrics["final/best_val_f1_macro"] = float(training_state.best_metric)
        monitor.log_hyperparameters(hparams, final_metrics if final_metrics else {"final/placeholder": 1})
    except Exception:
        pass

    # Final orientation safety report with recommendations
    try:
        if oh is not None:
            report_path = Path(config.log_dir) / "orientation_safety_report.json"
            oh.generate_safety_report(report_path)
    except Exception as e:
        logger.debug(f"Failed to write orientation_safety_report.json: {e}")

    # Final attention mask cache statistics
    try:
        from model_architecture import TransformerBlock
        cache_stats = TransformerBlock.get_cache_stats()
        if cache_stats['hits'] + cache_stats['misses'] > 0:
            logger.info(
                f"Final attention mask cache statistics:\n"
                f"  Total accesses: {cache_stats['hits'] + cache_stats['misses']}\n"
                f"  Cache hits: {cache_stats['hits']} ({cache_stats['hit_rate']:.2%})\n"
                f"  Cache misses: {cache_stats['misses']}\n"
                f"  Cached entries: {cache_stats['entries']}\n"
                f"  Cache size: {cache_stats['size_mb']:.2f} MB\n"
                f"  Performance gain: ~{cache_stats['hits'] * 0.01:.2f}ms saved from mask inversions"
            )
    except Exception as e:
        logger.debug(f"Failed to log final cache stats: {e}")

    # Guaranteed resource cleanup on all exit paths
    logger.debug("Cleaning up training resources...")

    # Close monitor (flushes TensorBoard)
    try:
        monitor.close()
        logger.debug("Monitor closed successfully")
    except Exception as e:
        logger.warning(f"Error closing monitor: {e}")

    # Stop background validators
    try:
        for _loader in (train_loader, val_loader):
            ds = getattr(_loader, "dataset", None)
            validator = getattr(ds, "validator", None) if ds is not None else None
            if validator is not None and hasattr(validator, "stop"):
                validator.stop()
        logger.debug("Background validators stopped")
    except Exception as e:
        logger.warning(f"Error stopping validators: {e}")

    # Clean up stats queue
    try:
        if stats_queue is not None:
            # Drain any remaining items
            while not stats_queue.empty():
                try:
                    stats_queue.get_nowait()
                except queue.Empty:
                    break
            # Close the queue
            stats_queue.close()
            stats_queue.join_thread()
            logger.debug("Stats queue cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up stats queue: {e}")

    logger.debug("Training resource cleanup complete")

def main():
    """Main entry point for training script."""
    parser = create_config_parser()
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Setup logging
    listener = setup_logging(
        log_level=config.log_level,
        log_dir=config.log_dir,
        log_to_file=config.file_logging_enabled,
        json_console=True, # Or get from config if you add it
        rank=config.training.local_rank,
        world_size=config.training.world_size,
    )

    if args.validate_only:
        try:
            config.validate()
            logger.info("Configuration is valid.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)

    try:
        train_with_orientation_tracking(config)
    finally:
        if listener:
            listener.stop()


if __name__ == "__main__":
    main()
