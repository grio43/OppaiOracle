#!/usr/bin/env python3
"""
Enhanced training script with comprehensive orientation handling for anime image tagger.
Demonstrates integration of the orientation handler with fail-fast behavior and statistics tracking.
"""

import logging
import os
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import multiprocessing as mp
import sys
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
    ) -> tuple[torch.Tensor, bool]:
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

    metric_computer = MetricComputer(num_labels=num_tags)

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
        'architecture_type', 'attention_bias', 'attention_probs_dropout_prob',
        'hidden_dropout_prob', 'initializer_range', 'num_groups', 'num_labels',
        'num_special_tokens', 'tags_per_group', 'use_cls_token', 'use_color_token',
        'use_line_token', 'use_style_token'
    }
    model_config = {k: v for k, v in model_config.items() if k not in _unused_config_keys}

    model = create_model(**model_config)
    # Use channels_last if requested (benefits conv/projection kernels)
    if getattr(config.training, "memory_format", "contiguous") == "channels_last":
        model = model.to(memory_format=torch.channels_last)
    model.to(device)

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

    # Log the model graph
    if config.training.use_tensorboard:
        try:
            sample_batch = next(iter(train_loader))
            images = sample_batch['images'].to(device)
            padding_mask = sample_batch.get('padding_mask', None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            monitor.log_model_graph(model, images, padding_mask)
        except Exception as e:
            logger.warning(f"Could not log model graph: {e}")

    # Detect and log SDPA backend for Flash Attention verification
    if config.model.use_flash_attention:
        logger.info("=" * 70)
        logger.info("Flash Attention Configuration:")
        logger.info(f"  use_flash_attention: {config.model.use_flash_attention}")

        # Detect available SDPA backends
        try:
            if hasattr(torch.nn.attention, 'SDPBackend'):
                import torch.nn.attention as attention

                logger.info("Checking available SDPA backends...")
                with torch.no_grad():
                    # Create test tensors
                    test_q = torch.randn(1, 4, 8, 16, device=device, dtype=torch.bfloat16)
                    test_k, test_v = test_q, test_q

                    # Test each backend
                    backends = [
                        ('FLASH_ATTENTION (FlashAttention-2)', 'FLASH_ATTENTION'),
                        ('EFFICIENT_ATTENTION (Memory-Efficient)', 'EFFICIENT_ATTENTION'),
                        ('CUDNN_ATTENTION (cuDNN)', 'CUDNN_ATTENTION'),
                        ('MATH (Fallback)', 'MATH'),
                    ]

                    active_backend = None
                    for backend_name, backend_attr in backends:
                        backend_enum = getattr(attention.SDPBackend, backend_attr, None)
                        if backend_enum is not None:
                            try:
                                with attention.sdpa_kernel([backend_enum]):
                                    _ = torch.nn.functional.scaled_dot_product_attention(test_q, test_k, test_v)
                                    logger.info(f"  ✓ {backend_name} - AVAILABLE")
                                    if active_backend is None:
                                        active_backend = backend_name
                            except Exception:
                                logger.info(f"  ✗ {backend_name} - not available")

                    if active_backend:
                        logger.info(f"\n  Primary backend: {active_backend}")
                        if 'FLASH_ATTENTION' in active_backend:
                            logger.info("  Status: FlashAttention-2 is ACTIVE - optimal performance!")
                        else:
                            logger.warning(f"  Warning: Using {active_backend} instead of FlashAttention-2")
                            logger.warning("  This may result in slower attention computation")
            else:
                logger.info("  Backend detection not available in this PyTorch version")
                logger.info(f"  PyTorch version: {torch.__version__}")
        except Exception as e:
            logger.warning(f"Could not detect SDPA backend: {e}")

        logger.info("=" * 70)

    # torch.compile() optimization (PyTorch 2.0+)
    # Provides 15-35% speedup through graph optimization and kernel fusion
    if getattr(config.training, "use_compile", False):
        logger.info("=" * 70)
        logger.info("Compiling model with torch.compile()...")
        logger.info("This will take 2-5 minutes on first forward pass but provides")
        logger.info("15-35% speedup for transformer training workloads.")

        compile_mode = getattr(config.training, "compile_mode", "default")
        compile_fullgraph = getattr(config.training, "compile_fullgraph", False)
        compile_dynamic = getattr(config.training, "compile_dynamic", True)

        logger.info(f"  compile_mode: {compile_mode}")
        logger.info(f"  fullgraph: {compile_fullgraph}")
        logger.info(f"  dynamic: {compile_dynamic}")

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
    else:
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
            ignore_index=0,  # Ignore <PAD> for tags
        ),
        rating_loss_fn=AsymmetricFocalLoss(
            alpha=rating_loss_cfg.alpha,
            clip=rating_loss_cfg.clip,
            gamma_neg=rating_loss_cfg.gamma_neg,
            gamma_pos=rating_loss_cfg.gamma_pos,
            label_smoothing=rating_loss_cfg.label_smoothing,
            ignore_index=None,  # Keep all rating classes (no pad)
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

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=total_updates,
        cycle_mult=1.0,
        max_lr=config.training.learning_rate,
        min_lr=getattr(config.training, "lr_end", 1e-6),
        warmup_steps=warmup_steps,
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
            if is_mid_epoch and resume_batch_idx > 0:
                logger.info("Resumed from %s (epoch=%s, step=%s, batch_in_epoch=%s) - mid-epoch resume",
                           ckpt_path, start_epoch, global_step, resume_batch_idx)
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

    # Track optimizer updates (optimizer steps), distinct from micro-steps (batches)
    # Maintain in training_state for resume compatibility - ensure all required fields exist
    _state_defaults = {
        'optimizer_updates': 0,
        'batch_in_epoch': 0,
        'is_epoch_boundary': True,
        'best_metric': 0.0,
    }
    for attr, default_val in _state_defaults.items():
        if not hasattr(training_state, attr):
            setattr(training_state, attr, default_val)

    # Create validation metrics once before training loop (CR-040 fix)
    # These will be reset each epoch instead of being recreated
    num_tags = len(vocab.index_to_tag)
    threshold = getattr(getattr(config, "threshold_calibration", {}), "default_threshold", 0.5)
    val_metrics = {
        'f1_macro': MultilabelF1Score(num_labels=num_tags, average="macro", threshold=threshold).to(device),
        'f1_micro': MultilabelF1Score(num_labels=num_tags, average="micro", threshold=threshold).to(device),
        'map_macro': MultilabelAveragePrecision(num_labels=num_tags, average="macro").to(device)
    }
    logger.info(f"Validation metrics initialized with {num_tags} tags, threshold={threshold}")

    # Initialize memory monitor to track RAM usage and prevent OOM
    mem_monitor = MemoryMonitor(warn_threshold_gb=115.0, critical_threshold_gb=125.0)
    logger.info("Memory monitor initialized (warn: 115 GB, critical: 125 GB)")

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
            if hasattr(val_loader.dataset, 'set_epoch'):
                val_loader.dataset.set_epoch(epoch)
                logger.debug(f"Val dataset epoch set to {epoch}")
        except Exception as e:
            logger.debug(f"Dataset set_epoch skipped: {e}")

        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # Use set_to_none for memory efficiency
        accum_count = 0  # Tracks accumulated batches (handles skipped batches)
        processed_batches = 0  # Excludes skipped batches for accurate loss averaging
        skipped_batches = 0

        with anomaly_ctx:
            for step, batch in enumerate(train_loader):
                # Skip already-processed batches when resuming mid-epoch
                if epoch == start_epoch and is_mid_epoch and step < resume_batch_idx:
                    # Fast-forward through already-processed batches
                    # RNG states are restored, so dataset order is identical
                    if step == 0:
                        logger.info(f"Resuming mid-epoch: skipping first {resume_batch_idx} batches (already processed)")
                    continue

                images = batch['images'].to(device, non_blocking=True)
                # accum defined above; used for correct grad-accum scaling
                if getattr(config.training, "memory_format", "contiguous") == "channels_last":
                    images = images.contiguous(memory_format=torch.channels_last)
                tag_labels = batch['tag_labels'].to(device, non_blocking=True)
                rating_labels = batch['rating_labels'].to(device, non_blocking=True)

                # Assert that input data is finite and labels are in range
                assert_finite(images, tag_labels, names=['images', 'tag_labels'], batch=batch, config=config)

                rating_labels, is_valid = rating_validator.validate_and_handle(
                    rating_labels, batch, global_step
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

                if getattr(config.debug, 'log_input_stats', False) and (global_step % config.training.logging_steps == 0):
                    monitor.log_scalar('train/image_min', images.min().item(), global_step)
                    monitor.log_scalar('train/image_max', images.max().item(), global_step)
                    monitor.log_scalar('train/image_mean', images.mean().item(), global_step)
                    logger.debug(
                        f"Input stats - min: {images.min().item():.6f}, "
                        f"mean: {images.mean().item():.6f}, max: {images.max().item():.6f}"
                    )

                with amp_autocast():
                    pmask = batch.get('padding_mask', None)
                    if pmask is not None:
                        pmask = pmask.to(device=device, dtype=torch.bool, non_blocking=True)
                    outputs = model(images, padding_mask=pmask)

                    if getattr(config.debug, 'log_activation_stats', False) and (global_step % config.training.logging_steps == 0):
                        tag_logits = outputs.get('tag_logits')
                        rating_logits = outputs.get('rating_logits')
                        if tag_logits is not None:
                            monitor.log_scalar('train/tag_logits_min', tag_logits.min().item(), global_step)
                            monitor.log_scalar('train/tag_logits_max', tag_logits.max().item(), global_step)
                            monitor.log_scalar('train/tag_logits_mean', tag_logits.mean().item(), global_step)
                            logger.debug(
                                f"Tag logits stats - min: {tag_logits.min().item():.6f}, "
                                f"mean: {tag_logits.mean().item():.6f}, max: {tag_logits.max().item():.6f}"
                            )
                        if rating_logits is not None:
                            monitor.log_scalar('train/rating_logits_min', rating_logits.min().item(), global_step)
                            monitor.log_scalar('train/rating_logits_max', rating_logits.max().item(), global_step)
                            monitor.log_scalar('train/rating_logits_mean', rating_logits.mean().item(), global_step)
                            logger.debug(
                                f"Rating logits stats - min: {rating_logits.min().item():.6f}, "
                                f"mean: {rating_logits.mean().item():.6f}, max: {rating_logits.max().item():.6f}"
                            )

                    # Assert that model outputs are finite before loss calculation
                    assert_finite(
                        outputs['tag_logits'],
                        outputs['rating_logits'],
                        names=['tag_logits', 'rating_logits'],
                        batch=batch,
                        outputs=outputs,
                        config=config
                    )

                    loss, losses = criterion(outputs['tag_logits'], outputs['rating_logits'], tag_labels, rating_labels)

                if not torch.isfinite(loss):
                    # Avoid calling .item() on non-finite tensor (can crash on some PyTorch versions)
                    logger.warning(f"Found non-finite loss at step {global_step}; skipping step")
                    if getattr(config.training.overflow_backoff_on_nan, "enabled", False):
                        factor = getattr(config.training.overflow_backoff_on_nan, "factor", 0.1)
                        MIN_LR = 1e-8  # Prevent learning rate from going to zero
                        for g in optimizer.param_groups:
                            g["lr"] = max(g["lr"] * factor, MIN_LR)
                            if g["lr"] == MIN_LR:
                                logger.warning(f"Learning rate hit minimum bound {MIN_LR}")
                    if accum_count > 0:
                        logger.warning(
                            f"Discarding {accum_count} accumulated gradient steps due to non-finite loss. "
                            f"Consider reducing learning rate or enabling gradient clipping."
                        )
                    optimizer.zero_grad(set_to_none=True)
                    # CRITICAL: Update scaler state when skipping batch to maintain consistency
                    # Without this, the scaler's internal loss scale may become stale
                    scaler.update()
                    accum_count = 0
                    skipped_batches += 1
                    continue

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
                        total_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        monitor.log_scalar('train/grad_norm', total_norm, global_step)

                    if getattr(config.training.gradient_clipping, 'enabled', True):
                        max_norm = getattr(config.training.gradient_clipping, 'max_norm', 1.0)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    grads_finite = all(
                        (p.grad is None) or torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
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
                            current_train_loss = (running_loss + loss.item()) / max(1, processed_batches + 1)
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
                                config=config.to_dict()
                            )
                            logger.info(
                                "Periodic save: optimizer_update=%s, global_step=%s",
                                training_state.optimizer_updates,
                                global_step,
                            )
                        except Exception as e:
                            logger.warning("Periodic save failed: %s", e)

                running_loss += loss.item()
                processed_batches += 1

                # Early soft stop check - handles step 0 and mid-accumulation
                # This runs EVERY iteration, not just after optimizer steps
                stop_requested = soft_stop_event.is_set() or stop_sentinel.exists()
                if stop_requested:
                    logger.info("Soft stop requested - saving checkpoint...")

                    # Calculate correct batch position for resume
                    if accum_count > 0:
                        # Discard partial gradients - they're incomplete
                        logger.info(f"Discarding {accum_count} incomplete accumulation steps")
                        optimizer.zero_grad(set_to_none=True)
                        # Restart from first batch of incomplete cycle
                        save_batch_position = step - accum_count + 1
                    else:
                        # Next batch to process
                        save_batch_position = step + 1

                    try:
                        current_train_loss = running_loss / max(1, processed_batches)
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
                            config=config.to_dict()
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
                            'processed_batches': processed_batches
                        }

                        try:
                            current_train_loss = state_snapshot['running_loss'] / max(1, state_snapshot['processed_batches'])
                        except Exception:
                            current_train_loss = float('nan')

                        # Update training state using frozen snapshot
                        training_state.epoch = state_snapshot['epoch']
                        training_state.global_step = state_snapshot['global_step']
                        training_state.train_loss = float(current_train_loss) if np.isfinite(current_train_loss) else current_train_loss
                        # Track mid-epoch position for resume
                        training_state.batch_in_epoch = step
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
                                config=config.to_dict()
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
                    # Histogram logging (gated by monitor config)
                    if config.training.use_tensorboard:
                        try:
                            monitor.log_param_and_grad_histograms(model, global_step)
                        except Exception:
                            pass

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

                # Memory monitoring (check every 100 steps)
                if global_step % 100 == 0:
                    try:
                        mem_stats = mem_monitor.check_memory()
                        # Log to TensorBoard for tracking trends
                        monitor.log_scalar('memory/system_used_gb', mem_stats['system_used_gb'], global_step)
                        monitor.log_scalar('memory/system_percent', mem_stats['system_percent'], global_step)
                        monitor.log_scalar('memory/process_total_gb', mem_stats['total_process_gb'], global_step)
                        monitor.log_scalar('memory/workers_gb', mem_stats['workers_gb'], global_step)
                    except Exception as e:
                        logger.debug(f"Memory monitoring failed: {e}")

                # Orientation health check (writes/refreshes unmapped_orientation_tags.txt)
                try:
                    if orientation_monitor is not None and oh is not None:
                        orientation_monitor.check_health(oh)
                except Exception:
                    pass

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

        avg_train_loss = running_loss / max(1, processed_batches)

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

        # Validation loop
        model.eval()
        val_loss = 0.0
        # Reset validation metrics for this epoch (CR-040 fix: reuse instead of recreate)
        try:
            for metric in val_metrics.values():
                metric.reset()
        except Exception as e:
            logger.warning(f"Failed to reset validation metrics, recreating: {e}")
            # CRITICAL: Clean up old metrics before recreating to prevent GPU memory leak
            # Move old metrics to CPU and delete to free GPU memory
            for name, metric in list(val_metrics.items()):
                try:
                    metric.cpu()  # Move to CPU to free GPU memory
                except Exception:
                    pass
            del val_metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Recreate metrics on failure to ensure clean state
            val_metrics = {
                'f1_macro': MultilabelF1Score(num_labels=num_tags, average="macro", threshold=threshold).to(device),
                'f1_micro': MultilabelF1Score(num_labels=num_tags, average="micro", threshold=threshold).to(device),
                'map_macro': MultilabelAveragePrecision(num_labels=num_tags, average="macro").to(device)
            }
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader):
                images = batch['images'].to(device, non_blocking=True)
                if getattr(config.training, "memory_format", "contiguous") == "channels_last":
                    images = images.contiguous(memory_format=torch.channels_last)
                tag_labels = batch['tag_labels'].to(device, non_blocking=True)
                rating_labels = batch['rating_labels'].to(device, non_blocking=True)

                with amp_autocast():
                    pmask = batch.get('padding_mask', None)
                    if pmask is not None:
                        pmask = pmask.to(device=device, dtype=torch.bool, non_blocking=True)
                    outputs = model(images, padding_mask=pmask)
                    loss, _ = criterion(outputs['tag_logits'], outputs['rating_logits'], tag_labels, rating_labels)
                val_loss += loss.item()

                # Update streaming metrics
                probs = torch.sigmoid(outputs['tag_logits'])
                targs = (tag_labels > 0.5).to(torch.long)
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
                    )

        avg_val_loss = val_loss / len(val_loader)
        val_f1_macro = val_metrics['f1_macro'].compute().item()
        val_f1_micro = val_metrics['f1_micro'].compute().item()
        val_mAP = val_metrics['map_macro'].compute().item()
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
            # Track best during burn-in to avoid losing a great model
            if val_f1_macro > training_state.best_metric:
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
                prev_best = training_state.best_metric
                training_state.best_metric = max(baseline, best_during_burnin)
                training_state.patience_counter = 0
                logger.info(
                    "Early-stopping burn-in complete (epochs=%d, strategy=%s). "
                    "Baseline set to %.4f (best during burn-in %.4f, prev best %.4f).",
                    burn_in_epochs, burn_in_strategy, baseline, best_during_burnin, prev_best,
                )
            # During burn-in: patience not updated, but best is still tracked
        else:
            if val_f1_macro > training_state.best_metric + es_threshold:
                training_state.best_metric = val_f1_macro
                training_state.patience_counter = 0
                training_state.best_epoch = epoch + 1
                is_best = True
            else:
                training_state.patience_counter += 1

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
                config=config.to_dict()
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
        import queue
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
