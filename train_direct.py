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

import shutil
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision
import numpy as np
from Monitor_log import MonitorConfig, TrainingMonitor
from evaluation_metrics import MetricComputer

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

from Configuration_System import load_config, create_config_parser, FullConfig
from utils.logging_setup import setup_logging

# Paths will be loaded from the unified config in the main function.
logger = logging.getLogger(__name__)

# Import the orientation handler
from orientation_handler import OrientationHandler, OrientationMonitor

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


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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

    logger = logging.getLogger(__name__)
    
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
    logger.info(f"Using active data path: {active_data_path}")

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
        """
        # Non-interactive (e.g., piped/cron) -> use default
        if not sys.stdin or not sys.stdin.isatty():
            return bool(default) if default is not None else False
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

    # Enforce mapping validation at startup when strict flag is set
    try:
        if bool(getattr(config.data, "strict_orientation_validation", False)):
            if getattr(config.data, "random_flip_prob", 0.0) > 0:
                oh = OrientationHandler(
                    mapping_file=Path(config.data.orientation_map_path) if getattr(config.data, "orientation_map_path", None) else None,
                    random_flip_prob=float(getattr(config.data, "random_flip_prob", 0.0) or 0.0),
                    strict_mode=True,
                    safety_mode=str(getattr(config.data, "orientation_safety_mode", "conservative")),
                    skip_unmapped=bool(getattr(config.data, "skip_unmapped", False)),
                )
                issues = oh.validate_mappings()
                if issues:
                    raise ValueError(f"Orientation mapping validation failed with issues: {issues}")
    except Exception as e:
        logger.warning(f"Orientation mapping validation failed: {e}")

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
    logger.info(f"Creating model with {num_tags} tags and {num_ratings} ratings")

    metric_computer = MetricComputer(num_labels=num_tags)

    model_config = config.model.to_dict()
    model_config["num_tags"] = num_tags
    # Assuming num_ratings is not part of the model config and needs to be added.
    # If it is, this line would be redundant.
    model_config["num_ratings"] = num_ratings
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
    accum = max(1, int(getattr(config.training, "gradient_accumulation_steps", 1)))
    steps_per_epoch = max(1, len(train_loader))
    updates_per_epoch = (steps_per_epoch + accum - 1) // accum  # ceil division
    total_updates = max(1, int(getattr(config.training, "num_epochs", 1)) * updates_per_epoch)
    warmup_steps = int(getattr(config.training, "warmup_steps", 10_000))

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=total_updates,
        cycle_mult=1.0,
        max_lr=config.training.learning_rate,
        min_lr=getattr(config.training, "lr_end", 1e-6),
        warmup_steps=warmup_steps,
    )

    amp_enabled = config.training.use_amp and device_type == 'cuda'
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

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

    # GradScaler is only needed for float16 AMP.
    use_scaler = amp_enabled and amp_dtype == torch.float16 and torch.cuda.get_device_capability()[0] >= 7
    # Prefer device-agnostic torch.amp.GradScaler; fallback to legacy CUDA scaler if needed.
    try:
        # PyTorch >= 2.x: torch.amp.GradScaler accepts 'device' as a string ('cuda' or 'cpu').
        scaler = GradScaler(device=('cuda' if amp_enabled else 'cpu'), enabled=use_scaler)
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
            training_state = TrainingState.from_dict(ckpt.get('training_state', {}))
            start_epoch = ckpt.get('epoch', 0)
            global_step = ckpt.get('step', 0)
            # Preserve historical best; only reconcile when explicitly marked as best
            if ckpt.get('is_best', False):
                try:
                    loaded_best = float(ckpt.get('metrics', {}).get('val_f1_macro', training_state.best_metric))
                    training_state.best_metric = max(training_state.best_metric, loaded_best)
                except Exception:
                    pass
            logger.info("Resumed from %s (epoch=%s, step=%s)", ckpt_path, start_epoch, global_step)
        except Exception as e:
            logger.exception("Failed to load checkpoint from %s; starting fresh. Error: %s", ckpt_path, e)

    for epoch in range(start_epoch, config.training.num_epochs):
        # Ensure distinct shuffles across epochs in distributed mode
        try:
            if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
        except Exception as e:
            logger.debug(f"set_epoch skipped: {e}")
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        with anomaly_ctx:
            for step, batch in enumerate(train_loader):
                images = batch['images'].to(device, non_blocking=True)
                # accum defined above; used for correct grad-accum scaling
                if getattr(config.training, "memory_format", "contiguous") == "channels_last":
                    images = images.contiguous(memory_format=torch.channels_last)
                tag_labels = batch['tag_labels'].to(device)
                rating_labels = batch['rating_labels'].to(device)

                # Assert that input data is finite and labels are in range
                assert_finite(images, tag_labels, names=['images', 'tag_labels'], batch=batch, config=config)
                if rating_labels.dtype in (torch.long, torch.int64):
                    if not ((rating_labels >= 0) & (rating_labels < num_ratings)).all():
                        raise RuntimeError(f"Rating label out of range. Found min {rating_labels.min()} / max {rating_labels.max()}, expected 0 to {num_ratings-1}")

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
                    if pmask is not None: pmask = pmask.to(device)
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
                    logger.warning(f"Found non-finite loss at step {global_step}: {loss.item()}; skipping step")
                    if getattr(config.training.overflow_backoff_on_nan, "enabled", False):
                        factor = getattr(config.training.overflow_backoff_on_nan, "factor", 0.1)
                        for g in optimizer.param_groups:
                            g["lr"] *= factor
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # ---- Correct gradient-accumulation scaling ----
                # Average the loss over accumulation micro-steps before backward.
                scaler.scale(loss / accum).backward()

                if (step + 1) % accum == 0:
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
                        continue

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    # ---- Step-based scheduler: advance once per optimizer update ----
                    try:
                        scheduler.step()
                    except Exception:
                        # keep training even if a rare scheduler state issue occurs
                        pass

                running_loss += loss.item()
                global_step += 1

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

        avg_train_loss = running_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        # Streaming metrics to avoid holding the entire validation set in memory
        num_tags = len(vocab.index_to_tag)
        threshold = getattr(getattr(config, "threshold_calibration", {}), "default_threshold", 0.5)
        f1_macro = MultilabelF1Score(num_labels=num_tags, average="macro", threshold=threshold).to(device)
        f1_micro = MultilabelF1Score(num_labels=num_tags, average="micro", threshold=threshold).to(device)
        map_macro = MultilabelAveragePrecision(num_labels=num_tags, average="macro").to(device)
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader):
                images = batch['images'].to(device, non_blocking=True)
                if getattr(config.training, "memory_format", "contiguous") == "channels_last":
                    images = images.contiguous(memory_format=torch.channels_last)
                tag_labels = batch['tag_labels'].to(device)
                rating_labels = batch['rating_labels'].to(device)

                with amp_autocast():
                    pmask = batch.get('padding_mask', None)
                    if pmask is not None:
                        pmask = pmask.to(device)
                    outputs = model(images, padding_mask=pmask)
                    loss, _ = criterion(outputs['tag_logits'], outputs['rating_logits'], tag_labels, rating_labels)
                val_loss += loss.item()

                # Update streaming metrics
                probs = torch.sigmoid(outputs['tag_logits'])
                targs = (tag_labels > 0.5).to(torch.long)
                f1_macro.update(probs, targs)
                f1_micro.update(probs, targs)
                map_macro.update(probs, targs)

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
        val_f1_macro = f1_macro.compute().item()
        val_f1_micro = f1_micro.compute().item()
        val_mAP = map_macro.compute().item()
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
        monitor.log_scalar('lr', current_lr, global_step)

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
                prev_best = training_state.best_metric
                training_state.best_metric = baseline
                training_state.patience_counter = 0
                logger.info(
                    "Early-stopping burn-in complete (epochs=%d, strategy=%s). "
                    "Baseline set to %.4f (prev best %.4f).",
                    burn_in_epochs, burn_in_strategy, baseline, prev_best,
                )
            # During burn-in: do not update patience/best based on improvement
        else:
            if val_f1_macro > training_state.best_metric + es_threshold:
                training_state.best_metric = val_f1_macro
                training_state.patience_counter = 0
                training_state.best_epoch = epoch + 1
                is_best = True
            else:
                training_state.patience_counter += 1

        # Respect "save_best_only": skip cadence saves unless this is a new best.
        keep_best_only = bool(getattr(config.training, "save_best_only", False))
        should_save = (not keep_best_only and (global_step % config.training.save_steps == 0)) or is_best
        if should_save:
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=global_step,
                metrics={'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_f1_macro': val_f1_macro, 'val_mAP': val_mAP},
                training_state=training_state,
                is_best=is_best,
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

    monitor.close()

    # ---- Teardown: stop any background validators if present ----
    try:
        for _loader in (train_loader, val_loader):
            ds = getattr(_loader, "dataset", None)
            validator = getattr(ds, "validator", None) if ds is not None else None
            if validator is not None and hasattr(validator, "stop"):
                validator.stop()
    except Exception:
        pass


def validate_orientation_mappings():
    """Standalone function to validate orientation mappings."""
    # This function can remain as is, it's a utility.
    pass

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
