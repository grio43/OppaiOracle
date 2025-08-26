#!/usr/bin/env python3
"""
Enhanced training script with comprehensive orientation handling for anime image tagger.
Demonstrates integration of the orientation handler with fail-fast behavior and statistics tracking.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import multiprocessing as mp
import sys
import random
from datetime import datetime
import torch.distributed as dist
from dataclasses import dataclass

import shutil
import torch
from torch.amp import GradScaler, autocast
import numpy as np
from Monitor_log import MonitorConfig, TrainingMonitor

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

from Configuration_System import load_config, create_config_parser, FullConfig

# Paths will be loaded from the unified config in the main function.
logger = logging.getLogger(__name__)

# Import the orientation handler
from orientation_handler import OrientationHandler

# Import base modules with error handling
try:
    from HDF5_loader import create_dataloaders
except ImportError as e:
    error_msg = (
        f"""MISSING REQUIRED FILE: HDF5_loader.py
Please ensure HDF5_loader.py exists in the current directory with create_dataloaders function.
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
from training_utils import CheckpointManager, TrainingState, setup_seed, log_sample_order_hash
from HDF5_loader import AugmentationStats, validate_dataset
from utils.logging_sanitize import ensure_finite_tensor

# Add after other imports

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
    
    # Create configuration
    config = {
        "data_dir": data_dir,
        "json_dir": json_dir,
        "vocab_path": vocab_path,
        "random_flip_prob": random_flip_prob,
        "orientation_map_path": orientation_map_path,  # Keep as Path object
        "strict_orientation": strict_orientation,
        "skip_unmapped": skip_unmapped,
        "dataloader_overrides": {
            "random_flip_prob": random_flip_prob,
            "orientation_map_path": orientation_map_path
        }
    }
    
    return config


def train_with_orientation_tracking(config: FullConfig):
    """Training loop with orientation handling and statistics tracking."""
    
    import tempfile

    logger = logging.getLogger(__name__)
    
    # Seeding & determinism
    seed, deterministic_mode = setup_seed(config.training.seed, config.training.deterministic)

    # Enable anomaly detection if configured (for debugging NaN gradients)
    if getattr(config.debug, 'detect_anomaly', False):
        logger.warning("PyTorch anomaly detection is enabled. This will slow down training.")
        torch.autograd.set_detect_anomaly(True)

    try:
        torch.use_deterministic_algorithms(deterministic_mode)
    except Exception:
        pass

    # Set up log directory
    log_dir = Path(config.log_dir or "./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    os.environ['OPPAI_LOG_DIR'] = str(log_dir)
    logger.info(f"Log directory: {log_dir}")

    log_queue = mp.Queue(maxsize=5000)

    is_primary = not dist.is_initialized() or dist.get_rank() == 0 if dist.is_available() else True

    _listener = None
    if is_primary:
        from HDF5_loader import CompressingRotatingFileHandler
        fh = CompressingRotatingFileHandler(
            log_dir / 'training.log',
            maxBytes=config.log_rotation_max_bytes,
            backupCount=config.log_rotation_backups,
            compress=True
        )
        formatter = logging.Formatter(config.log_format)
        fh.setFormatter(formatter)
        fh.setLevel(getattr(logging, config.log_level, logging.INFO))
        _listener = logging.handlers.QueueListener(log_queue, fh, respect_handler_level=True)
        _listener.start()

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

    if stats_queue:
        config.data.stats_queue = stats_queue

    train_loader, val_loader, vocab = create_dataloaders(
        data_config=config.data,
        validation_config=config.validation,
        vocab_path=Path(config.vocab_path),
        active_data_path=active_data_path,
        distributed=config.training.distributed,
        rank=config.training.local_rank,
        world_size=config.training.world_size,
        seed=seed,
        log_queue=log_queue,
    )

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

    model_config = config.model.to_dict()
    model_config["num_tags"] = num_tags
    # Assuming num_ratings is not part of the model config and needs to be added.
    # If it is, this line would be redundant.
    model_config["num_ratings"] = num_ratings
    model = create_model(**model_config)
    model.to(device)

    # Ensure tensorboard_dir points to a stable experiment root; TrainingMonitor will create a per-run subdir.
    config.tensorboard_dir = str(Path(getattr(config, "output_root", "experiments")) / getattr(config, "experiment_name", "default_experiment"))

    # Update monitor config with values from other parts of the config for backward compatibility
    if not hasattr(config, 'monitor'):
        # In case the config file is old and doesn't have a monitor section
        config.monitor = MonitorConfig()

    config.monitor.log_dir = str(log_dir)
    config.monitor.use_tensorboard = config.training.use_tensorboard
    config.monitor.tensorboard_dir = str(Path(config.output_root) / config.experiment_name)
    config.monitor.use_wandb = config.training.use_wandb

    monitor = TrainingMonitor(config.monitor)

    # --- TensorBoard: initial hparams snapshot ---
    try:
        to_dict = getattr(config, "to_dict", None)
        hparams = to_dict() if callable(to_dict) else (vars(config) if hasattr(config, "__dict__") else {})
        monitor.log_hyperparameters(hparams, {"init/placeholder": 0})
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
            alpha=config.training.focal_alpha,
            clip=config.training.focal_clip,
            gamma_neg=config.training.focal_gamma_neg,
            gamma_pos=config.training.focal_gamma_pos,
            label_smoothing=config.training.label_smoothing
        )
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

    amp_enabled = config.training.use_amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16

    # GradScaler is only needed for float16 AMP.
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = GradScaler(device='cuda', enabled=use_scaler)
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
    best_val_loss = float('inf')

    global_step = 0
    for epoch in range(config.training.num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            tag_labels = batch['tag_labels'].to(device)
            rating_labels = batch['rating_labels'].to(device)

            # Assert that input data is finite and labels are in range
            assert_finite(images, tag_labels, names=['images', 'tag_labels'], batch=batch, config=config)
            if rating_labels.dtype in (torch.long, torch.int64):
                if not ((rating_labels >= 0) & (rating_labels < num_ratings)).all():
                    raise RuntimeError(f"Rating label out of range. Found min {rating_labels.min()} / max {rating_labels.max()}, expected 0 to {num_ratings-1}")
            
            with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                pmask = batch.get('padding_mask', None)
                if pmask is not None: pmask = pmask.to(device)
                outputs = model(images, padding_mask=pmask)

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

            # Optional: automatic LR backoff when loss is non-finite (place right after computing loss)
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss detected at step {global_step}: {loss.item()}; reducing LR 10x and skipping step")
                for g in optimizer.param_groups:
                    g["lr"] *= 0.1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (step + 1) % config.training.gradient_accumulation_steps == 0:
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

                # --- Gradient Norm Monitoring ---
                if getattr(config.debug, 'log_gradient_norm', False) and (global_step % config.training.logging_steps == 0):
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    monitor.log_scalar('train/grad_norm', total_norm, global_step)

                # Clip gradients to prevent exploding gradients.
                if config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            global_step += 1

            if stats_queue:
                while not stats_queue.empty():
                    try:
                        stat_type, stats_data = stats_queue.get_nowait()
                        if stat_type == 'aug_stats':
                            monitor.log_augmentations(global_step, stats_data)
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.warning(f"Error processing stats queue: {e}")

            if global_step % config.training.logging_steps == 0:
                monitor.log_step(global_step, loss.item(), losses, optimizer.param_groups[0]['lr'], images.size(0))

        avg_train_loss = running_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader):
                images = batch['images'].to(device)
                tag_labels = batch['tag_labels'].to(device)
                rating_labels = batch['rating_labels'].to(device)
                
                with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    pmask = batch.get('padding_mask', None)
                    if pmask is not None: pmask = pmask.to(device)
                    outputs = model(images, padding_mask=pmask)
                    loss, _ = criterion(outputs['tag_logits'], outputs['rating_logits'], tag_labels, rating_labels)
                val_loss += loss.item()

                if val_step == 0 and config.training.use_tensorboard:
                    tag_preds = torch.sigmoid(outputs['tag_logits'])
                    tag_names = [vocab.index_to_tag[i] for i in range(len(vocab.index_to_tag))]
                    monitor.log_images(
                        step=global_step,
                        images=images,
                        predictions=tag_preds,
                        targets=tag_labels,
                        tag_names=tag_names,
                        prefix="val"
                    )
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        monitor.log_validation(global_step, {'loss': avg_val_loss})

        # --- TensorBoard: periodic flush ---
        try:
            monitor.flush()
        except Exception:
            pass

        # Checkpointing
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        if global_step % config.training.save_steps == 0 or is_best:
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch + 1,
                step=global_step,
                metrics={'train_loss': avg_train_loss, 'val_loss': avg_val_loss},
                is_best=is_best,
                config=config.to_dict()
            )

    # --- TensorBoard: final hparams snapshot ---
    try:
        to_dict = getattr(config, "to_dict", None)
        hparams = to_dict() if callable(to_dict) else (vars(config) if hasattr(config, "__dict__") else {})
        final_metrics = {}
        if 'avg_val_loss' in locals(): final_metrics["final/val_loss"] = float(avg_val_loss)
        if 'avg_train_loss' in locals(): final_metrics["final/train_loss"] = float(avg_train_loss)
        if 'best_val_metric' in locals(): final_metrics["final/best_val_metric"] = float(best_val_metric)
        monitor.log_hyperparameters(hparams, final_metrics if final_metrics else {"final/placeholder": 1})
    except Exception:
        pass

    monitor.close()
    if _listener:
        _listener.stop()


def validate_orientation_mappings():
    """Standalone function to validate orientation mappings."""
    # This function can remain as is, it's a utility.
    pass

def main():
    """Main entry point for training script."""
    parser = create_config_parser()
    args = parser.parse_args()
    
    config = load_config(args.config, args=args)

    # Setup logging
    level = getattr(logging, config.log_level, logging.INFO)
    fmt = config.log_format
    logging.basicConfig(level=level, format=fmt)

    if config.file_logging_enabled:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / 'train_direct.log',
            maxBytes=config.log_rotation_max_bytes,
            backupCount=config.log_rotation_backups,
        )
        handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(handler)

    if args.validate_only:
        try:
            config.validate()
            logger.info("Configuration is valid.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)

    train_with_orientation_tracking(config)


if __name__ == "__main__":
    main()
