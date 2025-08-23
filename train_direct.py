#!/usr/bin/env python3
"""
Enhanced training script with comprehensive orientation handling for anime image tagger.
Demonstrates integration of the orientation handler with fail-fast behavior and statistics tracking.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import multiprocessing as mp
import sys
import random
import torch.distributed as dist
from dataclasses import dataclass

import torch
from torch.amp import GradScaler, autocast
import numpy as np

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
from training_utils import CheckpointManager, TrainingState

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


def setup_orientation_aware_training(
    data_dir: Path,
    json_dir: Path,
    vocab_path: Path,
    orientation_map_path: Optional[Path] = None,
    random_flip_prob: float = 0.2,
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


def train_with_orientation_tracking():
    """Training loop with orientation handling and statistics tracking."""
    
    # All required modules are imported at module load time; ImportErrors are raised immediately.

    # Set up logger early so it's available for all messages
    import logging
    from logging.handlers import QueueListener, RotatingFileHandler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Use a proper multiprocessing.Queue for cross-process communication
    # The BoundedLevelAwareQueue uses threading primitives which cannot be
    # safely pickled and used in worker processes
    log_queue = mp.Queue(maxsize=5000)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(sh)
    
    # Enhanced configuration with orientation handling
    config = {
        "learning_rate": 1e-4,  # Reduced for stability
        "batch_size": 32,
        "gradient_accumulation": 2,
        "num_epochs": 8,
        "warmup_steps": 10_000,
        "weight_decay": 0.01,
        "label_smoothing": 0.05,
        "max_grad_norm": 1.0,  # Add gradient clipping
        "data_dir": Path("/media/andrewk/qnap-public/workspace/shard_00022/"),
        "json_dir": Path("/media/andrewk/qnap-public/workspace/shard_00022/"),
        "vocab_path": Path("vocabulary.json"),
        "num_workers": 4,  # Store in config for dataset initialization
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "amp": True,
        # Orientation-specific settings
        "random_flip_prob": 0.2,  # 20% chance of horizontal flip
        "orientation_map_path": Path("configs/orientation_map.json"),
        "strict_orientation": False,  # Changed: Don't fail on unmapped tags
        "skip_unmapped": True,  # Changed: DO skip unmapped for safety
        "orientation_safety_mode": "conservative",  # New: safe by default
        "log_orientation_stats": True,
        # Model configuration
        "checkpoint_dir": Path("checkpoints"),  # Add checkpoint directory
        "model_config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "patch_size": 16,
            "gradient_checkpointing": True,
        },
    }
    
    # Seeding & determinism BEFORE spawning workers/loaders
    # Enforce deterministic cuBLAS workspace requirement (CUDA >= 10.2)
    # NOTE: This environment variable must be set *before* any cuBLAS kernels run.
    # If it isn't set here, abort early with a clear message instead of failing deep in backward().
    if torch.cuda.is_available():
        ws = os.getenv("CUBLAS_WORKSPACE_CONFIG")
        if ws not in (":4096:8", ":16:8"):
            # Automatically set the environment variable for deterministic behavior
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            logger.info(
                "Setting CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic cuBLAS operations. "
                "To avoid this message, set the environment variable before running:\n"
                "    export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
                "Or run via: scripts/run_train_deterministic.sh"
            )
        else:
            logger.info(f"Using existing CUBLAS_WORKSPACE_CONFIG={ws}")
    
    # Add configuration option for deterministic mode
    deterministic_mode = config.get("deterministic_mode", True)
    if not deterministic_mode:
        logger.info("Deterministic mode disabled - training may have slight variations between runs")
    

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if deterministic_mode:
            torch.use_deterministic_algorithms(True)  # raise if an op is nondeterministic
        else:
            torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass

    # Set up file logging for primary process

    is_primary = True
    try:
        is_primary = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        is_primary = True

    _listener = None
    if is_primary:
        # Use rotating file handler with compression
        from HDF5_loader import CompressingRotatingFileHandler
        fh = CompressingRotatingFileHandler(
            'training_with_orientation.log',
            maxBytes=128 * 1024 * 1024,  # 128MB
            backupCount=5,
            compress=True
        )
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        _listener = QueueListener(log_queue, fh, respect_handler_level=True)
        _listener.start()
    
    # Validate and setup orientation handling
    try:
        orientation_config = setup_orientation_aware_training(
            data_dir=config["data_dir"],
            json_dir=config["json_dir"],
            vocab_path=config["vocab_path"],
            orientation_map_path=config.get("orientation_map_path"),
            random_flip_prob=config["random_flip_prob"],
            strict_orientation=config["strict_orientation"],
            safety_mode=config.get("orientation_safety_mode", "conservative"),
            skip_unmapped=config.get("skip_unmapped", False)
        )
        
        logger.info("Orientation handling configured successfully")
        # Convert Path objects to strings for JSON serialization
        config_for_logging = orientation_config.copy()
        if config_for_logging.get('orientation_map_path'):
            config_for_logging['orientation_map_path'] = str(config_for_logging['orientation_map_path'])
        logger.info(f"Orientation config: {json.dumps(config_for_logging, indent=2, default=str)}")
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Orientation setup failed: {e}")
        if config["strict_orientation"]:
            raise
        else:
            logger.warning("Continuing without proper orientation handling (not recommended)")
            config["random_flip_prob"] = 0  # Disable flips
    
    
    # Show orientation statistics warning only once
    orientation_stats_warning_shown = False
    if config["num_workers"] > 0 and config["random_flip_prob"] > 0 and not orientation_stats_warning_shown:
        logger.info("\n" + "="*60)
        logger.info("IMPORTANT: Orientation Statistics Limitation")
        logger.info("="*60)
        logger.info(f"With num_workers={config['num_workers']}, orientation statistics will be incomplete.")
        logger.info("Each worker tracks stats independently and they are not aggregated.")
        logger.info("For accurate orientation statistics, use num_workers=0.")
        logger.info("This does not affect training quality, only statistics reporting.")
        logger.info("="*60 + "\n")
        orientation_stats_warning_shown = True
    
    device = torch.device(config["device"])
    
    # Create dataloaders with orientation-aware configuration
    train_loader, val_loader, vocab = create_dataloaders(
        data_dir=config["data_dir"],
        json_dir=config["json_dir"],
        vocab_path=config["vocab_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        frequency_sampling=True,
        val_batch_size=None,
        config_updates={
            "random_flip_prob": config["random_flip_prob"],
            # Convert string back to Path if needed
            "orientation_map_path": Path(config.get("orientation_map_path")) if config.get("orientation_map_path") and isinstance(config.get("orientation_map_path"), str) else config.get("orientation_map_path"),
            "skip_unmapped": config.get("skip_unmapped", True),
            "strict_orientation_validation": config.get("strict_orientation", True),
            "num_workers": config["num_workers"]  # Pass to dataset for LMDB initialization
        },
        seed=seed,
        log_queue=log_queue,
        force_val_persistent_workers=False,
    )
    
    # Model setup
    num_tags = len(vocab.tag_to_index)
    num_ratings = len(vocab.rating_to_index)
    logger.info(f"Creating model with {num_tags} tags and {num_ratings} ratings")
    
    model = create_model(
        num_tags=num_tags,
        num_ratings=num_ratings,
        **config["model_config"]
    )
    model.to(device)
    
    # Loss and optimizer
    criterion = MultiTaskLoss(
        tag_loss_weight=0.9,
        rating_loss_weight=0.1,
        tag_loss_fn=AsymmetricFocalLoss(
            gamma_pos=1.0,
            gamma_neg=3.0,
            alpha=0.75,
            label_smoothing=config["label_smoothing"]
        )
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # AMP setup (prefer BF16 on modern NVIDIA GPUs like Blackwell when available)
    is_cuda = config["device"].startswith("cuda") and torch.cuda.is_available()
    device_type = "cuda" if is_cuda else "cpu"
    # DTYPE: BF16 on CUDA if supported; FP32 on CPU (safer); otherwise FP16
    if is_cuda and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device_type == "cpu":
        # Use FP32 on CPU for stability in production
        amp_dtype = torch.float32
        config["amp"] = False  # Disable AMP on CPU
        logger.info("Disabling AMP on CPU for stability")
    else:
        amp_dtype = torch.float16

    logger.info(f"Using dtype: {amp_dtype} for autocast")
    # GradScaler is only needed for FP16 on CUDA
    scaler = GradScaler(device='cuda') if (config["amp"] and is_cuda and amp_dtype == torch.float16) else None

    # Initialize checkpoint manager and training state
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config["checkpoint_dir"],
        max_checkpoints=5,
        keep_best=True,
        save_frequency=1
    )
    
    training_state = TrainingState()
    best_val_loss = float('inf')

    # Enable anomaly detection for debugging NaN issues (disable in production)
    if config.get("debug_mode", False):
        torch.autograd.set_detect_anomaly(True)

    # Helper function to safely get orientation stats
    def get_dataset_orientation_stats():
        """Get orientation stats from the dataset if available."""
        try:
            if hasattr(train_loader.dataset, 'get_orientation_stats'):
                return train_loader.dataset.get_orientation_stats()
        except Exception as e:
            logger.debug(f"Could not get orientation stats: {e}")
        
        return {
            'total_flips': 0,
            'skipped_flips': 0,
            'flip_rate': 0.0,
            'has_handler': False,
            'worker_local': True
        }
        
    # Training loop with orientation statistics
    global_step = 0
    orientation_stats_interval = 100  # Log orientation stats every N batches
    
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            tag_labels = batch['tag_labels'].to(device)
            rating_labels = batch['rating_labels'].to(device)
            
            # Validate inputs for NaN/Inf
            if torch.isnan(images).any() or torch.isinf(images).any():
                logger.error(f"NaN/Inf detected in input images at step {step}")
                continue
            if torch.isnan(tag_labels).any() or torch.isinf(tag_labels).any():
                logger.error(f"NaN/Inf detected in tag labels at step {step}")
                continue
            
            # Forward pass with mixed precision
            if config["amp"]:
                with autocast(device_type=device_type, enabled=True, dtype=amp_dtype):
                    pmask = batch.get('padding_mask', None)
                    if pmask is not None:
                        pmask = pmask.to(device)
                    outputs = model(images, padding_mask=pmask)
                    loss, losses = criterion(
                        outputs['tag_logits'],
                        outputs['rating_logits'],
                        tag_labels,
                        rating_labels,
                        sample_weights=None
                    )
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                pmask = batch.get('padding_mask', None)
                if pmask is not None:
                    pmask = pmask.to(device)
                outputs = model(images, padding_mask=pmask)
                loss, losses = criterion(
                    outputs['tag_logits'],
                    outputs['rating_logits'],
                    tag_labels,
                    rating_labels,
                    sample_weights=None
                )
                loss.backward()
            
            running_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % config["gradient_accumulation"] == 0:
                # Add gradient clipping before optimizer step
                if config.get("max_grad_norm", 0) > 0:
                    if config["amp"] and scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                
                if config["amp"] and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
        
        # Epoch summary
        avg_train_loss = running_loss / max(1, len(train_loader))
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}: Avg train loss = {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                tag_labels = batch['tag_labels'].to(device)
                rating_labels = batch['rating_labels'].to(device)
                
                pmask = batch.get('padding_mask', None)
                if pmask is not None:
                    pmask = pmask.to(device)
                outputs = model(images, padding_mask=pmask)
                loss, _ = criterion(
                    outputs['tag_logits'],
                    outputs['rating_logits'],
                    tag_labels,
                    rating_labels,
                    sample_weights=None
                )
                val_loss += loss.item()
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        
        # Check for NaN and handle gracefully
        if torch.isnan(torch.tensor(avg_val_loss)) or torch.isinf(torch.tensor(avg_val_loss)):
            logger.error(f"NaN or Inf detected in validation loss at epoch {epoch + 1}!")
            logger.error("Skipping checkpoint save for this epoch. Consider debugging with anomaly detection.")
            avg_val_loss = float('inf')  # Prevent best model update
        else:
            logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}: Avg val loss = {avg_val_loss:.4f}")

        # Update training state
        training_state.epoch = epoch + 1
        training_state.global_step = global_step
        training_state.train_loss = avg_train_loss
        training_state.val_loss = avg_val_loss

        # Check if this is the best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            training_state.best_metric = best_val_loss
            training_state.best_epoch = epoch + 1

        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=epoch + 1,
            step=global_step,
            metrics={"train_loss": avg_train_loss, "val_loss": avg_val_loss},
            training_state=training_state,
            is_best=is_best,
            config=config
        )

        # Log orientation statistics at end of epoch
        if config["random_flip_prob"] > 0:
            stats = get_dataset_orientation_stats()
            if stats['has_handler']:
                logger.info(
                    f"Epoch {epoch + 1} orientation stats (worker-local): "
                    f"Flips: {stats['total_flips']}, "
                    f"Skipped: {stats['skipped_flips']}, "
                    f"Flip rate: {stats['flip_rate']:.1%}"
                )
                if config["num_workers"] > 0:
                    logger.info(
                        "Note: These stats are from the main process only. "
                        "Actual flip counts across all workers will be higher."
                    )
    # Generate and save flip safety report if requested
    if config.get("generate_flip_report", False):
        try:
            if hasattr(train_loader.dataset, 'orientation_handler'):
                handler = train_loader.dataset.orientation_handler
                if handler:
                    report_path = Path("flip_safety_report.json")
                    report = handler.generate_safety_report(report_path)
                    logger.info(f"Flip safety report saved to {report_path}")
                    logger.info(f"Flip rate: {report['summary']['flip_rate']:.1%}")
                    logger.info(f"Block rate: {report['summary']['block_rate']:.1%}")
        except Exception as e:
            logger.warning(f"Could not generate flip safety report: {e}")
    

    # Log final warning about statistics limitation
    if config["num_workers"] > 0 and config["random_flip_prob"] > 0:
        logger.info("\n" + "="*60)
        logger.info("IMPORTANT: Orientation statistics limitation")
        logger.info("Statistics shown above are incomplete due to multi-worker data loading.")
        logger.info("For accurate flip statistics, re-run with num_workers=0")
        logger.info("="*60)
    
    logger.info("\nTraining complete with orientation-aware augmentation!")
    
    # Cleanup: Stop the QueueListener if it was started
    if _listener is not None:
        try:
            _listener.stop()
            logger.info("QueueListener stopped successfully")
        except Exception as e:
            logger.warning(f"Error stopping QueueListener: {e}")

    # Log queue drop statistics
    # Note: mp.Queue doesn't have get_drop_stats method
    # This was specific to BoundedLevelAwareQueue
    """
    try:
        if hasattr(log_queue, 'get_drop_stats'):
            drop_stats = log_queue.get_drop_stats()
            total_drops = sum(drop_stats.values())
            if total_drops > 0:
                logger.warning(f"Logging queue dropped {total_drops} messages: {drop_stats}")
    except Exception as e:
        logger.debug(f"Could not get queue drop stats: {e}")
    """

    logger.info("Training complete!")


def validate_orientation_mappings():
    """Standalone function to validate orientation mappings."""
    
    mapping_file = Path("configs/orientation_map.json")
    
    if not mapping_file.exists():
        print(f"Orientation map not found at {mapping_file}")
        return
    
    print(f"Validating orientation mappings from {mapping_file}")
    print("="*60)
    
    handler = OrientationHandler(
        mapping_file=mapping_file,
        random_flip_prob=0.5,
        strict_mode=False
    )
    
    # Run validation
    issues = handler.validate_mappings()
    
    if not issues:
        print("✓ All mappings are valid!")
    else:
        print("Issues found:")
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"\n{issue_type.replace('_', ' ').title()}:")
                for issue in issue_list:
                    print(f"  - {issue}")
    
    # Test some common tags
    test_tags = [
        "hair_over_left_eye",
        "looking_to_the_right",
        "left_hand_up",
        "asymmetrical_hair",
        "single_thighhigh",
        "text",
        "standing"
    ]

    print("\n" + "="*60)
    print("Testing sample tags:")
    for tag in test_tags:
        swapped = handler.swap_tag(tag)
        if swapped != tag:
            print(f"  {tag:30} -> {swapped}")
        else:
            print(f"  {tag:30} -> (no change)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_orientation_mappings()
    else:
        train_with_orientation_tracking()
