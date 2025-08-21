#!/usr/bin/env python3
"""
Enhanced training script with comprehensive orientation handling for anime image tagger.
Demonstrates integration of the orientation handler with fail-fast behavior and statistics tracking.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import multiprocessing as mp
import random
import torch.distributed as dist

import torch
from torch.cuda.amp import GradScaler, autocast
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
    
    # Enhanced configuration with orientation handling
    config = {
        "learning_rate": 4e-4,
        "batch_size": 32,
        "gradient_accumulation": 2,
        "num_epochs": 8,
        "warmup_steps": 10_000,
        "weight_decay": 0.01,
        "label_smoothing": 0.05,
        "data_dir": Path("data/images"),
        "json_dir": Path("data/annotations"),
        "vocab_path": Path("vocabulary.json"),
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "amp": True,
        # Orientation-specific settings
        "random_flip_prob": 0.2,  # 20% chance of horizontal flip
        "orientation_map_path": Path("configs/orientation_map.json"),
        "strict_orientation": True,  # Fail if orientation map missing
        "skip_unmapped": False,  # Don't skip images with unmapped tags
        "log_orientation_stats": True,
        # Model configuration
        "model_config": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "patch_size": 16,
            "gradient_checkpointing": True,
        },
    }
    
    # Seeding & determinism BEFORE spawning workers/loaders
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)  # raise if an op is nondeterministic
    except Exception:
        pass
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass

    # Queue-based, rank-safe logging
    from logging.handlers import QueueListener
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_queue = mp.Queue()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(sh)

    is_primary = True
    try:
        is_primary = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        is_primary = True

    if is_primary:
        fh = logging.FileHandler('training_with_orientation.log')
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
    
    # Initialize orientation handler for statistics tracking
    orientation_handler = None
    if config["random_flip_prob"] > 0 and config.get("orientation_map_path"):
        orientation_handler = OrientationHandler(
            mapping_file=config["orientation_map_path"],
            random_flip_prob=config["random_flip_prob"],
            strict_mode=config["strict_orientation"],
            skip_unmapped=config.get("skip_unmapped", False)
        )
    
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
            "strict_orientation_validation": config.get("strict_orientation", True)
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
    
    scaler = GradScaler() if config["amp"] else None
    
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
            
            # Forward pass with mixed precision
            if config["amp"]:
                with autocast():
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
                scaler.scale(loss).backward()
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
                if config["amp"]:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Log orientation statistics periodically
            if orientation_handler and step % orientation_stats_interval == 0:
                stats = orientation_handler.get_statistics()
                logger.info(
                    f"Orientation stats - Flips: {stats['total_flips']}, "
                    f"Skipped: {stats['skipped_flips']} ({stats['skip_rate']:.1%}), "
                    f"Mapped tags: {stats['num_mapped_tags']}, "
                    f"Unmapped: {stats['num_unmapped_tags']}"
                )
                
                if stats['unmapped_tags_sample']:
                    logger.warning(
                        f"Sample of unmapped orientation tags: {stats['unmapped_tags_sample'][:5]}"
                    )
            
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
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}: Avg val loss = {avg_val_loss:.4f}")
    
    # Final orientation statistics
    if orientation_handler:
        final_stats = orientation_handler.get_statistics()
        logger.info("\n" + "="*50)
        logger.info("FINAL ORIENTATION STATISTICS")
        logger.info("="*50)
        logger.info(f"Total horizontal flips performed: {final_stats['total_flips']}")
        logger.info(f"Flips skipped: {final_stats['skipped_flips']}")
        logger.info(f"Skip rate: {final_stats['skip_rate']:.2%}")
        logger.info(f"Unique tags mapped: {final_stats['num_mapped_tags']}")
        logger.info(f"Unique unmapped orientation tags found: {final_stats['num_unmapped_tags']}")
        
        if final_stats['unmapped_tags_sample']:
            logger.warning("\nUnmapped orientation tags (should be added to orientation_map.json):")
            for tag in final_stats['unmapped_tags_sample']:
                logger.warning(f"  - {tag}")
        
        # Save unmapped tags for future reference
        if final_stats['num_unmapped_tags'] > 0:
            unmapped_file = Path("unmapped_orientation_tags.json")
            with open(unmapped_file, 'w') as f:
                json.dump({
                    "unmapped_tags": list(orientation_handler.stats['unmapped_tags']),
                    "total_count": final_stats['num_unmapped_tags'],
                    "training_run": {
                        "epochs": config["num_epochs"],
                        "flip_prob": config["random_flip_prob"],
                        "total_flips": final_stats['total_flips']
                    }
                }, f, indent=2)
            logger.info(f"\nSaved unmapped tags to {unmapped_file}")
    
    logger.info("\nTraining complete with orientation-aware augmentation!")


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