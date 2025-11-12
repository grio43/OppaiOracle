#!/usr/bin/env python3
"""
Automatic training setup - drop this into your training script and it handles everything.
No configuration needed - it detects your dataset, GPU, and optimizes everything automatically.
"""

import logging
from typing import Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_training_automatically(
    train_loader,
    model,
    config,
    force_batch_size: Optional[int] = None,
    force_grad_accum: Optional[int] = None,
) -> Tuple[Any, Any, dict]:
    """Automatically setup optimizer and scheduler with zero configuration.

    This function analyzes your training setup and configures everything optimally:
    - Detects dataset size
    - Detects GPU memory
    - Recommends/adjusts batch size if needed
    - Configures AdamW8bit optimizer with adaptive hyperparameters
    - Selects and configures appropriate learning rate scheduler
    - Sets up warmup

    Just call this function and use the returned optimizer and scheduler!

    Args:
        train_loader: Your training DataLoader
        model: Your PyTorch model
        config: Your training configuration object
        force_batch_size: Override batch size detection (None = auto)
        force_grad_accum: Override gradient accumulation (None = auto)

    Returns:
        Tuple of (optimizer, scheduler, info_dict)
        - optimizer: Configured AdamW8bit optimizer
        - scheduler: Configured learning rate scheduler
        - info_dict: Dictionary with configuration details

    Example:
        >>> optimizer, scheduler, info = setup_training_automatically(
        ...     train_loader=train_loader,
        ...     model=model,
        ...     config=config
        ... )
        >>> # That's it! Now just use optimizer and scheduler in your training loop
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("   🤖 AUTOMATIC TRAINING SETUP")
    logger.info("=" * 70)
    logger.info("")

    # Step 1: Detect dataset size
    try:
        dataset_size = len(train_loader.dataset)
        logger.info(f"✓ Detected dataset size: {dataset_size:,} samples")
    except Exception as e:
        logger.warning(f"Could not detect dataset size: {e}")
        logger.warning("Falling back to batch-based estimation")
        dataset_size = len(train_loader) * getattr(config.data, 'batch_size', 32)
        logger.info(f"✓ Estimated dataset size: {dataset_size:,} samples")

    # Step 2: Extract configuration
    try:
        batch_size = force_batch_size or getattr(config.data, 'batch_size', None)
        grad_accum = force_grad_accum or getattr(config.training, 'gradient_accumulation_steps', None)
        num_epochs = getattr(config.training, 'num_epochs', 50)
        num_gpus = getattr(config.training, 'world_size', 1)

        # Detect image size from config
        image_size = getattr(getattr(config.data, 'image_size', None), 'height', 512)
        if image_size is None:
            image_size = getattr(config.data, 'image_height', 512)

        # Detect model size
        model_params = sum(p.numel() for p in model.parameters())
        if model_params < 50_000_000:  # < 50M params
            model_size = "small"
        elif model_params < 200_000_000:  # < 200M params
            model_size = "medium"
        else:
            model_size = "large"

        logger.info(f"✓ Model size: {model_size} ({model_params/1e6:.1f}M parameters)")

    except Exception as e:
        logger.warning(f"Error extracting config: {e}, using defaults")
        batch_size = None
        grad_accum = None
        num_epochs = 50
        num_gpus = 1
        image_size = 512
        model_size = "medium"

    # Step 3: Determine training goal based on dataset characteristics
    if dataset_size < 30000:
        # Small dataset - use exploration with restarts to find best solution
        training_goal = "exploration"
        logger.info(f"✓ Small dataset detected → using 'exploration' mode (cosine with restarts)")
    elif dataset_size < 100000:
        # Medium dataset - could benefit from exploration if training long enough
        if num_epochs >= 50:
            training_goal = "exploration"
            logger.info(f"✓ Medium dataset with many epochs → using 'exploration' mode")
        else:
            training_goal = "best_convergence"
            logger.info(f"✓ Medium dataset → using 'best_convergence' mode (simple cosine)")
    else:
        # Large dataset - use stable convergence
        training_goal = "best_convergence"
        logger.info(f"✓ Large dataset detected → using 'best_convergence' mode (simple cosine)")

    # Step 4: Auto-configure training
    try:
        from unified_training_config import get_complete_training_config, create_optimizer_and_scheduler

        # Let the system determine batch size if needed
        auto_batch = (batch_size is None or grad_accum is None)

        training_config, optim_kwargs, sched_kwargs = get_complete_training_config(
            dataset_size=dataset_size,
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_gpus=num_gpus,
            auto_batch_size=auto_batch,
            training_goal=training_goal,
            image_size=image_size,
            model_size=model_size,
        )

        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=model,
            config=training_config,
            optimizer_kwargs=optim_kwargs,
            scheduler_kwargs=sched_kwargs
        )

        # Build info dictionary
        info = {
            'dataset_size': dataset_size,
            'batch_size': training_config.batch_size,
            'gradient_accumulation_steps': training_config.gradient_accumulation_steps,
            'effective_batch_size': training_config.effective_batch_size,
            'steps_per_epoch': training_config.steps_per_epoch,
            'total_steps': training_config.total_steps,
            'learning_rate': training_config.learning_rate,
            'weight_decay': training_config.weight_decay,
            'warmup_steps': training_config.warmup_steps,
            'scheduler_type': training_config.scheduler_type.value,
            'training_goal': training_goal,
            'model_size': model_size,
            'model_parameters': model_params,
        }

        # Update config with recommended batch settings if auto-detected
        if auto_batch:
            logger.info("")
            logger.info("⚠️  AUTO-DETECTED BATCH SIZE:")
            logger.info(f"   Please update your config to use:")
            logger.info(f"   batch_size: {training_config.batch_size}")
            logger.info(f"   gradient_accumulation_steps: {training_config.gradient_accumulation_steps}")
            logger.info("")

        logger.info("=" * 70)
        logger.info("   ✓ AUTO-SETUP COMPLETE - Ready to train!")
        logger.info("=" * 70)
        logger.info("")

        return optimizer, scheduler, info

    except ImportError as e:
        logger.error(f"Could not import auto-configuration modules: {e}")
        logger.error("Falling back to manual configuration from config file")
        return _fallback_manual_setup(model, config)


def _fallback_manual_setup(model, config):
    """Fallback to manual configuration if auto-setup fails."""
    logger.warning("Using fallback manual configuration")

    try:
        import bitsandbytes as bnb
        from training_utils import CosineAnnealingWarmupRestarts

        # Get settings from config
        lr = getattr(config.training, 'learning_rate', 1e-4)
        weight_decay = getattr(config.training, 'weight_decay', 0.01)
        beta1 = getattr(config.training, 'adam_beta1', 0.9)
        beta2 = getattr(config.training, 'adam_beta2', 0.999)
        eps = getattr(config.training, 'adam_epsilon', 1e-8)

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )

        # Compute total steps for scheduler
        batch_size = getattr(config.data, 'batch_size', 32)
        grad_accum = getattr(config.training, 'gradient_accumulation_steps', 1)
        num_epochs = getattr(config.training, 'num_epochs', 50)

        # Estimate steps (won't be exact without dataset size)
        steps_per_epoch = 500  # Rough estimate
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * 0.05)

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_steps,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=lr * 0.01,
            warmup_steps=warmup_steps,
        )

        info = {
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'fallback': True,
        }

        logger.info("✓ Fallback configuration complete")

        return optimizer, scheduler, info

    except Exception as e:
        logger.error(f"Fallback configuration failed: {e}")
        raise RuntimeError("Could not configure optimizer and scheduler") from e


# Convenience function for even simpler usage
def auto_setup(train_loader, model, config):
    """Ultra-simple wrapper - just call this and you're done.

    Example:
        >>> optimizer, scheduler, _ = auto_setup(train_loader, model, config)
    """
    return setup_training_automatically(train_loader, model, config)


def print_recommendation_summary(info: dict):
    """Print a summary of the auto-configuration decisions.

    Call this after setup to see what was configured.

    Args:
        info: The info dict returned from setup_training_automatically
    """
    print("\n" + "=" * 70)
    print("   📊 TRAINING CONFIGURATION SUMMARY")
    print("=" * 70)

    print(f"\nDataset:")
    print(f"  Size: {info.get('dataset_size', 'unknown'):,} samples")
    print(f"  Batch size: {info.get('batch_size', 'unknown')}")
    print(f"  Gradient accumulation: {info.get('gradient_accumulation_steps', 'unknown')}")
    print(f"  Effective batch: {info.get('effective_batch_size', 'unknown')}")
    print(f"  Steps per epoch: {info.get('steps_per_epoch', 'unknown')}")

    print(f"\nOptimizer (AdamW8bit):")
    print(f"  Learning rate: {info.get('learning_rate', 'unknown'):.6f}")
    print(f"  Weight decay: {info.get('weight_decay', 'unknown'):.6f}")

    print(f"\nScheduler:")
    print(f"  Type: {info.get('scheduler_type', 'unknown')}")
    print(f"  Warmup steps: {info.get('warmup_steps', 'unknown'):,}")
    print(f"  Training goal: {info.get('training_goal', 'unknown')}")

    if 'model_size' in info:
        params_m = info.get('model_parameters', 0) / 1e6
        print(f"\nModel:")
        print(f"  Size: {info['model_size']} ({params_m:.1f}M params)")

    print("\n" + "=" * 70 + "\n")
