#!/usr/bin/env python3
"""
Unified training configuration that combines optimizer and scheduler setup.
This is the ONE function you need for complete AdamW8bit + scheduler configuration.
"""

import warnings

warnings.warn(
    "The 'unified_training_config.py' file is deprecated and will be removed in a future version. "
    "Please use the unified configuration system in 'Configuration_System.py'.",
    DeprecationWarning,
    stacklevel=2
)

import logging
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from training_config import (
    get_adamw8bit_config,
    get_recommended_batch_size,
    AdamW8bitConfig,
    get_scheduler_config,
    create_scheduler_from_config,
    recommend_scheduler,
    SchedulerType,
    SchedulerConfig
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Complete training configuration including optimizer and scheduler."""
    # Optimizer settings
    learning_rate: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float

    # Scheduler settings
    scheduler_type: SchedulerType
    warmup_steps: int
    total_steps: int
    min_lr: float

    # Batch settings
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    steps_per_epoch: int

    # Scheduler-specific
    scheduler_kwargs: dict


def get_complete_training_config(
    dataset_size: int,
    num_epochs: int,
    batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    num_gpus: int = 1,
    num_nodes: int = 1,
    # Optimizer options
    optimizer_config: Optional[AdamW8bitConfig] = None,
    # Scheduler options
    scheduler_type: Optional[SchedulerType] = None,
    training_goal: str = "best_convergence",
    # Batch size auto-detection
    auto_batch_size: bool = False,
    image_size: int = 512,
    model_size: str = "medium",
    # Advanced
    warmup_steps_override: Optional[int] = None,
) -> Tuple[TrainingConfig, dict, dict]:
    """Get complete training configuration for optimizer + scheduler.

    This is the main function to use. It handles:
    - Automatic batch size selection (optional)
    - Optimizer configuration (learning rate, weight decay, betas)
    - Scheduler selection and configuration
    - Warmup computation

    Args:
        dataset_size: Number of training samples
        num_epochs: Number of training epochs
        batch_size: Per-device batch size (None = auto-detect)
        gradient_accumulation_steps: Gradient accumulation (None = auto-compute)
        num_gpus: Number of GPUs
        num_nodes: Number of nodes for distributed training
        optimizer_config: Custom optimizer config (None = use defaults)
        scheduler_type: Force specific scheduler (None = auto-recommend)
        training_goal: Training objective ("best_convergence", "fast_training", "exploration", "fine_tuning")
        auto_batch_size: Automatically determine batch size based on GPU memory
        image_size: Image size for memory estimation
        model_size: Model size for memory estimation
        warmup_steps_override: Manual warmup steps (None = auto-compute)

    Returns:
        Tuple of (TrainingConfig, optimizer_kwargs, scheduler_kwargs)

    Example:
        >>> config, optim_kwargs, sched_kwargs = get_complete_training_config(
        ...     dataset_size=50000,
        ...     num_epochs=50,
        ...     auto_batch_size=True  # Let it figure out batch size
        ... )
        >>> # Then create optimizer and scheduler
        >>> optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate, **optim_kwargs)
        >>> scheduler = create_scheduler_from_config(optimizer, config.learning_rate, ...)
    """

    # Step 1: Determine batch size if needed
    if auto_batch_size or batch_size is None or gradient_accumulation_steps is None:
        logger.info("Auto-detecting optimal batch size...")
        recommended_batch, recommended_accum = get_recommended_batch_size(
            dataset_size=dataset_size,
            available_memory_gb=None,  # Auto-detect
            target_steps_per_epoch=500,
            image_size=image_size,
            model_size=model_size
        )

        if batch_size is None:
            batch_size = recommended_batch
            logger.info(f"Using auto-detected batch_size: {batch_size}")

        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = recommended_accum
            logger.info(f"Using auto-detected gradient_accumulation_steps: {gradient_accumulation_steps}")
    else:
        # Use provided values
        logger.info(f"Using provided batch_size={batch_size}, gradient_accumulation={gradient_accumulation_steps}")

    # Step 2: Get optimizer configuration
    lr, optim_kwargs, auto_warmup = get_adamw8bit_config(
        dataset_size=dataset_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        base_config=optimizer_config
    )

    # Step 3: Get scheduler configuration
    warmup_steps = warmup_steps_override if warmup_steps_override is not None else auto_warmup

    sched_config, sched_kwargs = get_scheduler_config(
        dataset_size=dataset_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_gpus=num_gpus,
        scheduler_type=scheduler_type,
        training_goal=training_goal,
        warmup_steps=warmup_steps
    )

    # Step 4: Build unified configuration
    effective_batch = batch_size * gradient_accumulation_steps * num_gpus * num_nodes
    steps_per_epoch = (dataset_size + effective_batch - 1) // effective_batch  # Ceiling division
    total_steps = steps_per_epoch * num_epochs

    # Compute min_lr
    if optimizer_config:
        min_lr_ratio = 0.01  # Default
    else:
        min_lr_ratio = 0.01

    min_lr = lr * min_lr_ratio

    config = TrainingConfig(
        learning_rate=lr,
        weight_decay=optim_kwargs['weight_decay'],
        betas=optim_kwargs['betas'],
        eps=optim_kwargs['eps'],
        scheduler_type=sched_config.scheduler_type,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=effective_batch,
        steps_per_epoch=steps_per_epoch,
        scheduler_kwargs=sched_kwargs
    )

    # Log unified summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("   COMPLETE TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Dataset & Batch Settings:")
    logger.info(f"  Dataset size:                {dataset_size:,} samples")
    logger.info(f"  Batch size (per device):     {batch_size}")
    logger.info(f"  Gradient accumulation:       {gradient_accumulation_steps}")
    logger.info(f"  Number of GPUs:              {num_gpus}")
    logger.info(f"  Effective batch size:        {effective_batch}")
    logger.info(f"  Steps per epoch:             {steps_per_epoch}")
    logger.info(f"  Number of epochs:            {num_epochs}")
    logger.info(f"  Total training steps:        {total_steps:,}")
    logger.info("")
    logger.info("Optimizer Settings (AdamW8bit):")
    logger.info(f"  Learning rate:               {lr:.6f}")
    logger.info(f"  Weight decay:                {optim_kwargs['weight_decay']:.6f}")
    logger.info(f"  Beta1:                       {optim_kwargs['betas'][0]}")
    logger.info(f"  Beta2:                       {optim_kwargs['betas'][1]:.6f}")
    logger.info(f"  Epsilon:                     {optim_kwargs['eps']}")
    logger.info("")
    logger.info("Scheduler Settings:")
    logger.info(f"  Scheduler type:              {sched_config.scheduler_type.value}")
    logger.info(f"  Warmup steps:                {warmup_steps:,} ({warmup_steps/total_steps:.1%})")
    logger.info(f"  Min learning rate:           {min_lr:.6f}")

    if sched_config.scheduler_type == SchedulerType.COSINE_RESTARTS:
        logger.info(f"  Number of cycles:            {sched_kwargs['num_cycles']}")
        logger.info(f"  First cycle length:          {sched_kwargs['first_cycle_steps']:,} steps")
        logger.info(f"  Cycle multiplier:            {sched_kwargs['cycle_mult']}")

    logger.info("")
    logger.info("Training Goal:                 " + training_goal)
    logger.info("=" * 70)
    logger.info("")

    return config, optim_kwargs, sched_kwargs


def create_optimizer_and_scheduler(
    model,
    config: TrainingConfig,
    optimizer_kwargs: dict,
    scheduler_kwargs: dict
):
    """Create optimizer and scheduler from unified configuration.

    Args:
        model: PyTorch model
        config: TrainingConfig from get_complete_training_config
        optimizer_kwargs: Optimizer kwargs from get_complete_training_config
        scheduler_kwargs: Scheduler kwargs from get_complete_training_config

    Returns:
        Tuple of (optimizer, scheduler)

    Example:
        >>> config, optim_kwargs, sched_kwargs = get_complete_training_config(...)
        >>> optimizer, scheduler = create_optimizer_and_scheduler(
        ...     model, config, optim_kwargs, sched_kwargs
        ... )
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for AdamW8bit optimizer. "
            "Install with: pip install bitsandbytes"
        )

    # Create optimizer
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=config.learning_rate,
        **optimizer_kwargs
    )

    # Create scheduler
    sched_config = SchedulerConfig(
        scheduler_type=config.scheduler_type,
        warmup_steps=config.warmup_steps,
        min_lr_ratio=config.min_lr / config.learning_rate
    )

    scheduler = create_scheduler_from_config(
        optimizer=optimizer,
        max_lr=config.learning_rate,
        config=sched_config,
        scheduler_kwargs=scheduler_kwargs
    )

    logger.info("✓ Optimizer and scheduler created successfully")

    return optimizer, scheduler


# Convenience function for the absolute easiest setup
def auto_configure_training(
    dataset_size: int,
    num_epochs: int = 50,
    training_goal: str = "best_convergence",
    image_size: int = 512,
    model_size: str = "medium"
) -> Tuple[TrainingConfig, dict, dict]:
    """The easiest way to get a complete training configuration.

    Just provide dataset size and let it handle everything else.

    Args:
        dataset_size: Number of training samples
        num_epochs: Number of training epochs (default: 50)
        training_goal: "best_convergence", "fast_training", "exploration", or "fine_tuning"
        image_size: Input image size (default: 512)
        model_size: "small", "medium", or "large" (default: "medium")

    Returns:
        Tuple of (TrainingConfig, optimizer_kwargs, scheduler_kwargs)

    Example:
        >>> # Literally just this:
        >>> config, optim_kwargs, sched_kwargs = auto_configure_training(dataset_size=50000)
        >>> # Done! Now create your optimizer and scheduler
    """
    logger.info("🚀 Auto-configuring training setup...")
    logger.info(f"   Dataset: {dataset_size:,} samples, {num_epochs} epochs")
    logger.info(f"   Goal: {training_goal}")

    return get_complete_training_config(
        dataset_size=dataset_size,
        num_epochs=num_epochs,
        auto_batch_size=True,  # Auto-detect everything
        training_goal=training_goal,
        image_size=image_size,
        model_size=model_size
    )
