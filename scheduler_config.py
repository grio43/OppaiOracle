#!/usr/bin/env python3
"""
Learning rate scheduler configuration for AdamW8bit optimizer.
Provides guidance on when to use different schedulers and integrates with optimizer_config.
"""

import warnings

warnings.warn(
    "The 'scheduler_config.py' file is deprecated and will be removed in a future version. "
    "Please use the unified configuration system in 'Configuration_System.py'.",
    DeprecationWarning,
    stacklevel=2
)

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    """Available learning rate scheduler types"""
    COSINE = "cosine"                          # Simple cosine decay
    COSINE_RESTARTS = "cosine_restarts"       # Cosine with warm restarts (SGDR)
    LINEAR = "linear"                          # Linear decay
    POLYNOMIAL = "polynomial"                  # Polynomial decay
    CONSTANT_WARMUP = "constant_warmup"       # Constant LR after warmup
    ONE_CYCLE = "one_cycle"                   # One-cycle policy


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers.

    This config helps you choose and configure the best scheduler for your training scenario.
    """

    # Scheduler type
    scheduler_type: SchedulerType = SchedulerType.COSINE

    # Warmup configuration
    warmup_steps: int = 1000
    warmup_strategy: str = "linear"  # "linear" or "constant"

    # Cosine-specific
    min_lr_ratio: float = 0.01  # min_lr = max_lr * min_lr_ratio

    # Cosine with restarts specific
    num_cycles: int = 3          # Number of cosine cycles
    cycle_mult: float = 2.0      # Multiplier for cycle length (1.0 = same length)
    restart_decay: float = 1.0   # Decay factor for max_lr after each restart

    # One-cycle specific
    pct_start: float = 0.3       # Percentage of training in warmup phase
    div_factor: float = 25.0     # Initial LR = max_lr / div_factor
    final_div_factor: float = 1e4  # Final LR = max_lr / final_div_factor


def recommend_scheduler(
    dataset_size: int,
    num_epochs: int,
    effective_batch_size: int,
    training_goal: str = "best_convergence"
) -> Tuple[SchedulerType, str]:
    """Recommend a scheduler based on training characteristics.

    Args:
        dataset_size: Number of training samples
        num_epochs: Number of training epochs
        effective_batch_size: Effective batch size (batch_size * grad_accum * num_gpus)
        training_goal: One of:
            - "best_convergence": Optimize for best final performance
            - "fast_training": Optimize for training speed
            - "exploration": More aggressive learning for finding good solutions
            - "fine_tuning": Conservative for fine-tuning pretrained models

    Returns:
        Tuple of (recommended_scheduler, explanation)
    """
    steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
    total_steps = steps_per_epoch * num_epochs

    # Short training runs (< 10k steps)
    if total_steps < 10000:
        if training_goal == "fine_tuning":
            return SchedulerType.LINEAR, (
                "Linear decay is simple and effective for short fine-tuning runs. "
                "Provides stable gradual LR reduction."
            )
        else:
            return SchedulerType.COSINE, (
                "Cosine decay works well for short training runs. "
                "Smooth decay without the overhead of restarts."
            )

    # Medium training runs (10k - 100k steps)
    elif total_steps < 100000:
        if training_goal == "exploration":
            return SchedulerType.COSINE_RESTARTS, (
                "Cosine with restarts helps exploration in medium-length training. "
                "Periodic LR spikes can help escape local minima. "
                "Use 2-3 cycles for this length."
            )
        elif training_goal == "fast_training":
            return SchedulerType.ONE_CYCLE, (
                "One-cycle policy can converge faster in medium-length training. "
                "Combines warmup, peak, and decay in one smooth cycle."
            )
        else:
            return SchedulerType.COSINE, (
                "Standard cosine decay is reliable for medium-length training. "
                "Well-tested and stable convergence."
            )

    # Long training runs (> 100k steps)
    else:
        if training_goal == "exploration":
            return SchedulerType.COSINE_RESTARTS, (
                "Cosine with restarts beneficial for very long training. "
                "Multiple cycles (3-5) help find better solutions. "
                "Save checkpoints at end of each cycle for ensemble."
            )
        elif training_goal == "fine_tuning":
            return SchedulerType.COSINE, (
                "Simple cosine decay for conservative long fine-tuning. "
                "Stable and predictable convergence."
            )
        else:
            if dataset_size > 100000:
                return SchedulerType.COSINE, (
                    "Cosine decay recommended for large datasets with long training. "
                    "More stable than restarts for large-scale training."
                )
            else:
                return SchedulerType.COSINE_RESTARTS, (
                    "Cosine with restarts can improve results on smaller datasets "
                    "with long training. Use 3-4 cycles."
                )


def compute_cycle_steps(
    total_steps: int,
    num_cycles: int,
    cycle_mult: float = 1.0
) -> List[int]:
    """Compute the length of each cycle for cosine annealing with restarts.

    Args:
        total_steps: Total training steps
        num_cycles: Number of cosine cycles
        cycle_mult: Multiplier for cycle length (1.0 = equal length cycles)

    Returns:
        List of cycle lengths

    Examples:
        >>> compute_cycle_steps(10000, 3, cycle_mult=1.0)
        [3333, 3333, 3334]  # Equal cycles

        >>> compute_cycle_steps(10000, 3, cycle_mult=2.0)
        [1429, 2857, 5714]  # Exponentially growing cycles
    """
    if cycle_mult == 1.0:
        # Equal-length cycles
        base_length = total_steps // num_cycles
        cycles = [base_length] * num_cycles
        # Add remainder to last cycle
        cycles[-1] += total_steps - sum(cycles)
        return cycles

    # Exponentially growing cycles
    # first_cycle * (1 + cycle_mult + cycle_mult^2 + ...) = total_steps
    # first_cycle * (cycle_mult^num_cycles - 1) / (cycle_mult - 1) = total_steps
    sum_mult = (cycle_mult ** num_cycles - 1) / (cycle_mult - 1)
    first_cycle = int(total_steps / sum_mult)

    cycles = []
    for i in range(num_cycles):
        cycle_len = int(first_cycle * (cycle_mult ** i))
        cycles.append(cycle_len)

    # Adjust last cycle to match total exactly
    cycles[-1] = total_steps - sum(cycles[:-1])

    return cycles


def get_scheduler_config(
    dataset_size: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    num_gpus: int = 1,
    scheduler_type: Optional[SchedulerType] = None,
    training_goal: str = "best_convergence",
    warmup_steps: Optional[int] = None,
) -> Tuple[SchedulerConfig, dict]:
    """Get complete scheduler configuration.

    Args:
        dataset_size: Number of training samples
        batch_size: Per-device batch size
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Gradient accumulation steps
        num_gpus: Number of GPUs
        scheduler_type: Force specific scheduler (None = auto-recommend)
        training_goal: Training objective (see recommend_scheduler)
        warmup_steps: Override warmup steps (None = auto-compute)

    Returns:
        Tuple of (SchedulerConfig, scheduler_kwargs dict)
    """
    effective_batch = batch_size * gradient_accumulation_steps * num_gpus
    steps_per_epoch = math.ceil(dataset_size / effective_batch)
    total_steps = steps_per_epoch * num_epochs

    # Auto-recommend if not specified
    if scheduler_type is None:
        scheduler_type, explanation = recommend_scheduler(
            dataset_size, num_epochs, effective_batch, training_goal
        )
        logger.info(f"Recommended scheduler: {scheduler_type.value}")
        logger.info(f"Reason: {explanation}")

    # Compute warmup if not provided
    if warmup_steps is None:
        # 5% of total steps, clamped to reasonable range
        warmup_steps = int(total_steps * 0.05)
        warmup_steps = max(500, min(warmup_steps, 10000))

    config = SchedulerConfig(
        scheduler_type=scheduler_type,
        warmup_steps=warmup_steps,
    )

    # Build scheduler-specific kwargs
    scheduler_kwargs = {
        'warmup_steps': warmup_steps,
        'total_steps': total_steps,
    }

    if scheduler_type == SchedulerType.COSINE:
        scheduler_kwargs['min_lr_ratio'] = config.min_lr_ratio

    elif scheduler_type == SchedulerType.COSINE_RESTARTS:
        # Compute optimal number of cycles for this training length
        if total_steps < 20000:
            num_cycles = 2
        elif total_steps < 100000:
            num_cycles = 3
        else:
            num_cycles = 4

        # Compute cycle lengths
        cycles = compute_cycle_steps(total_steps, num_cycles, config.cycle_mult)
        first_cycle = cycles[0]

        scheduler_kwargs.update({
            'first_cycle_steps': first_cycle,
            'cycle_mult': config.cycle_mult,
            'num_cycles': num_cycles,
            'cycles': cycles,
            'min_lr_ratio': config.min_lr_ratio,
        })

        logger.info(f"Cosine restarts: {num_cycles} cycles with lengths {cycles}")

    elif scheduler_type == SchedulerType.ONE_CYCLE:
        scheduler_kwargs.update({
            'pct_start': config.pct_start,
            'div_factor': config.div_factor,
            'final_div_factor': config.final_div_factor,
        })

    # Log configuration
    logger.info("=" * 60)
    logger.info("Learning Rate Scheduler Configuration")
    logger.info("=" * 60)
    logger.info(f"Scheduler type:           {scheduler_type.value}")
    logger.info(f"Total training steps:     {total_steps:,}")
    logger.info(f"Steps per epoch:          {steps_per_epoch:,}")
    logger.info(f"Warmup steps:             {warmup_steps:,} ({warmup_steps/total_steps:.1%})")

    if scheduler_type == SchedulerType.COSINE_RESTARTS:
        logger.info(f"Number of cycles:         {scheduler_kwargs['num_cycles']}")
        logger.info(f"First cycle length:       {scheduler_kwargs['first_cycle_steps']:,}")
        logger.info(f"Cycle multiplier:         {config.cycle_mult}")

    logger.info("=" * 60)

    return config, scheduler_kwargs


def create_scheduler_from_config(
    optimizer,
    max_lr: float,
    config: SchedulerConfig,
    scheduler_kwargs: dict
):
    """Create a PyTorch learning rate scheduler from configuration.

    This function works with your existing CosineAnnealingWarmupRestarts
    or can create other scheduler types.

    Args:
        optimizer: PyTorch optimizer
        max_lr: Maximum learning rate
        config: SchedulerConfig object
        scheduler_kwargs: Dictionary from get_scheduler_config

    Returns:
        PyTorch learning rate scheduler
    """
    from training_utils import CosineAnnealingWarmupRestarts

    scheduler_type = config.scheduler_type
    total_steps = scheduler_kwargs['total_steps']
    warmup_steps = scheduler_kwargs['warmup_steps']
    min_lr = max_lr * config.min_lr_ratio

    if scheduler_type == SchedulerType.COSINE:
        # Simple cosine decay (using CosineAnnealingWarmupRestarts with cycle_mult=1.0)
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_steps,
            cycle_mult=1.0,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
        )

    elif scheduler_type == SchedulerType.COSINE_RESTARTS:
        # Cosine with actual restarts
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=scheduler_kwargs['first_cycle_steps'],
            cycle_mult=scheduler_kwargs['cycle_mult'],
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
        )

    elif scheduler_type == SchedulerType.LINEAR:
        # Linear decay using built-in scheduler
        from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

        warmup_sched = ConstantLR(optimizer, factor=0.1, total_iters=warmup_steps)
        main_sched = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.min_lr_ratio,
            total_iters=total_steps - warmup_steps
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup_steps]
        )

    elif scheduler_type == SchedulerType.ONE_CYCLE:
        from torch.optim.lr_scheduler import OneCycleLR

        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=config.pct_start,
            div_factor=scheduler_kwargs['div_factor'],
            final_div_factor=scheduler_kwargs['final_div_factor'],
        )

    else:
        # Default to simple cosine
        logger.warning(f"Unknown scheduler type {scheduler_type}, using cosine")
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_steps,
            cycle_mult=1.0,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
        )


def get_scheduler_recommendation_summary() -> str:
    """Get a summary of when to use each scheduler type."""
    return """
Learning Rate Scheduler Recommendations
========================================

COSINE (Simple Cosine Decay)
----------------------------
When to use:
  ✓ Standard training runs (10k-100k steps)
  ✓ Large datasets (>100k samples)
  ✓ When you want stable, predictable convergence
  ✓ Fine-tuning pretrained models
  ✓ Production training (less experimental)

Pros: Stable, well-tested, smooth decay
Cons: May not find best solution for complex tasks

COSINE_RESTARTS (Cosine Annealing with Warm Restarts / SGDR)
-------------------------------------------------------------
When to use:
  ✓ Long training runs (>50k steps)
  ✓ Smaller datasets (<100k samples) with many epochs
  ✓ When you want to explore solution space
  ✓ Training from scratch (not fine-tuning)
  ✓ When you can save checkpoints at cycle ends

Pros: Can escape local minima, ensemble-like behavior
Cons: More hyperparameters, requires longer training

Recommended cycles:
  - Short runs (<20k steps): 2 cycles
  - Medium runs (20k-100k): 3 cycles
  - Long runs (>100k): 4-5 cycles

ONE_CYCLE (One-Cycle Policy)
----------------------------
When to use:
  ✓ Fast convergence needed
  ✓ Medium-length training (10k-50k steps)
  ✓ When you have good learning rate estimates
  ✓ Classification tasks

Pros: Fast convergence, good for time-limited training
Cons: Sensitive to hyperparameters, less exploration

LINEAR (Linear Decay)
--------------------
When to use:
  ✓ Short fine-tuning runs (<10k steps)
  ✓ Simple, interpretable schedule needed
  ✓ Conservative training

Pros: Simple, predictable
Cons: Less effective than cosine for long runs

For Your Anime Tagger Project:
==============================
Recommended: COSINE or COSINE_RESTARTS

Use COSINE if:
  - Dataset > 100k images
  - Training for convergence (not exploration)
  - Want stable, predictable training

Use COSINE_RESTARTS if:
  - Dataset < 100k images
  - Training for many epochs (>50)
  - Want to find best possible solution
  - Can train longer
  - Save checkpoints at cycle ends for ensemble
"""
