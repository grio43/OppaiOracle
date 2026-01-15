#!/usr/bin/env python3
"""
Unified training hyperparameter configuration.

Combines optimizer and scheduler configuration into a single module for easier
maintenance and AI-assisted development. Provides dataset-aware scaling and
automatic hyperparameter tuning.

This module includes:
- AdamW8bit optimizer configuration with dataset-aware scaling
- Learning rate scheduler configuration with training-goal aware recommendations
- Unified integration functions for complete training setup
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: OPTIMIZER CONFIGURATION
# =============================================================================

@dataclass
class AdamW8bitConfig:
    """Configuration for AdamW8bit optimizer with dataset-aware scaling.

    This configuration automatically adjusts hyperparameters based on:
    - Dataset size (number of training samples)
    - Effective batch size (batch_size * grad_accum * num_gpus)
    - Number of training epochs

    Key principles:
    1. Learning rate scales with sqrt(effective_batch_size) following linear scaling rule
    2. Warmup steps scale with dataset size to ensure stable initialization
    3. Weight decay adjusted based on model size and dataset size
    4. Beta2 adjusted for longer training runs
    """

    # Base hyperparameters (for reference batch size of 256)
    base_lr: float = 1e-4
    base_batch_size: int = 256

    # AdamW-specific parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    # Learning rate scaling
    lr_scaling_mode: str = "sqrt"  # "linear", "sqrt", or "none"

    # Warmup configuration
    warmup_ratio: float = 0.05  # 5% of total steps for warmup
    min_warmup_steps: int = 500
    max_warmup_steps: int = 10000

    # Weight decay scaling
    wd_scaling_mode: str = "inverse_sqrt"  # "fixed", "linear", "inverse_sqrt"
    min_weight_decay: float = 0.001
    max_weight_decay: float = 0.1


def compute_effective_batch_size(
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    num_gpus: int = 1,
    num_nodes: int = 1
) -> int:
    """Compute the effective batch size for optimization.

    Args:
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_gpus: Number of GPUs per node
        num_nodes: Number of nodes (for distributed training)

    Returns:
        Effective batch size (total samples per optimizer step)
    """
    return batch_size * gradient_accumulation_steps * num_gpus * num_nodes


def scale_learning_rate(
    base_lr: float,
    effective_batch_size: int,
    base_batch_size: int = 256,
    mode: str = "sqrt"
) -> float:
    """Scale learning rate based on effective batch size.

    Args:
        base_lr: Base learning rate (tuned for base_batch_size)
        effective_batch_size: Actual effective batch size
        base_batch_size: Reference batch size for base_lr
        mode: Scaling mode - "linear", "sqrt", or "none"

    Returns:
        Scaled learning rate

    References:
        - Linear scaling: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
        - Sqrt scaling: "BERT" (Devlin et al., 2018)
    """
    if mode == "none":
        return base_lr
    elif mode == "linear":
        # Linear scaling: lr ∝ batch_size
        return base_lr * (effective_batch_size / base_batch_size)
    elif mode == "sqrt":
        # Square root scaling: lr ∝ sqrt(batch_size)
        # More conservative, better for larger batches
        return base_lr * math.sqrt(effective_batch_size / base_batch_size)
    else:
        raise ValueError(f"Unknown scaling mode: {mode}")


def compute_warmup_steps(
    dataset_size: int,
    effective_batch_size: int,
    num_epochs: int,
    warmup_ratio: float = 0.05,
    min_steps: int = 500,
    max_steps: int = 10000
) -> int:
    """Compute number of warmup steps based on dataset characteristics.

    Args:
        dataset_size: Number of training samples
        effective_batch_size: Effective batch size
        num_epochs: Total number of training epochs
        warmup_ratio: Fraction of total training steps for warmup
        min_steps: Minimum warmup steps
        max_steps: Maximum warmup steps

    Returns:
        Number of warmup steps
    """
    steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
    total_steps = steps_per_epoch * num_epochs

    # Compute warmup as a ratio of total steps
    warmup_steps = int(total_steps * warmup_ratio)

    # Clamp to reasonable range
    warmup_steps = max(min_steps, min(warmup_steps, max_steps))

    ratio_str = f"({warmup_steps/total_steps:.1%})" if total_steps > 0 else "(N/A)"
    logger.info(
        f"Computed warmup schedule: {warmup_steps} steps "
        f"{ratio_str} of {total_steps} total steps"
    )

    return warmup_steps


def scale_weight_decay(
    base_wd: float,
    dataset_size: int,
    mode: str = "inverse_sqrt",
    min_wd: float = 0.001,
    max_wd: float = 0.1
) -> float:
    """Scale weight decay based on dataset size.

    Larger datasets benefit from less regularization (lower weight decay).
    Smaller datasets need more regularization to prevent overfitting.

    Args:
        base_wd: Base weight decay value
        dataset_size: Number of training samples
        mode: Scaling mode - "fixed", "linear", "inverse_sqrt"
        min_wd: Minimum weight decay
        max_wd: Maximum weight decay

    Returns:
        Scaled weight decay
    """
    if mode == "fixed":
        return base_wd

    # Reference dataset size (100k samples)
    ref_size = 100_000

    if mode == "inverse_sqrt":
        # weight_decay ∝ 1/sqrt(dataset_size)
        # More regularization for smaller datasets
        scale_factor = math.sqrt(ref_size / max(dataset_size, 1000))
        scaled_wd = base_wd * scale_factor
    elif mode == "linear":
        # weight_decay ∝ 1/dataset_size
        scale_factor = ref_size / max(dataset_size, 1000)
        scaled_wd = base_wd * scale_factor
    else:
        raise ValueError(f"Unknown weight decay scaling mode: {mode}")

    # Clamp to reasonable range
    scaled_wd = max(min_wd, min(scaled_wd, max_wd))

    return scaled_wd


def adjust_beta2_for_long_training(
    base_beta2: float,
    num_epochs: int,
    dataset_size: int,
    effective_batch_size: int
) -> float:
    """Adjust beta2 (second moment decay) for long training runs.

    For very long training runs, a slightly higher beta2 can help stabilize
    the optimizer by giving more weight to historical gradients.

    Args:
        base_beta2: Base beta2 value (typically 0.999)
        num_epochs: Number of training epochs
        dataset_size: Number of training samples
        effective_batch_size: Effective batch size

    Returns:
        Adjusted beta2 value
    """
    if effective_batch_size <= 0:
        return base_beta2
    total_steps = math.ceil(dataset_size / effective_batch_size) * num_epochs

    # For training runs > 100k steps, increase beta2 slightly
    if total_steps > 100_000:
        # Smoothly increase beta2 from 0.999 to 0.9999 for very long runs
        adjustment = min(0.0009, 0.0009 * (total_steps - 100_000) / 400_000)
        return min(base_beta2 + adjustment, 0.9999)

    return base_beta2


def get_adamw8bit_config(
    dataset_size: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    num_gpus: int = 1,
    num_nodes: int = 1,
    base_config: Optional[AdamW8bitConfig] = None
) -> Tuple[float, dict, int]:
    """Get complete AdamW8bit configuration scaled for dataset size.

    This is the main function you should use. It computes all optimizer
    hyperparameters based on your dataset characteristics.

    Args:
        dataset_size: Number of training samples
        batch_size: Per-device batch size
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Gradient accumulation steps
        num_gpus: Number of GPUs
        num_nodes: Number of nodes (for multi-node training)
        base_config: Base configuration to override defaults

    Returns:
        Tuple of (learning_rate, optimizer_kwargs_dict, warmup_steps)

    Example:
        >>> lr, optim_kwargs, warmup_steps = get_adamw8bit_config(
        ...     dataset_size=50000,
        ...     batch_size=32,
        ...     num_epochs=50,
        ...     gradient_accumulation_steps=2
        ... )
        >>> optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, **optim_kwargs)
    """
    # Validate inputs
    if dataset_size <= 0:
        raise ValueError(
            f"dataset_size must be positive, got {dataset_size}. "
            "Check that your dataset was loaded correctly."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")

    if base_config is None:
        base_config = AdamW8bitConfig()

    # 1. Compute effective batch size
    effective_batch = compute_effective_batch_size(
        batch_size, gradient_accumulation_steps, num_gpus, num_nodes
    )

    # 2. Scale learning rate
    lr = scale_learning_rate(
        base_config.base_lr,
        effective_batch,
        base_config.base_batch_size,
        base_config.lr_scaling_mode
    )

    # 3. Scale weight decay
    weight_decay = scale_weight_decay(
        base_config.weight_decay,
        dataset_size,
        base_config.wd_scaling_mode,
        base_config.min_weight_decay,
        base_config.max_weight_decay
    )

    # 4. Adjust beta2 for long training
    beta2 = adjust_beta2_for_long_training(
        base_config.beta2,
        num_epochs,
        dataset_size,
        effective_batch
    )

    # 5. Compute warmup steps
    warmup_steps = compute_warmup_steps(
        dataset_size,
        effective_batch,
        num_epochs,
        base_config.warmup_ratio,
        base_config.min_warmup_steps,
        base_config.max_warmup_steps
    )

    # Log the configuration
    steps_per_epoch = math.ceil(dataset_size / effective_batch)
    total_steps = steps_per_epoch * num_epochs

    logger.info("=" * 60)
    logger.info("AdamW8bit Configuration (Dataset-Adaptive)")
    logger.info("=" * 60)
    logger.info(f"Dataset size:              {dataset_size:,}")
    logger.info(f"Batch size (per-device):   {batch_size}")
    logger.info(f"Gradient accumulation:     {gradient_accumulation_steps}")
    logger.info(f"Effective batch size:      {effective_batch}")
    logger.info(f"Number of epochs:          {num_epochs}")
    logger.info(f"Steps per epoch:           {steps_per_epoch}")
    logger.info(f"Total training steps:      {total_steps:,}")
    logger.info("-" * 60)
    logger.info(f"Learning rate:             {lr:.6f}")
    logger.info(f"Weight decay:              {weight_decay:.6f}")
    logger.info(f"Beta1:                     {base_config.beta1}")
    logger.info(f"Beta2:                     {beta2:.6f}")
    logger.info(f"Epsilon:                   {base_config.eps}")
    ratio_str2 = f"({warmup_steps/total_steps:.1%})" if total_steps > 0 else "(N/A)"
    logger.info(f"Warmup steps:              {warmup_steps:,} {ratio_str2}")
    logger.info("=" * 60)

    optimizer_kwargs = {
        'betas': (base_config.beta1, beta2),
        'eps': base_config.eps,
        'weight_decay': weight_decay,
    }

    return lr, optimizer_kwargs, warmup_steps


def detect_gpu_memory() -> float:
    """Detect available GPU memory in GB.

    Returns:
        Available GPU memory in GB (or 24.0 as default if detection fails)
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory of first GPU
            total_memory = torch.cuda.get_device_properties(0).total_memory
            # Convert to GB and leave some headroom (use 90%)
            available_gb = (total_memory / (1024**3)) * 0.9
            logger.info(f"Detected GPU memory: {available_gb:.1f} GB available")
            return available_gb
    except Exception as e:
        logger.debug(f"Could not detect GPU memory: {e}")

    logger.warning("GPU memory detection failed, using default 24GB - verify this matches your hardware")
    return 24.0


def get_recommended_batch_size(
    dataset_size: int,
    available_memory_gb: Optional[float] = None,
    target_steps_per_epoch: int = 500,
    image_size: int = 512,
    model_size: str = "medium"
) -> Tuple[int, int]:
    """Recommend batch size and gradient accumulation based on constraints.

    Args:
        dataset_size: Number of training samples
        available_memory_gb: Available GPU memory in GB (None = auto-detect)
        target_steps_per_epoch: Desired number of steps per epoch
        image_size: Input image size (affects memory usage)
        model_size: Model size - "small", "medium", "large"

    Returns:
        Tuple of (batch_size, gradient_accumulation_steps)
    """
    # Auto-detect GPU memory if not provided
    if available_memory_gb is None:
        available_memory_gb = detect_gpu_memory()

    # Memory-based batch size recommendations for different image sizes
    # Format: (image_size, model_size) -> {memory_gb: batch_size}
    memory_recommendations = {
        (512, "small"): {8: 12, 12: 20, 16: 28, 24: 40, 32: 56, 40: 72, 48: 96},
        (512, "medium"): {8: 8, 12: 16, 16: 24, 24: 32, 32: 48, 40: 64, 48: 80},
        (512, "large"): {8: 4, 12: 8, 16: 12, 24: 20, 32: 32, 40: 48, 48: 64},
        (768, "medium"): {8: 4, 12: 8, 16: 12, 24: 20, 32: 32, 40: 40, 48: 56},
        (1024, "medium"): {8: 2, 12: 4, 16: 8, 24: 12, 32: 20, 40: 28, 48: 36},
    }

    # Find closest match for image size and model size
    key = (image_size, model_size)
    if key not in memory_recommendations:
        # Fallback to 512/medium
        logger.warning(f"No recommendation for image_size={image_size}, model={model_size}, using defaults")
        key = (512, "medium")

    memory_to_batch = memory_recommendations[key]

    # Find the largest batch size that fits in memory
    max_batch_size = 8  # Conservative default
    for mem_threshold, batch in sorted(memory_to_batch.items()):
        if available_memory_gb >= mem_threshold:
            max_batch_size = batch

    # Compute gradient accumulation to hit target steps per epoch
    effective_batch = max(1, dataset_size // target_steps_per_epoch)
    grad_accum = max(1, effective_batch // max_batch_size)

    # Adjust batch size if accumulation would be too large
    if grad_accum > 8:
        # Prefer larger batch size over excessive accumulation
        max_batch_size = min(128, effective_batch // 8)
        grad_accum = max(1, effective_batch // max_batch_size)

    # Ensure batch size is reasonable
    max_batch_size = max(4, min(max_batch_size, 128))
    grad_accum = max(1, min(grad_accum, 16))

    actual_effective = max_batch_size * grad_accum
    actual_steps = dataset_size // actual_effective

    logger.info("=" * 60)
    logger.info("Batch Size Recommendation")
    logger.info("=" * 60)
    logger.info(f"Dataset size:             {dataset_size:,}")
    logger.info(f"Available GPU memory:     {available_memory_gb:.1f} GB")
    logger.info(f"Image size:               {image_size}x{image_size}")
    logger.info(f"Model size:               {model_size}")
    logger.info("-" * 60)
    logger.info(f"Recommended batch_size:   {max_batch_size}")
    logger.info(f"Gradient accumulation:    {grad_accum}")
    logger.info(f"Effective batch size:     {actual_effective}")
    logger.info(f"Steps per epoch:          {actual_steps}")
    logger.info(f"Target steps per epoch:   {target_steps_per_epoch}")
    logger.info("=" * 60)

    return max_batch_size, grad_accum


# =============================================================================
# SECTION 2: SCHEDULER CONFIGURATION
# =============================================================================

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

    # Cosine-specific
    min_lr_ratio: float = 0.01  # min_lr = max_lr * min_lr_ratio

    # Cosine with restarts specific
    num_cycles: int = 3          # Number of cosine cycles
    cycle_mult: float = 2.0      # Multiplier for cycle length (1.0 = same length)

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
    ratio_str = f"({warmup_steps/total_steps:.1%})" if total_steps > 0 else "(N/A)"
    logger.info(f"Warmup steps:             {warmup_steps:,} {ratio_str}")

    if scheduler_type == SchedulerType.COSINE_RESTARTS:
        logger.info(f"Number of cycles:         {scheduler_kwargs['num_cycles']}")
        logger.info(f"First cycle length:       {scheduler_kwargs['first_cycle_steps']:,}")
        logger.info(f"Cycle multiplier:         {config.cycle_mult}")

    logger.info("=" * 60)

    return config, scheduler_kwargs


def create_scheduler_from_config(
    optimizer: "torch.optim.Optimizer",
    max_lr: float,
    config: SchedulerConfig,
    scheduler_kwargs: dict
) -> "torch.optim.lr_scheduler.LRScheduler":
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

    elif scheduler_type == SchedulerType.POLYNOMIAL:
        from torch.optim.lr_scheduler import PolynomialLR, SequentialLR, ConstantLR

        warmup_sched = ConstantLR(optimizer, factor=0.1, total_iters=warmup_steps)
        main_sched = PolynomialLR(
            optimizer,
            total_iters=total_steps - warmup_steps,
            power=2.0,  # Quadratic decay
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup_steps]
        )

    elif scheduler_type == SchedulerType.CONSTANT_WARMUP:
        from torch.optim.lr_scheduler import ConstantLR, SequentialLR

        # Warmup phase: ramp from 10% to 100% of max_lr
        warmup_sched = ConstantLR(optimizer, factor=0.1, total_iters=warmup_steps)
        # Main phase: constant at max_lr
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps)
        return SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup_steps]
        )

    else:
        # Default to simple cosine (should not reach here with valid SchedulerType)
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
  - Standard training runs (10k-100k steps)
  - Large datasets (>100k samples)
  - When you want stable, predictable convergence
  - Fine-tuning pretrained models
  - Production training (less experimental)

Pros: Stable, well-tested, smooth decay
Cons: May not find best solution for complex tasks

COSINE_RESTARTS (Cosine Annealing with Warm Restarts / SGDR)
-------------------------------------------------------------
When to use:
  - Long training runs (>50k steps)
  - Smaller datasets (<100k samples) with many epochs
  - When you want to explore solution space
  - Training from scratch (not fine-tuning)
  - When you can save checkpoints at cycle ends

Pros: Can escape local minima, ensemble-like behavior
Cons: More hyperparameters, requires longer training

Recommended cycles:
  - Short runs (<20k steps): 2 cycles
  - Medium runs (20k-100k): 3 cycles
  - Long runs (>100k): 4-5 cycles

ONE_CYCLE (One-Cycle Policy)
----------------------------
When to use:
  - Fast convergence needed
  - Medium-length training (10k-50k steps)
  - When you have good learning rate estimates
  - Classification tasks

Pros: Fast convergence, good for time-limited training
Cons: Sensitive to hyperparameters, less exploration

LINEAR (Linear Decay)
--------------------
When to use:
  - Short fine-tuning runs (<10k steps)
  - Simple, interpretable schedule needed
  - Conservative training

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
