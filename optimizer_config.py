#!/usr/bin/env python3
"""
Adaptive AdamW8bit optimizer configuration that scales based on dataset size.
Provides best-practice hyperparameters tuned for anime image tagging with automatic scaling.
"""

import warnings

warnings.warn(
    "The 'optimizer_config.py' file is deprecated and will be removed in a future version. "
    "Please use the unified configuration system in 'Configuration_System.py'.",
    DeprecationWarning,
    stacklevel=2
)

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


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

    logger.info(
        f"Computed warmup schedule: {warmup_steps} steps "
        f"({warmup_steps/total_steps:.1%} of {total_steps} total steps)"
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
    total_steps = (dataset_size / effective_batch_size) * num_epochs

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
    logger.info(f"Warmup steps:              {warmup_steps:,} ({warmup_steps/total_steps:.1%})")
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

    logger.info("GPU memory detection failed, using default 24GB")
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
        max_batch_size = min(64, effective_batch // 8)
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
