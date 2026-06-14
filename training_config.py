#!/usr/bin/env python3
"""
Training hyperparameter helpers.

Only the learning-rate batch-size scaling consumed by train_direct.py lives
here. The former dataset-aware "auto-config" system (AdamW8bit config builder,
weight-decay scaling, warmup computation, beta2 adjustment, batch-size and
scheduler recommendation) was removed: none of it was ever wired into training
(train_direct reads config.training.* directly), and weight decay is
intentionally FIXED at config.training.weight_decay — the old inverse-sqrt
dataset-size scaling is research-refuted for this project and must not return.
"""

import math


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
