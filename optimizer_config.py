#!/usr/bin/env python3
"""
DEPRECATED: This module has been merged into training_config.py

This file is kept for backward compatibility only.
Please update your imports to use training_config instead:

    # Old:
    from optimizer_config import get_adamw8bit_config, AdamW8bitConfig

    # New:
    from training_config import get_adamw8bit_config, AdamW8bitConfig
"""

import warnings

warnings.warn(
    "The 'optimizer_config.py' file is deprecated and will be removed in a future version. "
    "Please import from 'training_config.py' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from training_config for backward compatibility
from training_config import (
    AdamW8bitConfig,
    compute_effective_batch_size,
    scale_learning_rate,
    compute_warmup_steps,
    scale_weight_decay,
    adjust_beta2_for_long_training,
    get_adamw8bit_config,
    detect_gpu_memory,
    get_recommended_batch_size,
)

__all__ = [
    'AdamW8bitConfig',
    'compute_effective_batch_size',
    'scale_learning_rate',
    'compute_warmup_steps',
    'scale_weight_decay',
    'adjust_beta2_for_long_training',
    'get_adamw8bit_config',
    'detect_gpu_memory',
    'get_recommended_batch_size',
]
