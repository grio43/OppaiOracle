#!/usr/bin/env python3
"""
DEPRECATED: This module has been merged into training_config.py

This file is kept for backward compatibility only.
Please update your imports to use training_config instead:

    # Old:
    from scheduler_config import get_scheduler_config, SchedulerType

    # New:
    from training_config import get_scheduler_config, SchedulerType
"""

import warnings

warnings.warn(
    "The 'scheduler_config.py' file is deprecated and will be removed in a future version. "
    "Please import from 'training_config.py' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from training_config for backward compatibility
from training_config import (
    SchedulerType,
    SchedulerConfig,
    recommend_scheduler,
    compute_cycle_steps,
    get_scheduler_config,
    create_scheduler_from_config,
    get_scheduler_recommendation_summary,
)

__all__ = [
    'SchedulerType',
    'SchedulerConfig',
    'recommend_scheduler',
    'compute_cycle_steps',
    'get_scheduler_config',
    'create_scheduler_from_config',
    'get_scheduler_recommendation_summary',
]
