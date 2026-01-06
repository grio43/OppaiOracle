"""
DEPRECATED: This module is deprecated and will be removed in a future version.
The functionality has been moved to the CheckpointManager class in training_utils.py.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

warnings.warn(
    "The 'safe_checkpoint' module is deprecated and will be removed. "
    "Use 'training_utils.CheckpointManager' instead.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)


class InvalidCheckpointError(RuntimeError):
    """Raised when a checkpoint file contains unexpected objects."""
    pass


def safe_load_checkpoint(
    path: Union[str, Path],
    validate_values: bool = True,
    allow_nan: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    DEPRECATED: Load a checkpoint without executing arbitrary code.
    """
    from training_utils import CheckpointManager
    
    checkpoint_dir = Path(path).parent
    manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
    checkpoint = manager.load_checkpoint(checkpoint_path=str(path))
    if not checkpoint:
        raise FileNotFoundError(f"Could not load checkpoint from {path}")

    state_dict = checkpoint.pop('state_dict')
    meta = checkpoint
    return state_dict, meta