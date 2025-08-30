"""Utilities for securely loading model checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch


class InvalidCheckpointError(RuntimeError):
    """Raised when a checkpoint file contains unexpected objects."""
    pass


def safe_load_checkpoint(path: Union[str, Path]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load a checkpoint without executing arbitrary code.

    This uses ``torch.load`` with ``weights_only=True`` when available. It
    ensures the result is a dictionary of tensors and splits out metadata
    entries.

    Args:
        path: Path to the checkpoint file.

    Returns:
        A tuple of ``(state_dict, metadata)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise InvalidCheckpointError("Checkpoint does not contain a state_dict dictionary")

    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
        meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        # Backward-compat: accept older key
        state_dict = checkpoint["model_state_dict"]
        meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        # Fallback: assume entire mapping is a state_dict
        state_dict = checkpoint
        meta = {}

    if not all(torch.is_tensor(v) for v in state_dict.values()):
        raise InvalidCheckpointError("state_dict contains non-tensor values")

    return state_dict, meta
