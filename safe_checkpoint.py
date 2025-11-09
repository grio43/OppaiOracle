"""Utilities for securely loading model checkpoints."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class InvalidCheckpointError(RuntimeError):
    """Raised when a checkpoint file contains unexpected objects."""
    pass


def safe_load_checkpoint(
    path: Union[str, Path],
    validate_values: bool = True,
    allow_nan: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load a checkpoint without executing arbitrary code.

    This uses ``torch.load`` with ``weights_only=True`` when available. It
    ensures the result is a dictionary of tensors and splits out metadata
    entries.

    Args:
        path: Path to the checkpoint file.
        validate_values: If True, check for NaN/Inf in tensors.
        allow_nan: If True, allow NaN values (some models use them intentionally).

    Returns:
        A tuple of ``(state_dict, metadata)``.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        InvalidCheckpointError: If checkpoint is invalid or corrupted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch < 1.13 doesn't support weights_only
        warnings.warn(
            f"PyTorch version {torch.__version__} does not support weights_only=True. "
            f"Loading checkpoint with reduced security. "
            f"Upgrade to PyTorch >= 1.13 for safe checkpoint loading. "
            f"DO NOT load checkpoints from untrusted sources.",
            UserWarning,
            stacklevel=2
        )
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

    # Validate state_dict is not empty
    if len(state_dict) == 0:
        raise InvalidCheckpointError(
            "state_dict is empty - checkpoint may be corrupted"
        )

    # Validate all values are tensors
    if not all(torch.is_tensor(v) for v in state_dict.values()):
        raise InvalidCheckpointError("state_dict contains non-tensor values")

    # Optional: Check for corruption (NaN/Inf)
    if validate_values and not allow_nan:
        corrupt_keys = []
        for k, v in state_dict.items():
            if v.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                if torch.isnan(v).any():
                    corrupt_keys.append((k, "contains NaN"))
                elif torch.isinf(v).any():
                    corrupt_keys.append((k, "contains Inf"))

        if corrupt_keys:
            keys_str = ', '.join(f"{k} ({reason})" for k, reason in corrupt_keys[:5])
            if len(corrupt_keys) > 5:
                keys_str += f", ... and {len(corrupt_keys) - 5} more"
            raise InvalidCheckpointError(
                f"Checkpoint contains NaN/Inf in {len(corrupt_keys)} tensor(s): "
                f"{keys_str}. File may be corrupted."
            )

    return state_dict, meta
