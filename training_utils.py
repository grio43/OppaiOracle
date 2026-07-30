#!/usr/bin/env python3
"""
Training Utilities for Anime Image Tagger
Comprehensive training helpers including schedulers, checkpointing, and distributed training
"""

import os
import json
import logging
import math
import shutil
import threading
import tempfile
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, fields, asdict
from contextlib import nullcontext
import hashlib
from datetime import datetime
from collections import defaultdict, deque
import warnings
import random
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

try:
    import filelock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False
#
# NOTE:
# We dropped `pl_bolts` because it is incompatible with PyTorch Lightning >= 2.0.
# Use our vendored scheduler instead (behavior matches the one from pl_bolts).
from schedulers import LinearWarmupCosineLR as LinearWarmupCosineAnnealingLR
from torch.amp import GradScaler
import torch.backends.cudnn as cudnn

# Import ModelMetadata at module level for fail-fast behavior
try:
    from oppai_oracle.model_metadata import ModelMetadata
except ImportError as e:
    # Fallback to relative import if package not installed
    try:
        from model_metadata import ModelMetadata
    except ImportError:
        raise ImportError(
            "Could not import ModelMetadata. Please ensure the package is installed correctly with 'pip install -e .' from the project root."
        ) from e

logger = logging.getLogger(__name__)
LAST_CKPT_NAME = "last.pt"  # always maintained for crash-resume

class InvalidCheckpointError(RuntimeError):
    """Raised when a checkpoint file contains unexpected objects."""
    pass

# -----------------------------------------------------------------------------
def setup_seed(user_seed: Optional[int], deterministic: bool) -> tuple[int, bool]:
    """
    Opt-in seed: if None, derive a fresh seed from os.urandom, log it, and run non-deterministically.
    """
    if user_seed is None:
        user_seed = int.from_bytes(os.urandom(8), "big") % (2**31 - 1)
        logger.info(f"Training seed: {user_seed} (auto-generated)")
        deterministic = False
    else:
        logger.info(f"Training seed: {user_seed} (user-specified)")
    random.seed(user_seed)
    np.random.seed(user_seed % (2**32 - 1))
    torch.manual_seed(user_seed)
    # Allow runtime.yaml to override cudnn behavior
    det = bool(_RUNTIME.get("deterministic", deterministic))
    cudnn.deterministic = det
    cudnn.benchmark = bool(_RUNTIME.get("cudnn_benchmark", not det))
    return user_seed, bool(deterministic)


def log_sample_order_hash(dataloader, epoch: int, N: int = 128, max_batches: int = 8):
    """Log sha1 over first N sample identifiers to verify shuffle changed.

    Tries, in order:
      - batch['meta']['paths'] or ['image_paths'] if present
      - batch['image_id'] list

    Reads at most ``max_batches`` batches to avoid scanning a whole epoch when
    metadata is unavailable.
    """
    try:
        it = iter(dataloader)
        acc: list[str] = []
        batches_seen = 0
        while len(acc) < N and batches_seen < max_batches:
            batch = next(it)
            batches_seen += 1
            meta = batch.get("meta", {}) if isinstance(batch, dict) else {}
            paths = []
            if isinstance(meta, dict):
                paths = meta.get("paths") or meta.get("image_paths") or []
            if not paths:
                ids = None
                if isinstance(batch, dict):
                    ids = batch.get("image_id")
                if ids is not None:
                    if isinstance(ids, (list, tuple)):
                        paths = [str(x) for x in ids]
                    else:
                        paths = [str(ids)]
            if paths:
                acc.extend(map(str, paths))
        if acc:
            h = hashlib.sha1("|".join(acc[:N]).encode()).hexdigest()
            logger.info(f"epoch={epoch} sample_hash={h}")
        else:
            logger.debug("sample_hash skipped: no identifiers found in first %d batches", max_batches)
    except Exception as e:
        logger.debug(f"sample_hash logging skipped: {e}")


def _save_rng_states():
    """Capture Python, NumPy, Torch CPU and CUDA RNG states.

    Returns:
        Tuple of (py_state, np_state, torch_cpu_state, cuda_state).
        cuda_state is None if CUDA unavailable or errors occur.
    """
    py = random.getstate()
    np_state = np.random.get_state()
    torch_cpu = torch.get_rng_state()
    cuda = None

    if not torch.cuda.is_available():
        return py, np_state, torch_cpu, cuda

    # Try to capture CUDA state for all devices
    try:
        cuda = torch.cuda.get_rng_state_all()
        logger.debug(f"Captured CUDA RNG state for {len(cuda)} devices")
    except RuntimeError as e:
        # CUDA runtime error - try single device fallback
        logger.warning(f"Failed to capture all CUDA RNG states: {e}, trying single device")
        try:
            cuda = torch.cuda.get_rng_state()
            logger.debug("Captured CUDA RNG state for current device")
        except RuntimeError as e2:
            logger.error(f"Failed to capture CUDA RNG state: {e2}. RNG reproducibility may be affected.")
            cuda = None
    except Exception as e:
        # Unexpected error - log and continue
        logger.error(f"Unexpected error capturing CUDA RNG state: {type(e).__name__}: {e}")
        cuda = None

    return py, np_state, torch_cpu, cuda


def _restore_rng_states(states):
    """Restore Python, NumPy, Torch CPU and CUDA RNG states.

    Expects a tuple of (py_state, np_state, torch_cpu_state, cuda_state).
    Logs warnings for any restoration failures.

    Returns:
        Dict[str, bool]: Success status for each component
    """
    py, np_state, torch_cpu, cuda = states
    success = {}

    # Restore Python RNG
    try:
        random.setstate(py)
        success['python'] = True
    except Exception as e:
        logger.warning(f"Failed to restore Python RNG state: {type(e).__name__}: {e}")
        success['python'] = False

    # Restore NumPy RNG
    try:
        import numpy as _np
        # Accept both native NumPy state and packed state
        if isinstance(np_state, (tuple, list)) and len(np_state) >= 5 and not hasattr(np_state[1], "dtype"):
            # Packed form; rebuild ndarray then set state
            bitgen = np_state[0]
            state_list = np_state[1]
            pos = int(np_state[2])
            has_gauss = int(np_state[3])
            cached = float(np_state[4])
            try:
                arr = _np.array(state_list, dtype=_np.uint32)
            except Exception:
                arr = _np.array(state_list)
            _np.random.set_state((bitgen, arr, pos, has_gauss, cached))
        else:
            _np.random.set_state(np_state)  # type: ignore[arg-type]
        success['numpy'] = True
    except Exception as e:
        logger.warning(f"Failed to restore NumPy RNG state: {type(e).__name__}: {e}")
        success['numpy'] = False

    # Restore PyTorch CPU RNG
    try:
        torch.set_rng_state(torch_cpu)
        success['torch_cpu'] = True
    except Exception as e:
        logger.warning(f"Failed to restore PyTorch CPU RNG state: {type(e).__name__}: {e}")
        success['torch_cpu'] = False

    # Restore CUDA RNG
    try:
        if cuda is not None and torch.cuda.is_available():
            current_device_count = torch.cuda.device_count()

            if isinstance(cuda, list):
                saved_device_count = len(cuda)

                if saved_device_count == current_device_count:
                    # Exact match - restore all
                    torch.cuda.set_rng_state_all(cuda)
                    logger.debug(f"Restored CUDA RNG state for {saved_device_count} devices")
                    success['cuda'] = True
                elif saved_device_count > current_device_count:
                    # Saved with more GPUs - restore only for available devices
                    logger.warning(
                        f"Checkpoint saved with {saved_device_count} GPUs, "
                        f"but only {current_device_count} available. "
                        f"Restoring RNG state for first {current_device_count} devices only."
                    )
                    for i in range(current_device_count):
                        torch.cuda.set_rng_state(cuda[i], device=i)
                    success['cuda'] = True  # Partial but acceptable
                else:
                    # Saved with fewer GPUs - restore what we have, warn about others
                    logger.warning(
                        f"Checkpoint saved with {saved_device_count} GPUs, "
                        f"but {current_device_count} available. "
                        f"Restoring RNG state for first {saved_device_count} devices only. "
                        f"Devices {saved_device_count}-{current_device_count-1} will have fresh RNG state."
                    )
                    for i in range(saved_device_count):
                        torch.cuda.set_rng_state(cuda[i], device=i)
                    success['cuda'] = True  # Partial but acceptable
            else:
                # Single device state
                torch.cuda.set_rng_state(cuda)
                logger.debug("Restored CUDA RNG state for current device")
                success['cuda'] = True
        else:
            success['cuda'] = None  # Not applicable
    except RuntimeError as e:
        logger.warning(f"Failed to restore CUDA RNG state: {e}. Training may not be reproducible.")
        success['cuda'] = False
    except Exception as e:
        logger.error(f"Unexpected error restoring CUDA RNG state: {type(e).__name__}: {e}")
        success['cuda'] = False

    # Log summary
    failed = [k for k, v in success.items() if v is False]
    if failed:
        logger.warning(f"RNG state restoration incomplete. Failed components: {', '.join(failed)}")
    else:
        logger.debug("RNG state fully restored")

    return success


def _pack_np_state(np_state: tuple) -> tuple:
    """Convert NumPy RNG state to a pickle-safe tuple of builtins.

    NumPy returns (bit_generator: str, state: ndarray, pos: int, has_gauss: int, cached_gaussian: float).
    Replace the ndarray with a plain Python list to avoid dependency on NumPy object pickling semantics
    when loading with safe checkpoints.
    """
    try:
        if isinstance(np_state, tuple) and len(np_state) >= 5:
            bitgen = np_state[0]
            state_arr = np_state[1]
            pos = np_state[2]
            has_gauss = np_state[3]
            cached = np_state[4]
            try:
                state_list = state_arr.tolist() if hasattr(state_arr, "tolist") else list(state_arr)
            except Exception:
                # Best-effort fallback
                state_list = [int(x) for x in state_arr]
            return (bitgen, state_list, int(pos), int(has_gauss), float(cached))
    except Exception:
        pass
    return np_state


def _unpack_np_state(packed_state: tuple) -> tuple:
    """Rebuild NumPy RNG state tuple from packed builtins.

    Restores the second element back to an ndarray of dtype uint32/int64 as required by NumPy.
    """
    try:
        import numpy as _np  # local import
        if isinstance(packed_state, (tuple, list)) and len(packed_state) >= 5:
            bitgen = packed_state[0]
            state_list = packed_state[1]
            pos = packed_state[2]
            has_gauss = packed_state[3]
            cached = packed_state[4]
            try:
                arr = _np.array(state_list, dtype=_np.uint32)
            except Exception:
                arr = _np.array(state_list)
            return (bitgen, arr, int(pos), int(has_gauss), float(cached))
    except Exception:
        pass
    return packed_state


def _get_nested_value(obj, key_path: str, default=None):
    """Get a value from a nested dict/object using dot notation.

    Args:
        obj: Dict or object to search
        key_path: Dot-separated path like 'model.num_labels' or 'data.image_size'
        default: Value to return if key not found

    Returns:
        The value at the path, or default if not found
    """
    parts = key_path.split('.')
    current = obj

    for part in parts:
        if current is None:
            return default

        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part, None)
        else:
            return default

    return current if current is not None else default


_STATE_DICT_WRAPPER_PREFIXES = ('module.', '_orig_mod.')


def strip_state_dict_prefixes(key: str) -> str:
    """Strip DDP / torch.compile wrapper prefixes from one state-dict key.

    Training saves ``model.state_dict()`` from the torch.compile'd module
    (config.training.use_compile defaults to true), so EVERY key in a real
    checkpoint is prefixed with ``_orig_mod.``; DDP adds ``module.``. Both can be
    present and either order is possible, hence the loop.

    Uses removeprefix, not str.replace: an unbounded replace would also mangle
    the substring anywhere later in the key.
    """
    changed = True
    while changed:
        changed = False
        for prefix in _STATE_DICT_WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``state_dict`` with DDP/torch.compile wrapper prefixes removed.

    Every consumer that loads a raw checkpoint into a bare (uncompiled) module
    must call this first, or ``load_state_dict(strict=True)`` fails with a
    confusing "Missing key(s)" listing every parameter in the model.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(
        isinstance(k, str) and k.startswith(_STATE_DICT_WRAPPER_PREFIXES)
        for k in state_dict
    ):
        return state_dict
    # isinstance guard: a state dict mixing non-str keys must not raise here.
    # type(state_dict)(...) preserves OrderedDict and friends.
    return type(state_dict)(
        (strip_state_dict_prefixes(k) if isinstance(k, str) else k, v)
        for k, v in state_dict.items()
    )


def detect_architecture_from_state_dict(state_dict_keys: list[str]) -> Optional[str]:
    """Detect model architecture type from state dict key patterns.

    This is used to infer architecture from old checkpoints that may not have
    the architecture_type field in their config.

    Args:
        state_dict_keys: List of keys from the model state dict.

    Returns:
        'vit' if ViT architecture detected,
        None if architecture cannot be determined.
    """
    # Normalize keys: remove 'module.' prefix (DDP) and '_orig_mod.' prefix (torch.compile)
    normalized_keys = [strip_state_dict_prefixes(key) for key in state_dict_keys]

    # ViT architecture indicators (SimplifiedTagger)
    # ViT models have 'patch_embed', 'blocks', 'cls_token', 'pos_embed' at top level
    vit_indicators = [
        'patch_embed.',
        'blocks.',
        'cls_token',
        'pos_embed',
    ]

    has_vit_keys = any(
        any(key.startswith(indicator) for indicator in vit_indicators)
        for key in normalized_keys
    )

    return 'vit' if has_vit_keys else None


def validate_config_compatibility(
    checkpoint_config: dict,
    current_config,
    strict: bool = False,
    state_dict_keys: Optional[list[str]] = None
) -> tuple[bool, list[str]]:
    """Validate that critical config parameters match between checkpoint and current config.

    Args:
        checkpoint_config: Config dict from checkpoint
        current_config: Current config (dict or object with attributes)
        strict: If True, raise exception on critical mismatches. If False, just warn.
        state_dict_keys: Optional list of keys from checkpoint state dict for architecture detection.
                        Used to detect architecture when config field is missing.

    Returns:
        Tuple of (is_compatible, list of warning messages)

    Raises:
        ValueError: If strict=True and critical parameters don't match
    """
    if not checkpoint_config:
        # Even without config, we can try to detect architecture from state dict
        if state_dict_keys and strict:
            detected_arch = detect_architecture_from_state_dict(state_dict_keys)
            current_arch = _get_nested_value(current_config, 'model.architecture_type')
            if detected_arch and current_arch and detected_arch != current_arch:
                raise ValueError(
                    f"Architecture mismatch detected from checkpoint state dict.\n"
                    f"  - Detected from state dict: {detected_arch}\n"
                    f"  - Current config: {current_arch}\n"
                    f"Cannot resume {detected_arch} checkpoint with {current_arch} architecture.\n"
                    f"To start fresh, set training.resume_from='none' in config."
                )
        return True, ["No config in checkpoint - skipping validation"]

    warning_messages = []
    errors = []

    # Critical parameters that MUST match (will cause training issues if different)
    critical_params = [
        ('model.num_labels', 'Vocabulary size', 'Model head size mismatch will cause crashes'),
        ('data.patch_size', 'Patch size', 'Patch embedding size mismatch'),
        ('model.architecture_type', 'Architecture type', 'Architecture mismatch will cause crashes'),
    ]

    # Important parameters that SHOULD match (may cause subtle issues)
    important_params = [
        ('training.gradient_accumulation_steps', 'Gradient accumulation', 'Effective batch size changed'),
        # image_size is intentionally NOT critical: load_checkpoint bicubically
        # interpolates the positional embeddings across a resolution change
        # (e.g. Phase 1 320 -> Phase 2 448). Treating it as a critical+strict
        # mismatch made resume_from=latest/best catch the ValueError and SILENTLY
        # skip the checkpoint, restarting a multi-day run from scratch. patch_size
        # stays critical, so a non-divisible patch grid is still rejected.
        ('data.image_size', 'Image size', 'pos_embed will be bicubically interpolated to the new resolution'),
    ]

    # Check critical parameters
    for key_path, name, impact in critical_params:
        ckpt_val = _get_nested_value(checkpoint_config, key_path)
        curr_val = _get_nested_value(current_config, key_path)

        # Skip if either value is None (not set)
        if ckpt_val is None or curr_val is None:
            continue

        # Handle list/tuple comparison
        if isinstance(ckpt_val, (list, tuple)):
            ckpt_val = tuple(ckpt_val)
        if isinstance(curr_val, (list, tuple)):
            curr_val = tuple(curr_val)

        if ckpt_val != curr_val:
            msg = f"CRITICAL: {name} mismatch - checkpoint: {ckpt_val}, current: {curr_val}. {impact}"
            errors.append(msg)

    # Check architecture type - detect from state dict if not in config
    checkpoint_arch = _get_nested_value(checkpoint_config, 'model.architecture_type')
    current_arch = _get_nested_value(current_config, 'model.architecture_type')

    # If checkpoint config is missing architecture_type, try to detect from state dict
    if checkpoint_arch is None and state_dict_keys:
        detected_arch = detect_architecture_from_state_dict(state_dict_keys)
        if detected_arch:
            checkpoint_arch = detected_arch
            logger.info(f"Detected architecture '{detected_arch}' from checkpoint state dict keys "
                       f"(config missing architecture_type field)")

    # Validate architecture match - don't allow silent fallthrough
    if checkpoint_arch is not None and current_arch is not None:
        if checkpoint_arch != current_arch:
            msg = (f"CRITICAL: Architecture type mismatch - checkpoint: {checkpoint_arch}, "
                   f"current: {current_arch}. Cannot resume a checkpoint trained with a different architecture")
            errors.append(msg)
    elif checkpoint_arch is None and current_arch is not None and state_dict_keys:
        # Could not detect architecture from state dict - this is suspicious
        # Log a warning but don't block (might be a very old checkpoint format)
        logger.warning(
            f"Could not detect architecture from checkpoint. Current config uses '{current_arch}'. "
            f"If checkpoint was created with different architecture, loading will fail with cryptic errors."
        )

    def _get_norm_params(cfg, arch):
        mean_val = _get_nested_value(cfg, 'data.normalize_mean')
        std_val = _get_nested_value(cfg, 'data.normalize_std')
        return mean_val, std_val

    if checkpoint_arch is not None and current_arch is not None:
        ckpt_mean, ckpt_std = _get_norm_params(checkpoint_config, checkpoint_arch)
        curr_mean, curr_std = _get_norm_params(current_config, current_arch)

        if ckpt_mean is not None and curr_mean is not None:
            if isinstance(ckpt_mean, (list, tuple)):
                ckpt_mean = tuple(ckpt_mean)
            if isinstance(curr_mean, (list, tuple)):
                curr_mean = tuple(curr_mean)
            if ckpt_mean != curr_mean:
                warning_messages.append(
                    f"WARNING: Normalization mean changed - checkpoint: {ckpt_mean}, "
                    f"current: {curr_mean}. Input normalization differs"
                )

        if ckpt_std is not None and curr_std is not None:
            if isinstance(ckpt_std, (list, tuple)):
                ckpt_std = tuple(ckpt_std)
            if isinstance(curr_std, (list, tuple)):
                curr_std = tuple(curr_std)
            if ckpt_std != curr_std:
                warning_messages.append(
                    f"WARNING: Normalization std changed - checkpoint: {ckpt_std}, "
                    f"current: {curr_std}. Input normalization differs"
                )

        # Channel order mismatch is significant: legacy checkpoints predate
        # the field so a missing value is treated as RGB.
        ckpt_color = _get_nested_value(checkpoint_config, 'data.color_order') or 'RGB'
        curr_color = _get_nested_value(current_config, 'data.color_order') or 'RGB'
        if str(ckpt_color).upper() != str(curr_color).upper():
            warning_messages.append(
                f"WARNING: color_order changed - checkpoint: {ckpt_color}, "
                f"current: {curr_color}. Input channel order differs"
            )

    # Check important parameters
    for key_path, name, impact in important_params:
        ckpt_val = _get_nested_value(checkpoint_config, key_path)
        curr_val = _get_nested_value(current_config, key_path)

        if ckpt_val is None or curr_val is None:
            continue

        # Handle list/tuple comparison
        if isinstance(ckpt_val, (list, tuple)):
            ckpt_val = tuple(ckpt_val)
        if isinstance(curr_val, (list, tuple)):
            curr_val = tuple(curr_val)

        if ckpt_val != curr_val:
            msg = f"WARNING: {name} changed - checkpoint: {ckpt_val}, current: {curr_val}. {impact}"
            warning_messages.append(msg)

    # Log all warnings
    for msg in warning_messages:
        logger.warning(msg)

    # Handle errors based on strict mode
    if errors:
        for msg in errors:
            logger.error(msg)

        if strict:
            raise ValueError(
                "Config incompatibility detected on resume. Errors:\n" +
                "\n".join(f"  - {e}" for e in errors) +
                "\n\nTo start fresh, set training.resume_from='none' in config."
            )

    is_compatible = len(errors) == 0
    all_messages = errors + warning_messages

    return is_compatible, all_messages


def log_index_order_hash(dataloader, epoch: int, N: int = 128):
    """Log sha1 over first N indices from the DataLoader's sampler.

    - Avoids any data I/O by iterating the sampler (indices only).
    - Wraps with RNG save/restore to avoid perturbing global RNG state.
    """
    try:
        sampler = getattr(dataloader, 'sampler', None)
        if sampler is None:
            logger.debug("index_hash skipped: no sampler attached to dataloader")
            return
        # Save RNG states so consuming sampler RNG doesn't affect training order
        states = _save_rng_states()
        try:
            it = iter(sampler)
            acc_idx = []
            for _ in range(N):
                try:
                    acc_idx.append(int(next(it)))
                except StopIteration:
                    break
            if acc_idx:
                h = hashlib.sha1("|".join(map(str, acc_idx)).encode()).hexdigest()
                logger.info(f"epoch={epoch} index_hash={h}")
            else:
                logger.debug("index_hash skipped: sampler yielded no indices")
        finally:
            _restore_rng_states(states)
    except Exception as e:
        logger.debug(f"index_hash logging skipped: {e}")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

# Load canonical paths (vocab, logs, outputs) from unified_config.yaml (back-compat aware)
def _load_paths():
    """Read paths from configs/unified_config.yaml with fallbacks.
    Prefers top-level keys (vocab_path, log_dir, default_output_dir).
    Falls back to data.vocab_path / data.vocab_dir / data.output_dir."""
    try:
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "unified_config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    data = (cfg.get("data") or {})
    # vocabulary path: explicit → from data.vocab_path → from data.vocab_dir/vocabulary.json → repo default
    vp = cfg.get("vocab_path") or data.get("vocab_path")
    if not vp:
        vd = data.get("vocab_dir")
        vp = str((PROJECT_ROOT / vd / "vocabulary.json").resolve()) if vd else str((PROJECT_ROOT / "vocabulary.json").resolve())
    # logs & outputs
    ld = cfg.get("log_dir") or data.get("log_dir") or os.getenv("OPPAI_LOG_DIR", str((PROJECT_ROOT / "logs").resolve()))
    od = cfg.get("default_output_dir") or data.get("output_dir") or str((PROJECT_ROOT / "outputs").resolve())
    return {"vocab_path": vp, "log_dir": ld, "default_output_dir": od}

_paths_cfg = _load_paths()
VOCAB_PATH = Path(_paths_cfg["vocab_path"])
LOG_DIR = Path(_paths_cfg["log_dir"])
DEFAULT_OUTPUT_DIR = Path(_paths_cfg["default_output_dir"])

# Optional runtime toggles (determinism, CuBLAS/CuDNN, quiet logging)
# Default to empty dict for safety
_RUNTIME = {}

def _apply_runtime_config():
    global _RUNTIME
    try:
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "runtime.yaml").read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to load runtime.yaml, using defaults: {e}")
        cfg = {}
    rcfg = cfg.get("runtime", {}) or {}
    if rcfg.get("cublas_workspace_config"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", str(rcfg["cublas_workspace_config"]))
    if rcfg.get("quiet_mode"):
        logging.getLogger().setLevel(logging.WARNING)
    _RUNTIME = rcfg  # Update global
    return rcfg

# Initialize runtime config at module load
_apply_runtime_config()


@dataclass
class TrainingState:
    """Maintains complete training state"""
    epoch: int = 0
    global_step: int = 0
    optimizer_updates: int = 0
    best_metric: float = float('-inf')
    best_epoch: int = 0
    # Name of the scalar `best_metric` was measured with (config.training.
    # selection_metric). best_metric is meaningless without it: macro-F1 and mAP
    # differ by ~50x in magnitude on this task, so restoring a best_metric
    # recorded under one metric while selecting on another either freezes
    # is_best forever or makes the first epoch a guaranteed false "best".
    # Empty string = a checkpoint predating this field.
    selection_metric: str = ""

    # Epoch tracking for proper resume semantics
    completed_epochs: int = 0
    is_epoch_boundary: bool = True
    batch_in_epoch: int = 0
    sample_in_epoch: int = 0  # Batch-size agnostic sample index for resume

    # Loss tracking
    train_loss: float = 0.0
    val_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)

    # Validation metric tracking (real fields so asdict()/checkpoints preserve
    # them; the trainer's validation-skip branch reads them back on resume)
    val_f1_macro: float = 0.0
    val_mAP: float = 0.0
    
    # Metric tracking
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    learning_rates: List[float] = field(default_factory=list)
    
    # Early stopping
    patience_counter: int = 0
    should_stop: bool = False
    # Burn-in samples collected so far. Persisted because the burn-in baseline is a
    # summary (median/mean/max) over the WHOLE window: a resume inside the window that
    # restarted this list empty would compute the baseline from the post-resume subset
    # only, silently shifting the early-stopping reference point.
    burn_in_values: List[float] = field(default_factory=list)
    # global_step of the last validation pass. Persisted so the eval_steps cadence
    # survives a soft stop; when this reset to 0 on restart,
    # `global_step - last_validation_step >= eval_steps` was trivially true and the
    # first post-resume epoch always validated regardless of the configured cadence.
    last_validation_step: int = 0
    
    # Gradient accumulation
    accumulation_steps: int = 0
    effective_batch_size: int = 0
    
    # Training time
    total_training_time: float = 0.0
    epoch_times: List[float] = field(default_factory=list)
    
    # Checkpoint info
    last_checkpoint_step: int = 0
    checkpoints_saved: List[str] = field(default_factory=list)

    # Unified training phase (config.training.phase) recorded at save time so a
    # resume can detect a phase transition (0 = pre-phase-key checkpoint)
    phase: int = 0
    # ASL loss-state persistence (todos/ASL_plan.md SS8 row 2): gamma_neg,
    # gamma-step history/dwell bookkeeping, and telemetry EMAs. Owned (and
    # mutated in place) by asl_telemetry.ASLDriveManager; without this, any
    # gamma change silently reverts to the YAML value on restart.
    loss_state: Dict[str, Any] = field(default_factory=dict)

    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics history"""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        # Convert to regular dict to avoid serialization issues
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingState':
        """Load from dictionary.

        Filters to known field names so checkpoints carrying keys for
        removed/renamed fields (legacy runs) load instead of TypeError-ing.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
    
    def get_summary(self) -> str:
        """Get training state summary"""
        summary = f"Epoch: {self.epoch}, Step: {self.global_step}\n"
        summary += f"Best Metric: {self.best_metric:.4f} (Epoch {self.best_epoch})\n"
        summary += f"Train Loss: {self.train_loss:.4f}, Val Loss: {self.val_loss:.4f}\n"
        summary += f"Patience: {self.patience_counter}, Should Stop: {self.should_stop}"
        return summary


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing with warm restarts and linear warmup"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        # Clamp so the cosine denominator (cur_cycle_steps - warmup_steps) in get_lr
        # is always >= 1. For multi-cycle configs first_cycle_steps can be
        # <= warmup_steps, which would divide by zero/negative -> NaN/inf LR. This
        # mirrors the restart guard in step() below.
        self.cur_cycle_steps = max(first_cycle_steps, warmup_steps + 1)
        self.cycle = 0
        self.step_in_cycle = 0
        
        self.base_max_lr = max_lr

        # Warn about edge case that can cause cycle_steps to remain constant
        if cycle_mult == 1.0 and warmup_steps == 0:
            warnings.warn(
                "CosineAnnealingWarmupRestarts: cycle_mult=1.0 with warmup_steps=0 "
                "means cycle length never changes. This is valid but may not be intended.",
                UserWarning
            )

        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            # Honor each param group's own base LR (layer-wise decay / separate
            # head LR). When all groups share the configured max_lr (the common
            # case) this ratio is exactly 1.0, so this is a no-op there; the old
            # `for _ in self.base_lrs` returned max_lr for every group, silently
            # collapsing any intended per-group LR.
            group_max = self.max_lr * (base_lr / self.base_max_lr) if self.base_max_lr else self.max_lr
            if self.step_in_cycle < self.warmup_steps:
                # Linear warmup
                lr = (group_max - self.min_lr) * self.step_in_cycle / self.warmup_steps + self.min_lr
            else:
                # Cosine annealing
                lr = self.min_lr + (group_max - self.min_lr) * \
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) /
                                  (self.cur_cycle_steps - self.warmup_steps))) / 2
            lrs.append(lr)
        return lrs
    
    def state_dict(self):
        d = super().state_dict()
        d['step_in_cycle'] = self.step_in_cycle
        d['cycle'] = self.cycle
        d['cur_cycle_steps'] = self.cur_cycle_steps
        return d

    def load_state_dict(self, state_dict):
        # Extract custom keys before passing to super (which doesn't know about them)
        step_in_cycle = state_dict.pop('step_in_cycle', None)
        cycle = state_dict.pop('cycle', None)
        cur_cycle_steps = state_dict.pop('cur_cycle_steps', None)
        super().load_state_dict(state_dict)
        if step_in_cycle is not None:
            self.step_in_cycle = step_in_cycle
        if cycle is not None:
            self.cycle = cycle
        if cur_cycle_steps is not None:
            self.cur_cycle_steps = cur_cycle_steps

    def retarget(self, *, max_lr: float, warmup_steps: int, first_cycle_steps: int,
                 min_lr: Optional[float] = None, gamma: Optional[float] = None):
        """Re-point the schedule at a new geometry while preserving progress.

        For resuming a run whose LR schedule no longer matches the config. ``state_dict()``
        is the whole ``__dict__``, so ``load_state_dict`` restores the checkpoint's peak
        LR, ``base_lrs``, warmup length, cycle length, ``min_lr`` and ``gamma`` - every
        one of which the current config may have moved. Continuing with them means
        training on the previous run's schedule while the config claims otherwise.

        Progress is carried across as a FRACTION of the cycle rather than as an absolute
        update count. That is the only mapping that keeps the cosine landing on
        ``min_lr`` at the end of training when updates-per-epoch changes; carrying the
        absolute ``step_in_cycle`` would either truncate the anneal (larger batch) or
        overrun ``cur_cycle_steps`` and fire an unintended warm restart (smaller batch).

        ``min_lr`` / ``gamma`` are optional because neither depends on batch size, which
        is what this method was first written for. They are accepted anyway: the caller
        is now a general LR-schedule guard, and a resume that silently discarded a
        changed ``lr_end`` would anneal to the OLD floor while the config advertised the
        new one. Pass None to leave either at its restored value.

        Any gamma decay already earned by completed cycles is re-applied on top of the
        new peak.
        """
        old_cycle = max(1, int(self.cur_cycle_steps))
        new_cycle = max(int(first_cycle_steps), int(warmup_steps) + 1)
        progress = min(1.0, max(0.0, self.step_in_cycle / old_cycle))

        # Preserve per-group LR ratios (layer-wise decay / separate head LR) by scaling
        # base_lrs against the OLD peak rather than overwriting them with the new one.
        old_peak = float(self.base_max_lr) if self.base_max_lr else float(max_lr)
        ratio = (float(max_lr) / old_peak) if old_peak else 1.0
        self.base_lrs = [lr * ratio for lr in self.base_lrs]

        # Both must land BEFORE max_lr is recomputed below, which reads them.
        if min_lr is not None:
            self.min_lr = float(min_lr)
        if gamma is not None:
            self.gamma = float(gamma)

        self.base_max_lr = float(max_lr)
        self.max_lr = max(self.base_max_lr * (self.gamma ** self.cycle), self.min_lr)
        self.warmup_steps = int(warmup_steps)
        self.first_cycle_steps = int(first_cycle_steps)
        self.cur_cycle_steps = new_cycle
        self.step_in_cycle = min(new_cycle - 1, int(round(progress * new_cycle)))

        # Apply immediately: the next optimizer update happens before the next
        # scheduler.step(), so without this it would run at the stale restored LR.
        # 'initial_lr' is re-synced too - it is only read by _LRScheduler.__init__, but
        # it rides along in optimizer.state_dict() into every subsequent checkpoint, and
        # leaving it at the pre-retarget value plants a contradiction for whoever reads
        # that checkpoint next.
        new_lrs = self.get_lr()
        for group, lr, base in zip(self.optimizer.param_groups, new_lrs, self.base_lrs):
            group['lr'] = lr
            if 'initial_lr' in group:
                group['initial_lr'] = base
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        return new_lrs

    def step(self, epoch=None):
        if epoch is None:
            self.step_in_cycle += 1
            
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                # Calculate new cycle steps with minimum bound to prevent 0 or negative values
                new_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
                self.cur_cycle_steps = max(new_cycle_steps, max(1, self.warmup_steps + 1))
                # Apply gamma decay with floor to prevent max_lr from becoming too small
                self.max_lr = max(self.base_max_lr * (self.gamma ** self.cycle), self.min_lr)
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cycle = n
                    self.cur_cycle_steps = int(self.first_cycle_steps * self.cycle_mult ** n)
                    self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
            else:
                self.step_in_cycle = epoch
        
        super().step(epoch)

class AsyncCheckpointWriter:
    """
    Background thread for non-blocking checkpoint saves.

    Eliminates 30-90 second training stalls by moving torch.save() to a background thread.
    The main thread only needs to prepare the checkpoint dict (state_dict copies are unavoidable),
    then queues it for async saving.

    Usage:
        writer = AsyncCheckpointWriter()
        writer.save_async(checkpoint_dict, path, callback=on_complete)
        # ... training continues immediately ...
        writer.shutdown()  # Call at end of training
    """

    def __init__(self, max_queue_size: int = 2):
        """
        Initialize async checkpoint writer.

        Args:
            max_queue_size: Max pending saves before blocking. Default 2 prevents unbounded memory.
        """
        import queue
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="AsyncCheckpointWriter")
        self._shutdown_event = threading.Event()
        self._thread.start()
        self._pending_count = 0
        self._lock = threading.Lock()
        self._last_error = None

    def _worker(self):
        """Background worker that processes checkpoint saves."""
        import queue
        while not self._shutdown_event.is_set():
            # Wait for work with timeout to check shutdown flag
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:  # Shutdown sentinel — never counted in _pending_count
                break

            # From here the item was counted by save_async(). Guarantee the
            # _pending_count decrement runs no matter what fails below (including a
            # malformed item / unpack error) so wait_pending() can never hang on a
            # stuck counter, and record _last_error so a worker failure is never
            # silently swallowed.
            try:
                checkpoint, path, lock_context, callback = item

                # Atomic save with temp file
                temp_path = None
                try:
                    fd, temp_path = tempfile.mkstemp(
                        suffix='.tmp',
                        prefix='async_ckpt_',
                        dir=path.parent
                    )
                    os.close(fd)

                    with lock_context:
                        torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=True)
                        os.replace(temp_path, path)
                        temp_path = None  # Mark as consumed

                    if callback:
                        callback(path, success=True, error=None)

                except Exception as e:
                    self._last_error = e
                    logger.error(f"Async checkpoint save failed: {e}")
                    if callback:
                        callback(path, success=False, error=e)
                finally:
                    if temp_path and Path(temp_path).exists():
                        try:
                            Path(temp_path).unlink()
                        except OSError:
                            pass  # Best-effort cleanup, ignore filesystem errors

            except Exception as e:
                # Unexpected failure (e.g. a malformed queue item) — record it so it
                # is surfaced at shutdown instead of vanishing, then fall through to
                # the finally so the pending counter is still released.
                self._last_error = e
                logger.error(f"AsyncCheckpointWriter worker error: {e}")
            finally:
                with self._lock:
                    self._pending_count -= 1
                self._queue.task_done()

    def save_async(
        self,
        checkpoint: Dict[str, Any],
        path: Path,
        lock_context=None,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Queue a checkpoint for async saving.

        Args:
            checkpoint: The checkpoint dict to save (must be detached from GPU)
            path: Destination path
            lock_context: Optional file lock context manager
            callback: Optional callback(path, success, error) called on completion

        Returns:
            True if queued successfully, False if queue is full (caller should save sync)
        """
        if self._shutdown_event.is_set():
            return False

        if lock_context is None:
            lock_context = nullcontext()

        import queue
        with self._lock:
            self._pending_count += 1
        try:
            self._queue.put_nowait((checkpoint, path, lock_context, callback))
            return True
        except queue.Full:
            with self._lock:
                self._pending_count -= 1
            # Queue is full, caller should save synchronously
            return False

    def wait_pending(self, timeout: float = 300.0) -> bool:
        """Wait for all pending saves to complete with timeout support.

        Args:
            timeout: Maximum seconds to wait (default 300s = 5 minutes)

        Returns:
            True if all saves completed, False if timeout was reached
        """
        deadline = time.time() + timeout
        poll_interval = 0.1  # 100ms polling

        while True:
            with self._lock:
                if self._pending_count == 0:
                    return True

            if time.time() > deadline:
                with self._lock:
                    pending = self._pending_count
                logger.warning(
                    f"Timeout waiting for {pending} pending checkpoint save(s) after {timeout:.0f}s"
                )
                return False

            time.sleep(poll_interval)

    def shutdown(self, wait: bool = True, timeout: float = 300.0):
        """
        Shutdown the writer.

        Args:
            wait: If True, wait for pending saves to complete
            timeout: Max seconds to wait
        """
        if wait:
            self.wait_pending(timeout)
        self._shutdown_event.set()
        self._queue.put(None)  # Wake up worker
        self._thread.join(timeout=10.0)

    @property
    def pending_count(self) -> int:
        """Number of saves currently queued or in progress."""
        with self._lock:
            return self._pending_count

    @property
    def last_error(self) -> Optional[Exception]:
        """Last error encountered during async save."""
        return self._last_error


class CheckpointManager:
    """Manages model checkpoints"""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        keep_best: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_last: bool = True,
        async_save: bool = True,
        **_unused,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        # when True, only retain numbered checkpoints on best
        self.keep_best = keep_best
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_last = save_last

        self.checkpoints = []
        self.best_checkpoint = None

        # Async checkpoint writer for non-blocking saves
        # Eliminates 30-90 second training stalls during checkpoint saves
        self.async_save = async_save
        self._async_writer = AsyncCheckpointWriter() if async_save else None

        # Load existing checkpoints
        self._scan_existing_checkpoints()

    def _is_primary_process(self) -> bool:
        """Check if this is the primary process for checkpoint operations."""
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True
    
    def _scan_existing_checkpoints(self):
        """Scan directory for existing checkpoints"""
        if self.checkpoint_dir.exists():
            checkpoint_files = [
                p for p in self.checkpoint_dir.glob("checkpoint_*.pt")
                if p.exists()
            ]
            self.checkpoints = checkpoint_files
            self._sort_checkpoints_safe()

        # Find best checkpoint
        best_file = self.checkpoint_dir / "best_model.pt"
        if best_file.exists():
            self.best_checkpoint = best_file

    def _sort_checkpoints_safe(self):
        """Sort checkpoints by mtime, handling missing files gracefully."""
        # Filter out non-existent files
        self.checkpoints = [p for p in self.checkpoints if p.exists()]
        try:
            self.checkpoints.sort(key=lambda x: x.stat().st_mtime)
        except FileNotFoundError:
            self.checkpoints = [p for p in self.checkpoints if p.exists()]
            if self.checkpoints:
                self.checkpoints.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0)

    def _sync_save_checkpoint(self, checkpoint: Dict[str, Any], path: Path, lock_context) -> None:
        """Synchronously save checkpoint with atomic write and file locking."""
        temp_path = None
        try:
            with lock_context:
                fd, temp_path = tempfile.mkstemp(suffix='.tmp', prefix='checkpoint_', dir=self.checkpoint_dir)
                try:
                    os.close(fd)
                except Exception:
                    pass
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, path)
                temp_path = None  # Mark as consumed
        finally:
            if temp_path is not None:
                try:
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
                except Exception:
                    pass

    def shutdown(self, wait: bool = True, timeout: float = 300.0):
        """Shutdown async writer. Call at end of training."""
        if self._async_writer is not None:
            self._async_writer.shutdown(wait=wait, timeout=timeout)
            # Surface any async-save failure that would otherwise let the run end
            # "cleanly" with missing numbered/best checkpoints. last.pt is refreshed
            # from each numbered file after it is fully written (atomic os.replace),
            # so it is always a complete file - but on a failed async save it may be
            # one save behind, and the operator must know the async best/numbered
            # copies may be absent.
            if self._async_writer.last_error is not None:
                logger.error(
                    "Async checkpoint writer reported an unresolved error during the run: "
                    f"{self._async_writer.last_error!r}. Numbered/best async checkpoints may be "
                    "missing or incomplete - verify the checkpoint directory. (last.pt is "
                    "always a complete file but may be one save behind.)"
                )
            self._async_writer = None

    def _deep_to_cpu(self, obj):
        """Recursively move tensors to CPU and clone them to ensure thread safety."""
        if isinstance(obj, torch.Tensor):
            # detach() prevents tracking history, cpu() moves to host, clone() ensures it's a copy
            # (essential if original was already on CPU to avoid reference)
            return obj.detach().cpu().clone()
        elif isinstance(obj, dict):
            return {k: self._deep_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_to_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._deep_to_cpu(v) for v in obj)
        elif isinstance(obj, set):
            return {self._deep_to_cpu(v) for v in obj}
        else:
            return obj

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        training_state: TrainingState,
        is_best: bool = False,
        config: Optional[Dict] = None,
        train_loader: Optional['DataLoader'] = None,
        scaler: Optional[GradScaler] = None
    ) -> Optional[Path]:
        """Save a checkpoint"""

        # VALIDATION FIRST - fail fast before any work
        if not self._is_primary_process():
            return None

        # Validate config is provided
        if config is None:
            raise RuntimeError(
                "Configuration must be provided to save_checkpoint to embed preprocessing parameters. "
                "Please pass a config dict with the following required keys: "
                "normalize_mean, normalize_std, image_size, patch_size. "
                "This ensures checkpoints contain the exact preprocessing used during training."
            )

        # Validate required preprocessing parameters are present
        required_params = ['normalize_mean', 'normalize_std', 'image_size', 'patch_size']

        def get_param(cfg: Dict[str, Any], key: str):
            if key in cfg:
                return cfg[key]
            for section in ('data', 'model', 'inference', 'export', 'training'):
                sub = cfg.get(section)
                if sub is None:
                    continue
                # Support both dict and dataclass/object access
                if isinstance(sub, dict) and key in sub:
                    return sub[key]
                elif hasattr(sub, key):
                    return getattr(sub, key)
            return None

        def _get_architecture_type(cfg: Dict[str, Any]) -> Optional[str]:
            arch = None
            if isinstance(cfg, dict):
                model_cfg = cfg.get('model')
                if isinstance(model_cfg, dict):
                    arch = model_cfg.get('architecture_type')
                elif model_cfg is not None:
                    arch = getattr(model_cfg, 'architecture_type', None)
                if arch is None:
                    arch = cfg.get('architecture_type')
            else:
                model_cfg = getattr(cfg, 'model', None)
                if model_cfg is not None:
                    arch = getattr(model_cfg, 'architecture_type', None)
            return str(arch).lower() if arch else None

        architecture_type = _get_architecture_type(config) or 'vit'

        missing_params = [p for p in required_params if get_param(config, p) is None]
        if missing_params:
            raise RuntimeError(
                f"Missing required preprocessing parameters in config: {', '.join(missing_params)}. "
                f"All of {required_params} must be explicitly provided to ensure correct preprocessing."
            )

        # Validate preprocessing parameter types and values
        try:
            # Extract color_order from config (defaults to RGB, the project
            # default and how legacy artifacts read back elsewhere).
            color_order_raw = get_param(config, 'color_order') or 'RGB'
            color_order = str(color_order_raw).upper()
            if color_order not in {'RGB', 'BGR'}:
                raise ValueError(
                    f"color_order must be 'RGB' or 'BGR', got {color_order_raw!r}"
                )

            normalize_mean = tuple(get_param(config, 'normalize_mean'))
            normalize_std = tuple(get_param(config, 'normalize_std'))
            if len(normalize_mean) != 3 or len(normalize_std) != 3:
                raise ValueError("normalize_mean and normalize_std must have exactly 3 values")

            image_size = int(get_param(config, 'image_size'))
            patch_size = int(get_param(config, 'patch_size'))

            if image_size <= 0 or patch_size <= 0:
                raise ValueError("image_size and patch_size must be positive")
            if image_size % patch_size != 0:
                raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")

        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Invalid preprocessing parameters in config: {e}")

        # Validate vocabulary path exists
        vocab_path = Path(config.get('vocab_path', VOCAB_PATH) if config else VOCAB_PATH)
        if not vocab_path.exists():
            raise RuntimeError(
                f"Vocabulary file not found at {vocab_path}. "
                "Refusing to save a non self-contained checkpoint (fail-fast)."
            )

        # NOW proceed with checkpoint preparation
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'metrics': metrics,
            'training_state': training_state.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'is_best': bool(is_best),
        }
        
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['optimizer_class'] = type(optimizer).__name__

        if self.save_scheduler and scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint['scheduler_class'] = type(scheduler).__name__
            # Save critical params for validation (what state_dict doesn't capture)
            checkpoint['scheduler_params'] = {
                'total_steps': getattr(scheduler, 'total_steps', None),
                'warmup_steps': getattr(scheduler, 'warmup_steps', None),
                'first_cycle_steps': getattr(scheduler, 'first_cycle_steps', None),
                'max_lr': getattr(scheduler, 'max_lr', None),
                # base_max_lr is the CONFIGURED peak; max_lr is the live value after any
                # gamma decay from completed restarts. Only the former is comparable
                # across runs, so it is what the resume-time diff check reads.
                'base_max_lr': getattr(scheduler, 'base_max_lr', None),
                'min_lr': getattr(scheduler, 'min_lr', None),
            }

        # Save GradScaler state for AMP training continuity
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        if config is not None:
            # Convert config to dict to avoid pickling Enum types (PyTorch 2.6+ weights_only=True compatibility)
            checkpoint['config'] = config.to_dict() if hasattr(config, 'to_dict') else config
            # Store architecture type for checkpoint compatibility checking
            checkpoint['architecture_type'] = architecture_type

        # Embed RNG states to enable exact stream continuation on resume
        try:
            py_state, np_state, torch_cpu_state, cuda_state = _save_rng_states()
            # Pack numpy state into builtins to avoid object pickling concerns
            np_packed = _pack_np_state(np_state)
            # cuda_state can be a list (set_rng_state_all) or a tensor
            checkpoint['rng_states'] = {
                'py': py_state,
                'np': np_packed,
                'torch_cpu': torch_cpu_state,
                'cuda': cuda_state,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        except Exception as _rng_e:
            logger.debug("RNG state capture skipped: %s", _rng_e)

        # Save sampler state for O(1) mid-epoch resume
        # This allows ResumableSampler to skip directly to the resume batch
        # instead of iterating through all batches (which takes ~17min for 5000+ batches)
        sampler_state = None
        if train_loader is not None:
            sampler = getattr(train_loader, 'sampler', None)
            if hasattr(sampler, 'get_state'):
                sampler_state = sampler.get_state()
                # Add current batch position for resume
                sampler_state['batch_in_epoch'] = training_state.batch_in_epoch
                logger.debug(f"Saved sampler state: epoch={sampler_state['epoch']}, batch={sampler_state['batch_in_epoch']}")
        checkpoint['sampler_state'] = sampler_state

        # Provide a deterministic salt hint derived from stable checkpoint content
        try:
            # Validate timestamp format before using it (prevents injection)
            timestamp_str = checkpoint.get('timestamp', '')
            if not isinstance(timestamp_str, str):
                timestamp_str = ''

            # Validate ISO format (prevents injection)
            try:
                # This will raise ValueError if format is invalid
                datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                validated_timestamp = timestamp_str
            except (ValueError, AttributeError):
                # Invalid timestamp, use current time
                validated_timestamp = datetime.now().isoformat()
                logger.debug(f"Invalid timestamp in checkpoint, using current time")

            # Create salt from validated components
            salt_components = [
                str(int(epoch)),
                str(int(step)),
                validated_timestamp
            ]
            salt_src = '|'.join(salt_components)
            checkpoint['resume_salt_hint'] = int(hashlib.sha1(salt_src.encode()).hexdigest()[:8], 16)
        except Exception as e:
            logger.debug(f"Failed to create resume salt hint: {e}")
            # Don't include salt if validation fails
            pass

        # CRITICAL: Embed vocabulary and preprocessing directly into checkpoint
        if hasattr(model, 'module'):
            model_to_check = model.module
        else:
            model_to_check = model

        # Load and embed vocabulary (already validated at function start)
        checkpoint = ModelMetadata.embed_vocabulary(checkpoint, vocab_path)

        # Embed preprocessing parameters (already validated at function start)
        checkpoint = ModelMetadata.embed_preprocessing_params(
            checkpoint,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            image_size=image_size,
            patch_size=patch_size,
            color_order=color_order,
        )

        # Backwards compatibility info
        if hasattr(model_to_check, 'config'):
            num_tags = getattr(model_to_check.config, 'num_tags', None)
            if num_tags is not None:
                checkpoint['num_tags'] = num_tags
                checkpoint['vocabulary_info'] = {
                    'num_tags': num_tags,
                    'vocab_path': str(vocab_path),
                    'has_vocabulary': True,
                    'embedded': 'vocab_b64_gzip' in checkpoint
                }

        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save numbered checkpoint atomically unless enforcing best-only retention
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        wrote_numbered = False

        # Use file locking to prevent race conditions in distributed training
        lock_path = self.checkpoint_dir / ".checkpoint_write.lock"
        lock_context = filelock.FileLock(lock_path, timeout=60) if HAS_FILELOCK else nullcontext()

        should_save_best = bool(is_best)

        def _save_best_from_checkpoint(source_path: Path) -> None:
            if not should_save_best:
                return

            best_path = self.checkpoint_dir / "best_model.pt"
            temp_best = None
            best_lock = filelock.FileLock(lock_path, timeout=60) if HAS_FILELOCK else nullcontext()
            try:
                with best_lock:
                    fd_best, temp_best = tempfile.mkstemp(
                        suffix='.tmp',
                        prefix='best_',
                        dir=self.checkpoint_dir
                    )
                    try:
                        os.close(fd_best)
                    except Exception:
                        pass

                    shutil.copy2(source_path, temp_best)
                    os.replace(temp_best, best_path)
                    temp_best = None  # Mark as consumed
                    self.best_checkpoint = best_path
                    logger.info(f"Saved best model to {best_path}")
            except Exception as e:
                logger.warning(f"Failed to save best model to {best_path}: {e}")
            finally:
                if temp_best is not None:
                    try:
                        if Path(temp_best).exists():
                            Path(temp_best).unlink()
                    except Exception:
                        pass

        def _update_last_from_file(source_path: Path) -> None:
            """Refresh last.pt from an already-written checkpoint file.

            Hardlinks the just-written file to a temp name and atomically
            os.replace()s it onto last.pt - no second serialization, and last.pt is
            always a complete file (crash-safe; numbered checkpoints are never
            modified in place, only unlinked, so sharing the inode is safe). Falls
            back to a copy where hardlinks are unsupported (non-NTFS/cross-volume).
            """
            if not self.save_last:
                return
            last_path = self.checkpoint_dir / LAST_CKPT_NAME
            temp_last = None
            last_lock = filelock.FileLock(lock_path, timeout=60) if HAS_FILELOCK else nullcontext()
            try:
                with last_lock:
                    fd_last, temp_last = tempfile.mkstemp(
                        suffix='.tmp',
                        prefix='last_',
                        dir=self.checkpoint_dir
                    )
                    try:
                        os.close(fd_last)
                    except Exception:
                        pass
                    # os.link refuses to overwrite, so drop the mkstemp placeholder first
                    os.unlink(temp_last)
                    try:
                        os.link(source_path, temp_last)
                    except OSError:
                        shutil.copy2(source_path, temp_last)
                    os.replace(temp_last, last_path)
                    temp_last = None  # Mark as consumed
                    logger.debug(f"Updated {last_path} from {source_path}")
            except Exception as e:
                logger.warning(f"Failed to update {last_path} from {source_path}: {e}")
            finally:
                if temp_last is not None:
                    try:
                        if Path(temp_last).exists():
                            Path(temp_last).unlink()
                    except Exception:
                        pass

        # Callback for async save completion logging
        def _on_save_complete(path, success, error):
            if success:
                logger.info(f"Async checkpoint saved to {path}")
                _save_best_from_checkpoint(path)
                _update_last_from_file(path)
            else:
                logger.error(f"Async checkpoint save failed for {path}: {error}")

        # Single CPU deep copy shared by the numbered and last.pt saves below.
        # (Previously this was done twice per save - two ~GB-scale D2H copies.)
        cpu_checkpoint = None

        if not (self.keep_best and not is_best):
            # Use async save if available (eliminates 30-90s training stalls)
            if self._async_writer is not None:
                # Deep copy to CPU to ensure thread safety during async save
                # This prevents race conditions where model weights change while saving
                try:
                    cpu_checkpoint = self._deep_to_cpu(checkpoint)
                    if self._async_writer.save_async(cpu_checkpoint, checkpoint_path, lock_context, _on_save_complete):
                        wrote_numbered = True
                        self.checkpoints.append(checkpoint_path)
                        logger.debug(f"Queued async checkpoint save to {checkpoint_path}")
                        # Note: cpu_checkpoint ownership transferred to async writer, don't delete here
                        # (last.pt is refreshed from the numbered file in _on_save_complete)
                    else:
                        # Queue full, fall back to sync save (use CPU checkpoint to avoid VRAM spike)
                        logger.warning("Async checkpoint queue full, falling back to sync save")
                        self._sync_save_checkpoint(cpu_checkpoint, checkpoint_path, lock_context)
                        wrote_numbered = True
                        self.checkpoints.append(checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        _save_best_from_checkpoint(checkpoint_path)
                        _update_last_from_file(checkpoint_path)
                        cpu_checkpoint = None  # Free CPU copy after sync save completes
                except Exception as e:
                    logger.warning(f"Failed to prepare async checkpoint: {e}. Falling back to sync save.")
                    # Try to create CPU checkpoint for sync save to avoid VRAM spike
                    try:
                        cpu_checkpoint = self._deep_to_cpu(checkpoint)
                        self._sync_save_checkpoint(cpu_checkpoint, checkpoint_path, lock_context)
                        cpu_checkpoint = None  # Free CPU copy after saving
                    except Exception:
                        # Last resort: save with GPU tensors if CPU conversion also fails
                        self._sync_save_checkpoint(checkpoint, checkpoint_path, lock_context)
                    wrote_numbered = True
                    self.checkpoints.append(checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    _save_best_from_checkpoint(checkpoint_path)
                    _update_last_from_file(checkpoint_path)
            else:
                # Sync save (original behavior) - use CPU checkpoint to avoid VRAM spike
                cpu_checkpoint = self._deep_to_cpu(checkpoint)
                self._sync_save_checkpoint(cpu_checkpoint, checkpoint_path, lock_context)
                cpu_checkpoint = None  # Free CPU copy after sync save completes
                wrote_numbered = True
                self.checkpoints.append(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                _save_best_from_checkpoint(checkpoint_path)
                _update_last_from_file(checkpoint_path)
        else:
            logger.debug("save_best_only=True: skipping numbered checkpoint at step %s", step)

        # last.pt (crash-resume pointer) is normally produced from the just-written
        # numbered file via _update_last_from_file (hardlink/copy + atomic
        # os.replace) - no second serialization. Only when the numbered save was
        # skipped entirely (save_best_only and not best) does last.pt need its own
        # serialization; queue it on the async writer when available.
        if self.save_last and not wrote_numbered:
            last_path = self.checkpoint_dir / LAST_CKPT_NAME
            try:
                if cpu_checkpoint is None:
                    cpu_checkpoint = self._deep_to_cpu(checkpoint)
                if self._async_writer is not None and self._async_writer.save_async(
                    cpu_checkpoint, last_path, lock_context, None
                ):
                    logger.debug(f"Queued async checkpoint save to {last_path}")
                    # Ownership transferred to async writer
                else:
                    self._sync_save_checkpoint(cpu_checkpoint, last_path, lock_context)
            except Exception as e:
                logger.warning("Failed to update %s using CPU copy: %s. Retrying with original tensors.", last_path, e)
                try:
                    self._sync_save_checkpoint(checkpoint, last_path, lock_context)
                except Exception as e2:
                    logger.error("Failed to update %s: %s", last_path, e2)
        
        # Manage checkpoint limit
        if wrote_numbered:
            self._cleanup_old_checkpoints()
        
        # Update training state
        if wrote_numbered:
            training_state.checkpoints_saved.append(str(checkpoint_path))
        else:
            training_state.checkpoints_saved.append(str(self.checkpoint_dir / LAST_CKPT_NAME))
        training_state.last_checkpoint_step = step

        # Explicit cleanup to help garbage collector release GPU memory
        # (checkpoint is a local reference to the passed-in dict, deleting removes this reference)
        del checkpoint

        return checkpoint_path if wrote_numbered else None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding limit"""
        if not self._is_primary_process():
            return

        if self.max_checkpoints is None or self.max_checkpoints <= 0:
            return

        # Refresh and sort checkpoints
        self._refresh_checkpoint_list()
        self._sort_checkpoints_safe()

        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if oldest == self.best_checkpoint:
                continue
            # Don't check exists() before unlink to avoid TOCTOU race
            # FileNotFoundError is caught if another process deleted it
            try:
                oldest.unlink()
                logger.info(f"Removed old checkpoint: {oldest}")
            except FileNotFoundError:
                # File was already deleted (by another process or manually)
                pass
            except Exception as e:
                logger.warning(f"Warning: Could not delete {oldest}: {e}")

    def _refresh_checkpoint_list(self):
        """Refresh the checkpoint list to sync with disk state."""
        disk_checkpoints = set()
        if self.checkpoint_dir.exists():
            disk_checkpoints = {
                p.resolve() for p in self.checkpoint_dir.glob('checkpoint_*.pt')
                if p.exists()
            }

        # Keep only existing files from our list
        self.checkpoints = [p for p in self.checkpoints if p.exists()]

        # Add any new files from disk that we don't know about
        known_paths = {p.resolve() for p in self.checkpoints}
        for disk_path in disk_checkpoints:
            if disk_path not in known_paths:
                self.checkpoints.append(Path(disk_path))

    def _safe_load_checkpoint(
        self,
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
        except pickle.UnpicklingError as e:
            # PyTorch 2.6+ blocks custom classes with weights_only=True
            # Fall back to weights_only=False for checkpoints containing config objects
            warnings.warn(
                f"Checkpoint contains custom classes blocked by weights_only=True: {e}. "
                f"Loading with weights_only=False. This is safe for your own checkpoints. "
                f"DO NOT load checkpoints from untrusted sources.",
                UserWarning,
                stacklevel=2
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

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
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = torch.device('cpu'),
        scaler: Optional[GradScaler] = None,
        expected_vocab_sha256: Optional[str] = None,
        enforce_vocab_check: bool = False,
        allow_unverified_vocab_resume: bool = False,
    ) -> Dict:
        """Load a checkpoint.

        Args:
            expected_vocab_sha256: SHA256 of the vocabulary the caller is using right
                now. If the checkpoint embeds a different ``vocab_sha256``, we refuse
                to load — vocab indices would be misaligned and the model would train
                against the wrong tags silently. A falsy value or the sentinel
                ``"unknown"`` means the caller could not compute the current vocab
                hash, which is treated as unverifiable (see below).
            enforce_vocab_check: when True (the resume path), verification is
                MANDATORY: an unverifiable vocab (current SHA unknown, or the
                checkpoint embeds no SHA) refuses the load unless
                ``allow_unverified_vocab_resume`` is set. When False (inference /
                export / standalone validation), the check is best-effort — a
                concrete SHA mismatch still raises, but a missing SHA only warns.
            allow_unverified_vocab_resume: escape hatch for ``enforce_vocab_check``;
                set True to load a legacy checkpoint whose vocab cannot be verified —
                the caller then owns confirming the vocabulary matches.
        """

        if checkpoint_path is None:
            # Load latest checkpoint
            if self.checkpoints:
                checkpoint_path = self.checkpoints[-1]
            else:
                raise ValueError("No checkpoints found")

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict, meta = self._safe_load_checkpoint(checkpoint_path)

        # Refuse to resume into a model whose vocabulary has shifted under us.
        # Without this guard, indices in the checkpoint's tag head silently
        # address the wrong tags after any vocab regeneration.
        checkpoint_sha = meta.get('vocab_sha256')
        # "unknown" is the sentinel compute_vocab_sha256 returns when it could not
        # read the vocab; treat it the same as a missing value (unverifiable).
        current_known = bool(expected_vocab_sha256) and expected_vocab_sha256 != "unknown"
        checkpoint_known = bool(checkpoint_sha) and checkpoint_sha != "unknown"
        if current_known and checkpoint_known:
            # Both sides have a real hash: a mismatch is always fatal.
            if checkpoint_sha != expected_vocab_sha256:
                raise InvalidCheckpointError(
                    f"Vocabulary SHA mismatch loading {checkpoint_path}: "
                    f"checkpoint embeds {checkpoint_sha}, current vocab is "
                    f"{expected_vocab_sha256}. Tag indices would be misaligned. "
                    "Pin the matching vocabulary or retrain."
                )
        elif enforce_vocab_check:
            # Resume path: verification is mandatory. Could not verify because the
            # caller could not hash the current vocab, or the checkpoint embeds no
            # SHA (legacy). Fail CLOSED unless explicitly overridden — a silent
            # vocab mismatch scrambles every label index.
            if not current_known and not checkpoint_known:
                reason = "neither the current vocabulary nor the checkpoint provides a usable SHA256"
            elif not current_known:
                reason = "the current vocabulary SHA256 could not be computed"
            else:
                reason = "the checkpoint embeds no vocab_sha256 (legacy checkpoint)"
            if allow_unverified_vocab_resume:
                logger.warning(
                    "Resuming with UNVERIFIED vocabulary for %s: %s. Proceeding only "
                    "because training.allow_unverified_vocab_resume=True.",
                    checkpoint_path, reason,
                )
            else:
                raise InvalidCheckpointError(
                    f"Cannot verify vocabulary compatibility loading {checkpoint_path}: "
                    f"{reason}. Loading against a mismatched vocabulary silently "
                    "scrambles every label index. Pin the matching vocabulary, or set "
                    "training.allow_unverified_vocab_resume=true to override (legacy "
                    "checkpoints only)."
                )
        elif current_known and not checkpoint_known:
            # Best-effort (non-resume) caller asked to verify but the checkpoint has
            # no usable SHA — warn, preserving the original behavior.
            logger.warning(
                "Checkpoint at %s has no embedded vocab_sha256; cannot verify "
                "vocabulary compatibility.", checkpoint_path,
            )

        # Load model state (handles DDP and torch.compile key prefix mismatches)
        if model is not None:
            # Determine if model is wrapped (DDP/DataParallel)
            is_wrapped = hasattr(model, 'module')
            # Check if state dict keys have 'module.' prefix
            has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

            if is_wrapped and not has_module_prefix:
                # Loading unwrapped checkpoint into wrapped model: add 'module.' prefix
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                logger.debug("Added 'module.' prefix to state dict keys for DDP model")
            elif not is_wrapped and has_module_prefix:
                # Loading wrapped checkpoint into unwrapped model: remove 'module.' prefix
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                logger.debug("Removed 'module.' prefix from state dict keys for non-DDP model")

            # Handle torch.compile() wrapper (_orig_mod prefix)
            is_compiled = hasattr(model, '_orig_mod')
            has_orig_mod_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())

            if is_compiled and not has_orig_mod_prefix:
                # Loading non-compiled checkpoint into compiled model: add '_orig_mod.' prefix
                state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
                logger.info("Added '_orig_mod.' prefix to state dict keys for torch.compile() model")
            elif not is_compiled and has_orig_mod_prefix:
                # Loading compiled checkpoint into non-compiled model: remove '_orig_mod.' prefix
                state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
                logger.info("Removed '_orig_mod.' prefix from state dict keys for non-compiled model")

            # Filter out removed rating_head keys from old checkpoints
            rating_head_keys = [k for k in state_dict if 'rating_head' in k]
            if rating_head_keys:
                for k in rating_head_keys:
                    del state_dict[k]
                logger.info(f"Removed {len(rating_head_keys)} rating_head keys from checkpoint (rating merged into tags)")

            # Interpolate positional embeddings if resolution changed between checkpoint and model.
            # This enables loading a checkpoint trained at one resolution (e.g. 224) into a model
            # configured for a different resolution (e.g. 448) by bicubic-interpolating the
            # learned spatial positional embeddings to the new grid size.
            pos_embed_key = None
            for key in state_dict:
                if key.endswith('pos_embed'):
                    pos_embed_key = key
                    break
            if pos_embed_key is not None:
                saved_pos_embed = state_dict[pos_embed_key]
                # Get the model's expected pos_embed shape
                model_to_check = model
                if hasattr(model_to_check, 'module'):
                    model_to_check = model_to_check.module
                if hasattr(model_to_check, '_orig_mod'):
                    model_to_check = model_to_check._orig_mod
                model_pos_embed = None
                for name, param in model_to_check.named_parameters():
                    if name.endswith('pos_embed'):
                        model_pos_embed = param
                        break
                if model_pos_embed is not None and saved_pos_embed.shape != model_pos_embed.shape:
                    # Shape: (1, num_tokens, hidden_size) where num_tokens = num_patches + num_special_tokens
                    saved_len = saved_pos_embed.shape[1]
                    model_len = model_pos_embed.shape[1]
                    hidden_size = saved_pos_embed.shape[2]
                    # Determine number of non-patch (special) tokens by comparing with known grid sizes
                    saved_grid = int(math.isqrt(saved_len - 1))
                    # Special-token count is derived from the checkpoint, not hardcoded:
                    # the current model has only cls_token, but older checkpoints may carry
                    # more, so num_special = saved_len - saved_grid**2 stays generic.
                    num_special = saved_len - saved_grid * saved_grid
                    model_grid = int(math.isqrt(model_len - num_special))
                    if saved_grid * saved_grid + num_special == saved_len and model_grid * model_grid + num_special == model_len:
                        logger.info(
                            f"Interpolating pos_embed from {saved_grid}x{saved_grid} "
                            f"({saved_len} tokens) to {model_grid}x{model_grid} "
                            f"({model_len} tokens), {num_special} special token(s)"
                        )
                        # Separate special tokens (CLS, etc.) from patch embeddings
                        special_tokens = saved_pos_embed[:, :num_special, :]
                        patch_embed = saved_pos_embed[:, num_special:, :]
                        # Reshape to 2D spatial grid, interpolate, flatten back
                        patch_embed = patch_embed.reshape(1, saved_grid, saved_grid, hidden_size).permute(0, 3, 1, 2).float()
                        patch_embed = torch.nn.functional.interpolate(
                            patch_embed, size=(model_grid, model_grid), mode='bicubic', align_corners=False
                        )
                        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, model_grid * model_grid, hidden_size)
                        patch_embed = patch_embed.to(saved_pos_embed.dtype)
                        state_dict[pos_embed_key] = torch.cat([special_tokens, patch_embed], dim=1)
                        logger.info(f"pos_embed interpolated: {saved_pos_embed.shape} -> {state_dict[pos_embed_key].shape}")
                    else:
                        logger.warning(
                            f"pos_embed shape mismatch ({saved_pos_embed.shape} vs {model_pos_embed.shape}) "
                            f"but could not determine valid grid sizes for interpolation. "
                            f"Skipping interpolation — load_state_dict will likely fail."
                        )

            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                # Catch common mismatch errors (keys or shapes) to provide helpful guidance
                msg = str(e)
                if "Missing key(s)" in msg or "Unexpected key(s)" in msg or "size mismatch" in msg:
                    # Heuristic: if massive mismatch, it's likely an architecture change
                    # e.g. architecture change or vocab size change
                    logger.error("=" * 60)
                    logger.error("CRITICAL: Model architecture mismatch during checkpoint loading!")
                    logger.error("The checkpoint architecture does not match the current model.")
                    logger.error("This often happens when:")
                    logger.error("  1. Switching architectures")
                    logger.error("  2. Changing model size/dimensions (e.g. hidden_size, image_size)")
                    logger.error("  3. Changing vocabulary size without rebuilding the head")
                    logger.error("  4. Loading an old checkpoint into a modified model")
                    logger.error("")
                    logger.error("To fix this, you must start training fresh:")
                    logger.error("  Set training.resume_from='none' in your config")
                    logger.error("=" * 60)
                    raise RuntimeError(
                        f"Architecture mismatch (keys/shapes). Cannot resume from this checkpoint. "
                        f"Set training.resume_from='none' to start fresh. Original error: {e}"
                    ) from e
                raise  # Re-raise other runtime errors

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in meta:
            try:
                saved_opt_state = meta['optimizer_state_dict']

                # Validate param_groups count matches before attempting load
                saved_pg_count = len(saved_opt_state.get('param_groups', []))
                current_pg_count = len(optimizer.param_groups)
                if saved_pg_count != current_pg_count:
                    raise RuntimeError(
                        f"Optimizer param_groups count mismatch: checkpoint has {saved_pg_count}, "
                        f"current model has {current_pg_count}. This usually means the model architecture "
                        f"changed (e.g., different num_tags). To start fresh, set training.resume_from='none'."
                    )

                # Validate optimizer class matches (if metadata available)
                current_opt_class = type(optimizer).__name__
                saved_opt_class = meta.get('optimizer_class')

                if saved_opt_class and saved_opt_class != current_opt_class:
                    # AdamW family members are compatible with warnings
                    adamw_family = {'AdamW', 'AdamW8bit', 'PagedAdamW8bit', 'PagedAdamW32bit'}
                    if saved_opt_class in adamw_family and current_opt_class in adamw_family:
                        logger.warning(
                            f"Optimizer type changed: checkpoint={saved_opt_class}, current={current_opt_class}. "
                            f"State will be loaded but momentum/variance may not transfer optimally."
                        )
                    else:
                        raise RuntimeError(
                            f"Optimizer type mismatch: checkpoint has {saved_opt_class}, "
                            f"but current optimizer is {current_opt_class}. "
                            f"To start fresh, set training.resume_from='none'."
                        )

                # Load state dict (optimizer creates state entries as needed)
                optimizer.load_state_dict(saved_opt_state)

                # Move optimizer state to device (handles tensor-like objects)
                if hasattr(optimizer, 'state') and optimizer.state:
                    tensors_moved = 0
                    tensors_failed = 0
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            # Check for torch.Tensor and any tensor-like object with .to() method
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                                tensors_moved += 1
                            elif hasattr(v, 'to') and callable(getattr(v, 'to', None)):
                                # Handle tensor-like objects (e.g., bitsandbytes quantized tensors)
                                try:
                                    state[k] = v.to(device)
                                    tensors_moved += 1
                                except (TypeError, AttributeError) as e:
                                    tensors_failed += 1
                                    logger.debug(f"Could not move optimizer state '{k}' to {device}: {e}")
                    logger.debug(f"Moved {tensors_moved} optimizer state tensors to {device}")
                    if tensors_failed > 0:
                        logger.warning(
                            f"Failed to move {tensors_failed} optimizer state tensors to {device}. "
                            "Training may continue but optimizer momentum/variance could be on wrong device."
                        )

                    # Verify all tensors migrated successfully - fail fast if not
                    # Compare device types (e.g., 'cuda') not full device specs (e.g., 'cuda:0')
                    # because .to('cuda') moves to 'cuda:0' but device may be 'cuda' without index
                    target_device_type = device.type
                    wrong_device_count = 0
                    for state in optimizer.state.values():
                        for v in state.values():
                            if isinstance(v, torch.Tensor) and v.device.type != target_device_type:
                                wrong_device_count += 1
                    if wrong_device_count > 0:
                        raise RuntimeError(
                            f"{wrong_device_count} optimizer state tensors failed to migrate to {device}. "
                            f"Cannot continue - optimizer.step() would fail with device mismatch. "
                            f"To start fresh, set training.resume_from='none' in config."
                        )
                else:
                    logger.debug("Optimizer state loaded but empty (no tensors to migrate)")

            except Exception as e:
                logger.error(f"Failed to load optimizer state: {type(e).__name__}: {e}")
                raise RuntimeError(
                    f"Optimizer state restoration failed. This would cause loss of momentum/variance "
                    f"and training divergence. To start fresh, set training.resume_from='none'. Error: {e}"
                ) from e

        # Load scheduler state (with exception handling like optimizer)
        if scheduler is not None and 'scheduler_state_dict' in meta:
            try:
                saved_sched_state = meta['scheduler_state_dict']
                current_sched_type = type(scheduler).__name__

                # Basic validation: check for required keys
                if 'last_epoch' not in saved_sched_state:
                    raise RuntimeError(
                        f"Saved scheduler state is missing 'last_epoch' key. "
                        f"Checkpoint may be corrupted or from incompatible version."
                    )

                # Validate scheduler class matches (if metadata available)
                saved_sched_class = meta.get('scheduler_class')
                if saved_sched_class and saved_sched_class != current_sched_type:
                    raise RuntimeError(
                        f"Scheduler type mismatch: checkpoint has {saved_sched_class}, "
                        f"but current scheduler is {current_sched_type}. "
                        f"To start fresh, set training.resume_from='none'."
                    )

                # Check if scheduler config differs from checkpoint.
                # Note: load_state_dict() restores ALL scheduler attributes from the
                # checkpoint (including warmup_steps, first_cycle_steps, etc.), so
                # the checkpoint's LR curve is preserved for training continuity.
                # This info message alerts the user that their config would produce
                # different values, but the checkpoint values take precedence.
                saved_params = meta.get('scheduler_params', {})
                if saved_params:
                    # base_max_lr / min_lr belong in this list: load_state_dict restores
                    # the checkpoint's peak LR and base_lrs too, so a run that recomputed
                    # its LR (e.g. a batch-size change feeding scale_learning_rate) would
                    # otherwise train at the old rate with no message at all. This is a
                    # WARNING, not INFO - silently discarding the configured LR is the
                    # kind of thing that costs a training run.
                    param_diffs = []
                    for key in ['total_steps', 'warmup_steps', 'first_cycle_steps',
                                'base_max_lr', 'min_lr']:
                        saved_val = saved_params.get(key)
                        current_val = getattr(scheduler, key, None)
                        if saved_val is not None and current_val is not None and saved_val != current_val:
                            param_diffs.append(f"{key}: config={current_val} vs checkpoint={saved_val}")
                    if param_diffs:
                        logger.warning(
                            f"Scheduler config differs from checkpoint: {', '.join(param_diffs)}. "
                            f"Checkpoint values take precedence for training continuity - the "
                            f"config values above are NOT applied."
                        )

                scheduler.load_state_dict(saved_sched_state)
                logger.info(f"Scheduler state loaded successfully (type: {current_sched_type})")
            except Exception as e:
                logger.error(f"Failed to load scheduler state: {type(e).__name__}: {e}")
                raise RuntimeError(
                    f"Scheduler state restoration failed. LR schedule would restart from beginning. "
                    f"To start fresh, set training.resume_from='none'. Error: {e}"
                ) from e

        # Load GradScaler state for AMP training continuity
        if scaler is not None and 'scaler_state_dict' in meta:
            try:
                scaler.load_state_dict(meta['scaler_state_dict'])
                saved_scale = meta['scaler_state_dict'].get('scale', 'unknown')
                logger.info(f"GradScaler state loaded successfully (scale={saved_scale})")
            except Exception as e:
                logger.warning(f"Failed to load GradScaler state: {e}. Starting with default scale.")

        # Restore RNG states to ensure reproducible dataset shuffling
        if 'rng_states' in meta:
            try:
                rng_dict = meta['rng_states']

                # Validate RNG state structure before attempting restoration
                required_keys = ['py', 'np', 'torch_cpu']
                missing_keys = [k for k in required_keys if k not in rng_dict]
                if missing_keys:
                    raise RuntimeError(
                        f"RNG state dict is missing required keys: {missing_keys}. "
                        f"Checkpoint may be corrupted or from incompatible version."
                    )

                # Validate Python RNG state format (should be a tuple)
                if not isinstance(rng_dict['py'], tuple):
                    raise RuntimeError(
                        f"Python RNG state has invalid type: {type(rng_dict['py']).__name__}. "
                        f"Expected tuple. Checkpoint may be corrupted."
                    )

                # Validate torch_cpu state (should be a tensor-like)
                torch_cpu_state = rng_dict['torch_cpu']
                if not (isinstance(torch_cpu_state, torch.Tensor) or hasattr(torch_cpu_state, '__len__')):
                    raise RuntimeError(
                        f"PyTorch CPU RNG state has invalid type: {type(torch_cpu_state).__name__}. "
                        f"Checkpoint may be corrupted."
                    )

                # Validate NumPy state structure completely
                np_raw = rng_dict['np']
                if isinstance(np_raw, (tuple, list)):
                    if len(np_raw) < 5:
                        raise RuntimeError(
                            f"NumPy RNG state has only {len(np_raw)} elements, needs 5+. "
                            f"Checkpoint may be corrupted."
                        )
                    state_arr = np_raw[1]
                    if isinstance(state_arr, (list, np.ndarray)) and len(state_arr) < 100:
                        raise RuntimeError(
                            f"NumPy state array has only {len(state_arr)} elements, needs 624+ for MT19937. "
                            f"Checkpoint may be corrupted."
                        )

                # Validate CUDA state elements are tensors (if present)
                cuda_state = rng_dict.get('cuda')
                if cuda_state is not None and isinstance(cuda_state, list):
                    for i, elem in enumerate(cuda_state):
                        if not isinstance(elem, torch.Tensor):
                            raise RuntimeError(
                                f"CUDA RNG state element {i} is {type(elem).__name__}, expected torch.Tensor. "
                                f"Checkpoint may be corrupted."
                            )
                        if elem.dtype != torch.uint8:
                            logger.warning(f"CUDA RNG state element {i} has dtype {elem.dtype}, expected uint8")

                # Warn on CUDA version mismatch
                saved_cuda_ver = rng_dict.get('cuda_version')
                current_cuda_ver = torch.version.cuda if torch.cuda.is_available() else None
                if saved_cuda_ver and current_cuda_ver and saved_cuda_ver != current_cuda_ver:
                    logger.warning(
                        f"CUDA version changed: checkpoint={saved_cuda_ver}, current={current_cuda_ver}. "
                        f"GPU random operations may produce different sequences."
                    )

                # Unpack the numpy state if it was packed during save
                np_state = _unpack_np_state(rng_dict['np'])
                # Reconstruct tuple expected by _restore_rng_states
                rng_tuple = (rng_dict['py'], np_state, rng_dict['torch_cpu'], rng_dict.get('cuda'))
                success = _restore_rng_states(rng_tuple)

                # Check for critical failures and warn prominently
                critical_components = ['python', 'numpy', 'torch_cpu']
                critical_failed = [k for k in critical_components if success.get(k) is False]

                if critical_failed:
                    logger.error("=" * 60)
                    logger.error("CRITICAL: RNG state restoration failed for: %s", critical_failed)
                    logger.error("Mid-epoch resume will produce DIFFERENT data order!")
                    logger.error("Training may repeat or skip batches, affecting results.")
                    logger.error("=" * 60)
                    raise RuntimeError(
                        f"RNG state restoration failed for {critical_failed}. Mid-epoch resume would produce "
                        f"different data order, potentially repeating or skipping batches. "
                        f"To start fresh, set training.resume_from='none'."
                    )
                elif success.get('cuda') is False:
                    logger.warning("CUDA RNG restoration failed - GPU random operations may differ")

                logger.info(f"RNG state restoration results: {success}")
            except Exception as e:
                logger.error(f"Failed to restore RNG states: {type(e).__name__}: {e}")
                logger.error("Dataset shuffling WILL differ from original training run!")
                # Re-raise so the critical RNG-restore failure (and the validation
                # RuntimeErrors above) actually abort the resume. Previously this
                # broad except swallowed the RuntimeError raised just above for a
                # critical-component failure, defeating the entire guard and letting
                # the run silently proceed with the wrong data order.
                raise RuntimeError(
                    f"RNG state restoration failed ({type(e).__name__}: {e}). Mid-epoch "
                    f"resume would produce a different data order (repeating/skipping "
                    f"batches). To start fresh, set training.resume_from='none'."
                ) from e

        return {"state_dict": state_dict, **meta}
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        return self.best_checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint. Prefers crash-resume pointer."""
        last_path = self.checkpoint_dir / LAST_CKPT_NAME
        if last_path.exists():
            return last_path
        # Otherwise, refresh and choose newest by mtime
        self._refresh_checkpoint_list()
        existing = [p for p in self.checkpoints if p.exists()]
        return max(existing, key=lambda p: p.stat().st_mtime) if existing else None

    def peek_checkpoint_config(
        self,
        checkpoint_path: Union[str, Path],
        include_state_dict_keys: bool = False
    ) -> Union[Optional[Dict[str, Any]], Tuple[Optional[Dict[str, Any]], Optional[List[str]]]]:
        """Load only the config/metadata from a checkpoint without loading model weights.

        This enables fast architecture validation BEFORE attempting to load state_dict
        into the model, avoiding cryptic PyTorch errors on architecture mismatch.

        Args:
            checkpoint_path: Path to the checkpoint file.
            include_state_dict_keys: If True, also return the list of state dict keys
                                    for architecture detection from old checkpoints.

        Returns:
            If include_state_dict_keys is False:
                The config dict from the checkpoint, or None if not available.
            If include_state_dict_keys is True:
                A tuple of (config dict, list of state dict keys).
                Either or both may be None if not available.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            # Load checkpoint to CPU - we only need metadata, not model weights
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                # PyTorch < 1.13 doesn't support weights_only
                checkpoint = torch.load(path, map_location="cpu")
            except pickle.UnpicklingError:
                # PyTorch 2.6+ blocks custom classes with weights_only=True
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            if not isinstance(checkpoint, dict):
                if include_state_dict_keys:
                    return None, None
                return None

            config = checkpoint.get('config')

            if include_state_dict_keys:
                # Extract state dict keys for architecture detection
                state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
                state_dict_keys = list(state_dict.keys()) if state_dict else None
                return config, state_dict_keys

            return config
        except Exception as e:
            logger.warning(f"Could not peek checkpoint config from {path}: {e}")
            if include_state_dict_keys:
                return None, None
            return None

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        self._refresh_checkpoint_list()
        if not self.checkpoints:
            return None

        existing = [p for p in self.checkpoints if p.exists()]
        if not existing:
            return None

        # Prefer explicit crash-resume pointer first
        last_path = self.checkpoint_dir / LAST_CKPT_NAME
        candidates: List[Path] = []
        if last_path.exists():
            candidates.append(last_path)
        if existing:
            candidates.append(max(existing, key=lambda x: x.stat().st_mtime))

        for path in candidates:
            try:
                state_dict, meta = self._safe_load_checkpoint(path)
                return {"state_dict": state_dict, **meta}
            except (FileNotFoundError, ValueError, InvalidCheckpointError) as e:
                logger.warning("Failed to load latest checkpoint %s: %s", path, e)
                continue
        return None

    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint based on metric."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            try:
                state_dict, meta = self._safe_load_checkpoint(best_path)
                return {"state_dict": state_dict, **meta}
            except (FileNotFoundError, ValueError, InvalidCheckpointError) as e:
                logger.warning("Failed to load best checkpoint %s: %s", best_path, e)
        return None


class LearningRateSchedulerFactory:
    """Factory for creating learning rate schedulers"""
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str,
        num_epochs: int,
        steps_per_epoch: int = 0,
        warmup_epochs: int = 0,  # Deprecated, use warmup_steps
        warmup_steps: int = 0,
        min_lr: float = 1e-8,
        **kwargs
    ) -> _LRScheduler:
        """Create a learning rate scheduler
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler (cosine, linear, exponential, etc.)
            num_epochs: Total number of epochs (for epoch-based schedulers)
            steps_per_epoch: Number of steps per epoch (for step-based schedulers)
            warmup_epochs: Number of warmup epochs (deprecated, use warmup_steps)
            warmup_steps: Number of warmup steps
            min_lr: Minimum learning rate
            **kwargs: Additional scheduler-specific arguments
        """

        # For step-based schedulers
        if scheduler_type == 'cosine':
            return LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=num_epochs,
                warmup_start_lr=kwargs.get('warmup_start_lr', min_lr),
                eta_min=min_lr
            )

        elif scheduler_type == 'cosine_restarts':
            # Compute first_cycle_steps with fallback for when steps_per_epoch is 0
            first_cycle_steps = kwargs.get('first_cycle_steps', None)
            if first_cycle_steps is None or first_cycle_steps <= 0:
                if steps_per_epoch > 0 and num_epochs > 0:
                    first_cycle_steps = steps_per_epoch * num_epochs
                else:
                    raise ValueError(
                        f"cosine_restarts scheduler requires first_cycle_steps > 0, "
                        f"but got steps_per_epoch={steps_per_epoch}, num_epochs={num_epochs}"
                    )
            # Get max_lr from kwargs or optimizer defaults (with fallback)
            max_lr = kwargs.get('max_lr', None)
            if max_lr is None:
                try:
                    max_lr = optimizer.defaults.get('lr', optimizer.param_groups[0]['lr'])
                except (KeyError, IndexError):
                    raise ValueError(
                        "cosine_restarts scheduler requires max_lr or optimizer with lr set"
                    )
            return CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=first_cycle_steps,
                cycle_mult=kwargs.get('cycle_mult', 1.0),
                max_lr=max_lr,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                gamma=kwargs.get('gamma', 1.0)
            )
        
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        
        elif scheduler_type == 'multistep':
            return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=kwargs.get('milestones', [30, 60, 90]),
                gamma=kwargs.get('gamma', 0.1)
            )
        
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.1),
                patience=kwargs.get('patience', 10),
                threshold=kwargs.get('threshold', 0.0001),
                min_lr=min_lr
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class TrainingMetricsTracker:
    """Tracks and aggregates training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.epoch_metrics = defaultdict(list)
        self.global_step = 0
        self._lock = threading.RLock()  # Use RLock to allow recursive locking
        
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Update metrics"""
        with self._lock:
            if step is not None:
                self.global_step = step
            
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.metrics[key].append(value)
    
    def update_epoch(self, metrics: Dict[str, float]):
        """Update epoch-level metrics"""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.epoch_metrics[key].append(value)
    
    def get_average(self, metric_name: str) -> float:
        with self._lock:
            if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
                # Create a copy to avoid issues if deque is modified during mean calculation
                values = list(self.metrics[metric_name])
                return np.mean(values)
            return 0.0
    
    def get_last(self, metric_name: str) -> float:
        """Get last value of metric"""
        with self._lock:
            if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
                return self.metrics[metric_name][-1]
            return 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        with self._lock:
            summary = {}
            for key, values in self.metrics.items():
                if len(values) > 0:
                    # Create a copy to avoid issues during statistical calculations
                    values_copy = list(values)
                    summary[f'{key}_avg'] = np.mean(values_copy)
                    summary[f'{key}_std'] = np.std(values_copy)
                    summary[f'{key}_last'] = values_copy[-1]
            return summary
    
    def reset(self):
        """Reset metrics"""
        self.metrics.clear()
    
    def reset_epoch(self):
        """Reset epoch metrics"""
        self.epoch_metrics.clear()


class TrainingUtils:
    """Static utility functions for training"""
    
    @staticmethod
    def set_random_seed(seed: int, deterministic: bool = False):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'total_mb': total * 4 / 1024 / 1024,  # Assuming float32
            'trainable_mb': trainable * 4 / 1024 / 1024
        }
    
    @staticmethod
    def get_optimizer(
        model: nn.Module,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float = 0.01,
        **kwargs
    ) -> optim.Optimizer:
        """Create optimizer"""
        
        # Get parameters with weight decay exclusion
        params = TrainingUtils.get_parameter_groups(model, weight_decay)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(
                params,
                lr=learning_rate,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optimizer_type.lower() == 'adamw':
            # Standard PyTorch AdamW (32-bit, no external dependencies)
            # Use fused implementation if available (PyTorch 2.0+ on CUDA)
            extra_args = {}
            if torch.cuda.is_available() and 'fused' in optim.AdamW.__init__.__code__.co_varnames:
                 extra_args['fused'] = True

            return optim.AdamW(
                params,
                lr=learning_rate,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                **extra_args
            )

        elif optimizer_type.lower() == 'adamw8bit':
            # Memory-efficient 8-bit AdamW from bitsandbytes
            if bnb is None:
                raise ImportError(
                    "bitsandbytes is required for AdamW8bit optimizer. "
                    "Install it with: pip install bitsandbytes"
                )
            return bnb.optim.AdamW8bit(
                params,
                lr=learning_rate,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=weight_decay,
                block_wise=True,
            )
        
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(
                params,
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0.9),
                nesterov=kwargs.get('nesterov', True)
            )
        
        elif optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(
                params,
                lr=learning_rate,
                alpha=kwargs.get('alpha', 0.99),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optimizer_type.lower() == 'adan':
            from adan_optimizer import Adan
            return Adan(
                params,
                lr=learning_rate,
                betas=kwargs.get('betas', (0.98, 0.92, 0.99)),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=weight_decay,
                no_prox=kwargs.get('no_prox', False)
            )

        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    @staticmethod
    def get_cosine_scheduler(optimizer: optim.Optimizer, training_cfg) -> _LRScheduler:
        """Create CosineAnnealingWarmupRestarts scheduler from training config."""
        steps_per_epoch = getattr(training_cfg, "steps_per_epoch", 1)
        total_steps = steps_per_epoch * training_cfg.num_epochs
        warmup_epochs = int(getattr(training_cfg, "warmup_epochs", 5))
        warmup_steps = warmup_epochs * steps_per_epoch
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_steps,
            cycle_mult=1.0,
            max_lr=training_cfg.learning_rate,
            min_lr=getattr(training_cfg, "lr_end", 1e-6),
            warmup_steps=warmup_steps,
        )
    
    @staticmethod
    def get_parameter_groups(
        model: nn.Module,
        weight_decay: float = 0.01,
        layer_decay: Optional[float] = None
    ) -> List[Dict]:
        """Get parameter groups with proper weight decay and layer-wise learning rate decay"""
        
        # Parameters that should not have weight decay.
        # ViT convention (DeiT/MAE/timm): exclude position embeddings, special tokens,
        # and the patch projection in addition to bias/norm. Decaying pos_embed or
        # cls_token degrades the only token the head reads from.
        no_decay = ['bias', 'norm', 'pos_embed', 'cls_token', '_token', 'patch_embed']
        
        if layer_decay is None or layer_decay == 1.0:
            # Standard parameter groups
            params = [
                {
                    'params': [p for n, p in model.named_parameters() 
                              if not any(nd in n for nd in no_decay) and p.requires_grad],
                    'weight_decay': weight_decay
                },
                {
                    'params': [p for n, p in model.named_parameters() 
                              if any(nd in n for nd in no_decay) and p.requires_grad],
                    'weight_decay': 0.0
                }
            ]
        else:
            # Layer-wise learning rate decay
            params = TrainingUtils._get_layer_wise_params(model, weight_decay, layer_decay, no_decay)
        
        return params
    
    @staticmethod
    def _get_layer_wise_params(
        model: nn.Module,
        weight_decay: float,
        layer_decay: float,
        no_decay: List[str]
    ) -> List[Dict]:
        """Get parameters with layer-wise learning rate decay"""
        
        # Get depth of model
        def get_layer_id(name):
            if 'blocks' in name:
                # Extract layer number
                import re
                match = re.search(r'\.(\d+)\.', name)
                if match:
                    return int(match.group(1))
            return 0
        
        # Group parameters by layer
        layer_params = defaultdict(lambda: {'decay': [], 'no_decay': []})
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            layer_id = get_layer_id(name)
            
            if any(nd in name for nd in no_decay):
                layer_params[layer_id]['no_decay'].append(param)
            else:
                layer_params[layer_id]['decay'].append(param)
        
        # Create parameter groups with layer-wise decay
        max_layer = max(layer_params.keys()) if layer_params else 0
        params = []
        
        for layer_id in sorted(layer_params.keys()):
            layer_scale = layer_decay ** (max_layer - layer_id)
            
            if layer_params[layer_id]['decay']:
                params.append({
                    'params': layer_params[layer_id]['decay'],
                    'weight_decay': weight_decay,
                    'lr_scale': layer_scale
                })
            
            if layer_params[layer_id]['no_decay']:
                params.append({
                    'params': layer_params[layer_id]['no_decay'],
                    'weight_decay': 0.0,
                    'lr_scale': layer_scale
                })
        
        return params
    
    @staticmethod
    def compute_effective_batch_size(
        batch_size: int,
        accumulation_steps: int,
        world_size: int = 1
    ) -> int:
        """Compute effective batch size"""
        return batch_size * accumulation_steps * world_size
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in seconds to readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def save_training_config(config: Dict, output_dir: Path):
        """Save training configuration"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Saved training config to {config_path}")

    @staticmethod
    def worker_init_fn(worker_id: int):
        """
        Initializes the random number generators for each worker in a DataLoader.
        This ensures reproducibility across different workers.
        """
        # Use a unique seed for each worker based on the main process's seed
        # and the worker ID.
        worker_seed = torch.initial_seed() % (2**32 - 1) + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)
            torch.cuda.manual_seed_all(worker_seed)
        logger.debug(f"Worker {worker_id} initialized with seed {worker_seed}")


if __name__ == "__main__":
    # Test the utilities
    print("Testing Training Utilities...")
    
    # Test training state
    state = TrainingState()
    state.epoch = 5
    state.global_step = 1000
    state.update_metrics({'loss': 0.5, 'accuracy': 0.95})
    print(f"\nTraining State Summary:\n{state.get_summary()}")
    
    # Test checkpoint manager
    checkpoint_manager = CheckpointManager("./test_checkpoints")
    print(f"\nCheckpoint directory: {checkpoint_manager.checkpoint_dir}")
    
    # Test scheduler factory
    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = LearningRateSchedulerFactory.create_scheduler(
        optimizer,
        scheduler_type='cosine',
        num_epochs=100,
        steps_per_epoch=100,
        warmup_epochs=5
    )
    print(f"\nCreated scheduler: {type(scheduler).__name__}")
    
    # Test metrics tracker
    tracker = TrainingMetricsTracker()
    for i in range(10):
        tracker.update({'loss': 0.5 - i*0.01, 'accuracy': 0.8 + i*0.01})
    
    summary = tracker.get_summary()
    print(f"\nMetrics summary: {summary}")
    
    print("\n✓ All utilities tested successfully!")
