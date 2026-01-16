#!/usr/bin/env python3
"""
Training Utilities for Anime Image Tagger
Comprehensive training helpers including schedulers, checkpointing, and distributed training
"""

import os
import sys
import json
import gzip
import base64
import logging
import math
import shutil
import threading
import tempfile
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from contextlib import nullcontext
import hashlib
from datetime import datetime, timedelta
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
from torch.nn.parallel import DistributedDataParallel as DDP
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
from torch.amp import GradScaler, autocast
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
        import numpy as _np  # local import
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


def validate_config_compatibility(
    checkpoint_config: dict,
    current_config,
    strict: bool = False
) -> tuple[bool, list[str]]:
    """Validate that critical config parameters match between checkpoint and current config.

    Args:
        checkpoint_config: Config dict from checkpoint
        current_config: Current config (dict or object with attributes)
        strict: If True, raise exception on critical mismatches. If False, just warn.

    Returns:
        Tuple of (is_compatible, list of warning messages)

    Raises:
        ValueError: If strict=True and critical parameters don't match
    """
    if not checkpoint_config:
        return True, ["No config in checkpoint - skipping validation"]

    warning_messages = []
    errors = []

    # Critical parameters that MUST match (will cause training issues if different)
    critical_params = [
        ('model.num_labels', 'Vocabulary size', 'Model head size mismatch will cause crashes'),
        ('data.image_size', 'Image size', 'Images will be wrong size for model'),
        ('data.patch_size', 'Patch size', 'Patch embedding size mismatch'),
    ]

    # Important parameters that SHOULD match (may cause subtle issues)
    important_params = [
        ('data.normalize_mean', 'Normalization mean', 'Input normalization differs'),
        ('data.normalize_std', 'Normalization std', 'Input normalization differs'),
        ('training.gradient_accumulation_steps', 'Gradient accumulation', 'Effective batch size changed'),
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

    # Epoch tracking for proper resume semantics
    completed_epochs: int = 0
    is_epoch_boundary: bool = True
    batch_in_epoch: int = 0

    # Loss tracking
    train_loss: float = 0.0
    val_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    
    # Metric tracking
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    learning_rates: List[float] = field(default_factory=list)
    
    # Early stopping
    patience_counter: int = 0
    should_stop: bool = False
    
    # Gradient accumulation
    accumulation_steps: int = 0
    effective_batch_size: int = 0
    
    # Training time
    total_training_time: float = 0.0
    epoch_times: List[float] = field(default_factory=list)
    
    # Checkpoint info
    last_checkpoint_step: int = 0
    checkpoints_saved: List[str] = field(default_factory=list)
    
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
        """Load from dictionary"""
        return cls(**data)
    
    def get_summary(self) -> str:
        """Get training state summary"""
        summary = f"Epoch: {self.epoch}, Step: {self.global_step}\n"
        summary += f"Best Metric: {self.best_metric:.4f} (Epoch {self.best_epoch})\n"
        summary += f"Train Loss: {self.train_loss:.4f}, Val Loss: {self.val_loss:.4f}\n"
        summary += f"Patience: {self.patience_counter}, Should Stop: {self.should_stop}"
        return summary


class DistributedTrainingHelper:
    """Helper for distributed training setup and management"""
    
    def __init__(self, local_rank: int = -1, backend: str = 'nccl'):
        self.local_rank = local_rank
        self.backend = backend
        self.is_distributed = False
        self.world_size = 1
        self.rank = 0
        self.device = None
        
    def setup(self) -> torch.device:
        """Set up distributed training and return device for this process.

        Returns:
            torch.device: The GPU or CPU assigned to the current process.
        """
        if 'WORLD_SIZE' in os.environ:
            self.is_distributed = True
            
            # Get distributed parameters
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            
            if self.local_rank == -1:
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    init_method='env://'
                )
            
            # Setup device
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            
            logger.info(f"Distributed training: Rank {self.rank}/{self.world_size}, "
                       f"Local rank {self.local_rank}")
        else:
            # Single GPU or CPU training
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Single device training on {self.device}")
        
        return self.device
    
    def wrap_model(self, model: nn.Module, sync_bn: bool = True) -> nn.Module:
        """Wrap model for distributed training"""
        if self.is_distributed:
            # Convert BatchNorm to SyncBatchNorm
            if sync_bn and torch.cuda.is_available():
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
            # Wrap with DDP
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            logger.info("Model wrapped with DistributedDataParallel")
        
        return model
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_distributed:
            dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.rank == 0
    
    def barrier(self):
        """Synchronize all processes"""
        if self.is_distributed:
            dist.barrier()
    
    def reduce_tensor(self, tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
        """Reduce tensor across all processes"""
        if not self.is_distributed:
            return tensor
        
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        
        if average:
            rt /= self.world_size
        
        return rt
    
    def gather_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all processes"""
        if not self.is_distributed:
            return [tensor]
        
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        
        return gathered


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
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0
        
        self.base_max_lr = max_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            # Linear warmup
            return [(self.max_lr - self.min_lr) * self.step_in_cycle / self.warmup_steps + self.min_lr
                    for _ in self.base_lrs]
        else:
            # Cosine annealing
            return [self.min_lr + (self.max_lr - self.min_lr) * 
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                    (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for _ in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            self.step_in_cycle += 1
            
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
                self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
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

class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if mode == 'min' else -np.Inf
        
    def __call__(self, val_score: float, model: nn.Module = None) -> bool:
        """Check if should stop training
        
        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_score: float, model: nn.Module = None):
        """Saves model when validation score improves"""
        if self.verbose:
            if self.mode == 'min':
                delta = self.val_score_min - val_score
            else:
                delta = val_score - self.val_score_min
            logger.info(f'Validation score improved ({self.val_score_min:.6f} --> {val_score:.6f}, delta={delta:.6f})')

        # Update the tracked best score (val_score_min is misleading name for 'max' mode)
        self.val_score_min = val_score
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if self.mode == 'min' else -np.Inf


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
        while not self._shutdown_event.is_set():
            try:
                # Wait for work with timeout to check shutdown flag
                import queue
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is None:  # Shutdown signal
                    break

                checkpoint, path, lock_context, callback = item

                try:
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

                finally:
                    with self._lock:
                        self._pending_count -= 1
                    self._queue.task_done()

            except Exception as e:
                logger.error(f"AsyncCheckpointWriter worker error: {e}")

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
        try:
            self._queue.put_nowait((checkpoint, path, lock_context, callback))
            # Increment AFTER successful queue operation to avoid race condition
            with self._lock:
                self._pending_count += 1
            return True
        except queue.Full:
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

    def shutdown(self):
        """Shutdown async writer. Call at end of training."""
        if self._async_writer is not None:
            self._async_writer.shutdown(wait=True)
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
        train_loader: Optional['DataLoader'] = None
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

        missing_params = [p for p in required_params if get_param(config, p) is None]
        if missing_params:
            raise RuntimeError(
                f"Missing required preprocessing parameters in config: {', '.join(missing_params)}. "
                f"All of {required_params} must be explicitly provided to ensure correct preprocessing."
            )

        # Validate preprocessing parameter types and values
        try:
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
                'min_lr': getattr(scheduler, 'min_lr', None),
            }
        
        if config is not None:
            # Convert config to dict to avoid pickling Enum types (PyTorch 2.6+ weights_only=True compatibility)
            checkpoint['config'] = config.to_dict() if hasattr(config, 'to_dict') else config

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

        # Callback for async save completion logging
        def _on_save_complete(path, success, error):
            if success:
                logger.info(f"Async checkpoint saved to {path}")
            else:
                logger.error(f"Async checkpoint save failed for {path}: {error}")

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
                    else:
                        # Queue full, fall back to sync save
                        logger.warning("Async checkpoint queue full, falling back to sync save")
                        self._sync_save_checkpoint(checkpoint, checkpoint_path, lock_context)
                        wrote_numbered = True
                        self.checkpoints.append(checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to prepare async checkpoint: {e}. Falling back to sync save.")
                    self._sync_save_checkpoint(checkpoint, checkpoint_path, lock_context)
                    wrote_numbered = True
                    self.checkpoints.append(checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            else:
                # Sync save (original behavior)
                self._sync_save_checkpoint(checkpoint, checkpoint_path, lock_context)
                wrote_numbered = True
                self.checkpoints.append(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        else:
            logger.debug("save_best_only=True: skipping numbered checkpoint at step %s", step)

        # Always update last.pt atomically for crash-resume
        # Use sync save for last.pt to ensure crash recovery works
        if self.save_last:
            last_path = self.checkpoint_dir / LAST_CKPT_NAME
            try:
                self._sync_save_checkpoint(checkpoint, last_path, lock_context)
            except Exception as e:
                logger.warning("Failed to update %s: %s", last_path, e)
        
        # Save best model if applicable (use atomic copy via temp file with locking)
        if is_best and self.keep_best and wrote_numbered:
            best_path = self.checkpoint_dir / "best_model.pt"
            temp_best = None
            try:
                with lock_context:  # Use same lock for DDP safety
                    fd_best, temp_best = tempfile.mkstemp(suffix='.tmp', prefix='best_', dir=self.checkpoint_dir)
                    try:
                        os.close(fd_best)
                    except Exception:
                        pass

                    # Copy to temp, then atomic rename
                    shutil.copy2(checkpoint_path, temp_best)
                    os.replace(temp_best, best_path)
                    temp_best = None  # Mark as consumed
                    self.best_checkpoint = best_path
                    logger.info(f"Saved best model to {best_path}")
            except Exception as e:
                logger.warning(f"Failed to save best model to {best_path}: {e}")
            finally:
                # Clean up temp file if it wasn't consumed
                if temp_best is not None:
                    try:
                        if Path(temp_best).exists():
                            Path(temp_best).unlink()
                    except Exception:
                        pass
        
        # Manage checkpoint limit
        if wrote_numbered:
            self._cleanup_old_checkpoints()
        
        # Update training state
        if wrote_numbered:
            training_state.checkpoints_saved.append(str(checkpoint_path))
        else:
            training_state.checkpoints_saved.append(str(self.checkpoint_dir / LAST_CKPT_NAME))
        training_state.last_checkpoint_step = step
        
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
        device: torch.device = torch.device('cpu')
    ) -> Dict:
        """Load a checkpoint"""
        
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

            model.load_state_dict(state_dict)

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

                # Warn on critical scheduler param changes (what state_dict doesn't capture)
                saved_params = meta.get('scheduler_params', {})
                if saved_params:
                    param_warnings = []
                    for key in ['total_steps', 'warmup_steps', 'first_cycle_steps']:
                        saved_val = saved_params.get(key)
                        current_val = getattr(scheduler, key, None)
                        if saved_val is not None and current_val is not None and saved_val != current_val:
                            param_warnings.append(f"{key}: {saved_val} -> {current_val}")
                    if param_warnings:
                        logger.warning(
                            f"Scheduler parameters changed since checkpoint: {', '.join(param_warnings)}. "
                            f"LR schedule may not match original training run."
                        )

                scheduler.load_state_dict(saved_sched_state)
                logger.info(f"Scheduler state loaded successfully (type: {current_sched_type})")
            except Exception as e:
                logger.error(f"Failed to load scheduler state: {type(e).__name__}: {e}")
                raise RuntimeError(
                    f"Scheduler state restoration failed. LR schedule would restart from beginning. "
                    f"To start fresh, set training.resume_from='none'. Error: {e}"
                ) from e

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


class MixedPrecisionTrainer:
    """Helper for mixed precision training"""
    
    def __init__(
        self,
        use_amp: bool = True,
        amp_dtype: str = 'bfloat16',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        if use_amp and not torch.cuda.is_available():
            raise RuntimeError("bfloat16 AMP requested but CUDA is not available.")
        self.use_amp = use_amp and torch.cuda.is_available()

        # Enforce bfloat16 to avoid fp32/fp16 VRAM usage.
        self.amp_dtype = torch.bfloat16
        if self.use_amp:
            amp_dtype_norm = str(amp_dtype).lower()
            if amp_dtype_norm not in {"bfloat16", "bf16"}:
                raise ValueError(f"MixedPrecisionTrainer requires bfloat16, got '{amp_dtype}'.")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("bfloat16 AMP requested but CUDA device does not support bf16.")
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # GradScaler is only needed for float16 AMP, which we do not allow here.
        self.scaler = None
    
    def train_step(
        self,
        model: nn.Module,
        batch: Dict,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        accumulation_step: int
    ) -> Tuple[torch.Tensor, Dict]:
        """Execute a single training step with mixed precision
        
        Returns:
            Loss value and additional outputs
        """
        # Forward pass with autocast
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with autocast(device_type=device_type, enabled=self.use_amp, dtype=self.amp_dtype):
            outputs = model(batch['image'], labels=batch.get('labels'))
            
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = criterion(outputs, batch['labels'])
            
            # Scale loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping and optimizer step
        if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)
            
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
            if self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        return loss.item() * self.gradient_accumulation_steps, outputs
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing"""
        state = {
            'use_amp': self.use_amp,
            'amp_dtype': str(self.amp_dtype),
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_grad_norm': self.max_grad_norm
        }
        if self.scaler is not None:
            state['scaler_state'] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint"""
        if 'scaler_state' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler_state'])


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
                eps=kwargs.get('eps', 1e-8)
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
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_steps,
            cycle_mult=1.0,
            max_lr=training_cfg.learning_rate,
            min_lr=getattr(training_cfg, "lr_end", 1e-6),
            warmup_steps=training_cfg.warmup_steps,
        )
    
    @staticmethod
    def get_parameter_groups(
        model: nn.Module,
        weight_decay: float = 0.01,
        layer_decay: Optional[float] = None
    ) -> List[Dict]:
        """Get parameter groups with proper weight decay and layer-wise learning rate decay"""
        
        # Parameters that should not have weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'layer_norm', 'ln']
        
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
            if 'blocks' in name or 'layers' in name:
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
    
    # Test distributed helper
    dist_helper = DistributedTrainingHelper()
    device = dist_helper.setup()
    print(f"Device: {device}")
    print(f"Is distributed: {dist_helper.is_distributed}")
    
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
