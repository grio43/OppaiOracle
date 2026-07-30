#!/usr/bin/env python3
"""
Training script for the anime image tagger.
"""

import gc
import logging
import math
import os
import re

# Set CUDA allocator config to reduce memory fragmentation
# Must be set BEFORE any torch/CUDA imports
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import time
from pathlib import Path
from typing import Any, List, Optional, Tuple
import multiprocessing as mp
import sys
import queue
from datetime import datetime
from contextlib import nullcontext
import signal
import threading

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision
import numpy as np


from Monitor_log import MonitorConfig, TrainingMonitor
from evaluation_metrics import FrequencyBucketMetrics, ThresholdCalibrator
from asl_telemetry import ASLDriveManager

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

from Configuration_System import load_config, create_config_parser, FullConfig
from utils.logging_setup import setup_logging

# Paths will be loaded from the unified config in the main function.
logger = logging.getLogger(__name__)

# Periodic NaN/Inf check interval (steps). Configurable via NAN_CHECK_INTERVAL_STEPS env var.
# Default: 50 steps - balances early detection with minimal GPU sync overhead.
# Set to 0 to disable periodic checks (GradScaler still catches NaN gradients post-backward).
NAN_CHECK_EVERY_STEPS = int(os.getenv('NAN_CHECK_INTERVAL_STEPS', '50'))

# Candidate model-selection scalars, keyed by config.training.selection_metric.
# Each entry maps (val_f1_macro, val_f1_micro, val_mAP) -> the scalar that drives
# best-checkpoint selection and early stopping. The F1 variants are evaluated at
# a single frozen global threshold and are therefore calibration-coupled; val_mAP
# is threshold-free and is the default. See the selection_value comment in the
# epoch loop for why this matters to the ASL gamma_neg schedule.
_SELECTION_METRICS = {
    'val_mAP': lambda f1_macro, f1_micro, mAP: mAP,
    'val_f1_macro': lambda f1_macro, f1_micro, mAP: f1_macro,
    'val_f1_micro': lambda f1_macro, f1_micro, mAP: f1_micro,
}

# Import base modules with error handling
try:
    from dataset_loader import create_dataloaders
except ImportError as e:
    error_msg = (
        f"""MISSING REQUIRED FILE: dataset_loader.py
Please ensure dataset_loader.py exists in the current directory with create_dataloaders function.
Import error: {e}"""
    )
    raise ImportError(error_msg)

try:
    from model_architecture import create_model, initialize_tag_head_bias
except ImportError as e:
    error_msg = (
        f"""MISSING REQUIRED FILE: model_architecture.py
Please ensure model_architecture.py exists in the current directory with create_model function.
Import error: {e}"""
    )
    raise ImportError(error_msg)

# Import training utilities for checkpointing
from training_utils import (
    CheckpointManager,
    LAST_CKPT_NAME,
    TrainingState,
    setup_seed,
    CosineAnnealingWarmupRestarts,
    validate_config_compatibility,
)
from training_utils import VOCAB_PATH as DEFAULT_VOCAB_PATH
from vocabulary import create_vocabulary_from_datasets  # NEW: rebuild vocab each run
from dataset_loader import validate_dataset

try:
    from loss_functions import MultiTaskLoss, AsymmetricFocalLoss
except ImportError as e:
    error_msg = (
        f"""MISSING REQUIRED FILE: loss_functions.py
Please ensure loss_functions.py exists in the current directory with MultiTaskLoss and AsymmetricFocalLoss classes.
Import error: {e}"""
    )
    raise ImportError(error_msg)


def assert_finite(*tensors, names=None, batch=None, outputs=None, config=None):
    """Assert that all tensors are finite, with optional debugging hooks."""
    if names is None:
        names = [f"Tensor {i}" for i in range(len(tensors))]

    for name, t in zip(names, tensors):
        if t is not None and hasattr(t, 'dtype') and t.is_floating_point():
            if not torch.isfinite(t).all():
                # Non-finite value detected, attempt to perform debug actions
                if config and hasattr(config, 'debug') and config.debug.enabled:
                    logger.error(f"Non-finite detected in '{name}'. Debug mode enabled, attempting to save context.")

                    # Log batch info if available and enabled
                    if config.debug.log_batch_info_on_error and batch:
                        # Log available metadata, avoiding large tensors
                        batch_info = {k: v for k, v in batch.items() if not isinstance(v, torch.Tensor) or v.numel() < 10}
                        logger.error(f"Problematic batch info: {batch_info}")

                    # Dump tensors to file if available and enabled
                    if config.debug.dump_tensors_on_error and batch and outputs:
                        dump_dir = Path(config.output_root) / config.experiment_name / "debug_dumps"
                        dump_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        dump_path = dump_dir / f"non_finite_dump_{name}_{timestamp}.pt"

                        dump_data = {
                            'failed_tensor_name': name,
                            'batch': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                            'outputs': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                        }

                        try:
                            torch.save(dump_data, dump_path)
                            logger.error(f"Saved debug tensors for failed tensor '{name}' to: {dump_path}")
                        except Exception as e:
                            logger.error(f"Failed to save debug tensor dump: {e}")

                # Always raise the error to halt training
                raise RuntimeError(f"Non-finite detected in {name}")


def _loss_to_float(value: Any) -> float:
    """Convert a loss value to a Python float, safely handling torch tensors."""
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            value = value.mean()
        return float(value.cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            value = value.mean()
        return float(value)
    return float(value)


def _compute_class_weights(
    vocab,
    strategy: str,
    clip_min: float = 0.05,
    clip_max: float = 5.0,
    beta: float = 0.9999,
) -> list:
    """Compute per-class weights from tag frequencies.

    Args:
        vocab: TagVocabulary with tag_frequencies and index_to_tag.
        strategy: Weighting strategy ("inverse_sqrt" or "effective_number").
        clip_min: Floor for weights (prevents near-zero).
        clip_max: Cap for weights (prevents extreme values).
        beta: Beta parameter for effective_number strategy (Cui et al. 2019).

    Returns:
        List of floats, length = vocab size, suitable for AsymmetricFocalLoss.
    """
    import math

    vocab_size = len(vocab.tag_to_index)
    freqs = vocab.tag_frequencies
    raw_weights = []

    for idx in range(vocab_size):
        tag = vocab.index_to_tag.get(idx, "")
        # PAD (0) and UNK (1) get zero weight — already excluded by ignore_indices
        if idx <= 1:
            raw_weights.append(0.0)
            continue
        freq = freqs.get(tag, 0)
        if freq <= 0:
            raw_weights.append(1.0)
            continue

        if strategy == "inverse_sqrt":
            raw_weights.append(1.0 / math.sqrt(freq))
        elif strategy == "effective_number":
            # Cui et al. 2019: w = (1-beta) / (1-beta^freq)
            if freq > 1000:
                # Log-space for large freq to avoid float overflow
                log_term = freq * math.log(beta)
                effective_n = (1.0 - math.exp(log_term)) / (1.0 - beta)
            else:
                effective_n = (1.0 - beta ** freq) / (1.0 - beta)
            raw_weights.append(1.0 / effective_n)
        else:
            raise ValueError(f"Unknown class_weight_strategy: {strategy}")

    # Normalize non-special weights to mean=1.0
    active_weights = [w for w in raw_weights if w > 0]
    if active_weights:
        mean_w = sum(active_weights) / len(active_weights)
        if mean_w > 0:
            raw_weights = [w / mean_w if w > 0 else 0.0 for w in raw_weights]

    # Log pre-clip distribution percentiles
    active_pre_clip = sorted(w for w in raw_weights if w > 0)
    if active_pre_clip:
        n = len(active_pre_clip)
        logger.info(
            f"Class weights ({strategy}) pre-clip: "
            f"p1={active_pre_clip[n//100]:.4f} p10={active_pre_clip[n//10]:.4f} "
            f"p50={active_pre_clip[n//2]:.4f} p90={active_pre_clip[9*n//10]:.4f} "
            f"p99={active_pre_clip[99*n//100]:.4f}"
        )

    # Clip active weights
    weights = []
    for w in raw_weights:
        if w > 0:
            weights.append(max(clip_min, min(clip_max, w)))
        else:
            weights.append(0.0)

    active = [w for w in weights if w > 0]
    if active:
        logger.info(
            f"Computed class weights (strategy={strategy}): "
            f"min={min(active):.4f}, max={max(active):.4f}, "
            f"mean={sum(active)/len(active):.4f}, active_tags={len(active)}/{vocab_size}"
        )
    else:
        logger.warning(f"No active class weights computed (vocab_size={vocab_size})")
    return weights


def train_with_orientation_tracking(config: FullConfig):
    """Main training loop."""

    import tempfile
    from utils.memory_monitor import MemoryMonitor

    logger = logging.getLogger(__name__)

    def _normalize_experiment_name(name: str, arch: str) -> Tuple[str, bool]:
        if not name:
            return arch, False
        tokens = [t for t in re.split(r"[_-]+", name) if t]
        had_arch_token = any(t.lower() == "vit" for t in tokens)
        sep = "-" if "-" in name and "_" not in name else "_"
        base_tokens = [t for t in tokens if t.lower() != "vit"]
        base = sep.join(base_tokens)
        resolved = arch if not base else f"{base}{sep}{arch}"
        return resolved, had_arch_token

    architecture_type = str(getattr(config.model, 'architecture_type', 'vit') or 'vit').lower()
    config.model.architecture_type = architecture_type
    base_experiment = str(getattr(config, 'experiment_name', '') or 'experiment')
    resolved_experiment, had_arch_token = _normalize_experiment_name(base_experiment, architecture_type)
    legacy_experiment = None
    if resolved_experiment != base_experiment:
        logger.info("Adjusted experiment_name for architecture: %s -> %s", base_experiment, resolved_experiment)
        if not had_arch_token:
            legacy_experiment = base_experiment
    config.experiment_name = resolved_experiment
    
    # --- Soft stop support (signals + sentinel files) -----------------------
    # Save a checkpoint at the next safe point (optimizer step) and exit.
    # Use threading.Event for signal-safe flag (CR-033)
    soft_stop_event = threading.Event()
    # Escalation state owned EXCLUSIVELY by the signal handler. It must not be
    # soft_stop_event: the STOP_TRAINING sentinel poll also sets that event, so
    # reusing it would make the operator's FIRST Ctrl+C look like a second signal
    # whenever a sentinel stop was already in flight - hard-aborting the run (and
    # possibly a torch.save in progress) instead of letting it checkpoint.
    _signal_seen = threading.Event()

    def _soft_stop_handler(signum, frame):
        """Signal-safe handler - only sets an atomic event (first signal).

        IMPORTANT: Do NOT use logging or any non-reentrant functions here.
        Signal handlers can deadlock if they try to acquire locks held by
        the interrupted code. Only atomic operations are safe.

        A SECOND signal escalates to a hard abort. soft_stop_event is only polled
        inside the training step loop, so between the first signal and the next
        safe point the process can be uninterruptible for minutes (Arrow/metadata
        cache build, torch.compile, a full validation pass). Without escalation the
        operator has no way out short of SIGKILL, which loses the checkpoint.
        """
        if _signal_seen.is_set():
            # Restore the default dispositions first so a third signal can never
            # re-enter this handler, then raise into the interrupted frame (this is
            # exactly what signal.default_int_handler does) so `finally` blocks and
            # the checkpoint-writer shutdown still run.
            try:
                signal.signal(signal.SIGINT, signal.default_int_handler)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
            except Exception:
                pass
            # os.write on fd 2 is async-signal-safe (unlike logging, which takes locks)
            try:
                os.write(2, b"\nSecond stop signal received - aborting immediately.\n")
            except Exception:
                pass
            raise KeyboardInterrupt
        _signal_seen.set()
        soft_stop_event.set()
        # Write to stderr is relatively safe (no locks in Python's signal handling)
        # but even this should be minimal. The actual message will be logged
        # when the training loop checks soft_stop_event.
        try:
            os.write(2, b"\nSoft stop requested - will checkpoint at the next safe point. "
                        b"Signal again to abort immediately.\n")
        except Exception:
            pass

    try:
        signal.signal(signal.SIGINT, _soft_stop_handler)
        signal.signal(signal.SIGTERM, _soft_stop_handler)
    except Exception as _e:
        logger.debug("Signal handler install skipped: %s", _e)

    # --- Manual TensorBoard image logging hotkey (press 'i' to log images) ---
    def _keyboard_listener(log_dir: Path, stop_event: threading.Event):
        """Background thread that listens for hotkey presses.

        Press 'i' to trigger immediate TensorBoard image logging.
        This creates a sentinel file that the training loop checks.
        Non-blocking and won't interrupt training.

        Alternative: manually create LOG_IMAGES_NOW file in log_dir.
        """
        image_sentinel = log_dir / "LOG_IMAGES_NOW"

        # Try Windows-specific keyboard handling
        try:
            import msvcrt  # Windows-only
            while not stop_event.is_set():
                try:
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                        if key == 'i':
                            image_sentinel.touch()
                            print(f"\n[Hotkey] 'i' pressed - will log images to TensorBoard at next step...")
                except Exception:
                    pass  # Ignore individual keypress errors
                stop_event.wait(0.1)  # Check every 100ms
            return
        except ImportError:
            pass  # Not on Windows, try Unix approach

        # Unix/Linux: use select-based approach
        # Note: requires terminal in raw mode for immediate response
        try:
            import sys
            import select
            import tty
            import termios

            # Check if stdin is a real terminal
            if not sys.stdin.isatty():
                return  # stdin redirected, can't read keys

            # Save terminal settings and set raw mode for immediate keypress
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())  # cbreak mode: immediate input, no echo
                while not stop_event.is_set():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
                        if key == 'i':
                            image_sentinel.touch()
                            print(f"\n[Hotkey] 'i' pressed - will log images to TensorBoard at next step...")
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            pass  # Keyboard monitoring not available (e.g., no terminal)

    # Start keyboard listener as daemon thread (won't block exit)
    _kb_thread = threading.Thread(
        target=_keyboard_listener,
        args=(Path(config.log_dir), soft_stop_event),
        daemon=True,
        name="KeyboardListener"
    )
    _kb_thread.start()
    logger.info("Keyboard hotkey listener started: press 'i' to log images to TensorBoard")

    # Seeding & determinism
    seed, deterministic_mode = setup_seed(config.training.seed, config.training.deterministic)

    use_anomaly = (
        getattr(config.training, "enable_anomaly_detection", False)
        or getattr(config.debug, "detect_anomaly", False)
    )
    anomaly_ctx = torch.autograd.detect_anomaly(check_nan=True) if use_anomaly else nullcontext()

    try:
        torch.use_deterministic_algorithms(deterministic_mode)
    except Exception:
        pass

    # Allow cuDNN to pick the fastest kernels when not in strict-deterministic mode
    torch.backends.cudnn.benchmark = bool(getattr(config.training, "benchmark", True))
    
    # Enable TensorFloat-32 (TF32) for massive speedup on Ampere+ GPUs
    # This uses Tensor Cores for FP32 matmuls (Linear layers) and convolutions
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TensorFloat-32 (TF32) enabled for matmul and cuDNN")

    # Ensure log_dir exists early for sentinel files and diagnostics
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    # Ensure output_root exists for tensorboard and checkpoints
    Path(config.output_root).mkdir(parents=True, exist_ok=True)

    # Find the active data path from storage_locations
    active_location = next((loc for loc in config.data.storage_locations if loc.get('enabled')), None)

    if not active_location:
        error_msg = (
            "No enabled storage location found in your configuration.\n"
            "Please ensure you are providing a configuration file using the --config argument, for example:\n"
            "  python train_direct.py --config configs/unified_config.yaml\n"
            "And that your configuration file has an enabled entry under data.storage_locations."
        )
        raise ValueError(error_msg)

    active_data_path = Path(active_location['path'])

    # Validate path exists and is accessible (CR-036)
    if not active_data_path.exists():
        raise FileNotFoundError(
            f"Active data path does not exist: {active_data_path}\n"
            f"Please ensure the path in your configuration is correct and the storage is mounted."
        )

    if not active_data_path.is_dir():
        raise NotADirectoryError(
            f"Active data path is not a directory: {active_data_path}\n"
            f"Expected a directory containing training images and annotations."
        )

    if not os.access(active_data_path, os.R_OK):
        raise PermissionError(
            f"Active data path is not readable: {active_data_path}\n"
            f"Please check file permissions."
        )

    logger.info(f"Using active data path: {active_data_path} (validated)")

    # --- Prompt to (re)build vocabulary at startup ---------------------------------
    # Decide where the vocabulary should live and whether we already have one
    vocab_dest = Path(getattr(config, "vocab_path", str(DEFAULT_VOCAB_PATH)))
    check_path = vocab_dest / "vocabulary.json" if vocab_dest.is_dir() else vocab_dest
    has_vocab = check_path.exists()

    def _ask_yes_no(prompt: str, default: Optional[bool]) -> bool:
        """
        Simple Y/N prompt. If not attached to a TTY, fall back to the default.
        default=True  -> [Y/n]
        default=False -> [y/N]
        default=None  -> [y/n]

        Environment variables (CR-001):
        - OO_AUTO_REBUILD_VOCAB: "1", "true", "yes" for auto-rebuild
        - OO_NON_INTERACTIVE: "1" to force non-interactive mode
        """
        # Check environment variable override (CR-001 fix)
        env_override = os.environ.get("OO_AUTO_REBUILD_VOCAB")
        if env_override:
            result = env_override.lower() in ("1", "true", "yes")
            logger.info(f"Using environment variable OO_AUTO_REBUILD_VOCAB={env_override} -> {result}")
            return result

        # Force non-interactive mode via environment variable
        if os.environ.get("OO_NON_INTERACTIVE", "").lower() in ("1", "true", "yes"):
            result = bool(default) if default is not None else False
            logger.info(f"Non-interactive mode via OO_NON_INTERACTIVE -> using default: {result}")
            return result

        # Non-interactive (e.g., piped/cron) -> use default
        if not sys.stdin or not sys.stdin.isatty():
            result = bool(default) if default is not None else False
            logger.info(f"Non-TTY detected -> using default: {result}")
            return result

        choices = " [Y/n] " if default is True else (" [y/N] " if default is False else " [y/n] ")
        ans = input(prompt + choices).strip().lower()
        if ans in ("y", "yes"): return True
        if ans in ("n", "no"):  return False
        return bool(default) if default is not None else False

    # Default choice: build if missing, otherwise skip
    rebuild = _ask_yes_no(
        "Build a new tag vocabulary from dataset JSONs?",
        default=(not has_vocab)
    )

    # If the user declines but there is no vocabulary, build anyway to avoid crash
    if not has_vocab and not rebuild:
        logger.warning(
            "No vocabulary found at %s but rebuild was declined; "
            "building now to avoid a startup failure.", check_path
        )
        rebuild = True

    if rebuild:
        try:
            logger.info("Rebuilding tag vocabulary from dataset at %s", active_data_path)
            # Scans recursively for *.json sidecars
            rebuilt_vocab = create_vocabulary_from_datasets(
                [active_data_path],
                min_frequency=getattr(config.data, 'vocab_min_frequency', 125)
            )
            vocab_file = (vocab_dest / "vocabulary.json") if vocab_dest.is_dir() else vocab_dest
            rebuilt_vocab.save_vocabulary(vocab_file)
            logger.info("Vocabulary rebuilt with %d tags -> %s",
                        len(rebuilt_vocab.tag_to_index), vocab_file)
        except Exception as e:
            logger.error("Failed to (re)build vocabulary: %s", e)
            raise
    else:
        logger.info("Using existing vocabulary at %s", check_path)
    # -------------------------------------------------------------------------------

    stats_queue = mp.Queue(maxsize=1000) if config.training.use_tensorboard else None
    device = torch.device(config.training.device)
    device_type = device.type

    # Expose stats queue to dataloaders for optional telemetry
    config.data.stats_queue = stats_queue

    train_loader, val_loader, vocab = create_dataloaders(
        data_config=config.data,
        validation_config=config.validation,
        vocab_path=Path(config.vocab_path),
        active_data_path=active_data_path,
        seed=seed,
        debug_config=config.debug,
        architecture_type=config.model.architecture_type,
        patch_size=getattr(config.model, 'patch_size', None),
    )

    # NOTE: in-train validation subsampling is controlled by data.max_val_samples
    # (applied inside create_dataloaders). validation.max_samples is only consumed
    # by the standalone validation_loop runner.

    # Set initial epoch before any DataLoader access (prevents worker spawn warnings)
    if hasattr(train_loader.dataset, 'set_epoch'):
        train_loader.dataset.set_epoch(0)
    if val_loader is not None and hasattr(val_loader.dataset, 'set_epoch'):
        val_loader.dataset.set_epoch(0)

    # Pre-training validation
    if getattr(config.debug, 'validate_input_data', False):
        logger.info("Starting pre-training input validation...")
        # NOTE: validate_dataset() is currently a no-op placeholder; these calls
        # do NOT actually validate inputs/labels (it warns to that effect).
        validate_dataset(train_loader, vocab, config, num_batches_to_check=10)
        validate_dataset(val_loader, vocab, config, num_batches_to_check=5)

    num_tags = len(vocab.tag_to_index)

    # Sync config.model.num_labels with actual vocabulary size
    # This is deferred until after vocabulary is loaded (config validation allows 0)
    if config.model.num_labels == 0:
        config.model.num_labels = num_tags
        logger.debug(f"Set config.model.num_labels to vocabulary size: {num_tags}")
    elif config.model.num_labels != num_tags:
        raise ValueError(
            f"CRITICAL: config.model.num_labels ({config.model.num_labels}) does not match "
            f"vocabulary size ({num_tags}). This mismatch will cause model architecture errors. "
            f"Please update your config or use a compatible vocabulary/checkpoint."
        )

    logger.info(f"Creating model with {num_tags} tags (including rating tags)")

    # Determine architecture type (only ViT is supported)
    architecture_type = getattr(config.model, 'architecture_type', 'vit')
    logger.info(f"Creating model with architecture: {architecture_type}")

    model_config = config.model.to_dict()
    model_config["num_tags"] = num_tags
    # Bridge the user-facing dropout knob to the architecture field name. The YAML
    # exposes model.hidden_dropout_prob, but VisionTransformerConfig's MLP/projection
    # dropout field is named `dropout`; without this remap the YAML value is stripped
    # below and MLP dropout silently stays at the dataclass default (0.1).
    model_config["dropout"] = model_config.get(
        "hidden_dropout_prob", model_config.get("dropout", 0.1)
    )

    # Filter out config keys that are in unified_config.yaml but not used by VisionTransformerConfig.
    # These remain in the YAML purely for legacy reasons (grouped-prediction prototype was
    # never shipped, and architecture_type is kept as a forward-compat selector); strip them
    # so create_model() doesn't choke on unknown kwargs.
    _unused_config_keys = {
        'architecture_type',
        'hidden_dropout_prob', 'initializer_range', 'num_groups', 'num_labels',
        'tags_per_group', 'swin_config'
    }
    model_config = {k: v for k, v in model_config.items() if k not in _unused_config_keys}

    model = create_model(**model_config)

    # Initialize tag_head bias with log-prior (RetinaNet technique).
    # Checkpoint resume will overwrite this via load_state_dict.
    initialize_tag_head_bias(
        model,
        index_to_tag=vocab.index_to_tag,
        tag_frequencies=vocab.tag_frequencies,
        total_samples=len(train_loader.dataset),
    )

    # Move model to device first, then apply dtype conversion
    # NOTE: channels_last memory format is applied LATER (after checkpoint loading and dtype conversion)
    # to ensure it's not lost during transformations. See the channels_last application block below.
    model.to(device)

    # Sync monitor config from training config (training.* is the single source of truth)
    if not hasattr(config, 'monitor'):
        # In case the config file is old and doesn't have a monitor section
        config.monitor = MonitorConfig()

    config.monitor.log_dir = config.log_dir
    config.monitor.use_tensorboard = config.training.use_tensorboard
    # Only set a default if not provided in config
    if not getattr(config.monitor, "tensorboard_dir", None):
        config.monitor.tensorboard_dir = str(Path(config.output_root) / config.experiment_name)
    # Ensure tensorboard directory exists before initializing monitor
    Path(config.monitor.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    config.monitor.normalize_mean = tuple(getattr(config.data, 'normalize_mean', (0.5, 0.5, 0.5)))
    config.monitor.normalize_std = tuple(getattr(config.data, 'normalize_std', (0.5, 0.5, 0.5)))

    monitor = TrainingMonitor(config.monitor)

    # --- TensorBoard: initial hparams snapshot ---
    try:
        to_dict = getattr(config, "to_dict", None)
        hparams = to_dict() if callable(to_dict) else (
            vars(config) if hasattr(config, "__dict__") else {}
        )
        monitor.log_hyperparameters(hparams, {"init/placeholder": 0})
    except Exception:
        pass

    # Log loss hyperparameters
    tag_loss_cfg = config.training.tag_loss
    loss_hparams = {
        "tag_loss": tag_loss_cfg.to_dict() if hasattr(tag_loss_cfg, "to_dict") else vars(tag_loss_cfg),
    }
    logger.info(f"Loss hyperparameters: {loss_hparams}")
    try:
        monitor.log_hyperparameters(loss_hparams, {"loss/init": 0})
    except Exception:
        pass

    # NOTE: TensorBoard model graph logging is deferred until AFTER torch.compile()
    # to avoid stride mismatch issues. See the model graph logging block below.

    # Detect and log Flex Attention configuration
    if config.model.use_flex_attention:
        logger.info("=" * 70)
        logger.info("Flex Attention Configuration:")
        logger.info(f"  PyTorch version: {torch.__version__}")

        # Check Flex Attention availability
        if hasattr(torch.nn.attention, 'flex_attention'):
            from torch.nn.attention.flex_attention import flex_attention
            logger.info("  Status: Flex Attention - AVAILABLE")
            logger.info(f"  Block size: {getattr(config.model, 'flex_block_size', 128)}")

            # Quick test of Flex Attention kernel
            try:
                with torch.no_grad():
                    test_q = torch.randn(1, 1, 16, 64, device=device, dtype=torch.bfloat16)
                    _ = flex_attention(test_q, test_q, test_q)
                    logger.info("  Test: Flex Attention kernel - WORKING")
            except Exception as e:
                logger.error(f"  Test: Flex Attention - FAILED: {e}")
        else:
            logger.error("  Flex Attention not available - requires PyTorch 2.5+")
            raise RuntimeError("Flex Attention requires PyTorch 2.5 or newer")

        logger.info("  Note: Flex Attention benefits from torch.compile() - kernel fusion enabled")
        logger.info("=" * 70)

    # torch.compile() optimization (PyTorch 2.0+)
    # Provides 15-35% speedup through graph optimization and kernel fusion
    # NOTE: Actual compilation is DEFERRED until after checkpoint loading to ensure
    # inductor kernels are compiled for the correct weight strides (channels_last).
    # Loading a checkpoint can reset tensor strides, causing stride mismatch errors.
    use_compile = getattr(config.training, "use_compile", False)
    if use_compile:
        # Check if Triton is available (required for torch.compile with inductor backend)
        try:
            import triton
            logger.info(f"Triton {triton.__version__} available for torch.compile")
        except ImportError:
            logger.warning("torch.compile() requires Triton but it's not installed")
            logger.warning("Install with: pip install triton-windows (Windows) or pip install triton (Linux)")
            logger.warning("Training will proceed without compilation - expect ~15-35% slower training")
            use_compile = False

    # Save compile settings - actual compilation happens after checkpoint loading
    compile_mode = getattr(config.training, "compile_mode", "default") if use_compile else None
    compile_fullgraph = getattr(config.training, "compile_fullgraph", False) if use_compile else None
    compile_dynamic = getattr(config.training, "compile_dynamic", True) if use_compile else None

    if not use_compile:
        logger.info("torch.compile() disabled (use_compile=false in config)")

    # Compute class weights from tag frequencies if configured
    class_weights = tag_loss_cfg.class_weights  # Manual override takes priority
    if class_weights is None and tag_loss_cfg.class_weight_strategy is not None:
        class_weights = _compute_class_weights(
            vocab=vocab,
            strategy=tag_loss_cfg.class_weight_strategy,
            clip_min=tag_loss_cfg.class_weight_clip_min,
            clip_max=tag_loss_cfg.class_weight_clip_max,
            beta=tag_loss_cfg.class_weight_beta,
        )

    criterion = MultiTaskLoss(
        tag_loss_fn=AsymmetricFocalLoss(
            alpha=tag_loss_cfg.alpha,
            clip=tag_loss_cfg.clip,
            gamma_neg=tag_loss_cfg.gamma_neg,
            gamma_pos=tag_loss_cfg.gamma_pos,
            label_smoothing=tag_loss_cfg.label_smoothing,
            ignore_indices=[0, 1],  # Ignore <PAD> (0) and <UNK> (1) for tags
            class_weights=class_weights,
        ),
    )
    criterion = criterion.to(device)
    from training_utils import TrainingUtils
    # Construct betas tuple based on the selected optimizer
    betas = (config.training.adam_beta1, config.training.adam_beta2)
    if config.training.optimizer == 'adan':
        beta3 = getattr(config.training, 'adan_beta3', 0.99)
        betas = betas + (beta3,)

    base_lr = config.training.learning_rate

    # Scale learning rate based on effective batch size
    from training_config import scale_learning_rate
    lr_scaling_mode = getattr(config.training, 'lr_scaling_mode', 'sqrt')
    lr_base_batch_size = int(getattr(config.training, 'lr_base_batch_size', 256))
    effective_batch_size = config.data.batch_size * int(getattr(config.training, 'gradient_accumulation_steps', 1))
    effective_learning_rate = scale_learning_rate(
        base_lr=base_lr,
        effective_batch_size=effective_batch_size,
        base_batch_size=lr_base_batch_size,
        mode=lr_scaling_mode,
    )
    logger.info(
        f"LR scaling: base {base_lr:.2e} × {lr_scaling_mode}({effective_batch_size}/{lr_base_batch_size}) "
        f"= {effective_learning_rate:.2e}"
    )

    optimizer = TrainingUtils.get_optimizer(
        model,
        optimizer_type=config.training.optimizer,
        learning_rate=effective_learning_rate,
        weight_decay=config.training.weight_decay,
        betas=betas,
        eps=config.training.adam_epsilon
    )

    # ---- LR scheduler: STEP-BASED semantics ----
    # Interpret warmup / cycle lengths in optimizer updates (not epochs).

    # Validate gradient accumulation steps (CR-038)
    try:
        accum_raw = getattr(config.training, "gradient_accumulation_steps", 1)
        accum = int(accum_raw)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid gradient_accumulation_steps in config: {accum_raw!r}. "
            f"Must be a positive integer. Error: {e}"
        )

    if accum < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, got {accum}. "
            f"Use 1 to disable gradient accumulation."
        )

    # Warn if accumulation is suspiciously high
    batch_size = config.data.batch_size
    if accum > batch_size:
        logger.warning(
            f"gradient_accumulation_steps ({accum}) > batch_size ({batch_size}). "
            f"This is unusual and may indicate a configuration error."
        )

    # Validate num_epochs
    try:
        num_epochs = int(getattr(config.training, "num_epochs", 1))
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid num_epochs in config: must be a positive integer. Error: {e}"
        )

    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

    steps_per_epoch = max(1, len(train_loader))
    updates_per_epoch = (steps_per_epoch + accum - 1) // accum  # ceil division
    total_updates = num_epochs * updates_per_epoch

    logger.info(
        f"Scheduler setup: {num_epochs} epochs, {steps_per_epoch} steps/epoch, "
        f"{accum}x gradient accumulation = {updates_per_epoch} optimizer updates/epoch "
        f"({total_updates} total updates)"
    )
    warmup_epochs = int(getattr(config.training, "warmup_epochs", 5))
    warmup_steps = warmup_epochs * updates_per_epoch
    logger.info(f"Warmup: {warmup_epochs} epochs = {warmup_steps} optimizer updates")
    num_cycles = int(getattr(config.training, "num_cycles", 1))
    cycle_decay = float(getattr(config.training, "cycle_decay", 0.9))

    # For multiple cycles, first_cycle_steps = total_updates / num_cycles
    # This ensures each cycle is roughly equal in length
    first_cycle_steps = total_updates // num_cycles if num_cycles > 1 else total_updates

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=1.0,  # Equal cycle lengths
        max_lr=effective_learning_rate,  # Scaled base learning rate
        min_lr=getattr(config.training, "lr_end", 1e-6),
        warmup_steps=warmup_steps,
        gamma=cycle_decay,  # Decay max_lr by this factor after each restart
    )

    if num_cycles > 1:
        logger.info(
            f"SGDR scheduler: {num_cycles} cycles of ~{first_cycle_steps} steps each, "
            f"gamma={cycle_decay} (LR decays by {(1-cycle_decay)*100:.0f}% per restart)"
        )

    amp_enabled = bool(config.training.use_amp) and device_type == 'cuda'
    amp_dtype_name = str(getattr(config.training, "amp_dtype", "bfloat16")).lower()
    if config.training.use_amp:
        if device_type != 'cuda':
            raise RuntimeError("bfloat16 AMP requested but CUDA device is not available.")
        if amp_dtype_name not in {"bfloat16", "bf16"}:
            raise ValueError(f"Only bfloat16 AMP is supported, got '{amp_dtype_name}'.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 AMP requested but CUDA device does not support bf16.")
    # Master weights stay in fp32 even though AMP runs bf16 — bf16 has only 7 mantissa
    # bits, so storing master weights in bf16 makes the smallest representable update at
    # |w|=1 ≈ 2^-7 (~7.8e-3); with lr~5e-4 and typical gradients, optimizer updates of
    # ~5e-7 would silently round to zero. Autocast handles bf16 forward/backward without
    # requiring bf16 storage.
    amp_dtype = torch.bfloat16

    # Provide an autocast wrapper compatible with both torch.amp and torch.cuda.amp
    # Older PyTorch versions do not accept the 'device_type' argument.
    from contextlib import contextmanager
    try:
        # Probe signature (do not enter context)
        _probe_ctx = autocast(device_type=device_type, enabled=False, dtype=amp_dtype)
        def amp_autocast():
            return autocast(device_type=device_type, enabled=amp_enabled, dtype=amp_dtype)
    except TypeError:  # Older API without device_type
        try:
            from torch.cuda.amp import autocast as cuda_autocast  # type: ignore
            def amp_autocast():
                return cuda_autocast(enabled=amp_enabled, dtype=amp_dtype)
        except Exception:
            @contextmanager
            def amp_autocast():
                yield

    # GradScaler is only needed for float16 AMP on Volta+ GPUs (compute capability >= 7)
    # Note: bfloat16 does NOT require GradScaler because it has a much larger dynamic range
    # (8 exponent bits vs float16's 5), which naturally avoids the underflow/overflow issues
    # that GradScaler addresses for float16. When using bfloat16, enabled=False is intentional.
    use_scaler = False
    if amp_enabled and amp_dtype == torch.float16:
        if torch.cuda.is_available():
            try:
                capability = torch.cuda.get_device_capability()
                use_scaler = capability[0] >= 7
                if not use_scaler:
                    logger.info(f"CUDA device capability {capability[0]}.{capability[1]} < 7.0. GradScaler disabled.")
            except Exception as e:
                logger.warning(f"Could not determine CUDA capability: {e}. GradScaler disabled.")
        else:
            logger.warning("AMP enabled but CUDA not available. GradScaler disabled.")
    # Create GradScaler - only specify device when using CUDA for AMP
    # GradScaler is only meaningful for CUDA; CPU always uses disabled scaler
    scaler_device = 'cuda' if (use_scaler and torch.cuda.is_available()) else None
    try:
        # PyTorch >= 2.x: torch.amp.GradScaler accepts optional 'device' kwarg
        if scaler_device:
            scaler = GradScaler(device=scaler_device, enabled=use_scaler)
        else:
            scaler = GradScaler(enabled=use_scaler)
    except TypeError:
        # Older torch.amp.GradScaler without 'device' kwarg
        try:
            scaler = GradScaler(enabled=use_scaler)
        except Exception:
            # Very old versions: use legacy CUDA GradScaler (may emit deprecation on newer torch)
            from torch.cuda.amp import GradScaler as CudaGradScaler  # type: ignore
            scaler = CudaGradScaler(enabled=use_scaler)
    if amp_enabled:
        logger.info(f"AMP enabled with dtype={amp_dtype} and GradScaler={'enabled' if use_scaler else 'disabled'}.")
    else:
        logger.info("AMP disabled.")

    checkpoint_dir = Path(config.output_root) / config.experiment_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=config.training.save_total_limit,
        keep_best=config.training.save_best_only
    )

    training_state = TrainingState()
    patience = getattr(config.training, "early_stopping_patience", None)
    es_threshold = getattr(config.training, "early_stopping_threshold", 0.0)
    # Early-stopping burn-in to avoid first-epoch outlier triggering patience
    burn_in_epochs = int(getattr(config.training, "early_stopping_burn_in_epochs", 0) or 0)
    burn_in_strategy = str(getattr(config.training, "early_stopping_burn_in_strategy", "median")).lower()
    _burn_in_vals = []  # collect val metric during burn-in window (re-seeded from checkpoint below)
    global_step = 0
    _last_image_log_step = -1  # Guard against duplicate image logging within accumulation window
    start_epoch = 0
    # Track mid-epoch resume info (for resuming from exact batch position).
    # resume_sample_idx MUST be initialized here even though it is only meaningful
    # when a checkpoint loads: it is referenced unconditionally in the mid-epoch
    # resume guard below, and only `is_mid_epoch` short-circuiting kept a fresh run
    # from raising NameError.
    resume_batch_idx = 0
    resume_sample_idx = 0
    is_mid_epoch = False
    ckpt_original_batch_size = None  # For batch-size agnostic resume from legacy checkpoints
    ckpt_sampler_state = None  # Saved sampler state (carries dataset-size guard for resume)
    # (batch_size, gradient_accumulation_steps) recorded in the resumed checkpoint, or
    # None for a fresh run / phase transition / checkpoint without an embedded config.
    ckpt_effective_batch = None
    # True only when load_checkpoint() actually overwrote the live scheduler's __dict__
    # with the checkpoint's. This is what arms the LR-schedule guard below, and it is
    # deliberately NOT the same condition as `ckpt_effective_batch is not None`: the
    # guard now also compares scheduler-derived geometry (warmup_steps,
    # first_cycle_steps, base_max_lr), which is only meaningful if those attributes came
    # from the checkpoint. A fresh run, a phase transition (scheduler is passed as None,
    # so it keeps this config's geometry), or a checkpoint saved with
    # save_scheduler=False must all leave the guard disarmed - otherwise it would report
    # "drift" against a scheduler that this very config just constructed.
    ckpt_scheduler_restored = False
    phase_transition = False  # Set when checkpoint phase != config.training.phase
    # Soft-stop sentinel files (located in log_dir)
    stop_sentinel = Path(config.log_dir) / "STOP_TRAINING"
    save_sentinel = Path(config.log_dir) / "SAVE_CHECKPOINT"
    image_log_sentinel = Path(config.log_dir) / "LOG_IMAGES_NOW"
    # Clear any leftover STOP_TRAINING sentinel from a previous run so a resume does
    # not immediately soft-stop again. Unlike SAVE_CHECKPOINT / LOG_IMAGES_NOW (which
    # are one-shot and unlinked on consumption), STOP_TRAINING was never cleared
    # anywhere, so a single soft-stop would block every subsequent invocation until
    # the file was deleted by hand. A genuine stop for THIS run is requested by
    # (re)creating the file after startup.
    try:
        stop_sentinel.unlink()
        logger.info("Cleared stale STOP_TRAINING sentinel from a previous run.")
    except FileNotFoundError:
        pass
    except Exception as _stop_sentinel_exc:
        logger.debug("Could not clear stale STOP_TRAINING sentinel: %s", _stop_sentinel_exc)
    early_exit = False
    soft_stop_pending = False  # defer stop until accumulation completes
    # A pending soft stop defers the epoch-boundary accumulation flush so the partial
    # window can be completed at the start of the NEXT epoch. That deferral is allowed
    # exactly once: without this cap, an epoch that ends with a partial window every
    # time (or a short/empty epoch) would carry the stop forward indefinitely and the
    # run could reach the end of training having never written the soft-stop checkpoint.
    soft_stop_carried = False
    # Microbatches elapsed since the stop was deferred, used to bound the wait when
    # repeatedly-skipped batches keep the accumulation window from ever closing.
    soft_stop_wait_steps = 0
    save_now_pending = False  # SAVE_CHECKPOINT sentinel latched, waiting for a safe point
    # PERF: Throttle filesystem sentinel checks to reduce syscall overhead in hot loop
    # Check every 10 steps (increased responsiveness)
    SENTINEL_CHECK_INTERVAL = 10

    def _find_resume_checkpoint(checkpoint_dir: Path, mode: str) -> Optional[Path]:
        if not checkpoint_dir.exists():
            return None
        if mode == "best":
            best_path = checkpoint_dir / "best_model.pt"
            if best_path.exists():
                return best_path
        last_path = checkpoint_dir / LAST_CKPT_NAME
        if last_path.exists():
            return last_path
        candidates = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # --- Resume logic controlled by config.training.resume_from ---
    resume_opt = str(getattr(config.training, "resume_from", "latest")).strip().lower()
    ckpt_path = None

    if resume_opt in ("", "none", "false", "off"):
        logger.info("Resume disabled by config (training.resume_from=%r). Starting fresh.", resume_opt)
    elif resume_opt == "latest":
        ckpt_path = checkpoint_manager.get_latest_checkpoint()
        if ckpt_path is None:
            logger.warning(
                "resume_from='latest' was set but no checkpoint found in %s. "
                "Starting fresh — verify experiment_name and output_root in config.",
                checkpoint_dir
            )
    elif resume_opt == "best":
        ckpt_path = checkpoint_manager.get_best_checkpoint()
        if ckpt_path is None:
            # Fallback to latest if user asked for best but it's missing
            logger.warning("Requested resume_from='best' but best_model.pt not found; trying latest instead.")
            ckpt_path = checkpoint_manager.get_latest_checkpoint()
            if ckpt_path is None:
                logger.info("No checkpoint available. Starting fresh.")
    else:
        # Treat as explicit path
        try_path = Path(getattr(config.training, "resume_from"))
        if try_path.exists():
            ckpt_path = try_path
        else:
            logger.warning("Requested resume_from path does not exist: %s; starting fresh.", try_path)

    if ckpt_path is None and legacy_experiment and resume_opt in ("latest", "best"):
        legacy_dir = Path(config.output_root) / legacy_experiment / "checkpoints"
        legacy_ckpt = _find_resume_checkpoint(legacy_dir, resume_opt)
        if legacy_ckpt is not None:
            logger.info(
                "Found legacy checkpoint for resume_from=%s at %s; resuming into %s.",
                resume_opt,
                legacy_ckpt,
                checkpoint_dir,
            )
            ckpt_path = legacy_ckpt

    if ckpt_path:
        try:
            # --- Pre-load architecture validation (fail-fast before loading weights) ---
            # Peek at checkpoint config AND state dict keys to detect architecture mismatch
            # BEFORE attempting to load state_dict, which would fail with a cryptic error.
            # State dict keys are needed for old checkpoints that lack architecture_type field.
            ckpt_config_preview, ckpt_state_dict_keys = checkpoint_manager.peek_checkpoint_config(
                ckpt_path, include_state_dict_keys=True
            )
            # Always run validation - even without config, we can detect architecture from state dict
            try:
                is_compatible, messages = validate_config_compatibility(
                    checkpoint_config=ckpt_config_preview or {},
                    current_config=config,
                    strict=True,  # Fail on critical mismatches (architecture_type, num_labels, etc.)
                    state_dict_keys=ckpt_state_dict_keys
                )
            except ValueError as e:
                if resume_opt in ("latest", "best"):
                    logger.warning(
                        "Skipping incompatible checkpoint %s; starting fresh. Error: %s",
                        ckpt_path,
                        e
                    )
                    ckpt_path = None
                else:
                    raise
            if ckpt_path and messages:
                for msg in messages:
                    logger.info("Pre-load validation: %s", msg)
            # --- End pre-load validation ---

            # --- Phase-transition detection (unified training.phase key) ---
            # Flipping config.training.phase (e.g. 1 -> 2 at the 320->448
            # resolution switch) and resuming triggers a transition: weights
            # load, but optimizer/scheduler/scaler start FRESH (FixRes-style
            # re-warmup) and epoch counters reset to 0. gamma_neg carries over
            # frozen from the checkpoint's loss state (ASL_plan SS3: P1->P2
            # freeze through re-warmup). Checkpoints predating the phase key
            # record no phase and resume normally.
            current_phase = int(getattr(config.training, 'phase', 0) or 0)
            ckpt_phase_raw = ((ckpt_config_preview or {}).get('training') or {}).get('phase')
            phase_transition = bool(
                ckpt_path is not None
                and current_phase > 0
                and ckpt_phase_raw is not None
                and int(ckpt_phase_raw) != current_phase
            )
            if phase_transition:
                logger.info("=" * 70)
                logger.info(
                    "PHASE TRANSITION: checkpoint phase %s -> config phase %s",
                    ckpt_phase_raw, current_phase,
                )
                logger.info(
                    "Model weights load from the checkpoint; optimizer/scheduler/"
                    "scaler start fresh (re-warmup); epoch counters reset to 0; "
                    "gamma_neg is carried over frozen from the checkpoint loss state."
                )
                logger.info("=" * 70)

            if not ckpt_path:
                logger.info("Resume skipped due to incompatible checkpoint.")
            else:
                # Pin checkpoint to current vocabulary by SHA256 so a regenerated
                # vocab can never silently misalign tag-head indices on resume.
                try:
                    from schemas import compute_vocab_sha256
                    current_vocab_sha = compute_vocab_sha256(
                        vocab_path=Path(getattr(config, 'vocab_path', str(DEFAULT_VOCAB_PATH)))
                    )
                except Exception as _vocab_e:
                    logger.warning("Could not compute current vocab sha for resume guard: %s", _vocab_e)
                    current_vocab_sha = None
                ckpt = checkpoint_manager.load_checkpoint(
                    checkpoint_path=ckpt_path,
                    model=model,
                    # Phase transition: optimizer/scheduler/scaler start fresh
                    # (re-warmup at the new resolution, ASL_plan SS3 / FixRes)
                    optimizer=None if phase_transition else optimizer,
                    scheduler=None if phase_transition else scheduler,
                    device=device,
                    scaler=None if phase_transition else scaler,
                    expected_vocab_sha256=current_vocab_sha,
                    enforce_vocab_check=True,
                    allow_unverified_vocab_resume=getattr(
                        config.training, 'allow_unverified_vocab_resume', False
                    ),
                )
                if not ckpt:
                    raise RuntimeError(f"Checkpoint returned empty data from {ckpt_path}")
                training_state = TrainingState.from_dict(ckpt.get('training_state', {}))

                # best_metric is only comparable against a value measured with the
                # SAME selection metric. This must run on the restored TrainingState
                # itself (not just the checkpoint `metrics` dict, and not only when
                # is_best is set): resuming `last.pt` — which is is_best=False and
                # carries a macro-F1-scale best_metric ~0.013 — while selecting on
                # val_mAP (~0.68) would make the very next epoch a guaranteed false
                # "best" and overwrite best_model.pt with unvetted weights.
                _sel_now = str(getattr(config.training, 'selection_metric', 'val_mAP'))
                # An empty recorded metric means the checkpoint predates this
                # field, and back then selection was hardcoded to val_f1_macro --
                # so "" is KNOWN to be val_f1_macro, not unknown. Treating it as
                # unknown would needlessly discard a comparable historical best
                # for a val_f1_macro run.
                _sel_ckpt = str(getattr(training_state, 'selection_metric', '') or '') or 'val_f1_macro'
                if _sel_ckpt != _sel_now and math.isfinite(training_state.best_metric):
                    _stale_best = training_state.best_metric
                    training_state.best_metric = float('-inf')
                    training_state.patience_counter = 0
                    # Before falling back to -inf (which makes the next validated
                    # epoch a "best" by construction and overwrites best_model.pt
                    # regardless of quality), try to recover a correctly-scaled
                    # best from the sibling best checkpoint, which records all
                    # metrics by name.
                    _recovered = None
                    try:
                        _best_sib = Path(ckpt_path).parent / 'best_model.pt'
                        if _best_sib.exists() and _best_sib != Path(ckpt_path):
                            _sib_metrics = (checkpoint_manager.peek_checkpoint_metrics(_best_sib)
                                            if hasattr(checkpoint_manager, 'peek_checkpoint_metrics')
                                            else (torch.load(_best_sib, map_location='meta',
                                                             weights_only=False).get('metrics') or {}))
                            if _sel_now in _sib_metrics:
                                _recovered = float(_sib_metrics[_sel_now])
                    except Exception as _sib_e:
                        logger.debug("Could not read sibling best_model.pt metrics: %s", _sib_e)
                    if _recovered is not None and math.isfinite(_recovered):
                        training_state.best_metric = _recovered
                        logger.warning(
                            "Resume: checkpoint's best_metric=%.6f was measured with %s but this "
                            "run selects on %s. Re-seeded best_metric=%.6f from the sibling "
                            "best_model.pt's recorded %s, so it is not overwritten by an "
                            "arbitrary first epoch.",
                            _stale_best, _sel_ckpt, _sel_now, _recovered, _sel_now,
                        )
                    else:
                        logger.warning(
                            "Resume: checkpoint's best_metric=%.6f was measured with %s but this "
                            "run selects on %s, and no %s value could be recovered from a sibling "
                            "best checkpoint. best_metric reset to -inf and patience to 0 -- NOTE "
                            "the first validated epoch will therefore be recorded as best and will "
                            "overwrite best_model.pt. Back it up first if you need the old one.",
                            _stale_best, _sel_ckpt, _sel_now, _sel_now,
                        )
                training_state.selection_metric = _sel_now

                if phase_transition:
                    # Epochs are phase-local: reset all counters/histories but
                    # carry the ASL loss state (gamma_neg stays frozen at its
                    # previous-phase landing value through the re-warmup).
                    training_state = TrainingState(
                        phase=current_phase,
                        loss_state=dict(getattr(training_state, 'loss_state', {}) or {}),
                    )
                    global_step = 0
                    start_epoch = 0
                    resume_batch_idx = 0
                    resume_sample_idx = 0
                    is_mid_epoch = False
                    ckpt_sampler_state = None
                    logger.info(
                        "Phase transition applied: epoch/step counters reset; "
                        "starting phase %d at epoch 1 with gamma_neg=%s from the "
                        "checkpoint loss state.",
                        current_phase,
                        (training_state.loss_state or {}).get('gamma_neg', '<YAML>'),
                    )
                else:
                    global_step = ckpt.get('step', 0)
                    # Preserve historical best; only reconcile when explicitly marked as best
                    if ckpt.get('is_best', False):
                        try:
                            # best_metric must be read back under the SAME metric that
                            # will drive selection this run. Reading val_f1_macro while
                            # selecting on val_mAP (or vice versa) seeds best_metric on a
                            # different scale, which either freezes is_best forever or
                            # makes every epoch a "new best".
                            _sel = str(getattr(config.training, 'selection_metric', 'val_mAP'))
                            _ckpt_metrics = ckpt.get('metrics', {}) or {}
                            _ckpt_sel = _ckpt_metrics.get('selection_metric')
                            if _ckpt_sel is not None and _ckpt_sel != _sel:
                                logger.warning(
                                    "Checkpoint was selected on %r but this run selects on %r; "
                                    "discarding the stored best_metric rather than comparing "
                                    "across scales.", _ckpt_sel, _sel,
                                )
                            elif _sel in _ckpt_metrics:
                                loaded_best = float(_ckpt_metrics[_sel])
                                training_state.best_metric = max(training_state.best_metric, loaded_best)
                            else:
                                logger.warning(
                                    "Checkpoint metrics carry no %r entry; leaving best_metric "
                                    "at %s (the first validated epoch will re-seed it).",
                                    _sel, training_state.best_metric,
                                )
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Could not parse best metric from checkpoint: {e}")
                    # Extract mid-epoch resume info if available
                    resume_batch_idx = getattr(training_state, 'batch_in_epoch', 0)
                    resume_sample_idx = getattr(training_state, 'sample_in_epoch', 0)  # Batch-size agnostic
                    is_mid_epoch = not getattr(training_state, 'is_epoch_boundary', True)

                    # NOTE: a mid-epoch save taken on (or past) the LAST batch of an epoch
                    # records batch_in_epoch == steps_per_epoch, and resuming it sets a
                    # sampler offset at the end of the shuffled index list, so the replayed
                    # epoch yields ZERO batches. That epoch is deliberately NOT promoted to
                    # the next one here: the model state is real and its validation /
                    # best-checkpoint pass is still worth running (skipping it would mean a
                    # stop on the final epoch resumes into range(N, N) and exits having
                    # never validated). The only actual damage was avg_train_loss collapsing
                    # to 0.0, which is handled at the epoch-loss computation below.

                    # Checkpoints store epoch+1 (1-based). Convert to 0-based loop index.
                    # For mid-epoch saves: resume the same epoch (subtract 1, then batch-skip handles the rest)
                    # For epoch-boundary saves: epoch is fully completed, start the NEXT epoch (no -1 needed)
                    if is_mid_epoch:
                        start_epoch = ckpt.get('epoch', 1) - 1
                        if start_epoch < 0:
                            start_epoch = 0
                        logger.info(f"Mid-epoch resume: continuing epoch {start_epoch} (0-based)")
                    else:
                        start_epoch = ckpt.get('epoch', 1)  # 1-based completed epoch = correct 0-based next epoch
                        if start_epoch < 0:
                            start_epoch = 0

                    if is_mid_epoch and resume_batch_idx > 0:
                        logger.info("Resumed from %s (epoch=%s, step=%s, batch_in_epoch=%s) - mid-epoch resume",
                                   ckpt_path, start_epoch, global_step, resume_batch_idx)
                        # Warn about persistent_workers limitation with mid-epoch resume
                        # Workers maintain RNG state that cannot be serialized - augmentations will differ
                        if getattr(config.data, 'persistent_workers', False):
                            logger.warning(
                                "Resuming mid-epoch with persistent_workers=True: per-worker RNG state "
                                "cannot be serialized. Augmentations may differ from original run. "
                                "Set persistent_workers=False for fully reproducible mid-epoch resume."
                            )
                    else:
                        logger.info("Resumed from %s (epoch=%s, step=%s)", ckpt_path, start_epoch, global_step)

                    # Extract original batch_size for legacy checkpoint resume (batch-size agnostic)
                    # Note: Critical config validation (architecture_type, num_labels, etc.) was already
                    # performed BEFORE load_checkpoint() via peek_checkpoint_config() to fail fast.
                    # Arm the LR-schedule guard below. Mirrors load_checkpoint()'s own
                    # restore condition (`scheduler is not None and
                    # 'scheduler_state_dict' in meta`); we are already in the
                    # non-phase-transition branch, so `scheduler` was passed in non-None
                    # and the only remaining question is whether the checkpoint carried
                    # scheduler state at all.
                    ckpt_scheduler_restored = 'scheduler_state_dict' in ckpt

                    ckpt_config = ckpt.get('config', {})
                    if ckpt_config:
                        ckpt_original_batch_size = ckpt_config.get('data', {}).get('batch_size')
                        if ckpt_original_batch_size and ckpt_original_batch_size != config.data.batch_size:
                            # NOTE: this value is a LEGACY fallback only. Checkpoints
                            # written by current code carry sample_in_epoch, which is
                            # already batch-size agnostic and is preferred by the
                            # mid-epoch resume path below; ckpt_original_batch_size is
                            # consulted solely when sample_in_epoch is absent (0).
                            logger.info(
                                "Checkpoint was trained with batch_size=%s, current batch_size=%s "
                                "(used for the resume offset only if the checkpoint predates "
                                "sample_in_epoch).",
                                ckpt_original_batch_size, config.data.batch_size,
                            )

                        # Record the checkpoint's EFFECTIVE batch for the LR-geometry
                        # guard below. Only recorded here - the policy is applied
                        # OUTSIDE this try/except, whose handler rewrites every
                        # exception as "Checkpoint loading failed", which would bury
                        # the operator instructions the guard needs to print.
                        _ckpt_bs = (ckpt_config.get('data', {}) or {}).get('batch_size')
                        _ckpt_accum = (ckpt_config.get('training', {}) or {}).get(
                            'gradient_accumulation_steps'
                        )
                        if _ckpt_bs and _ckpt_accum:
                            ckpt_effective_batch = (int(_ckpt_bs), int(_ckpt_accum))

                    # Keep sampler state so the mid-epoch resume path can verify the
                    # dataset size hasn't changed before applying a sample offset.
                    ckpt_sampler_state = ckpt.get('sampler_state')

                # Release checkpoint memory after extraction - model/optimizer states already loaded
                del ckpt
                gc.collect()
        except Exception as e:
            # CRITICAL: Don't silently continue with uninitialized state
            # This could overwrite existing checkpoints with bad data
            logger.exception("Failed to load checkpoint from %s. Error: %s", ckpt_path, e)
            raise RuntimeError(
                f"Checkpoint loading failed for {ckpt_path}. "
                f"To start fresh, set training.resume_from='none' in config. Error: {e}"
            ) from e

    # --- LR-schedule change guard -----------------------------------------------
    # optimizer.load_state_dict() restores the saved per-group lr, and
    # scheduler.load_state_dict() restores the checkpoint's ENTIRE LR geometry
    # (base_max_lr, base_lrs, warmup_steps, first_cycle_steps, cur_cycle_steps) because
    # _LRScheduler.state_dict() is the whole __dict__. So resuming with ANY of the
    # inputs to that geometry changed silently keeps the OLD calibration: the
    # effective_learning_rate / warmup_steps / first_cycle_steps computed above are all
    # discarded, the cosine no longer lands on lr_end at the end of the run, and if the
    # schedule SHRANK the extra updates can overrun cur_cycle_steps and fire an
    # unintended SGDR warm restart - which a num_cycles=1 config assumes can never
    # happen.
    #
    # This guard originally fired ONLY on a change in effective batch size
    # (batch_size x gradient_accumulation_steps). That was too narrow: warmup_steps and
    # first_cycle_steps are derived from updates_per_epoch, which is derived from
    # len(train_loader) - so ADDING OR REMOVING TRAINING IMAGES invalidates the schedule
    # just as thoroughly, at an unchanged batch size, and the batch-only trigger could
    # not see that axis at all. The narrow scope was a deliberate temporary concession
    # (broadening it would have hard-errored a then-live resume); that run has since
    # finished, so all four dimensions are checked now.
    #
    # The checkpoint side of the comparison is read off the LIVE scheduler rather than
    # off the checkpoint dict, because the scheduler object is the thing that will
    # actually drive training - it is the ground truth for what load_state_dict()
    # installed, including any key the save path did not record.
    #
    # Deliberately outside the try/except above: that handler rewrites everything as
    # "Checkpoint loading failed", which would bury the instructions printed here.
    # Armed only when scheduler state was genuinely restored, so a phase transition
    # (scheduler passed as None, geometry already fresh from this config) never trips it.
    if ckpt_scheduler_restored:
        # (human-readable dimension, current-config value, checkpoint/scheduler value)
        _drifts = []
        _ckpt_bs = _ckpt_accum = None
        if ckpt_effective_batch is not None:
            _ckpt_bs, _ckpt_accum = ckpt_effective_batch
            _ckpt_effective = _ckpt_bs * _ckpt_accum
            if _ckpt_effective != effective_batch_size:
                _drifts.append((
                    'effective batch size',
                    f"{config.data.batch_size}x{accum}={effective_batch_size}",
                    f"{_ckpt_bs}x{_ckpt_accum}={_ckpt_effective}",
                ))
        # Each dimension is skipped when the scheduler does not expose it. Older
        # checkpoints predate base_max_lr, and a missing value is NOT evidence of drift -
        # treating None as a mismatch would hard-error every legacy resume for a
        # dimension we simply cannot observe.
        _sched_warmup = getattr(scheduler, 'warmup_steps', None)
        if _sched_warmup is not None and int(_sched_warmup) != int(warmup_steps):
            _drifts.append(('warmup_steps', int(warmup_steps), int(_sched_warmup)))
        _sched_cycle = getattr(scheduler, 'first_cycle_steps', None)
        if _sched_cycle is not None and int(_sched_cycle) != int(first_cycle_steps):
            _drifts.append(('first_cycle_steps', int(first_cycle_steps), int(_sched_cycle)))
        _sched_peak = getattr(scheduler, 'base_max_lr', None)
        # math.isclose, not !=: base_max_lr is a float that has been through
        # scale_learning_rate(), a torch save/load round-trip and possibly a retarget(),
        # so an exact comparison would report phantom drift on bit-level noise.
        if _sched_peak is not None and not math.isclose(
            float(_sched_peak), float(effective_learning_rate), rel_tol=1e-9
        ):
            _drifts.append((
                'base_max_lr (peak LR)',
                f"{float(effective_learning_rate):.6e}",
                f"{float(_sched_peak):.6e}",
            ))
        # min_lr and gamma are restored by load_state_dict exactly like the four
        # dimensions above. They were skipped while this guard was scoped to batch-size
        # changes (neither depends on batch size) - but it is now a general LR-schedule
        # guard, and leaving them out meant a run that changed training.lr_end or
        # training.cycle_decay resumed onto the OLD values with the guard silent, the
        # anneal landing on the previous floor while the config advertised the new one.
        _cfg_min_lr = float(getattr(config.training, 'lr_end', 1e-6))
        _sched_min_lr = getattr(scheduler, 'min_lr', None)
        if _sched_min_lr is not None and not math.isclose(
            float(_sched_min_lr), _cfg_min_lr, rel_tol=1e-9
        ):
            _drifts.append((
                'min_lr (lr_end)',
                f"{_cfg_min_lr:.6e}",
                f"{float(_sched_min_lr):.6e}",
            ))
        # Default must match the one the scheduler was CONSTRUCTED with (line ~839).
        # A different fallback here would report phantom gamma drift the moment the
        # field went missing, since the two would disagree by construction.
        _cfg_gamma = float(getattr(config.training, 'cycle_decay', 0.9))
        _sched_gamma = getattr(scheduler, 'gamma', None)
        if _sched_gamma is not None and not math.isclose(
            float(_sched_gamma), _cfg_gamma, rel_tol=1e-9
        ):
            _drifts.append(('gamma (cycle_decay)', _cfg_gamma, float(_sched_gamma)))

        if _drifts:
            _policy = str(getattr(
                config.training, 'on_lr_schedule_change', 'error'
            )).strip().lower()
            _detail = (
                "LR schedule geometry changed between the checkpoint and this config: "
                + "; ".join(
                    f"{_name}: current {_cur} vs checkpoint {_old}"
                    for _name, _cur, _old in _drifts
                )
                + ". The checkpoint's LR geometry (peak LR, warmup length, cycle length) "
                  "was calibrated for the old values and is restored verbatim by "
                  "optimizer/scheduler load_state_dict()."
            )
            if _policy == 'keep':
                logger.warning(
                    "%s Continuing with the CHECKPOINT's LR geometry "
                    "(training.on_lr_schedule_change='keep'): this run will train on the "
                    "previous schedule's learning rate and its cosine will not land on "
                    "lr_end. The configured %.3e peak / %s warmup / %s cycle are discarded.",
                    _detail, effective_learning_rate, warmup_steps, first_cycle_steps,
                )
            elif _policy == 'rescale':
                try:
                    # retarget() covers every dimension checked above in one call - peak
                    # LR, warmup length, cycle length, min_lr and gamma - carrying
                    # progress across as a FRACTION of the cycle, so there is no
                    # per-dimension dispatch here.
                    _new_lrs = scheduler.retarget(
                        max_lr=effective_learning_rate,
                        warmup_steps=warmup_steps,
                        first_cycle_steps=first_cycle_steps,
                        min_lr=_cfg_min_lr,
                        gamma=_cfg_gamma,
                    )
                except Exception as _rt_e:
                    raise RuntimeError(
                        f"Failed to retarget the LR schedule after an LR-schedule "
                        f"change: {_rt_e}. Set training.on_lr_schedule_change='keep' to "
                        f"resume with the checkpoint's schedule instead."
                    ) from _rt_e
                logger.warning(
                    "%s Retargeting to the current config "
                    "(training.on_lr_schedule_change='rescale'): peak LR -> %.3e, "
                    "warmup -> %s updates, cycle -> %s updates, progress preserved at "
                    "%.1f%% (step_in_cycle=%s), live LR now %.3e.",
                    _detail, effective_learning_rate, warmup_steps,
                    scheduler.cur_cycle_steps,
                    100.0 * scheduler.step_in_cycle / max(1, scheduler.cur_cycle_steps),
                    scheduler.step_in_cycle,
                    _new_lrs[0] if _new_lrs else float('nan'),
                )
            else:
                # "Restore the old values" is only actionable per drifted dimension, so
                # the hints are built from _drifts rather than hardcoded: telling an
                # operator to restore a batch size that never moved sends them hunting
                # for a change that is not there.
                _restore_hints = ""
                if _ckpt_bs is not None and any(
                    _n == 'effective batch size' for _n, _, _ in _drifts
                ):
                    _restore_hints += (
                        f"  restore data.batch_size={_ckpt_bs} and "
                        f"training.gradient_accumulation_steps={_ckpt_accum}\n"
                    )
                if any(_n in ('warmup_steps', 'first_cycle_steps') for _n, _, _ in _drifts):
                    _restore_hints += (
                        f"  restore whatever moved updates_per_epoch (currently "
                        f"{updates_per_epoch}) - the training-set size / dataset filters, "
                        f"training.num_epochs ({num_epochs}), training.warmup_epochs "
                        f"({warmup_epochs}) or training.num_cycles ({num_cycles})\n"
                    )
                if any(_n.startswith('base_max_lr') for _n, _, _ in _drifts):
                    _restore_hints += (
                        f"  restore training.learning_rate / lr_scaling_mode / "
                        f"lr_base_batch_size so the scaled peak matches the checkpoint\n"
                    )
                if any(_n.startswith('min_lr') for _n, _, _ in _drifts):
                    _restore_hints += (
                        f"  restore training.lr_end (currently {_cfg_min_lr:.6e})\n"
                    )
                if any(_n.startswith('gamma') for _n, _, _ in _drifts):
                    _restore_hints += (
                        f"  restore training.cycle_decay (currently {_cfg_gamma})\n"
                    )
                raise RuntimeError(
                    f"{_detail}\n"
                    f"Refusing to resume: doing so would train on the wrong learning-rate "
                    f"schedule with no visible sign. Also re-check training.eval_steps "
                    f"({getattr(config.training, 'eval_steps', 0)}) against the new updates_per_epoch "
                    f"({updates_per_epoch}) - eval_steps >= updates_per_epoch drops "
                    f"validation to every other epoch.\n"
                    f"Choose one of:\n"
                    f"  training.on_lr_schedule_change: 'rescale'  - retarget peak LR / "
                    f"warmup / cycle to the current config, preserving progress as a "
                    f"fraction of the cycle (recommended)\n"
                    f"  training.on_lr_schedule_change: 'keep'     - keep the "
                    f"checkpoint's schedule (pre-fix behaviour)\n"
                    f"{_restore_hints}"
                    f"  training.resume_from: 'none'               - start fresh"
                )

    # --- ASL gamma_neg drive + always-on telemetry (todos/ASL_plan.md SS3/SS5/SS8) ---
    # Must run AFTER resume: it reconciles gamma_neg (checkpoint loss state wins
    # over YAML; gamma_neg_override applies a guarded manual step) and pushes the
    # result into the live criterion. Its state dict is shared BY REFERENCE with
    # training_state.loss_state so every checkpoint save persists gamma_neg +
    # telemetry EMAs (without this, gamma silently reverts to YAML on restart).
    # Fail-fast: without the drive manager the run degrades to fixed YAML gamma
    # with no gates - exactly V1's loss (ASL_plan SS8).
    asl_drive = ASLDriveManager(
        config=config,
        criterion=criterion,
        vocab=vocab,
        device=device,
        state=dict(getattr(training_state, 'loss_state', {}) or {}),
        start_epoch=start_epoch,
        monitor=monitor,
    )
    training_state.loss_state = asl_drive.state
    training_state.phase = int(getattr(config.training, 'phase', 0) or 0)

    def _ensure_conv2d_channels_last(model: torch.nn.Module, logger) -> int:
        """Force all Conv2d weights to channels_last format.

        model.to(memory_format=torch.channels_last) can silently fail to convert
        weights after checkpoint loading or dtype conversions. This explicitly
        verifies and forces the conversion for torch.compile compatibility.

        Returns number of tensors that needed fixing.
        """
        fixed_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.weight.ndim == 4:
                weight = module.weight
                if not weight.is_contiguous(memory_format=torch.channels_last):
                    old_stride = weight.stride()
                    module.weight.data = weight.contiguous(memory_format=torch.channels_last)
                    new_stride = module.weight.stride()
                    logger.warning(
                        f"Conv2d '{name}' had wrong strides {old_stride}, "
                        f"forced to channels_last: {new_stride}"
                    )
                    fixed_count += 1
        return fixed_count

    # Apply channels_last memory format AFTER all model transformations (device, dtype, checkpoint)
    # CRITICAL: This is the ONLY place channels_last should be applied. Earlier applications
    # (e.g., before bfloat16 conversion) are lost because dtype conversion creates new tensors.
    # torch.load() also restores tensors in contiguous (channels_first) format.
    # This must happen BEFORE torch.compile to ensure correct kernel generation.
    # Cache this for use throughout training (avoids repeated getattr calls)
    use_channels_last = getattr(config.training, "memory_format", "contiguous") == "channels_last"

    # CONFLICT RESOLUTION: channels_last is incompatible with torch.compile's inductor backend
    # due to stride mismatch issues in compiled kernels. Auto-disable channels_last when compile
    # is enabled since compile provides greater performance benefits (15-35% vs 5-15%).
    if use_channels_last and use_compile:
        logger.warning(
            "channels_last memory format is incompatible with torch.compile() due to stride "
            "mismatch issues in the inductor backend. Automatically disabling channels_last. "
            "torch.compile provides 15-35% speedup vs channels_last's 5-15%, so this is the "
            "optimal configuration. To use channels_last instead, set training.compile=false."
        )
        use_channels_last = False

    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        logger.info("Applied channels_last memory format to model")
        # Verify and force Conv2d weights - model.to() can fail silently after
        # checkpoint loading due to dtype mismatches or PyTorch bugs
        fixed = _ensure_conv2d_channels_last(model, logger)
        if fixed > 0:
            logger.warning(f"Fixed {fixed} Conv2d layer(s) with incorrect memory format")
        else:
            logger.info("All Conv2d layers verified as channels_last")

    # Now apply torch.compile() - AFTER checkpoint loading and memory format conversion
    # This ensures inductor kernels are compiled for the actual weights with correct strides
    if use_compile:
        logger.info("=" * 70)
        logger.info("Compiling model with torch.compile()...")
        logger.info("This will take 2-5 minutes on first forward pass but provides")
        logger.info("15-35% speedup for transformer training workloads.")

        logger.info(f"  compile_mode: {compile_mode}")
        logger.info(f"  fullgraph: {compile_fullgraph}")
        logger.info(f"  dynamic: {compile_dynamic}")

        try:
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic
            )
            logger.info("Model compiled successfully!")
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}")
            logger.warning("Continuing with eager mode (uncompiled)...")

        logger.info("=" * 70)

        # NOTE: no dummy-input warmup forward here. A no_grad/no-autocast warmup
        # compiles an *inference* graph (Dynamo guards on grad mode and autocast
        # state), so the first real training batch triggers a full recompile anyway
        # - the warmup was pure additive startup cost. A fwd+bwd warmup is also
        # undesirable: it would perturb RNG state and break resume reproducibility.

    # NOTE: TensorBoard model-graph logging was removed: it consumed a throwaway
    # next(iter(train_loader)) batch, which spins up all workers early and perturbs
    # restored RNG/data order on resume (and was inactive with use_compile=true).

    # NOTE: no "ensure required TrainingState attrs exist" backfill here. TrainingState
    # is a dataclass with defaults for every field and TrainingState.from_dict() filters
    # unknown keys, so optimizer_updates/batch_in_epoch/is_epoch_boundary/best_metric are
    # always present - the old hasattr() loop could never fire.

    # Burn-in samples and validation cadence are restored from the checkpoint so they
    # survive a soft stop. Both were previously process-local: last_validation_step
    # reset to 0 on every restart, making `global_step - last_validation_step >=
    # eval_steps` trivially true and forcing an off-cadence validation after every
    # resume; _burn_in_vals restarted empty, so the burn-in baseline was computed from
    # only the post-resume subset.
    _burn_in_vals = [float(v) for v in (getattr(training_state, 'burn_in_values', None) or [])]

    # Create validation metrics once before training loop (CR-040 fix)
    # These will be reset each epoch instead of being recreated
    # Single source of truth for the prediction threshold (also used by the
    # bucketed metrics below and the standalone validation runner).
    threshold = float(config.inference.prediction_threshold)
    selection_metric = str(getattr(config.training, 'selection_metric', 'val_mAP'))
    if selection_metric not in _SELECTION_METRICS:
        raise ValueError(
            f"training.selection_metric={selection_metric!r} is not one of "
            f"{sorted(_SELECTION_METRICS)}"
        )
    logger.info(
        "Model selection / early stopping driven by %s%s",
        selection_metric,
        "" if selection_metric == 'val_mAP' else
        f" (threshold-coupled at prediction_threshold={threshold}; a calibration "
        f"shift such as an ASL gamma_neg step will move it independently of ranking)",
    )
    # Stamp it into the state so every checkpoint records which scale best_metric
    # is on (see the resume reconciliation above).
    training_state.selection_metric = selection_metric
    skip_metric_cols = 2  # PAD=0, UNK=1 — consistent with loss ignore_indices and bucketed metrics
    num_metric_labels = num_tags - skip_metric_cols
    # Per-class metrics (average=None) so we can filter classes with zero positives
    # in the validation draw before macro-averaging. Without this, an 18-24K-class
    # long-tailed vocabulary leaves thousands of classes unrepresented in any
    # ~30k-sample draw, each contributing AP=0 and pulling the macro toward zero.
    # An integer `thresholds` puts the AP metric in binned mode (constant memory):
    # the default thresholds=None retains every update's preds/targets on GPU
    # (~7 GB at 30K x 19.3K labels) and concatenates them at compute().
    # The count is configurable because binning biases mAP DOWN and that bias
    # responds to rank-preserving calibration shifts — which matters now that
    # val_mAP is the default selection metric. See ValidationConfig.ap_thresholds.
    ap_thresholds = int(getattr(config.validation, 'ap_thresholds', None) or 200)
    # Persistent state; torchmetrics also keeps a deepcopy in _defaults, hence x2.
    _ap_state_mb = 2 * ap_thresholds * num_metric_labels * 4 * 8 / 1e6
    # The transient per-update intermediate is what actually constrains this.
    # _multilabel_precision_recall_curve_update holds several (B, L, T) tensors at
    # once (preds_t, unique_mapping, a bool mask and the (N,3) index tensor from
    # the >=0 filter), not one: measured ~50 bytes per element on torchmetrics
    # 1.8.2 at both T=200 (9.34 GB) and T=500 (23.57 GB). Counting a single int64
    # understates the peak 6.3x. The AP metric is fed VALIDATION batches.
    _AP_BYTES_PER_ELEM = 50
    _ap_val_bs = int(getattr(getattr(config.validation, 'dataloader', None), 'batch_size', 0)
                     or config.data.batch_size)
    _ap_peak_gb = (_ap_val_bs * num_metric_labels * ap_thresholds * _AP_BYTES_PER_ELEM) / 1e9
    val_metrics = {
        'f1_macro_per_class': MultilabelF1Score(num_labels=num_metric_labels, average=None, threshold=threshold).to(device),
        'f1_micro': MultilabelF1Score(num_labels=num_metric_labels, average="micro", threshold=threshold).to(device),
        'map_per_class': MultilabelAveragePrecision(num_labels=num_metric_labels, average=None, thresholds=ap_thresholds).to(device)
    }
    val_pos_counts = torch.zeros(num_metric_labels, dtype=torch.long, device=device)
    logger.info(
        f"Validation metrics initialized with {num_metric_labels} labels "
        f"(skipping {skip_metric_cols} special tokens), threshold={threshold}, "
        f"ap_thresholds={ap_thresholds} (~{_ap_state_mb:.0f} MB resident, "
        f"~{_ap_peak_gb:.1f} GB transient peak per AP update)"
    )
    if _ap_peak_gb > 12.0:
        logger.warning(
            "validation.ap_thresholds=%d implies a ~%.1f GB transient allocation per AP "
            "update at val batch_size=%d. This will thrash or OOM on most GPUs - lower it.",
            ap_thresholds, _ap_peak_gb, _ap_val_bs,
        )

    # Initialize memory monitor to track RAM usage and prevent OOM
    mem_monitor = MemoryMonitor(warn_threshold_gb=115.0, critical_threshold_gb=125.0)
    logger.info("Memory monitor initialized (warn: 115 GB, critical: 125 GB)")

    # Track last validation step for step-based validation frequency (restored from
    # the checkpoint - see the burn-in/cadence note above).
    last_validation_step = int(getattr(training_state, 'last_validation_step', 0) or 0)
    eval_steps = getattr(config.training, 'eval_steps', 0) or 0  # 0 means validate every epoch
    # `eval_steps < updates_per_epoch` is what guarantees at least one validation per
    # epoch. That guarantee used to be unconditional (validation was forced whenever
    # epoch == start_epoch), so nothing depended on the margin; now it does. Anything
    # that shrinks updates_per_epoch - a smaller corpus, a larger batch_size, a larger
    # gradient_accumulation_steps - can silently drop the cadence to every other epoch,
    # which with early_stopping_burn_in_epochs also starves the burn-in window.
    if eval_steps > 0 and eval_steps >= updates_per_epoch:
        logger.warning(
            "training.eval_steps=%s is >= updates_per_epoch=%s: validation will NOT run "
            "every epoch (roughly every %.1f epochs). Early stopping then counts validated "
            "epochs rather than epochs, and the burn-in window (%s epochs) may collect "
            "fewer samples than configured. Set eval_steps below %s to validate every epoch.",
            eval_steps, updates_per_epoch, eval_steps / max(1, updates_per_epoch),
            burn_in_epochs, updates_per_epoch,
        )
    else:
        logger.info(
            "Validation cadence: eval_steps=%s vs %s optimizer updates/epoch "
            "(%s per epoch)",
            eval_steps, updates_per_epoch,
            "every epoch" if eval_steps == 0 else "at least once",
        )

    # NOTE: use_channels_last is defined earlier (before torch.compile) and cached for use here

    # Create dedicated CUDA stream for H2D transfers to enable pipelining
    # This allows H2D transfers to overlap with compute from the previous batch
    h2d_stream = torch.cuda.Stream() if device.type == 'cuda' else None
    if h2d_stream is not None:
        logger.info("H2D transfer stream created for async CPU→GPU pipelining")

    # GPU scalar accumulator: avoids per-microbatch GPU→CPU sync that breaks compile overlap.
    # fp32 keeps precision over thousands of microbatches per epoch.
    running_loss = torch.zeros((), device=device, dtype=torch.float32)
    processed_batches = 0  # Excludes skipped batches for accurate loss averaging
    total_train_samples = 0  # Track total samples for proper per-sample loss averaging
    skipped_batches = 0
    accum_count = 0  # Tracks accumulated batches (handles skipped batches)

    for epoch in range(start_epoch, config.training.num_epochs):
        # Ensure distinct shuffles across epochs in distributed mode
        # CRITICAL: This must succeed in distributed training or gradients will be corrupted
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
            logger.debug(f"Set distributed sampler epoch to {epoch}")

        # Set epoch on datasets for epoch-varying flip decisions
        # This ensures augmentation diversity across epochs while maintaining determinism
        try:
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)
                logger.debug(f"Train dataset epoch set to {epoch}")
            if val_loader is not None and hasattr(val_loader.dataset, 'set_epoch'):
                val_loader.dataset.set_epoch(epoch)
                logger.debug(f"Val dataset epoch set to {epoch}")
        except Exception as e:
            logger.debug(f"Dataset set_epoch skipped: {e}")

        model.train()
        carry_accum = soft_stop_pending and accum_count > 0
        if carry_accum:
            logger.info(
                "Soft stop pending - carrying %s accumulated microbatches into next epoch.",
                accum_count,
            )
        else:
            running_loss.zero_()  # In-place reset of GPU scalar accumulator
            processed_batches = 0  # Excludes skipped batches for accurate loss averaging
            total_train_samples = 0  # Track total samples for proper per-sample loss averaging
            skipped_batches = 0
            optimizer.zero_grad(set_to_none=True)  # Use set_to_none for memory efficiency
            accum_count = 0  # Tracks accumulated batches (handles skipped batches)

        with anomaly_ctx:
            # Mid-epoch resume setup (before creating iterator to avoid double-init)
            start_step = 0
            # Whether the O(1) sampler offset was actually applied. The slow
            # batch-skipping fallback below MUST key off this rather than off
            # `start_step == 0`: when the recorded sample offset is smaller than the
            # CURRENT batch size (only possible if batch_size grew since the checkpoint),
            # start_step floors to 0 even though the offset was applied, and the fallback
            # would then consume resume_batch_idx further batches on top of it - silently
            # dropping up to new_bs^2/old_bs samples from the replayed epoch.
            sampler_offset_applied = False
            # `step` restarts at floor(sample_offset / batch_size), but the sampler starts
            # at sample_offset exactly. When the two disagree (again, only after a
            # batch_size change) every later save in this epoch would record a
            # sample_in_epoch that is `sample_offset % batch_size` samples behind the true
            # position. This carries the remainder so the recorded position stays exact.
            sample_pos_correction = 0
            if epoch == start_epoch and is_mid_epoch and (resume_batch_idx > 0 or resume_sample_idx > 0):
                # Try instant resume via ResumableSampler (O(1) instead of O(n) batch iteration)
                sampler = getattr(train_loader, 'sampler', None)
                # Dataset-size-change guard (mirrors ResumableSampler.load_state):
                # a sample offset recorded against a different dataset size would
                # address different samples, so refuse to apply a stale offset and
                # restart the epoch instead. Also disables the skip-by-index
                # fallback below, which is equally stale.
                saved_total = (ckpt_sampler_state or {}).get('total_size')
                current_total = getattr(sampler, 'total_size', None)
                if saved_total is not None and current_total is not None and saved_total != current_total:
                    logger.warning(
                        "Dataset size changed from %s to %s since checkpoint was saved. "
                        "Mid-epoch sample offset cannot be applied safely - resuming from epoch start.",
                        saved_total, current_total,
                    )
                    resume_batch_idx = 0
                    resume_sample_idx = 0
                elif hasattr(sampler, 'set_start_index'):
                    # Set sampler offset BEFORE creating iterator
                    # Note: set_start_index expects SAMPLE index, not batch index
                    # Use sample_in_epoch if available (batch-size agnostic), otherwise calculate
                    if resume_sample_idx > 0:
                        sample_offset = resume_sample_idx
                        start_step = sample_offset // train_loader.batch_size
                    else:
                        # Use original batch_size from checkpoint for legacy checkpoints
                        # (those predating sample_in_epoch), so the offset is still in
                        # sample space rather than in the old run's batch units.
                        # NOTE: deliberately NOT named effective_batch_size - that is a
                        # function-level local meaning batch_size * accum (samples per
                        # optimizer update, used for LR scaling and the effective-batch
                        # guard). Reusing the name here reassigned it to a plain
                        # batch_size mid-run; harmless only because every reader happens
                        # to run earlier, which is not a property worth relying on.
                        legacy_batch_size = ckpt_original_batch_size or train_loader.batch_size
                        sample_offset = resume_batch_idx * legacy_batch_size
                        start_step = sample_offset // train_loader.batch_size
                    sampler.set_start_index(sample_offset)
                    sampler_offset_applied = True
                    sample_pos_correction = sample_offset - start_step * train_loader.batch_size
                    logger.info(
                        "Resuming mid-epoch at batch %s (sample offset %s, instant via sampler)%s",
                        start_step, sample_offset,
                        (f" - offset is not a multiple of batch_size={train_loader.batch_size}; "
                         f"carrying a {sample_pos_correction}-sample correction so later saves "
                         f"record the exact position")
                        if sample_pos_correction else "",
                    )

            # Create iterator (sampler start_index already set if mid-epoch resume)
            # Workers spawn synchronously here; persistent_workers=True keeps them alive for subsequent epochs
            train_iter = iter(train_loader)

            # Fallback path for mid-epoch resume without ResumableSampler
            if epoch == start_epoch and is_mid_epoch and resume_batch_idx > 0 and not sampler_offset_applied:
                # No ResumableSampler - must iterate through batches (slow fallback).
                # This skips resume_batch_idx batches at the CURRENT batch size, so it is
                # only faithful when batch_size is unchanged; the sampler path above is
                # the batch-size-agnostic one and handles every configuration this repo
                # actually builds (create_dataloaders always installs a ResumableSampler).
                logger.info(f"Resuming mid-epoch: skipping {resume_batch_idx} batches (fallback mode, no ResumableSampler)...")
                skip_start = time.time()

                for i in range(resume_batch_idx):
                    next(train_iter)  # Consume and discard - batch still gets loaded
                    if (i + 1) % 500 == 0:
                        elapsed = time.time() - skip_start
                        rate = (i + 1) / elapsed
                        remaining = (resume_batch_idx - i - 1) / rate
                        logger.info(f"  Skip progress: {i + 1}/{resume_batch_idx} batches (~{remaining:.0f}s remaining)")

                skip_elapsed = time.time() - skip_start
                logger.info(f"Skip complete: {resume_batch_idx} batches in {skip_elapsed:.1f}s")
                start_step = resume_batch_idx

            for step, batch in enumerate(train_iter, start=start_step):
                # Poll the sentinels FIRST, before any of the `continue` paths below
                # (failed load / NaN loss / non-finite grad) can skip the rest of the
                # body. Only the LATCHES live here - the checkpoint saves stay at the end
                # of the body, where the accumulation window is in a known state and
                # `step` is safe to record as the resume position. With the polls at the
                # end, a failure pattern that happened to cover the throttled poll steps
                # meant the sentinels were never even read, so the request went unnoticed
                # until the epoch boundary (STOP_TRAINING) or indefinitely (SAVE_CHECKPOINT).
                if step % SENTINEL_CHECK_INTERVAL == 0:
                    if stop_sentinel.exists():
                        soft_stop_event.set()
                    if save_sentinel.exists():
                        save_now_pending = True

                # Filter out error samples that failed to load (zero-valued samples corrupt gradients)
                error_flags = batch.get('error')
                if error_flags is not None and isinstance(error_flags, torch.Tensor) and error_flags.any():
                    valid_mask = ~error_flags
                    num_errors = error_flags.sum().item()
                    if valid_mask.sum() == 0:
                        logger.warning(f"Skipping batch {global_step}: all {num_errors} samples failed to load")
                        skipped_batches += 1
                        continue
                    # Filter batch to only valid samples
                    logger.debug(f"Filtering {num_errors} error samples from batch {global_step}")
                    batch = {
                        k: v[valid_mask] if isinstance(v, torch.Tensor) and v.size(0) == len(error_flags) else v
                        for k, v in batch.items()
                    }

                # Transfer tensors to GPU
                # Use dedicated H2D stream to overlap transfers with compute from previous batch
                pmask = batch.get('padding_mask', None)
                h2d_ctx = torch.cuda.stream(h2d_stream) if h2d_stream is not None else nullcontext()
                with h2d_ctx:
                    images = batch['images'].to(device, non_blocking=True)
                    if use_channels_last:
                        images = images.contiguous(memory_format=torch.channels_last)
                    tag_labels = batch['tag_labels'].to(device, non_blocking=True)
                    if pmask is not None:
                        pmask = pmask.to(device=device, dtype=torch.bool, non_blocking=True)

                # Sync H2D stream before compute (ensures all transfers complete before model forward)
                if h2d_stream is not None:
                    torch.cuda.current_stream().wait_stream(h2d_stream)
                    # record_stream marks these allocations as in-use by the default
                    # stream. Without it the caching allocator considers them owned by
                    # the side stream only and can recycle the memory for next batch's
                    # H2D copy while backward still reads the activations (classic
                    # prefetcher data race).
                    images.record_stream(torch.cuda.current_stream())
                    tag_labels.record_stream(torch.cuda.current_stream())
                    if pmask is not None:
                        pmask.record_stream(torch.cuda.current_stream())

                # Assert that input data is finite and labels are in range (only when debug enabled to avoid GPU sync)
                if config.debug.enabled:
                    assert_finite(images, tag_labels, names=['images', 'tag_labels'], batch=batch, config=config)

                if config.debug.enabled and getattr(config.debug, 'log_input_stats', False) and (global_step % config.training.logging_steps == 0):
                    # OPTIMIZED: Single GPU sync via .tolist() instead of multiple .item() calls
                    with torch.no_grad():
                        img_stats = torch.stack([images.min(), images.max(), images.mean()]).cpu().tolist()
                    img_min, img_max, img_mean = img_stats
                    monitor.log_scalar('train/image_min', img_min, global_step)
                    monitor.log_scalar('train/image_max', img_max, global_step)
                    monitor.log_scalar('train/image_mean', img_mean, global_step)
                    logger.debug(f"Input stats - min: {img_min:.6f}, mean: {img_mean:.6f}, max: {img_max:.6f}")

                with nullcontext():
                    with amp_autocast():
                        outputs = model(images, padding_mask=pmask)

                        if config.debug.enabled and getattr(config.debug, 'log_activation_stats', False) and (global_step % config.training.logging_steps == 0):
                            tag_logits = outputs.get('tag_logits')
                            with torch.no_grad():
                                if tag_logits is not None:
                                    t_min, t_max, t_mean = torch.stack([tag_logits.min(), tag_logits.max(), tag_logits.mean()]).cpu().tolist()
                                    monitor.log_scalar('train/tag_logits_min', t_min, global_step)
                                    monitor.log_scalar('train/tag_logits_max', t_max, global_step)
                                    monitor.log_scalar('train/tag_logits_mean', t_mean, global_step)
                                    logger.debug(f"Tag logits stats - min: {t_min:.6f}, mean: {t_mean:.6f}, max: {t_max:.6f}")

                        # Assert that model outputs are finite before loss calculation (only when debug enabled to avoid GPU sync)
                        if config.debug.enabled:
                            assert_finite(
                                outputs['tag_logits'],
                                names=['tag_logits'],
                                batch=batch,
                                outputs=outputs,
                                config=config
                            )

                        loss, losses = criterion(outputs['tag_logits'], tag_labels)

                    # global_step only increments on optimizer updates, so all periodic
                    # per-step work below is gated on the update boundary (the microbatch
                    # that completes the accumulation window). Without this gate, every
                    # microbatch of a qualifying window fires the same event ~accum times
                    # (duplicate TB writes, GPU syncs, psutil scans).
                    is_update_boundary = (accum_count + 1 >= accum)
                    # Anticipated post-increment step: losses dict is deleted before
                    # global_step increments, so we predict the final step value here
                    # to align extraction with the logging check below.
                    anticipated_step = (global_step + 1) if is_update_boundary else global_step

                    # Periodic pre-backward NaN/Inf check on GPU tensor (avoids per-step sync overhead)
                    # This catches NaN loss early before backward pass corrupts gradients.
                    # The check runs every N updates (configurable via NAN_CHECK_INTERVAL_STEPS env var).
                    # Set NAN_CHECK_EVERY_STEPS=0 to disable (GradScaler still catches NaN gradients post-backward).
                    if NAN_CHECK_EVERY_STEPS > 0 and is_update_boundary and anticipated_step % NAN_CHECK_EVERY_STEPS == 0:
                        # Use torch.isfinite on GPU - this triggers a sync but only periodically
                        if not torch.isfinite(loss):
                            # Log with available info before potentially crashing .item() call
                            logger.error(
                                f"Pre-backward NaN/Inf loss detected at step {global_step} (periodic check). "
                                f"Skipping batch to prevent gradient corruption."
                            )
                            monitor.log_scalar('train/nan_inf_loss_detected', 1.0, global_step)
                            # Discard accumulated gradients to prevent corruption
                            if accum_count > 0:
                                logger.warning(
                                    f"Discarding {accum_count} accumulated gradient steps due to NaN/Inf loss."
                                )
                            # NOTE: no scaler.update() here - it would assert on an enabled
                            # fp16 scaler because no inf-checks were recorded (update()
                            # requires a preceding unscale_/step). Skipping the batch and
                            # zeroing gradients is sufficient.
                            optimizer.zero_grad(set_to_none=True)
                            accum_count = 0
                            skipped_batches += 1
                            del loss, losses, outputs
                            continue

                    # CR-041: Extract loss scalars BEFORE backward to allow early tensor deletion
                    # This prevents VRAM leak from retaining computation graphs during gradient accumulation
                    # Extract raw loss first, then scale - avoids double division (backward also divides by accum)
                    loss_detached = loss.detach()
                    batch_size_current = images.size(0)
                    
                    # Optimization: Avoid per-step CPU-GPU sync (loss.item())
                    # Only sync when strictly necessary (logging or periodic NaN check),
                    # and only on the update boundary so each event fires exactly once
                    # per qualifying optimizer update (see is_update_boundary above).
                    should_log = is_update_boundary and (anticipated_step == 1 or anticipated_step % config.training.logging_steps == 0)
                    should_check_nan = is_update_boundary and (NAN_CHECK_EVERY_STEPS > 0 and anticipated_step % NAN_CHECK_EVERY_STEPS == 0)
                    
                    loss_item = None
                    losses_items = {}
                    
                    if should_log or should_check_nan:
                        loss_item = loss_detached.item()
                        # Note: losses_items contains UNSCALED component losses
                        losses_items = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

                        # NaN/Inf loss detection
                        if not np.isfinite(loss_item):
                            logger.error(
                                f"NaN/Inf loss detected at step {global_step}: loss={loss_item}, "
                                f"components={losses_items}"
                            )
                            monitor.log_scalar('train/nan_inf_loss_detected', 1.0, global_step)
                            # Discard accumulated gradients to prevent corruption
                            if accum_count > 0:
                                logger.warning(
                                    f"Discarding {accum_count} accumulated gradient steps due to NaN/Inf loss."
                                )
                            optimizer.zero_grad(set_to_none=True)
                            accum_count = 0
                            skipped_batches += 1
                            del loss, losses, outputs, loss_detached
                            continue

                    # Save detached logits for potential image logging before we delete outputs
                    # These are lightweight copies that don't retain the computation graph
                    _saved_tag_logits = outputs['tag_logits'].detach()

                    # Divide loss by accumulation steps BEFORE backward so each micro-batch
                    # contributes equally-scaled gradients during accumulation.
                    scaled_loss = loss / accum if accum > 1 else loss
                    if use_scaler:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                # CR-041: Free computation graph IMMEDIATELY after backward to prevent VRAM leak
                del loss, scaled_loss, losses, outputs
                # tag_logits is only defined when log_activation_stats is enabled
                try:
                    del tag_logits
                except NameError:
                    pass

                accum_count += 1

                if accum_count >= accum:
                    if use_scaler:
                        scaler.unscale_(optimizer)

                    if config.debug.enabled and getattr(config.debug, 'log_gradient_norm', False) and (global_step % config.training.logging_steps == 0):
                        # Compute gradient norm using foreach operations (avoids memory spike from concatenation)
                        grads = [p.grad for p in model.parameters() if p.grad is not None]
                        if grads:
                            # Use _foreach_norm for efficient per-tensor norms, then combine
                            norms = torch._foreach_norm(grads, ord=2)
                            total_norm = torch.stack(norms).norm(2).item()
                            del norms, grads
                        else:
                            total_norm = 0.0
                        monitor.log_scalar('train/grad_norm', total_norm, global_step)

                    # Gradient clipping and non-finite gradient detection.
                    # clip_grad_norm_ returns the total gradient norm BEFORE clipping,
                    # which we use to detect NaN/Inf gradients. This works regardless
                    # of AMP dtype (bfloat16 disables GradScaler, so scale-based
                    # detection is a no-op — this is the real protection).
                    grad_clip_cfg = getattr(config.training, 'gradient_clipping', None)
                    if grad_clip_cfg and getattr(grad_clip_cfg, 'enabled', True):
                        max_norm = getattr(grad_clip_cfg, 'max_norm', 1.0)
                    else:
                        max_norm = float('inf')  # No clipping, but still compute norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    if not torch.isfinite(grad_norm):
                        logger.warning(
                            "Non-finite gradient norm at step %s - skipping optimizer update "
                            "to prevent weight corruption", global_step
                        )
                        monitor.log_scalar('train/nan_grad_skipped', 1.0, global_step)
                        optimizer.zero_grad(set_to_none=True)
                        accum_count = 0
                        # This boundary microbatch had a finite loss and its forward/backward
                        # ran; only the optimizer update was discarded (tracked via the
                        # nan_grad_skipped scalar above). Account its loss like the other
                        # microbatches of the window so epoch loss averaging stays correct,
                        # and free the detached loss tensor before skipping.
                        running_loss += loss_detached.float() * batch_size_current
                        total_train_samples += batch_size_current
                        processed_batches += 1
                        del loss_detached
                        continue

                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)  # Use set_to_none for memory efficiency
                    accum_count = 0
                    global_step += 1
                    try:
                        scheduler.step()
                    except Exception as sched_exc:
                        # keep training even if a rare scheduler state issue occurs
                        logger.warning(f"Scheduler step failed at global_step={global_step}: {sched_exc}")

                    # ASL telemetry sample (todos/ASL_plan.md SS5): rides the
                    # already-detached logits of the boundary microbatch, only on
                    # optimizer-update boundaries (accumulation-window hygiene).
                    # Internally throttled to every interval_updates updates.
                    try:
                        asl_drive.on_update(_saved_tag_logits, tag_labels, global_step, epoch)
                    except Exception as _asl_exc:
                        logger.warning(f"ASL telemetry update failed at step {global_step}: {_asl_exc}")

                    # Count optimizer updates and handle periodic checkpointing
                    try:
                        training_state.optimizer_updates += 1
                        save_every = int(getattr(getattr(config, 'training', {}), 'save_steps', 0) or 0)
                    except Exception:
                        save_every = 0

                    if save_every > 0 and (training_state.optimizer_updates % save_every == 0):
                        try:
                            # Include current batch in sample count for accurate averaging
                            # Uses loss_detached (tensor) instead of loss_item (float/None) to avoid sync
                            current_train_loss = (running_loss + (loss_detached * batch_size_current)) / max(1, total_train_samples + batch_size_current)
                        except Exception:
                            current_train_loss = float('nan')

                        current_train_loss_value = _loss_to_float(current_train_loss)
                        training_state.epoch = epoch + 1
                        training_state.global_step = global_step
                        training_state.train_loss = current_train_loss_value
                        # Track mid-epoch position for resume. Batch `step` has been
                        # consumed, so resume at the NEXT batch (consistent with the
                        # soft-stop and one-shot save paths) - recording `step` would
                        # retrain one batch and shift accumulation windows on resume.
                        training_state.batch_in_epoch = step + 1
                        training_state.sample_in_epoch = (
                            (step + 1) * train_loader.batch_size + sample_pos_correction
                        )
                        training_state.is_epoch_boundary = False
                        training_state.completed_epochs = epoch

                        try:
                            checkpoint_manager.save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch + 1,
                                step=global_step,
                                metrics={'train_loss': current_train_loss_value},
                                training_state=training_state,
                                is_best=False,
                                config=config.to_dict(),
                                train_loader=train_loader,
                                scaler=scaler,
                            )
                            logger.info(
                                "Periodic save: optimizer_update=%s, global_step=%s",
                                training_state.optimizer_updates,
                                global_step,
                            )
                        except Exception as e:
                            logger.warning("Periodic save failed: %s", e)

                # Accumulate loss weighted by batch size for proper per-sample averaging.
                # GPU-resident accumulation avoids per-microbatch sync; the .float() cast
                # promotes bf16 loss to fp32 inside the running sum to preserve precision
                # across thousands of microbatches per epoch.
                running_loss += loss_detached.float() * batch_size_current
                total_train_samples += batch_size_current
                processed_batches += 1
                
                # Explicitly delete detached loss to free VRAM
                del loss_detached

                # Early soft stop check - only stop at safe points after accumulation.
                # The sentinel poll is throttled (exists() is a syscall in the hot loop)
                # and lives at the TOP of this loop body; its RESULT is latched into
                # soft_stop_event, which is read here every microbatch. Testing the
                # throttled expression directly was a real bug: the save below
                # additionally requires accum_count == 0, i.e.
                # step % SENTINEL_CHECK_INTERVAL == 0 AND step % accum == accum - 1 in the
                # same iteration. By CRT that is solvable only when
                # gcd(SENTINEL_CHECK_INTERVAL, accum) == 1 - at accum=8 or 10 the two
                # conditions could NEVER coincide, so STOP_TRAINING latched
                # soft_stop_pending (disabling the epoch-boundary flush) and then never
                # actually stopped. Latching makes the stop land on the very next
                # accumulation boundary for any accum.
                stop_requested = soft_stop_event.is_set()
                if stop_requested:
                    if soft_stop_pending:
                        soft_stop_wait_steps += 1
                    # A healthy window closes within `accum` microbatches, so this bound is
                    # never reached in normal operation. It exists because the four
                    # `continue` paths above (all-samples-failed batch, NaN loss, non-finite
                    # grad) skip this check entirely, and two of them RESET accum_count. If
                    # such failures recur at a period that resonates with `accum`, no window
                    # ever closes, accum_count never returns to 0, and the stop request would
                    # be ignored until the epoch boundary - potentially six figures of
                    # microbatches away. Such a run is making zero optimizer progress anyway,
                    # so discarding one partial window to honour the stop is strictly better.
                    stall_limit = max(4 * accum, 32)
                    if accum_count > 0 and soft_stop_wait_steps <= stall_limit:
                        if not soft_stop_pending:
                            remaining = max(0, accum - accum_count)
                            logger.info(
                                "Soft stop requested - waiting for accumulation to finish (%s remaining microbatches).",
                                remaining
                            )
                            soft_stop_pending = True
                            soft_stop_wait_steps = 0
                    else:
                        if accum_count > 0:
                            logger.warning(
                                "Soft stop: accumulation window has not closed after %s "
                                "microbatches (accum=%s, accum_count=%s) - discarding the "
                                "partial window and stopping now. Repeated skipped batches "
                                "(failed loads / NaN loss / non-finite gradients) are the "
                                "usual cause; check train/nan_inf_loss_detected and "
                                "train/skipped_batches.",
                                soft_stop_wait_steps, accum, accum_count,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            accum_count = 0
                        logger.info("Soft stop requested - saving checkpoint...")

                        # Resume at the next batch to avoid reprocessing partial accumulation windows
                        save_batch_position = step + 1

                        try:
                            current_train_loss = running_loss / max(1, total_train_samples)
                        except Exception:
                            current_train_loss = float('nan')

                        current_train_loss_value = _loss_to_float(current_train_loss)
                        # Update training state for checkpoint
                        training_state.epoch = epoch + 1
                        training_state.global_step = global_step
                        training_state.train_loss = current_train_loss_value
                        training_state.batch_in_epoch = save_batch_position
                        training_state.sample_in_epoch = (
                            save_batch_position * train_loader.batch_size + sample_pos_correction
                        )
                        training_state.is_epoch_boundary = False
                        training_state.completed_epochs = epoch

                        # CR-043: Clear GPU memory before soft stop checkpoint save
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        try:
                            checkpoint_manager.save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch + 1,
                                step=global_step,
                                metrics={'train_loss': current_train_loss_value},
                                training_state=training_state,
                                is_best=False,
                                config=config.to_dict(),
                                train_loader=train_loader,
                                scaler=scaler,
                            )
                            logger.info(
                                "Soft stop checkpoint saved at global_step=%s, batch_in_epoch=%s (accum_count was %s)",
                                global_step, save_batch_position, accum_count
                            )
                        except Exception as e:
                            logger.error("Soft stop: failed to save checkpoint: %s", e)

                        early_exit = True
                        break

                # One-shot save handling (without stopping) - only at safe points after
                # an optimizer step. Same latch-then-act split as the soft stop above:
                # the throttled poll (hoisted to the top of this loop body) sets a sticky
                # flag, and the flag - not the throttled expression - gates the save.
                # Requiring the poll step and the accumulation boundary to coincide meant
                # SAVE_CHECKPOINT was never honored, and never unlinked so the file
                # lingered, for any accum sharing a factor with SENTINEL_CHECK_INTERVAL.
                if accum_count == 0 and global_step > 0 and save_now_pending:
                    save_now_pending = False
                    # Freeze running_loss to a Python float at snapshot time so the dict
                    # is decoupled from the GPU tensor (which keeps mutating in-place).
                    state_snapshot = {
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'step': step + 1,
                        'running_loss': _loss_to_float(running_loss),
                        'processed_batches': processed_batches,
                        'total_train_samples': total_train_samples
                    }

                    try:
                        current_train_loss = state_snapshot['running_loss'] / max(1, state_snapshot['total_train_samples'])
                    except Exception:
                        current_train_loss = float('nan')

                    current_train_loss_value = _loss_to_float(current_train_loss)
                    # Update training state using frozen snapshot
                    training_state.epoch = state_snapshot['epoch']
                    training_state.global_step = state_snapshot['global_step']
                    training_state.train_loss = current_train_loss_value
                    # Track mid-epoch position for resume
                    training_state.batch_in_epoch = step + 1  # Next batch to process (consistent with soft stop)
                    training_state.sample_in_epoch = (
                        (step + 1) * train_loader.batch_size + sample_pos_correction
                    )
                    training_state.is_epoch_boundary = False
                    training_state.completed_epochs = epoch

                    # CR-043: Clear GPU memory before one-shot checkpoint save
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Save checkpoint (updates last.pt atomically)
                    try:
                        checkpoint_manager.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=state_snapshot['epoch'],
                            step=state_snapshot['global_step'],
                            metrics={'train_loss': current_train_loss_value},
                            training_state=training_state,
                            is_best=False,
                            config=config.to_dict(),
                            train_loader=train_loader,
                            scaler=scaler,
                        )
                        logger.info("One-shot save: checkpoint written at step %s.", state_snapshot['global_step'])
                    except Exception as e:
                        logger.warning("One-shot save: failed to write checkpoint: %s", e)

                    # Clear the one-shot save sentinel
                    try:
                        save_sentinel.unlink()
                    except Exception:
                        pass

                # Log every N optimizer updates (throttled) and ensure first-update write.
                # should_log is True only on the boundary microbatch of a qualifying
                # window, so this fires exactly once per qualifying update (skipped
                # updates never reach here - they `continue` above).
                if should_log:
                    # Uses pre-extracted loss_item and losses_items (tensors already deleted after backward)
                    monitor.log_step(
                        global_step,
                        loss_item,
                        losses_items,
                        optimizer.param_groups[0]['lr'],
                        batch_size_current,
                    )
                # Memory monitoring (check every 2000 updates to reduce psutil overhead)
                # psutil calls are ~1-5ms each; at scale this adds up.
                # accum_count == 0 here means an optimizer update completed on THIS
                # microbatch (it is reset right after stepping), so the check fires
                # once per qualifying update instead of once per microbatch.
                if accum_count == 0 and global_step > 0 and global_step % 2000 == 0:
                    try:
                        mem_stats = mem_monitor.check_memory()
                        # Log to TensorBoard for tracking trends
                        monitor.log_scalar('memory/system_used_gb', mem_stats['system_used_gb'], global_step)
                        monitor.log_scalar('memory/system_percent', mem_stats['system_percent'], global_step)
                        monitor.log_scalar('memory/process_total_gb', mem_stats['total_process_gb'], global_step)
                        monitor.log_scalar('memory/workers_gb', mem_stats['workers_gb'], global_step)
                    except Exception as e:
                        logger.debug(f"Memory monitoring failed: {e}")

                # Step-based training image logging for TensorBoard
                # Also supports manual trigger via 'i' hotkey (creates LOG_IMAGES_NOW sentinel)
                # Sentinel check throttled to every 10 steps to reduce syscall overhead
                image_log_steps = getattr(config.monitor.tb_image_logging, 'image_log_steps', 0)
                manual_image_trigger = (global_step % 10 == 0) and image_log_sentinel.exists()
                should_log_images = (
                    config.training.use_tensorboard
                    and global_step != _last_image_log_step
                    and (
                        manual_image_trigger
                        or (image_log_steps > 0 and global_step % image_log_steps == 0 and global_step > 0)
                    )
                )
                if should_log_images:
                    _last_image_log_step = global_step
                    # Clear manual trigger sentinel if it was used
                    if manual_image_trigger:
                        try:
                            image_log_sentinel.unlink()
                        except FileNotFoundError:
                            pass
                        logger.info(f"Manual image logging triggered at step {global_step}")
                    try:
                        with torch.no_grad():
                            probs = torch.sigmoid(_saved_tag_logits)
                            tag_names = [vocab.index_to_tag.get(i, vocab.unk_token) for i in range(len(vocab.index_to_tag))]
                            raw_image_ids = batch.get("image_id")
                            image_ids: Optional[List[str]] = None
                            if raw_image_ids is not None:
                                image_ids = [
                                    str(x) if x is not None else None for x in raw_image_ids
                                ]
                            monitor.log_predictions(
                                step=global_step,
                                images=images,
                                predictions=probs,
                                targets=tag_labels,
                                tag_names=tag_names,
                                prefix="train",
                                max_images=config.monitor.tb_image_logging.max_samples,
                                topk=config.monitor.tb_image_logging.topk,
                                image_ids=image_ids,
                            )
                            logger.info(f"Logged {config.monitor.tb_image_logging.max_samples} training images to TensorBoard at step {global_step}")
                    except Exception as e:
                        logger.warning(f"Failed to log training images: {e}")

            if stats_queue:
                # Non-blocking drain of augmentation stats (accept both tuple and bare dict)
                while True:
                    try:
                        item = stats_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.warning(f"Error reading stats queue: {e}")
                        break

                    if isinstance(item, tuple) and len(item) == 2:
                        stat_type, stats_data = item
                    elif isinstance(item, dict):
                        # Back-compat: bare payload treated as aug_stats
                        stat_type, stats_data = 'aug_stats', item
                    else:
                        continue

                    if stat_type == 'aug_stats':
                        # Normalize keys to monitor schema and de-dupe semantics
                        sd = dict(stats_data)
                        if 'flip_total' not in sd and 'total_flips' in sd:
                            sd['flip_total'] = sd['total_flips']
                        if 'flip_safe' not in sd and 'safe_flips' in sd:
                            sd['flip_safe'] = sd['safe_flips']
                        if 'flip_skipped_text' not in sd and 'blocked_by_text' in sd:
                            sd['flip_skipped_text'] = sd['blocked_by_text']
                        if 'blocked_by_safety' in sd:
                            sd.setdefault('flip_skipped_unmapped', sd['blocked_by_safety'])
                            sd.setdefault('flip_blocked_safety', sd['blocked_by_safety'])
                        monitor.log_augmentations(global_step, sd)

            # Logging moved into inner loop (above) to avoid missing epoch-boundary steps.

            # Flush incomplete gradient accumulation at epoch boundary.
            # When steps_per_epoch is not divisible by accum, the last few batches
            # accumulate gradients but never reach the accum threshold. Rather than
            # silently discarding them, perform a partial optimizer step with
            # appropriately scaled gradients.
            # A pending soft stop normally defers the flush (the partial window is
            # carried into the next epoch - see the epoch-boundary handler below),
            # but on the FINAL epoch there is no next epoch: flush here so the
            # accumulated work is not lost and the soft-stop checkpoint still saves.
            # The same applies once we have ALREADY carried a window across one
            # boundary (soft_stop_carried): carrying again would let the stop request
            # ride to the end of training without ever checkpointing, and would keep
            # `carry_accum` suppressing the per-epoch reset of running_loss /
            # total_train_samples / processed_batches, turning avg_train_loss into a
            # cumulative average over every epoch since the request.
            is_final_epoch = (epoch + 1 >= config.training.num_epochs)
            defer_flush = soft_stop_pending and not is_final_epoch and not soft_stop_carried
            if accum_count > 0 and not defer_flush:
                logger.info(
                    "Epoch %s: flushing %s/%s accumulated micro-batches at epoch boundary",
                    epoch + 1, accum_count, accum,
                )
                try:
                    # Each micro-batch divided its loss by the FULL accum, but only
                    # accum_count < accum micro-batches accumulated into this partial
                    # window. Rescale the summed gradients by accum/accum_count so this
                    # flush step carries the same effective magnitude as a full window
                    # (otherwise it is under-weighted by accum_count/accum, biasing the
                    # update and effective LR of every non-divisible epoch boundary).
                    # Applied before unscale_; the constant commutes with the GradScaler.
                    if accum > 1 and 0 < accum_count < accum:
                        _partial_scale = accum / accum_count
                        for _p in model.parameters():
                            if _p.grad is not None:
                                _p.grad.mul_(_partial_scale)
                    scaler.unscale_(optimizer)
                    grad_clip_cfg = getattr(config.training, 'gradient_clipping', None)
                    if grad_clip_cfg and getattr(grad_clip_cfg, 'enabled', True):
                        max_norm = getattr(grad_clip_cfg, 'max_norm', 1.0)
                    else:
                        max_norm = float('inf')
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        accum_count = 0
                        global_step += 1
                        try:
                            scheduler.step()
                        except Exception as sched_exc:
                            logger.warning("Scheduler step failed during epoch-boundary flush at global_step=%s: %s", global_step, sched_exc)
                        training_state.optimizer_updates += 1
                    else:
                        logger.warning("Non-finite gradient norm during epoch-boundary flush - discarding partial accumulation")
                        optimizer.zero_grad(set_to_none=True)
                        accum_count = 0
                except Exception as e:
                    logger.warning("Epoch-boundary accumulation flush failed: %s", e)
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0

            # Clear GPU cache after epoch-boundary flush before validation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # If a soft stop was requested, exit training before validation
        # Also check sentinel at epoch boundary in case we missed it in the loop (due to throttling)
        stop_requested = soft_stop_event.is_set() or stop_sentinel.exists()
        if early_exit or stop_requested:
            if early_exit:
                logger.info("Soft stop engaged. Exiting training loop before validation.")
                break

            if accum_count > 0:
                # Only reachable when the flush above deferred (defer_flush): on the
                # final epoch, or after one carry, accum_count is already 0 here.
                soft_stop_pending = True
                soft_stop_carried = True
                logger.info(
                    "Soft stop detected at epoch boundary with %s accumulated microbatches. "
                    "Continuing into next epoch to finish accumulation before stopping.",
                    accum_count,
                )
                continue

            logger.info("Soft stop detected at epoch boundary. Saving checkpoint before exit.")

            try:
                current_train_loss = running_loss / max(1, total_train_samples)
            except Exception:
                current_train_loss = float('nan')

            current_train_loss_value = _loss_to_float(current_train_loss)
            training_state.epoch = epoch + 1
            training_state.global_step = global_step
            training_state.train_loss = current_train_loss_value
            training_state.batch_in_epoch = 0
            training_state.sample_in_epoch = 0
            training_state.is_epoch_boundary = True
            training_state.completed_epochs = epoch + 1

            # CR-043: Clear GPU memory before soft stop checkpoint save
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    step=global_step,
                    metrics={'train_loss': current_train_loss_value},
                    training_state=training_state,
                    is_best=False,
                    config=config.to_dict(),
                    train_loader=train_loader,
                    scaler=scaler,
                )
                logger.info(
                    "Soft stop checkpoint saved at epoch boundary (epoch=%s, global_step=%s)",
                    epoch + 1, global_step
                )
            except Exception as e:
                logger.error("Soft stop: failed to save checkpoint at epoch boundary: %s", e)
            break

        # Clear mid-epoch resume flag after completing the first resumed epoch
        if epoch == start_epoch and is_mid_epoch:
            logger.info(f"Completed resumed epoch {epoch + 1} - cleared mid-epoch flag")
            is_mid_epoch = False

        force_validate = False
        if total_train_samples > 0:
            avg_train_loss = _loss_to_float(running_loss / total_train_samples)  # Per-sample average; sync OK at epoch end
        else:
            # Zero batches trained this epoch. Usually this means we resumed from a
            # checkpoint whose batch_in_epoch lands at/after the end of the epoch: the
            # sampler offset consumes the whole index list, so the replayed epoch yields
            # nothing. (It is also reachable if every batch took a `continue` path -
            # all-samples-failed or NaN loss - which is a data/numerics incident, not a
            # resume artefact.) Dividing by max(1, 0) reported a train_loss of 0.0 that
            # was never measured, and overwrote training_state.train_loss and the
            # TensorBoard series with it. Carry the checkpoint's value forward instead.
            avg_train_loss = float(getattr(training_state, 'train_loss', 0.0) or 0.0)
            # Force validation. A zero-batch epoch does not advance global_step, so the
            # eval_steps cadence below cannot fire on its own and this replayed epoch
            # would produce no training, no validation and no checkpoint at all - on the
            # final epoch that means the resumed process exits having done nothing. The
            # validation pass on the restored model is the entire point of replaying it.
            force_validate = True
            logger.warning(
                "Epoch %s processed 0 training batches (%s skipped). Reusing the "
                "checkpoint's train_loss=%.4f rather than reporting 0.0, and forcing "
                "validation so the restored model is still evaluated.",
                epoch + 1, skipped_batches, avg_train_loss,
            )

        # Log skipped batch statistics for monitoring
        if skipped_batches > 0:
            skip_rate = skipped_batches / (processed_batches + skipped_batches)
            logger.info(f"Epoch {epoch+1}: Skipped {skipped_batches} batches ({skip_rate:.2%} of total)")
            monitor.log_scalar('train/skipped_batches', skipped_batches, global_step)
            monitor.log_scalar('train/skip_rate', skip_rate, global_step)

        # Check if we should run validation this epoch (based on eval_steps)
        # eval_steps=0 means validate every epoch; otherwise validate every N steps
        has_val_loader = val_loader is not None
        should_validate = has_val_loader and (
            eval_steps == 0  # 0 means always validate at epoch end
            # Always validate the first epoch of a FRESH run, so the selection metric has
            # a seed value. `epoch == start_epoch` alone also fired on every resume, which
            # made restoring last_validation_step pointless: the resumed epoch validated
            # unconditionally and then re-anchored the cadence to that global_step.
            or (epoch == start_epoch and last_validation_step == 0)
            or force_validate  # zero-batch replayed epoch: global_step cannot advance
            or global_step - last_validation_step >= eval_steps
        )

        if not should_validate:
            # Skip validation, use cached values from training state
            avg_val_loss = getattr(training_state, 'val_loss', 0.0) or 0.0
            val_f1_macro = getattr(training_state, 'val_f1_macro', 0.0) or 0.0
            val_f1_micro = val_f1_macro  # Approximation when skipping
            val_mAP = getattr(training_state, 'val_mAP', 0.0) or 0.0
            if not has_val_loader:
                logger.info(
                    f"Epoch {epoch+1}: Skipping validation (no validation loader). "
                    f"Using cached: val_loss={avg_val_loss:.4f}, F1={val_f1_macro:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}: Skipping validation (last at step {last_validation_step}, "
                    f"next at step {last_validation_step + eval_steps}). "
                    f"Using cached: val_loss={avg_val_loss:.4f}, F1={val_f1_macro:.4f}"
                )
        else:
            last_validation_step = global_step
            # Mirror into training_state at the point of change so every checkpoint
            # written afterwards (periodic, one-shot, soft stop, best) carries the
            # cadence forward - see TrainingState.last_validation_step.
            training_state.last_validation_step = last_validation_step
            # Validation loop
            model.eval()
            val_loss = torch.tensor(0.0, device=device)  # Keep on GPU to avoid per-batch sync
            # Reset validation metrics for this epoch (CR-040 fix: reuse instead of recreate)
            for metric in val_metrics.values():
                metric.reset()
            total_val_samples = 0  # Track samples for proper loss averaging
            # CPU accumulation feeds the bucketed TB diagnostics and threshold
            # calibration; skip it entirely when neither consumer is active
            # (otherwise ~N x 19K matrices are transferred, held, and discarded).
            calibration_enabled = bool(
                getattr(config, 'threshold_calibration', None)
                and config.threshold_calibration.enabled
            )
            accumulate_val_cpu = config.training.use_tensorboard or calibration_enabled
            all_val_probs = []  # Accumulate for frequency-bucketed metrics / calibration
            all_val_targs = []
            val_h2d_stream = torch.cuda.Stream() if device.type == 'cuda' else None
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader):
                    # Filter out error samples that failed to load
                    error_flags = batch.get('error')
                    if error_flags is not None and isinstance(error_flags, torch.Tensor) and error_flags.any():
                        valid_mask = ~error_flags
                        if valid_mask.sum() == 0:
                            continue  # Skip entirely failed batches
                        batch = {
                            k: v[valid_mask] if isinstance(v, torch.Tensor) and v.size(0) == len(error_flags) else v
                            for k, v in batch.items()
                        }

                    # Transfer tensors to GPU on dedicated H2D stream (overlap with previous batch compute)
                    pmask = batch.get('padding_mask', None)
                    val_h2d_ctx = torch.cuda.stream(val_h2d_stream) if val_h2d_stream is not None else nullcontext()
                    with val_h2d_ctx:
                        images = batch['images'].to(device, non_blocking=True)
                        if use_channels_last:
                            images = images.contiguous(memory_format=torch.channels_last)
                        tag_labels = batch['tag_labels'].to(device, non_blocking=True)
                        if pmask is not None:
                            pmask = pmask.to(device=device, dtype=torch.bool, non_blocking=True)
                    # Sync H2D stream before compute
                    if val_h2d_stream is not None:
                        torch.cuda.current_stream().wait_stream(val_h2d_stream)
                        # See the train-loop comment: mark side-stream allocations as
                        # in-use by the default stream so the allocator cannot recycle
                        # them for the next batch's H2D copy while still being read.
                        images.record_stream(torch.cuda.current_stream())
                        tag_labels.record_stream(torch.cuda.current_stream())
                        if pmask is not None:
                            pmask.record_stream(torch.cuda.current_stream())
                    total_val_samples += images.size(0)  # Count actual samples processed

                    with amp_autocast():
                        outputs = model(images, padding_mask=pmask)
                        loss, _ = criterion(outputs['tag_logits'], tag_labels)
                    # Accumulate loss weighted by batch size for proper per-sample averaging
                    val_loss = val_loss + loss.detach() * images.size(0)

                    # Update streaming metrics (keep on GPU to avoid per-batch sync/transfer)
                    probs = torch.sigmoid(outputs['tag_logits']).float()
                    # Targets must be int/long for torchmetrics (mAP uses precision-recall curves)
                    targs = tag_labels.long()
                    # Skip PAD/UNK columns (indices 0,1) for streaming metrics
                    metric_probs = probs[:, skip_metric_cols:]
                    metric_targs = targs[:, skip_metric_cols:]
                    val_metrics['f1_macro_per_class'].update(metric_probs, metric_targs)
                    val_metrics['f1_micro'].update(metric_probs, metric_targs)
                    val_metrics['map_per_class'].update(metric_probs, metric_targs)
                    val_pos_counts += metric_targs.sum(dim=0)
                    if accumulate_val_cpu:
                        # Compact dtypes for host accumulation: fp16 probs (cast back to
                        # fp32 by the consumers) and bool targets (vs int64 = 8 bytes per
                        # {0,1} value) cut host RAM and PCIe traffic ~5x.
                        all_val_probs.append(probs.to(torch.float16).to('cpu', non_blocking=True))
                        all_val_targs.append(targs.to(torch.bool).to('cpu', non_blocking=True))

                    if val_step == 0 and config.training.use_tensorboard:
                        tag_names = [vocab.index_to_tag.get(i, vocab.unk_token) for i in range(len(vocab.index_to_tag))]
                        monitor.log_predictions(
                            step=global_step,
                            images=images,
                            predictions=probs,
                            targets=tag_labels,
                            tag_names=tag_names,
                            prefix="val",
                            max_images=config.monitor.tb_image_logging.max_samples,
                            topk=config.monitor.tb_image_logging.topk,
                        )

            # Compute metrics (now on CPU to prevent VRAM accumulation)
            val_loss_avg = (val_loss / max(1, total_val_samples)).cpu()

            # Per-class compute, then mean over classes that actually had a positive
            # in this validation draw — see val_metrics initialization above.
            per_class_f1 = val_metrics['f1_macro_per_class'].compute()
            per_class_ap = val_metrics['map_per_class'].compute()
            keep_classes = (val_pos_counts > 0)
            num_supported = int(keep_classes.sum().item())
            if num_supported > 0:
                val_f1_macro = per_class_f1[keep_classes].float().mean().item()
                val_mAP = per_class_ap[keep_classes].float().mean().item()
            else:
                val_f1_macro = 0.0
                val_mAP = 0.0
            val_f1_micro = val_metrics['f1_micro'].compute().item()
            avg_val_loss = val_loss_avg.item()  # already CPU
            logger.debug(
                f"Macro metrics averaged over {num_supported}/{num_metric_labels} "
                f"classes with positive support this epoch."
            )

            # Reset metrics for next epoch
            for metric in val_metrics.values():
                metric.reset()
            val_pos_counts.zero_()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                f"Val F1(macro): {val_f1_macro:.4f}, Val F1(micro): {val_f1_micro:.4f}, Val mAP: {val_mAP:.4f}"
            )
            monitor.log_validation(global_step, {'loss': avg_val_loss, 'f1_macro': val_f1_macro, 'f1_micro': val_f1_micro, 'mAP': val_mAP})
            monitor.log_scalar('train/loss_epoch', avg_train_loss, global_step)

            # Frequency-bucketed diagnostic metrics + threshold calibration (both
            # consume the CPU-accumulated probability/target matrices)
            try:
                cat_probs = None
                cat_targs = None
                if all_val_probs:
                    cat_probs = torch.cat(all_val_probs, dim=0).float()
                    del all_val_probs
                    cat_targs = torch.cat(all_val_targs, dim=0)
                    del all_val_targs
                    freq_bins = getattr(config.validation, 'frequency_bins', None) or [300, 500, 1000, 5000, 10000, float('inf')]
                    tag_names = [vocab.index_to_tag.get(i, vocab.unk_token) for i in range(len(vocab.index_to_tag))]

                if cat_probs is not None and config.training.use_tensorboard:
                    pred_thr = float(config.inference.prediction_threshold)
                    mean_active = (cat_probs[:, skip_metric_cols:] > pred_thr).float().sum(dim=1).mean().item()
                    monitor.log_scalar('val/mean_active', mean_active, global_step)
                    bucket_metrics = FrequencyBucketMetrics(
                        tag_frequencies=vocab.tag_frequencies,
                        frequency_bins=freq_bins,
                        tag_names=tag_names,
                        skip_indices=[0, 1],
                    )
                    bucketed_results = bucket_metrics.compute_bucketed_metrics(
                        cat_probs, cat_targs, threshold=pred_thr,
                    )
                    for bucket_name, metrics in bucketed_results.items():
                        for metric_name, value in metrics.items():
                            monitor.log_scalar(f"val_bucketed/{bucket_name}/{metric_name}", value, global_step)
                    bucket_summary = ", ".join(
                        f"{b}: F1={m['f1_macro']:.3f} ({int(m['num_tags'])} tags)"
                        for b, m in bucketed_results.items() if m['num_tags'] > 0
                    )
                    logger.info(f"Bucketed metrics: {bucket_summary}")

                # ASL val telemetry (todos/ASL_plan.md SS5): non-GT score
                # histogram (clip watch band), sibling-gap per confusable group,
                # val-side dp_hard and per-decile EPR. Pure consumer of the
                # already-accumulated CPU prob/target matrices.
                if cat_probs is not None:
                    try:
                        asl_drive.compute_val(cat_probs, cat_targs, global_step, epoch)
                    except Exception as _asl_exc:
                        logger.warning(f"ASL val telemetry failed: {_asl_exc}")

                # Threshold calibration (reuses accumulated tensors). Deliberately NOT
                # gated on use_tensorboard: calibrated_thresholds is a training artifact
                # consumed at inference, not a logging concern - only the TB scalar
                # writes below remain TB-gated.
                if cat_probs is not None and calibration_enabled:
                    try:
                        calibrator = ThresholdCalibrator(
                            mode=config.threshold_calibration.mode,
                            default_threshold=config.threshold_calibration.default_threshold,
                            search_min=config.threshold_calibration.search_min,
                            search_max=config.threshold_calibration.search_max,
                            search_step=config.threshold_calibration.search_step,
                        )
                        calibrated_thresholds = calibrator.calibrate(
                            cat_probs, cat_targs, tag_names=tag_names,
                            skip_indices=[0, 1], frequency_bins=freq_bins,
                            tag_frequencies=vocab.tag_frequencies,
                        )
                        save_path = config.threshold_calibration.save_path
                        ThresholdCalibrator.save(calibrated_thresholds, save_path)
                        logger.info(f"Calibrated thresholds saved to {save_path}")
                        thresh_summary = ", ".join(
                            f"{k}: {v:.3f}" for k, v in calibrated_thresholds.items()
                        )
                        logger.info(f"Calibrated thresholds ({config.threshold_calibration.mode}): {thresh_summary}")
                        if config.training.use_tensorboard:
                            for name, thresh_val in calibrated_thresholds.items():
                                monitor.log_scalar(f"val_threshold/{name}", thresh_val, global_step)
                    except Exception as e:
                        logger.warning(f"Failed to calibrate thresholds: {e}")
            except Exception as e:
                logger.warning(f"Failed to compute bucketed metrics: {e}")
            finally:
                # Lists may already be deleted after torch.cat above
                try:
                    del all_val_probs, all_val_targs
                except NameError:
                    pass
                try:
                    del cat_probs, cat_targs
                except NameError:
                    pass

        # Restore train mode after validation so checkpoint saving
        # and any inter-epoch operations see consistent model state (dropout/batchnorm).
        model.train()

        # Scheduler already stepped per optimizer update; just read the last LR here.
        try:
            current_lr = scheduler.get_last_lr()[0]
        except Exception:
            current_lr = optimizer.param_groups[0]['lr']
        # Note: Learning rate is already logged in monitor.log_step() during training

        training_state.epoch = epoch + 1
        training_state.global_step = global_step
        training_state.train_loss = avg_train_loss
        training_state.val_loss = avg_val_loss
        training_state.val_f1_macro = val_f1_macro
        training_state.val_mAP = val_mAP
        training_state.learning_rates.append(current_lr)

        # Model-selection scalar. Configurable because val_f1_macro is evaluated
        # at a single FROZEN global threshold (config.inference.prediction_threshold)
        # and is therefore calibration-coupled: any change that shifts the global
        # probability scale without changing ranking -- notably an ASL gamma_neg
        # step -- moves it by far more than a real epoch's ranking improvement,
        # and in the wrong direction (lowering gamma_neg pushes probabilities down,
        # so a genuine gain reads as a large drop, burning patience). val_mAP is
        # threshold-free and is the default. See selection_metric in the config.
        selection_value = _SELECTION_METRICS[selection_metric](
            val_f1_macro, val_f1_micro, val_mAP
        )

        # --- TensorBoard: periodic flush ---
        try:
            monitor.flush()
        except Exception:
            pass

        # Checkpointing and early stopping on the configured selection metric
        is_best = False
        # Handle burn-in (ignore early-stopping decisions for first N epochs)
        if burn_in_epochs > 0 and (epoch + 1) <= burn_in_epochs:
            # Only record epochs where validation actually ran. On a skipped epoch
            # selection_value is a stale cached metric re-read from training_state, so
            # appending it double-counts the previous epoch and drags the baseline
            # toward it - and now that the window is checkpointed, that duplicate would
            # survive restarts instead of dying with the process.
            if should_validate:
                _burn_in_vals.append(float(selection_value))
                # Mirror into training_state at the point of change so a soft stop inside
                # the burn-in window resumes with the full window, not just the tail.
                training_state.burn_in_values = list(_burn_in_vals)
            prev_best_for_log = training_state.best_metric  # Capture before any modifications
            # Track best during burn-in to avoid losing a great model. Only act on
            # epochs where validation actually ran; on a validation-skipped epoch
            # the selection metric is a stale cached value and must not move best/patience.
            if should_validate and selection_value > training_state.best_metric + es_threshold:
                training_state.best_metric = selection_value
                training_state.best_epoch = epoch + 1
                # Don't save is_best during burn-in: early metrics are unreliable and
                # best_metric gets reset at burn-in end, leaving a stale "best" checkpoint.
            # On the last burn-in epoch, reset baseline to a robust summary.
            # `_burn_in_vals` can be empty if no epoch in the window ever validated
            # (eval_steps cadence, or no val_loader): np.max([]) raises and np.median([])
            # is nan, either of which would corrupt best_metric. Leave the baseline alone
            # in that case - the first validated epoch re-seeds it.
            if (epoch + 1) == burn_in_epochs and not _burn_in_vals:
                logger.warning(
                    "Early-stopping burn-in ended (epochs=%d) with no validated epoch; "
                    "leaving best_metric at %s and zeroing patience.",
                    burn_in_epochs, training_state.best_metric,
                )
                training_state.patience_counter = 0
            elif (epoch + 1) == burn_in_epochs:
                try:
                    if burn_in_strategy == "last":
                        baseline = float(_burn_in_vals[-1])
                    elif burn_in_strategy == "mean":
                        baseline = float(np.mean(_burn_in_vals))
                    elif burn_in_strategy == "max":
                        baseline = float(np.max(_burn_in_vals))
                    else:  # default: median
                        baseline = float(np.median(_burn_in_vals))
                except Exception:
                    baseline = float(np.median(_burn_in_vals))
                # Keep the better of baseline or actual best achieved during burn-in
                best_during_burnin = float(np.max(_burn_in_vals))
                prev_best = prev_best_for_log  # Use value captured before any modifications this epoch
                training_state.best_metric = max(baseline, best_during_burnin)
                training_state.patience_counter = 0
                logger.info(
                    "Early-stopping burn-in complete (epochs=%d, strategy=%s). "
                    "Baseline set to %.4f (best during burn-in %.4f, prev best %.4f).",
                    burn_in_epochs, burn_in_strategy, baseline, best_during_burnin, prev_best,
                )
            # During burn-in: patience not updated, but best is still tracked
        else:
            # LR-aware early stopping: only count patience when LR has dropped
            # significantly within a cycle (in the "fine-tuning" phase)
            current_lr = scheduler.get_last_lr()[0]
            cycle_max_lr = scheduler.max_lr  # Already accounts for gamma decay
            lr_ratio = current_lr / cycle_max_lr if cycle_max_lr > 0 else 1.0

            # Only update best/patience on epochs where validation actually ran.
            # When validation is skipped (eval_steps cadence), the selection metric
            # is a stale cached value: it never beats best, so without this guard the
            # `elif lr_ratio < 0.5` branch would advance patience toward the early-
            # stop limit on an epoch that produced no new metric.
            if should_validate and selection_value > training_state.best_metric + es_threshold:
                training_state.best_metric = selection_value
                training_state.patience_counter = 0
                training_state.best_epoch = epoch + 1
                is_best = True
            elif should_validate and lr_ratio < 0.5:
                # Only count patience when LR < 50% of cycle max (fine-tuning phase)
                training_state.patience_counter += 1
            # else: During warmup/early-cycle phase, don't increment patience
            # This prevents false early stops during cosine-induced plateaus

        # Respect "save_best_only": skip cadence saves unless this is a new best.
        # Only handle best-at-epoch saves here; periodic saves happen in-loop
        if is_best:
            # Ensure GPU state is consistent and free memory before checkpoint save
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Mark as epoch boundary since this is saved at end of epoch
            training_state.is_epoch_boundary = True
            training_state.batch_in_epoch = 0
            training_state.sample_in_epoch = 0
            training_state.completed_epochs = epoch + 1
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=global_step,
                metrics={
                    'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
                    'val_f1_macro': val_f1_macro, 'val_f1_micro': val_f1_micro,
                    'val_mAP': val_mAP,
                    # Records WHICH scalar made this the best checkpoint, so a
                    # resume can refuse to compare across metric scales.
                    'selection_metric': selection_metric,
                },
                training_state=training_state,
                is_best=True,
                config=config.to_dict(),
                train_loader=train_loader,
                scaler=scaler,
            )

        if patience and training_state.patience_counter >= patience:
            logger.info("Early stopping triggered: no improvement in %s for %s epochs", selection_metric, patience)
            break

    # Clear the one-shot SAVE_CHECKPOINT sentinel on every exit path. It is normally
    # unlinked when consumed, but a soft stop / early stop that breaks out while the
    # latch is still pending would leave the file on disk, and the NEXT run would then
    # write a checkpoint at its first safe point for no reason. (STOP_TRAINING has the
    # equivalent guard at startup; this one is cheaper to do on the way out.)
    try:
        save_sentinel.unlink()
        logger.info("Cleared pending SAVE_CHECKPOINT sentinel on exit.")
    except FileNotFoundError:
        pass
    except Exception as _save_sentinel_exc:
        logger.debug("Could not clear SAVE_CHECKPOINT sentinel: %s", _save_sentinel_exc)

    # --- TensorBoard: final hparams snapshot ---
    try:
        to_dict = getattr(config, "to_dict", None)
        hparams = to_dict() if callable(to_dict) else (vars(config) if hasattr(config, "__dict__") else {})
        final_metrics = {}
        if 'avg_val_loss' in locals():
            final_metrics["final/val_loss"] = float(avg_val_loss)
        if 'avg_train_loss' in locals():
            final_metrics["final/train_loss"] = float(avg_train_loss)
        final_metrics[f"final/best_{selection_metric}"] = float(training_state.best_metric)
        monitor.log_hyperparameters(hparams, final_metrics if final_metrics else {"final/placeholder": 1})
    except Exception:
        pass

    # Guaranteed resource cleanup on all exit paths
    logger.debug("Cleaning up training resources...")

    # Shutdown async checkpoint writer first (waits for pending saves)
    try:
        checkpoint_manager.shutdown(wait=True, timeout=300.0)
        logger.debug("Checkpoint manager shutdown successfully")
    except Exception as e:
        logger.warning(f"Error shutting down checkpoint manager: {e}")

    # Synchronize CUDA and clear cache before final cleanup
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.debug("CUDA synchronized and cache cleared")
    except Exception as e:
        logger.warning(f"Error during CUDA cleanup: {e}")

    # Close monitor (flushes TensorBoard)
    try:
        monitor.close()
        logger.debug("Monitor closed successfully")
    except Exception as e:
        logger.warning(f"Error closing monitor: {e}")

    # Stop background validators
    try:
        for _loader in (train_loader, val_loader):
            ds = getattr(_loader, "dataset", None)
            validator = getattr(ds, "validator", None) if ds is not None else None
            if validator is not None and hasattr(validator, "stop"):
                validator.stop()
        logger.debug("Background validators stopped")
    except Exception as e:
        logger.warning(f"Error stopping validators: {e}")

    # Clean up stats queue
    try:
        if stats_queue is not None:
            # Drain any remaining items
            while not stats_queue.empty():
                try:
                    stats_queue.get_nowait()
                except queue.Empty:
                    break
            # Close the queue
            stats_queue.close()
            stats_queue.join_thread()
            logger.debug("Stats queue cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up stats queue: {e}")

    logger.debug("Training resource cleanup complete")

def main():
    """Main entry point for training script."""
    parser = create_config_parser()
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Setup logging
    listener = setup_logging(
        log_level=config.log_level,
        log_dir=config.log_dir,
        log_to_file=config.file_logging_enabled,
        json_console=True, # Or get from config if you add it
    )

    if args.validate_only:
        try:
            config.validate()
            logger.info("Configuration is valid.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)

    try:
        train_with_orientation_tracking(config)
    finally:
        if listener:
            listener.stop()


if __name__ == "__main__":
    main()
