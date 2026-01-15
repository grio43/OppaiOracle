#!/usr/bin/env python3
"""
================================================================================
DEPRECATED: This file is deprecated and will be removed in a future version.
            Please use 'train_direct.py' for training instead.
================================================================================

Training script using PyTorch Lightning.

Builds a real DataModule from dataset_loader.create_dataloaders and aligns
Lightning settings with the unified configuration.

This file is kept for backwards compatibility with existing scripts that may
reference it. New code should use train_direct.py which offers better
performance and more features.
"""

# Module-level deprecation marker for programmatic detection
__deprecated__ = True
__deprecated_since__ = "2025-01"
__replacement__ = "train_direct.py"

import warnings

warnings.warn(
    "The 'train_lightning.py' file is deprecated and will be removed in a future version. "
    "Please use 'train_direct.py' for training.",
    DeprecationWarning,
    stacklevel=2
)

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

logger = logging.getLogger(__name__)
import torch

from Configuration_System import load_config
from lightning_module import LitOppai
from vocab_utils import load_vocab, compute_vocab_hash


class OppaiDataModule(pl.LightningDataModule):
    """Wrap dataset_loader.create_dataloaders() to supply actual data."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.train_loader: torch.utils.data.DataLoader | None = None
        self.val_loader: torch.utils.data.DataLoader | None = None
        self.vocab = None

    def setup(self, stage: str | None = None) -> None:
        from dataset_loader import create_dataloaders  # lazy import to avoid side effects at import-time

        # Validate config structure early
        if not hasattr(self.config, "data"):
            raise ValueError(
                "Configuration missing 'data' section. "
                "Please provide valid unified_config.yaml"
            )

        data_cfg = self.config.data

        if not hasattr(self.config, "validation"):
            raise ValueError(
                "Configuration missing 'validation' section. "
                "Please provide valid unified_config.yaml"
            )

        val_cfg = self.config.validation

        # Validate storage locations
        if not hasattr(data_cfg, "storage_locations"):
            raise ValueError(
                "Configuration missing data.storage_locations. "
                "Please specify data storage paths in config."
            )

        storage_locations = data_cfg.storage_locations

        if not storage_locations:
            raise ValueError(
                "No storage locations configured. "
                "Please add at least one storage location to config.data.storage_locations"
            )

        # Find enabled storage location
        active_data_path: str | None = None

        for i, loc in enumerate(storage_locations):
            # Validate location structure
            if not hasattr(loc, "enabled"):
                raise ValueError(
                    f"Storage location {i} missing 'enabled' field"
                )
            if not hasattr(loc, "path"):
                raise ValueError(
                    f"Storage location {i} missing 'path' field"
                )

            if loc.enabled:
                active_data_path = str(loc.path)
                logger.info(f"Using storage location: {active_data_path}")
                break

        if not active_data_path:
            raise ValueError(
                "No enabled storage location found in config.data.storage_locations. "
                "Please enable at least one storage location."
            )

        vocab_path = getattr(self.config, "vocab_path", None)
        if not vocab_path:
            # Fallback: allow data.vocab_dir/vocabulary.json
            vp = Path(getattr(getattr(self.config, "data", None), "vocab_dir", "./")) / "vocabulary.json"
            vocab_path = str(vp)

        self.train_loader, self.val_loader, self.vocab = create_dataloaders(
            data_cfg,
            val_cfg,
            vocab_path,
            active_data_path,
            distributed=bool(getattr(getattr(self.config, "training", None), "distributed", False)),
            rank=int(getattr(getattr(self.config, "training", None), "local_rank", -1) or -1),
            world_size=int(getattr(getattr(self.config, "training", None), "world_size", 1) or 1),
            seed=int(getattr(getattr(self.config, "training", None), "seed", 42) or 42),
            debug_config=getattr(self.config, "debug", None),
        )

    def train_dataloader(self):
        assert self.train_loader is not None, "Call setup() before requesting train_dataloader"
        return self.train_loader

    def val_dataloader(self):
        assert self.val_loader is not None, "Call setup() before requesting val_dataloader"
        return self.val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OppaiOracle model with Lightning")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    # DataModule
    datamodule = OppaiDataModule(config)

    # Determine vocab size from canonical vocabulary file or fall back to model config
    vocab_file = Path(getattr(config, "vocab_path", "./vocabulary.json"))
    if vocab_file.exists():
        vocab_list = load_vocab(str(vocab_file))
        vocab_size = len(vocab_list)
        _ = compute_vocab_hash(vocab_list)
    else:
        # Fallback to model.num_labels if provided, else zero
        vocab_size = int(getattr(getattr(config, "model", None), "num_labels", 0) or 0)

    model = LitOppai(config=config, vocab_size=vocab_size)

    # Validate required config attributes for checkpointing
    if not hasattr(config, "output_root"):
        raise ValueError(
            "Configuration missing 'output_root'. "
            "Please specify output directory in config."
        )
    if not hasattr(config, "experiment_name"):
        raise ValueError(
            "Configuration missing 'experiment_name'. "
            "Please specify experiment name in config."
        )

    # Checkpointing & callbacks
    checkpoint_dir = Path(config.output_root) / config.experiment_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(checkpoint_dir), save_top_k=-1)
    _training_cfg = getattr(config, "training", None)
    early_stop_patience = int(getattr(_training_cfg, "early_stopping_patience", 3) or 3)
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=early_stop_patience,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # Optional resume
    resume_from = str(getattr(getattr(config, "training", None), "resume_from", "none") or "none").lower()
    ckpt_path: str | None = None
    if resume_from not in {"none", "no"}:
        if resume_from in {"latest", "last"}:
            candidate = checkpoint_dir / "last.ckpt"
            ckpt_path = str(candidate) if candidate.exists() else None
        elif resume_from == "best":
            # Find checkpoint with best validation loss by parsing filenames
            # ModelCheckpoint saves as: epoch=X-step=Y-val_loss=Z.ckpt or val/loss=Z
            # Use regex for robust parsing that handles various filename patterns
            best_ckpt = None
            best_loss = float("inf")

            # Regex pattern to match val_loss=X.XXX or val/loss=X.XXX
            # Captures the numeric value (possibly negative, with decimals)
            loss_pattern = re.compile(r'(?:val_loss|val/loss)=(-?[\d.]+)')

            for ckpt in checkpoint_dir.glob("*.ckpt"):
                name = ckpt.stem
                match = loss_pattern.search(name)
                if match:
                    try:
                        loss_val = float(match.group(1))
                        if loss_val < best_loss:
                            best_loss = loss_val
                            best_ckpt = ckpt
                    except ValueError:
                        # Could not parse loss value, skip this checkpoint
                        logger.debug(f"Could not parse loss from checkpoint: {name}")
                        continue

            if best_ckpt is None:
                # Fallback to most recent by modification time
                ckpts = list(checkpoint_dir.glob("*.ckpt"))
                if ckpts:
                    best_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
                    logger.info(f"No val_loss in checkpoint names, using most recent: {best_ckpt.name}")
            ckpt_path = str(best_ckpt) if best_ckpt else None
        else:
            candidate_path = Path(resume_from)
            ckpt_path = str(candidate_path) if candidate_path.exists() else None

    # Strictness for resuming
    _training = getattr(config, "training", None)
    resume_strict = bool(getattr(_training, "resume_strict", False)) if _training else False
    if hasattr(model, "strict_loading"):
        model.strict_loading = resume_strict
    else:
        logger.debug("Model does not support strict_loading attribute")

    # Map AMP settings to PL precision strings
    use_amp = bool(getattr(getattr(config, "training", None), "use_amp", True))
    amp_dtype = str(getattr(getattr(config, "training", None), "amp_dtype", "bfloat16")).lower()
    if use_amp:
        if amp_dtype not in {"bfloat16", "bf16"}:
            raise ValueError(f"Only bfloat16 AMP is supported, got '{amp_dtype}'.")
        if not torch.cuda.is_available():
            raise RuntimeError("bfloat16 AMP requested but CUDA is not available.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 AMP requested but CUDA device does not support bf16.")
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    # Grad accumulation & clipping
    accumulate = int(getattr(getattr(config, "training", None), "gradient_accumulation_steps", 1) or 1)
    _train = getattr(config, "training", None)
    _clip = getattr(_train, "gradient_clipping", None) if _train is not None else None
    clip_enabled = bool(_clip and getattr(_clip, "enabled", False))
    clip_val = float(getattr(_clip, "max_norm", 0.0)) if clip_enabled else 0.0

    trainer = pl.Trainer(
        max_epochs=int(getattr(getattr(config, "training", None), "num_epochs", 1) or 1),
        precision=precision,
        accumulate_grad_batches=accumulate,
        gradient_clip_val=clip_val,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=int(getattr(getattr(config, "training", None), "logging_steps", 50) or 50),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
