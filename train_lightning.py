#!/usr/bin/env python3
"""Training script using PyTorch Lightning.

Builds a real DataModule from dataset_loader.create_dataloaders and aligns
Lightning settings with the unified configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from safetensors.torch import save_file
import torch

from Configuration_System import load_config
from lightning_module import LitOppai
from vocab_utils import load_vocab, compute_vocab_hash, diff_vocab  # noqa: F401 - imported for side effects/logging


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

        data_cfg = getattr(self.config, "data", None)
        val_cfg = getattr(self.config, "validation", None)

        # Pick first enabled storage location
        active_data_path: str | None = None
        storage_locations = getattr(data_cfg, "storage_locations", []) or []
        for loc in storage_locations:
            try:
                enabled = bool(loc.get("enabled", False)) if isinstance(loc, dict) else bool(getattr(loc, "enabled", False))
                path = loc.get("path") if isinstance(loc, dict) else getattr(loc, "path", None)
            except Exception:
                enabled, path = False, None
            if enabled and path:
                active_data_path = str(path)
                break
        if not active_data_path:
            raise RuntimeError("No enabled storage location found in config.data.storage_locations")

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

    # Checkpointing & callbacks
    checkpoint_dir = Path(config.output_root) / config.experiment_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(checkpoint_dir), save_top_k=-1)
    early_stopping = pl.callbacks.EarlyStopping(monitor="val/loss", mode="min", patience=int(getattr(getattr(config, "training", None), "early_stopping_patience", 3) or 3))
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # Optional resume
    resume_from = str(getattr(getattr(config, "training", None), "resume_from", "none") or "none").lower()
    ckpt_path: str | None = None
    if resume_from not in {"none", "no"}:
        if resume_from in {"latest", "last"}:
            candidate = checkpoint_dir / "last.ckpt"
            ckpt_path = str(candidate) if candidate.exists() else None
        elif resume_from == "best":
            ckpts = sorted(checkpoint_dir.glob("*.ckpt"))
            ckpt_path = str(ckpts[-1]) if ckpts else None
        else:
            candidate_path = Path(resume_from)
            ckpt_path = str(candidate_path) if candidate_path.exists() else None

    # Strictness for resuming
    try:
        model.strict_loading = bool(getattr(getattr(config, "training", None), "resume_strict", False))
    except Exception:
        pass

    # Map AMP settings to PL precision strings
    use_amp = bool(getattr(getattr(config, "training", None), "use_amp", True))
    amp_dtype = str(getattr(getattr(config, "training", None), "amp_dtype", "bfloat16")).lower()
    if use_amp and amp_dtype in {"bfloat16", "bf16"}:
        precision = "bf16-mixed"
    elif use_amp:
        precision = "16-mixed"
    else:
        precision = 32

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
