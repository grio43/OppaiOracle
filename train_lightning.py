#!/usr/bin/env python3
"""Training script using PyTorch Lightning."""

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
    """Minimal Lightning DataModule placeholder."""

    def __init__(self, config: any):
        super().__init__()
        self.config = config

    def setup(self, stage: str | None = None) -> None:  # pragma: no cover - placeholder
        pass

    def train_dataloader(self):  # pragma: no cover - placeholder
        return torch.utils.data.DataLoader([])

    def val_dataloader(self):  # pragma: no cover - placeholder
        return torch.utils.data.DataLoader([])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OppaiOracle model with Lightning")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Build DataModule and LightningModule
    datamodule = OppaiDataModule(config)
    # Determine the size of the tag vocabulary.  Prefer reading from the
    # canonical vocabulary file if it exists in the project artifacts.  If not
    # available, fall back to the configured num_tags.
    vocab_size = 0
    try:
        vocab_file = Path(config.project.artifacts.vocab_file)
    except Exception:
        vocab_file = None
    if vocab_file is not None and vocab_file.exists():
        vocab_list = load_vocab(str(vocab_file))
        vocab_size = len(vocab_list)
        _ = compute_vocab_hash(vocab_list)
    else:
        vocab_size = getattr(config, "num_tags", 0)

    model = LitOppai(config=config, vocab_size=vocab_size)

    checkpoint_dir = Path(getattr(config.project.artifacts, "checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(checkpoint_dir), save_top_k=-1)
    early_stopping = pl.callbacks.EarlyStopping(monitor="val/loss", mode="min", patience=3)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # Determine checkpoint path for resuming. Supports "none", "latest"/"last",
    # "best", or an explicit path.
    resume_from = getattr(getattr(config.training, "resume_from", None), "__str__", lambda: None)()
    ckpt_path: str | None = None
    if resume_from and resume_from.lower() not in {"none", "no"}:
        lower = resume_from.lower()
        if lower in {"latest", "last"}:
            candidate = checkpoint_dir / "last.ckpt"
            ckpt_path = str(candidate) if candidate.exists() else None
        elif lower == "best":
            ckpts = sorted(checkpoint_dir.glob("*.ckpt"))
            ckpt_path = str(ckpts[-1]) if ckpts else None
        else:
            candidate_path = Path(resume_from)
            ckpt_path = str(candidate_path) if candidate_path.exists() else None

    # Pass strict_loading flag to model
    model.strict_loading = getattr(config.training, "resume_strict", False)

    trainer = pl.Trainer(
        max_epochs=getattr(config.training, "num_epochs", 1),
        precision=getattr(config.trainer, "precision", 32),
        accumulate_grad_batches=getattr(config.trainer, "accumulate_grad_batches", 1),
        gradient_clip_val=getattr(config.trainer, "gradient_clip_val", 0.0),
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=getattr(getattr(config, "trainer", {}), "log_every_n_steps", 50),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
