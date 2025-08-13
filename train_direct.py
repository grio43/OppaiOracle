#!/usr/bin/env python3
"""
Direct training script for the anime image tagger.

This entry point wires together the dataset, model, loss functions and
optimiser into a simple training loop.  It demonstrates how to work with the
orientation‑aware dataloader defined in ``HDF5_loader.py`` and how to
instantiate the model with a dynamic tag dimension derived from the
vocabulary.  Mixed precision training and gradient accumulation are
supported.

The configuration dictionary at the top of ``train_direct`` should be
customised for your hardware and dataset.  In particular, update the
``data_dir``, ``json_dir`` and ``vocab_path`` fields to point at your
images, annotation JSONs and vocabulary file respectively.  The batch size,
number of epochs and learning rate can be tuned based on VRAM capacity and
convergence behaviour.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import torch
from torch.cuda.amp import GradScaler, autocast

from HDF5_loader import create_dataloaders
from model_architecture import create_model
from loss_functions import MultiTaskLoss, AsymmetricFocalLoss


def train_direct() -> None:
    # Basic configuration.  Adjust these values for your environment.
    config: Dict[str, Any] = {
        "learning_rate": 4e-4,
        "batch_size": 32,
        "gradient_accumulation": 2,
        "num_epochs": 8,
        "warmup_steps": 10_000,
        "weight_decay": 0.01,
        "label_smoothing": 0.05,
        "data_dir": Path("data/images"),
        "json_dir": Path("data/annotations"),
        "vocab_path": Path("vocabulary.json"),
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "amp": True,
        # Validation batch size factor.  If None, defaults to the training batch size.
        "val_batch_size": None,
    }
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    device = torch.device(config["device"])
    # Create dataloaders and vocabulary
    train_loader, val_loader, vocab = create_dataloaders(
        data_dir=config["data_dir"],
        json_dir=config["json_dir"],
        vocab_path=config["vocab_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        frequency_sampling=True,
        val_batch_size=config["val_batch_size"],
    )
    # Determine the number of tags and ratings from the vocabulary
    num_tags = len(vocab.tag_to_index)
    num_ratings = len(vocab.rating_to_index)
    logger.info(f"Creating model with {num_tags} tags and {num_ratings} ratings")
    # Instantiate model.  Other architectural parameters can be overridden here.
    model = create_model(
        hidden_size=1280,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_tags=num_tags,
        num_ratings=num_ratings,
        patch_size=16,
        gradient_checkpointing=True,
    )
    model.to(device)
    # Loss function
    criterion = MultiTaskLoss(
        tag_loss_weight=0.9,
        rating_loss_weight=0.1,
        tag_loss_fn=AsymmetricFocalLoss(
            gamma_pos=1.0,
            gamma_neg=3.0,
            alpha=0.75,
            label_smoothing=config["label_smoothing"],
        ),
    )
    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    # Mixed precision scaler
    scaler = GradScaler() if config["amp"] else None
    # Training loop
    global_step = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            tag_labels = batch['tag_labels'].to(device)
            rating_labels = batch['rating_labels'].to(device)
            if config["amp"]:
                with autocast():
                    outputs = model(images)
                    loss, losses = criterion(
                        outputs['tag_logits'],
                        outputs['rating_logits'],
                        tag_labels,
                        rating_labels,
                        sample_weights=None,
                    )
                scaler.scale(loss).backward()
            else:
                outputs = model(images)
                loss, losses = criterion(
                    outputs['tag_logits'],
                    outputs['rating_logits'],
                    tag_labels,
                    rating_labels,
                    sample_weights=None,
                )
                loss.backward()
            running_loss += loss.item()
            # Gradient accumulation
            if (step + 1) % config["gradient_accumulation"] == 0:
                if config["amp"]:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            global_step += 1
        avg_train_loss = running_loss / max(1, len(train_loader))
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}: Avg train loss = {avg_train_loss:.4f}")
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                tag_labels = batch['tag_labels'].to(device)
                rating_labels = batch['rating_labels'].to(device)
                outputs = model(images)
                loss, _ = criterion(
                    outputs['tag_logits'],
                    outputs['rating_logits'],
                    tag_labels,
                    rating_labels,
                    sample_weights=None,
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}: Avg val loss = {avg_val_loss:.4f}")
    logger.info("Training complete.")


if __name__ == "__main__":
    from typing import Any
    train_direct()