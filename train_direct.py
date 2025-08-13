#!/usr/bin/env python3
"""
Direct Training Script for Anime Image Tagger
This script instantiates the model and runs a training loop using the simplified
data loader. It incorporates several fixes:
  * Reduced batch size to fit within modest GPU memory budgets (e.g. 32)
  * Mixed precision (AMP) training enabled by default
  * Optional gradient accumulation for effective larger batches
  * Patch size adjusted to 16 to better divide the 640×640 input
  * Orientation-aware augmentation is handled in the data loader
"""

import logging
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast

from model_architecture import create_model
from loss_functions import MultiTaskLoss, AsymmetricFocalLoss
from HDF5_loader import create_dataloaders


def train_direct() -> None:
    # Configuration dictionary; tweak values as needed
    config: Dict = {
        "learning_rate": 4e-4,
        "batch_size": 32,  # reduced for better memory usage
        "gradient_accumulation": 2,
        "num_epochs": 8,
        "warmup_steps": 10000,
        "weight_decay": 0.01,
        "label_smoothing": 0.05,
        # Data locations (update these paths according to your dataset)
        "data_dir": Path("data/images"),
        "json_dir": Path("data/annotations"),
        "vocab_path": Path("vocabulary.json"),
        "num_workers": 4,
        # Device configuration
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # Mixed precision training
        "amp": True
    }

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    device = torch.device(config["device"])

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_dir=config["data_dir"],
        json_dir=config["json_dir"],
        vocab_path=config["vocab_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        frequency_sampling=True
    )

    # Create model with adjusted patch size and gradient checkpointing enabled
    model = create_model(
        hidden_size=1280,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_tags=100000,
        num_ratings=5,
        patch_size=16,
        gradient_checkpointing=True
    )
    model.to(device)

    # Loss function: multi-task combining tag and rating losses
    criterion = MultiTaskLoss(
        tag_loss_weight=0.9,
        rating_loss_weight=0.1,
        tag_loss_fn=AsymmetricFocalLoss(
            gamma_pos=1.0,
            gamma_neg=3.0,
            alpha=0.75,
            label_smoothing=config["label_smoothing"]
        )
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
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
            # Forward pass with or without AMP
            if config["amp"]:
                with autocast():
                    outputs = model(images)
                    loss, losses = criterion(
                        outputs['tag_logits'],
                        outputs['rating_logits'],
                        tag_labels,
                        rating_labels
                    )
                scaler.scale(loss).backward()
            else:
                outputs = model(images)
                loss, losses = criterion(
                    outputs['tag_logits'],
                    outputs['rating_logits'],
                    tag_labels,
                    rating_labels
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
        # End of epoch logging
        avg_train_loss = running_loss / len(train_loader)
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
                    rating_labels
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}: Avg val loss = {avg_val_loss:.4f}")

    logger.info("Training complete.")


if __name__ == "__main__":
    train_direct()