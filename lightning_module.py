from __future__ import annotations

import logging
import torch
import pytorch_lightning as pl
from typing import Any, Dict, List, Tuple

from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision

from torch.optim.lr_scheduler import LinearLR

from model_architecture import create_model
from loss_functions import MultiTaskLoss, AsymmetricFocalLoss
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
try:
    # MetricComputer is retained for backward compatibility.  If present,
    # downstream code can still compute other project-specific metrics.
    from evaluation_metrics import MetricComputer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    MetricComputer = None

logger = logging.getLogger(__name__)


class LitOppai(pl.LightningModule):
    """PyTorch Lightning module for OppaiOracle.

    Aligns with model outputs keyed as 'tag_logits' and 'rating_logits', and
    normalizes incoming batches from dict or tuple into (images, targets).
    """

    def __init__(self, config: Any, vocab_size: int, *, drop_shape_mismatches: bool = True, dataset_size: int = None):
        super().__init__()
        self.config = config
        self.drop_shape_mismatches = drop_shape_mismatches
        self.dataset_size = dataset_size  # Store for adaptive optimizer config
        # Whether to require strict key matching when loading checkpoints
        self.strict_loading = bool(getattr(getattr(config, "training", None), "resume_strict", False))

        # Build model from config; ensure vocab size is set
        model_args = getattr(config, "model", {})
        if not isinstance(model_args, dict):
            model_args = vars(model_args) if hasattr(model_args, "__dict__") else {}
        model_args = dict(model_args)
        model_args["num_tags"] = vocab_size
        self.model = create_model(**model_args)

        # Loss function for tags and ratings
        self.criterion = MultiTaskLoss(tag_loss_fn=AsymmetricFocalLoss())

        # Torchmetrics collections for training and validation
        threshold_cfg = getattr(config, "threshold_calibration", None)
        if threshold_cfg is not None:
            if hasattr(threshold_cfg, "default_threshold"):
                threshold = threshold_cfg.default_threshold
            elif isinstance(threshold_cfg, dict):
                threshold = threshold_cfg.get("default_threshold", 0.5)
            else:
                threshold = 0.5
        else:
            threshold = 0.5
        self.train_metrics = MetricCollection(
            {
                "f1_macro": MultilabelF1Score(num_labels=vocab_size, average="macro", threshold=threshold),
                "mAP": MultilabelAveragePrecision(num_labels=vocab_size),
            },
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val/")

        # Optional legacy metric computer
        self.metric_computer = MetricComputer() if MetricComputer is not None else None

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Normalize dataloader batches to (images, targets) with canonical keys.

        Accepts either a dict (as returned by our DatasetLoaders) or a tuple
        of (images, targets_dict). Canonical target keys are 'tag' and 'rating'.
        """
        if isinstance(batch, dict):
            images = batch.get("images") or batch.get("image")
            targets: Dict[str, torch.Tensor] = {}
            if "padding_mask" in batch:
                targets["padding_mask"] = batch["padding_mask"]
            tag = batch.get("tag") or batch.get("tag_labels")
            if tag is not None:
                targets["tag"] = tag
            rating = batch.get("rating") or batch.get("rating_labels")
            if rating is not None:
                targets["rating"] = rating
            return images, targets
        else:
            images, targets = batch
            if isinstance(targets, dict):
                if "tag" not in targets and "tag_labels" in targets:
                    targets["tag"] = targets.pop("tag_labels")
                if "rating" not in targets and "rating_labels" in targets:
                    targets["rating"] = targets.pop("rating_labels")
            return images, targets

    def forward(self, images: torch.Tensor, padding_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the underlying model."""
        if padding_mask is not None:
            return self.model(images, padding_mask=padding_mask)
        return self.model(images)

    def training_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]] | Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, targets = self._unpack_batch(batch)
        padding = targets.get("padding_mask")
        outputs = self(images, padding_mask=padding)
        # Align with model outputs
        tag_logits = outputs.get("tag_logits") or outputs.get("tag")
        rating_logits = outputs.get("rating_logits") or outputs.get("rating")
        tag_targets = targets.get("tag")
        rating_targets = targets.get("rating")

        # Validate we have at least tag predictions (required)
        if tag_logits is None:
            available_keys = list(outputs.keys())
            raise RuntimeError(
                f"Model outputs missing tag predictions. "
                f"Expected 'tag_logits' or 'tag', but got keys: {available_keys}. "
                f"This usually indicates a model architecture mismatch."
            )

        if tag_targets is None:
            available_keys = list(targets.keys())
            raise RuntimeError(
                f"Batch targets missing tag labels. "
                f"Expected 'tag' key, but got keys: {available_keys}. "
                f"This indicates a dataloader issue."
            )

        # Compute loss - handle optional ratings
        if rating_logits is not None and rating_targets is not None:
            loss, _ = self.criterion(tag_logits, rating_logits, tag_targets, rating_targets)
        else:
            # Compute tag loss only when ratings are absent
            loss = self.criterion.tag_loss_fn(tag_logits, tag_targets)

        # Update training metrics
        preds = torch.sigmoid(tag_logits)
        targs = tag_targets.to(dtype=preds.dtype)
        metrics_dict = self.train_metrics(preds, targs)
        self.log_dict(metrics_dict, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]] | Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        images, targets = self._unpack_batch(batch)
        padding = targets.get("padding_mask")
        outputs = self(images, padding_mask=padding)

        tag_logits = outputs.get("tag_logits") or outputs.get("tag")
        rating_logits = outputs.get("rating_logits") or outputs.get("rating")
        tag_targets = targets.get("tag")
        rating_targets = targets.get("rating")

        # Validate we have at least tag predictions (required)
        if tag_logits is None:
            available_keys = list(outputs.keys())
            raise RuntimeError(
                f"Validation: Model outputs missing tag predictions. "
                f"Expected 'tag_logits' or 'tag', but got keys: {available_keys}"
            )

        if tag_targets is None:
            available_keys = list(targets.keys())
            raise RuntimeError(
                f"Validation: Batch targets missing tag labels. "
                f"Expected 'tag' key, but got keys: {available_keys}"
            )

        # Compute loss - handle optional ratings
        if rating_logits is not None and rating_targets is not None:
            loss, _ = self.criterion(tag_logits, rating_logits, tag_targets, rating_targets)
        else:
            # Compute tag loss only when ratings are absent
            loss = self.criterion.tag_loss_fn(tag_logits, tag_targets)

        # Update metrics (logged in on_validation_epoch_end)
        preds = torch.sigmoid(tag_logits)
        targs = tag_targets.to(dtype=preds.dtype)
        self.val_metrics.update(preds, targs)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics.reset()

    def on_train_epoch_end(self) -> None:
        # Compute and log epoch-level training metrics before reset
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.train_metrics.reset()

    def configure_optimizers(self):
        if bnb is None:
            raise ImportError(
                "bitsandbytes is required for AdamW8bit optimizer. "
                "Install it with: pip install bitsandbytes"
            )

        # Get training config
        training_cfg = getattr(self.config, "training", None)
        data_cfg = getattr(self.config, "data", None)

        # Extract parameters with defaults
        batch_size = getattr(data_cfg, "batch_size", 32)
        num_epochs = getattr(training_cfg, "num_epochs", 50)
        grad_accum = getattr(training_cfg, "gradient_accumulation_steps", 1)

        warmup_steps = 0

        # Use adaptive optimizer configuration if dataset_size is available
        if self.dataset_size is not None and self.dataset_size > 0:
            try:
                from optimizer_config import get_adamw8bit_config

                lr, optim_kwargs, warmup_steps = get_adamw8bit_config(
                    dataset_size=self.dataset_size,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    gradient_accumulation_steps=grad_accum,
                    num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                )

                logger.info(f"Using adaptive AdamW8bit config: lr={lr:.6f}, warmup={warmup_steps}")
                optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=lr, **optim_kwargs)

            except ImportError as e:
                logger.warning(f"Could not import optimizer_config: {e}. Using fallback configuration.")
                lr = getattr(training_cfg, "learning_rate", 1e-4)
                optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=lr)
        else:
            # Fallback to config-based learning rate
            logger.info("Dataset size not provided, using config learning rate")
            lr = getattr(training_cfg, "learning_rate", 1e-4)
            weight_decay = getattr(training_cfg, "weight_decay", 0.01)
            beta1 = getattr(training_cfg, "adam_beta1", 0.9)
            beta2 = getattr(training_cfg, "adam_beta2", 0.999)
            eps = getattr(training_cfg, "adam_epsilon", 1e-8)

            optimizer = bnb.optim.AdamW8bit(
                self.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )

        # Create warmup scheduler if warmup_steps > 0
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": warmup_scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }

        return optimizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Remove mismatched shape entries from the checkpoint state dict.

        This method validates shape mismatches and drops parameters that are safe
        to reinitialize (e.g., classification heads when vocab_size changes).
        Unexpected mismatches trigger warnings to help catch configuration errors.
        """
        if not self.drop_shape_mismatches or self.strict_loading:
            return

        # Define patterns for parameters that are safe to drop on shape mismatch
        # These are typically the final classification heads that change with vocab_size
        SAFE_MISMATCH_PATTERNS = [
            'tag_head',      # Tag classification head
            'rating_head',   # Rating classification head
            'fc.weight',     # Final FC layer
            'fc.bias',
            'classifier',    # Generic classifier layer
        ]

        def is_safe_to_drop(key: str) -> bool:
            """Check if a parameter key is safe to drop on shape mismatch."""
            return any(pattern in key for pattern in SAFE_MISMATCH_PATTERNS)

        state_dict = checkpoint.get("state_dict", {})
        current_state = self.state_dict()
        safe_removed: List[str] = []
        unsafe_mismatches: List[Tuple[str, Tuple, Tuple]] = []

        for k in list(state_dict.keys()):
            if k in current_state:
                ckpt_tensor = state_dict[k]
                current_tensor = current_state[k]
                if ckpt_tensor.shape != current_tensor.shape:
                    if is_safe_to_drop(k):
                        logger.info(
                            f"Dropping expected shape mismatch in '{k}': "
                            f"checkpoint={ckpt_tensor.shape} -> current={current_tensor.shape}"
                        )
                        safe_removed.append(k)
                        del state_dict[k]
                    else:
                        # Unexpected mismatch - this might indicate a problem
                        unsafe_mismatches.append((k, tuple(ckpt_tensor.shape), tuple(current_tensor.shape)))
                        # Still drop it since drop_shape_mismatches=True
                        del state_dict[k]

        # Report unsafe mismatches with detailed guidance
        if unsafe_mismatches:
            warning_msg = (
                f"Found {len(unsafe_mismatches)} shape mismatch(es) in unexpected layer(s). "
                f"These will be DROPPED and randomly reinitialized:\n"
            )
            for key, ckpt_shape, curr_shape in unsafe_mismatches:
                warning_msg += f"  - {key}: {ckpt_shape} -> {curr_shape}\n"
            warning_msg += (
                "\nThis may indicate:\n"
                "  1. Intentional architecture change (expected)\n"
                "  2. Corrupted checkpoint (check file integrity)\n"
                "  3. Wrong checkpoint for this model (verify paths)\n"
                "\nIf this is intentional, you can ignore this warning. "
                "Otherwise, fix the architecture mismatch or use strict_loading=True to fail fast."
            )
            logger.warning(warning_msg)

        total_removed = len(safe_removed) + len(unsafe_mismatches)
        if total_removed > 0:
            logger.info(
                f"Dropped {total_removed} parameter(s) during checkpoint load "
                f"({len(safe_removed)} expected, {len(unsafe_mismatches)} unexpected)"
            )
