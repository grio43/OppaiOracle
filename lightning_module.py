import torch
import pytorch_lightning as pl
from typing import Any, Dict, Tuple

from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision

from model_architecture import create_model
from loss_functions import MultiTaskLoss, AsymmetricFocalLoss
try:
    # MetricComputer is retained for backward compatibility.  If present,
    # downstream code can still compute other project-specific metrics.
    from evaluation_metrics import MetricComputer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    MetricComputer = None


class LitOppai(pl.LightningModule):
    """PyTorch Lightning module for OppaiOracle."""

    def __init__(self, config: Any, vocab_size: int, *, drop_shape_mismatches: bool = True):
        super().__init__()
        self.config = config
        self.drop_shape_mismatches = drop_shape_mismatches
        # Whether to require strict key matching when loading checkpoints
        self.strict_loading = bool(getattr(getattr(config, "training", None), "resume_strict", False))

        # Build model from config; ensure vocab size is set
        model_args = getattr(config, "model", {})
        if hasattr(model_args, "__dict__"):
            model_args = vars(model_args)
        model_args = dict(model_args)
        model_args["num_tags"] = vocab_size
        self.model = create_model(**model_args)

        # Loss function for tags and ratings
        self.criterion = MultiTaskLoss(tag_loss_fn=AsymmetricFocalLoss())

        # Torchmetrics collections for training and validation
        threshold = getattr(getattr(config, "threshold_calibration", {}), "default_threshold", 0.5)
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

    def forward(self, images: torch.Tensor, padding_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the underlying model."""
        if padding_mask is not None:
            return self.model(images, padding_mask=padding_mask)
        return self.model(images)

    def training_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        padding = targets.get("padding_mask")
        outputs = self(images, padding_mask=padding)
        loss, _ = self.criterion(outputs["tag"], outputs["rating"], targets["tag"], targets["rating"])

        # Update metrics for tag predictions if present
        if "tag" in outputs and "tag" in targets:
            preds = torch.sigmoid(outputs["tag"])
            targs = targets["tag"].to(dtype=preds.dtype)
            metrics_dict = self.train_metrics(preds, targs)
            self.log_dict(metrics_dict, prog_bar=True, on_step=True, on_epoch=False)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> Dict[str, Any]:
        images, targets = batch
        padding = targets.get("padding_mask")
        outputs = self(images, padding_mask=padding)
        loss, _ = self.criterion(outputs["tag"], outputs["rating"], targets["tag"], targets["rating"])

        if "tag" in outputs and "tag" in targets:
            preds = torch.sigmoid(outputs["tag"])
            targs = targets["tag"].to(dtype=preds.dtype)
            metrics_dict = self.val_metrics(preds, targs)
            self.log_dict(metrics_dict, prog_bar=True, on_step=False, on_epoch=False)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def configure_optimizers(self):
        lr = getattr(getattr(self.config, "training", None), "learning_rate", 1e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Remove mismatched shape entries from the checkpoint state dict."""
        if not self.drop_shape_mismatches or self.strict_loading:
            return
        state_dict = checkpoint.get("state_dict", {})
        current_state = self.state_dict()
        removed_keys: list[str] = []
        for k in list(state_dict.keys()):
            if k in current_state:
                ckpt_tensor = state_dict[k]
                current_tensor = current_state[k]
                if ckpt_tensor.shape != current_tensor.shape:
                    removed_keys.append(k)
                    del state_dict[k]
        if removed_keys:
            self.log(
                "resume/mismatched_keys_dropped",
                len(removed_keys),
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
