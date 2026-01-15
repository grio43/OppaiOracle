import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from torchmetrics.functional.classification import (
    multilabel_f1_score,
    multilabel_average_precision,
)


@dataclass
class MetricComputer:
    """Compute macro/micro F1 and mAP for multilabel classification.

    Args:
        num_labels: Total number of labels in the vocabulary.
        threshold: Threshold for converting probabilities to binary predictions.
        skip_indices: Optional list of label indices to exclude from metric computation
                      (e.g., [0, 1] to skip PAD and UNK tokens).
    """
    num_labels: int
    threshold: float = 0.5
    skip_indices: Optional[List[int]] = None

    # Private field to cache the keep mask
    _keep_mask: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _effective_num_labels: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        # Using stateless functional metrics to avoid state carryover & resets.
        # Pre-compute the mask for filtering indices
        if self.skip_indices:
            self._keep_mask = torch.ones(self.num_labels, dtype=torch.bool)
            for idx in self.skip_indices:
                if 0 <= idx < self.num_labels:
                    self._keep_mask[idx] = False
            self._effective_num_labels = int(self._keep_mask.sum().item())
        else:
            self._keep_mask = None
            self._effective_num_labels = self.num_labels

    def compute_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, **_: Dict) -> Dict[str, float]:
        """Return macro/micro F1 and mAP metrics.

        If skip_indices was specified, those columns are filtered out before computing metrics.
        """
        # Accept probabilities or logits; TorchMetrics will sigmoid if logits are detected.
        preds = predictions.detach().float()
        # TorchMetrics (multilabel) requires integer {0,1} targets; binarize if floats.
        targs = targets.detach()
        if targs.dtype.is_floating_point:
            targs = (targs > 0.5).to(torch.long)
        else:
            targs = targs.to(torch.long)

        # Filter out skip_indices columns if specified
        if self._keep_mask is not None:
            # Move mask to same device as tensors if needed
            keep_mask = self._keep_mask.to(preds.device)
            preds = preds[:, keep_mask]
            targs = targs[:, keep_mask]
            effective_labels = self._effective_num_labels
        else:
            effective_labels = self.num_labels

        f1_macro = multilabel_f1_score(
            preds, targs, num_labels=effective_labels, average="macro", threshold=self.threshold
        ).item()
        f1_micro = multilabel_f1_score(
            preds, targs, num_labels=effective_labels, average="micro", threshold=self.threshold
        ).item()
        mAP = multilabel_average_precision(
            preds, targs, num_labels=effective_labels, average="macro"
        ).item()
        return {"f1_macro": f1_macro, "f1_micro": f1_micro, "mAP": mAP}
