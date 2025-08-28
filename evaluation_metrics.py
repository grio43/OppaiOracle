import torch
from dataclasses import dataclass
from typing import Dict
from torchmetrics.functional.classification import (
    multilabel_f1_score,
    multilabel_average_precision,
)


@dataclass
class MetricComputer:
    """Compute macro/micro F1 and mAP for multilabel classification."""
    num_labels: int
    threshold: float = 0.5

    def __post_init__(self) -> None:
        # Using stateless functional metrics to avoid state carryover & resets.
        pass

    def compute_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, **_: Dict) -> Dict[str, float]:
        """Return macro/micro F1 and mAP metrics."""
        # Accept probabilities or logits; TorchMetrics will sigmoid if logits are detected.
        preds = predictions.detach().float()
        # TorchMetrics (multilabel) requires integer {0,1} targets; binarize if floats.
        targs = targets.detach()
        if targs.dtype.is_floating_point:
            targs = (targs > 0.5).to(torch.long)
        else:
            targs = targs.to(torch.long)

        f1_macro = multilabel_f1_score(
            preds, targs, num_labels=self.num_labels, average="macro", threshold=self.threshold
        ).item()
        f1_micro = multilabel_f1_score(
            preds, targs, num_labels=self.num_labels, average="micro", threshold=self.threshold
        ).item()
        mAP = multilabel_average_precision(
            preds, targs, num_labels=self.num_labels, average="macro"
        ).item()
        return {"f1_macro": f1_macro, "f1_micro": f1_micro, "mAP": mAP}
