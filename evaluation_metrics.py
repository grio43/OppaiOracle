import torch
from dataclasses import dataclass
from typing import Dict
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision


@dataclass
class MetricComputer:
    """Compute macro/micro F1 and mAP for multilabel classification."""
    num_labels: int
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self._macro_f1 = MultilabelF1Score(self.num_labels, average="macro", threshold=self.threshold)
        self._micro_f1 = MultilabelF1Score(self.num_labels, average="micro", threshold=self.threshold)
        self._map = MultilabelAveragePrecision(self.num_labels, average="macro")

    def compute_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, **_: Dict) -> Dict[str, float]:
        """Return macro/micro F1 and mAP metrics."""
        preds = predictions.detach()
        targs = targets.detach()
        return {
            "f1_macro": self._macro_f1(preds, targs).item(),
            "f1_micro": self._micro_f1(preds, targs).item(),
            "mAP": self._map(preds, targs).item(),
        }
