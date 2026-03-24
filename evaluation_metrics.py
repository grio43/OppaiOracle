import torch
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from torchmetrics.functional.classification import (
    multilabel_f1_score,
    multilabel_average_precision,
)
# Note: multilabel_precision and multilabel_recall were removed from imports.
# The compute_per_tag_metrics method now computes these directly from the confusion
# matrix for better performance (single pass instead of 3 separate passes).

# Type alias for averaging modes
AveragingMode = Literal["micro", "macro", "weighted"]


@dataclass
class MetricComputer:
    """Compute macro/micro F1 and mAP for multilabel classification.

    Args:
        num_labels: Total number of labels in the vocabulary.
        threshold: Threshold for converting probabilities to binary predictions.
            Default is 0.5, but this may not be optimal for all models. Consider
            using find_optimal_threshold() on validation data to determine the
            best threshold for your specific model and dataset.
        skip_indices: Optional list of label indices to exclude from metric computation
                      (e.g., [0, 1] to skip PAD and UNK tokens).
        mAP_average: Averaging mode for mAP computation. Options:
            - "macro": Equal weight to all labels (default for backward compatibility).
                       May over-emphasize rare tags in imbalanced datasets.
            - "micro": Aggregate contributions across all labels. Better for imbalanced
                       datasets as it weights by sample count.
            - "weighted": Weight by support (number of true instances per label).
                          Good compromise for imbalanced multi-label classification.

    Note on threshold selection:
        The default threshold of 0.5 assumes symmetric decision boundaries, which
        may not be optimal for:
        - Imbalanced datasets where positive class is rare
        - Models trained with class weights or focal loss
        - Applications where precision/recall trade-off favors one over the other
        Use find_optimal_threshold() to tune based on validation performance.

    Note on averaging modes for mAP:
        - macro: Treats all labels equally, which may give disproportionate weight
                 to rare tags that have few samples.
        - micro: Computes global metrics by counting total true positives, etc.
                 Better reflects overall performance on imbalanced datasets.
        - weighted: Weights each label's contribution by its support, balancing
                    between macro and micro approaches.
    """
    num_labels: int
    threshold: float = 0.5
    skip_indices: Optional[List[int]] = None
    mAP_average: AveragingMode = "macro"

    # Private field to cache the keep mask
    _keep_mask: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _effective_num_labels: int = field(default=0, init=False, repr=False)
    _skip_indices_set: Optional[set] = field(default=None, init=False, repr=False)
    # PERFORMANCE OPTIMIZATION: Cache the device where _keep_mask was last transferred to.
    # This avoids redundant CPU->GPU transfers on every compute_all_metrics() call.
    # The mask is small (~num_labels bools) but the transfer overhead adds up over many calls.
    _keep_mask_cached_device: Optional[torch.device] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Using stateless functional metrics to avoid state carryover & resets.
        # Pre-compute the mask for filtering indices
        if self.skip_indices:
            self._keep_mask = torch.ones(self.num_labels, dtype=torch.bool)
            self._skip_indices_set = set(self.skip_indices)
            for idx in self.skip_indices:
                if 0 <= idx < self.num_labels:
                    self._keep_mask[idx] = False
            self._effective_num_labels = int(self._keep_mask.sum().item())
        else:
            self._keep_mask = None
            self._skip_indices_set = None
            self._effective_num_labels = self.num_labels

        # Validate mAP_average
        valid_averages = ("micro", "macro", "weighted")
        if self.mAP_average not in valid_averages:
            raise ValueError(f"mAP_average must be one of {valid_averages}, got '{self.mAP_average}'")

    def compute_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, **_: Dict) -> Dict[str, float]:
        """Return macro/micro F1 and mAP metrics.

        If skip_indices was specified, those columns are filtered out before computing metrics.
        """
        # Accept probabilities or logits; TorchMetrics will sigmoid if logits are detected.
        preds = predictions.detach()  # Keep native dtype (bfloat16)
        # TorchMetrics (multilabel) requires integer {0,1} targets; binarize if floats.
        targs = targets.detach()
        if targs.dtype.is_floating_point:
            targs = (targs > 0.5).to(torch.long)
        else:
            targs = targs.to(torch.long)

        # Filter out skip_indices columns if specified
        if self._keep_mask is not None:
            # PERFORMANCE OPTIMIZATION: Cache keep_mask on the target device after first transfer.
            # This avoids redundant .to(device) calls which cause CPU-GPU sync overhead.
            # The mask is moved in-place and cached, so subsequent calls skip the transfer.
            if self._keep_mask_cached_device != preds.device:
                self._keep_mask = self._keep_mask.to(preds.device)
                self._keep_mask_cached_device = preds.device
            preds = preds[:, self._keep_mask]
            targs = targs[:, self._keep_mask]
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
            preds, targs, num_labels=effective_labels, average=self.mAP_average
        ).item()
        return {"f1_macro": f1_macro, "f1_micro": f1_micro, "mAP": mAP}

    def compute_all_metrics_at_threshold(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float,
        **_: Dict
    ) -> Dict[str, float]:
        """Compute metrics at a specific threshold (for threshold optimization).

        This is a convenience method that computes metrics at a given threshold
        without modifying the instance's default threshold.

        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            threshold: Threshold to use for this computation

        Returns:
            Dict with f1_macro, f1_micro, and mAP metrics
        """
        preds = predictions.detach()
        targs = targets.detach()
        if targs.dtype.is_floating_point:
            targs = (targs > 0.5).to(torch.long)
        else:
            targs = targs.to(torch.long)

        if self._keep_mask is not None:
            # PERFORMANCE OPTIMIZATION: Use cached device-resident mask (see compute_all_metrics)
            if self._keep_mask_cached_device != preds.device:
                self._keep_mask = self._keep_mask.to(preds.device)
                self._keep_mask_cached_device = preds.device
            preds = preds[:, self._keep_mask]
            targs = targs[:, self._keep_mask]
            effective_labels = self._effective_num_labels
        else:
            effective_labels = self.num_labels

        f1_macro = multilabel_f1_score(
            preds, targs, num_labels=effective_labels, average="macro", threshold=threshold
        ).item()
        f1_micro = multilabel_f1_score(
            preds, targs, num_labels=effective_labels, average="micro", threshold=threshold
        ).item()
        mAP = multilabel_average_precision(
            preds, targs, num_labels=effective_labels, average=self.mAP_average
        ).item()
        return {"f1_macro": f1_macro, "f1_micro": f1_micro, "mAP": mAP}

    def find_optimal_threshold(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metric: str = "f1_macro",
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """Find the optimal threshold that maximizes the specified metric.

        This method searches over a range of thresholds to find the one that
        maximizes the given metric. Use this on validation data to tune the
        threshold for your specific model.

        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            metric: Which metric to optimize. Options: "f1_macro", "f1_micro".
                    Note: mAP is threshold-independent so cannot be optimized.
            thresholds: Optional list of thresholds to try. If None, uses
                        [0.1, 0.15, 0.2, ..., 0.9] (17 values).

        Returns:
            Tuple of (optimal_threshold, best_metric_value)

        Example:
            >>> metric_computer = MetricComputer(num_labels=100, skip_indices=[0, 1])
            >>> optimal_thresh, best_f1 = metric_computer.find_optimal_threshold(
            ...     val_predictions, val_targets, metric="f1_macro"
            ... )
            >>> # Update threshold for future computations
            >>> metric_computer.threshold = optimal_thresh
        """
        if metric not in ("f1_macro", "f1_micro"):
            raise ValueError(f"metric must be 'f1_macro' or 'f1_micro', got '{metric}'")

        if thresholds is None:
            thresholds = [0.1 + 0.05 * i for i in range(17)]  # 0.1 to 0.9 in 0.05 steps

        best_threshold = 0.5
        best_value = -1.0

        for thresh in thresholds:
            metrics = self.compute_all_metrics_at_threshold(predictions, targets, thresh)
            value = metrics[metric]
            if value > best_value:
                best_value = value
                best_threshold = thresh

        return best_threshold, best_value

    def compute_per_tag_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        tag_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-tag precision, recall, F1, and support.

        This method is optimized to filter out skip_indices (PAD/UNK) before
        computing metrics, avoiding unnecessary computation for excluded labels.

        Args:
            predictions: Model predictions (logits or probabilities), shape (N, num_labels)
            targets: Ground truth labels, shape (N, num_labels)
            tag_names: Optional list of tag names for indexing results. If skip_indices
                       is set, tag_names should still correspond to the original indices
                       (before filtering).

        Returns:
            Dict mapping tag name/index to metrics dict with 'precision', 'recall', 'f1', 'support'
        """
        preds = predictions.detach()
        targs = targets.detach()
        if targs.dtype.is_floating_point:
            targs = (targs > 0.5).to(torch.long)
        else:
            targs = targs.to(torch.long)

        # Filter out skip_indices before computing metrics for efficiency
        if self._keep_mask is not None:
            # PERFORMANCE OPTIMIZATION: Use cached device-resident mask (see compute_all_metrics)
            if self._keep_mask_cached_device != preds.device:
                self._keep_mask = self._keep_mask.to(preds.device)
                self._keep_mask_cached_device = preds.device
            preds_filtered = preds[:, self._keep_mask]
            targs_filtered = targs[:, self._keep_mask]
            effective_labels = self._effective_num_labels

            # Build mapping from filtered index back to original index
            original_indices = [i for i in range(self.num_labels) if i not in self._skip_indices_set]
        else:
            preds_filtered = preds
            targs_filtered = targs
            effective_labels = self.num_labels
            original_indices = list(range(self.num_labels))

        # OPTIMIZATION: Compute confusion matrix components (TP, FP, FN) once and derive
        # precision, recall, and F1 from them. This avoids 3 separate passes over the data
        # that would each redundantly compute the same confusion matrix internally.

        # Binarize predictions using threshold
        preds_binary = (preds_filtered > self.threshold).to(torch.long)

        # Compute confusion matrix components per label (sum over samples, dim=0)
        # TP: predicted positive AND actually positive
        tp = (preds_binary * targs_filtered).sum(dim=0).float()
        # FP: predicted positive BUT actually negative
        fp = (preds_binary * (1 - targs_filtered)).sum(dim=0).float()
        # FN: predicted negative BUT actually positive
        fn = ((1 - preds_binary) * targs_filtered).sum(dim=0).float()

        # Compute metrics from confusion matrix with epsilon to avoid division by zero
        eps = 1e-8
        precision_per_label = tp / (tp + fp + eps)
        recall_per_label = tp / (tp + fn + eps)
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1_per_label = 2 * tp / (2 * tp + fp + fn + eps)

        # Support = number of positive samples per label (same as tp + fn)
        support_per_label = targs_filtered.sum(dim=0)

        per_tag_metrics = {}
        for filtered_idx, original_idx in enumerate(original_indices):
            tag_key = tag_names[original_idx] if tag_names and original_idx < len(tag_names) else str(original_idx)
            per_tag_metrics[tag_key] = {
                'precision': precision_per_label[filtered_idx].item(),
                'recall': recall_per_label[filtered_idx].item(),
                'f1': f1_per_label[filtered_idx].item(),
                'support': support_per_label[filtered_idx].item(),
            }

        return per_tag_metrics
