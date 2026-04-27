import json
import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union
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
        The default threshold of 0.2653 matches the P=R threshold used by
        competing v2.0 models for comparable evaluation. Note that this may not
        be optimal for:
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
    threshold: float = 0.2653
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

        Cast predictions to fp32 before computing metrics — bf16 has 7 mantissa bits,
        which loses precision near the threshold and in PR-curve sums across many
        rare classes.
        """
        # Accept probabilities or logits; TorchMetrics will sigmoid if logits are detected.
        # Cast to fp32 for stable thresholding and PR-curve accumulation.
        preds = predictions.detach().float()
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

        preds, targs, effective_labels = self._drop_zero_positive_classes(preds, targs)

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

    @staticmethod
    def _drop_zero_positive_classes(
        preds: torch.Tensor, targs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Remove classes with zero positive targets in this draw.

        torchmetrics returns AP=0 for any class with no positives, so under macro
        averaging across an 18-24K-class long-tailed vocabulary, a 30k-sample
        validation draw leaves thousands of unrepresented classes contributing
        zeros. The macro score then reflects vocabulary sparsity rather than
        model quality. Filtering keeps macro averaging meaningful.
        """
        if targs.numel() == 0:
            return preds, targs, preds.size(1)
        positive_per_class = targs.sum(dim=0)
        keep = positive_per_class > 0
        effective = int(keep.sum().item())
        if effective == 0 or effective == preds.size(1):
            return preds, targs, preds.size(1)
        return preds[:, keep], targs[:, keep], effective

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
        preds = predictions.detach().float()
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

        preds, targs, effective_labels = self._drop_zero_positive_classes(preds, targs)

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


@dataclass
class FrequencyBucketMetrics:
    """Compute metrics broken down by tag frequency buckets.

    Implements LVIS-style frequency-bucketed evaluation (AP_rare, AP_common, AP_frequent)
    to diagnose whether model performance degrades for rare vs common tags.

    Args:
        tag_frequencies: Dict mapping tag name to occurrence count in training data.
        frequency_bins: Bin edges for frequency buckets (required, no default).
            Example: [300, 500, 1000, 5000, 10000, float('inf')]
            Creates buckets: [300-499], [500-999], [1000-4999], [5000-9999], [10000+]
        tag_names: Ordered list of tag names matching model output indices.
        skip_indices: Indices to exclude from metric computation (e.g., [0, 1] for PAD/UNK).
    """
    tag_frequencies: Dict[str, int]
    frequency_bins: List[float]
    tag_names: List[str]
    skip_indices: Optional[List[int]] = None

    # Computed in __post_init__
    _bucket_assignments: Dict[str, List[int]] = field(default_factory=dict, init=False, repr=False)
    _bucket_names: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        skip_set = set(self.skip_indices) if self.skip_indices else set()
        bins = self.frequency_bins

        # Build bucket names from bin edges
        self._bucket_names = []
        for i in range(len(bins) - 1):
            low = int(bins[i])
            high = bins[i + 1]
            if high == float('inf'):
                self._bucket_names.append(f"{low}+")
            else:
                self._bucket_names.append(f"{low}-{int(high) - 1}")

        # Assign each tag index to a bucket
        self._bucket_assignments = {name: [] for name in self._bucket_names}
        for idx, tag_name in enumerate(self.tag_names):
            if idx in skip_set:
                continue
            freq = self.tag_frequencies.get(tag_name, 0)
            for i in range(len(bins) - 1):
                if bins[i] <= freq < bins[i + 1]:
                    self._bucket_assignments[self._bucket_names[i]].append(idx)
                    break

    def compute_bucketed_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.2653,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-bucket F1 (macro/micro), mAP, tag count, and mean support.

        Args:
            predictions: Probability tensor, shape (N, num_labels).
            targets: Binary target tensor, shape (N, num_labels).
            threshold: Threshold for binarizing predictions.

        Returns:
            Dict mapping bucket name to {f1_macro, f1_micro, mAP, num_tags, mean_support}.
        """
        targs = targets.detach()
        if targs.dtype.is_floating_point:
            targs = (targs > 0.5).to(torch.long)
        else:
            targs = targs.to(torch.long)
        preds = predictions.detach()

        results: Dict[str, Dict[str, float]] = {}
        for bucket_name, indices in self._bucket_assignments.items():
            num_tags = len(indices)
            if num_tags == 0:
                results[bucket_name] = {
                    "f1_macro": 0.0, "f1_micro": 0.0, "mAP": 0.0,
                    "num_tags": 0, "mean_support": 0.0,
                }
                continue

            idx_tensor = torch.tensor(indices, dtype=torch.long, device=preds.device)
            bucket_preds = preds[:, idx_tensor]
            bucket_targs = targs[:, idx_tensor]

            f1_macro = multilabel_f1_score(
                bucket_preds, bucket_targs, num_labels=num_tags,
                average="macro", threshold=threshold,
            ).item()
            f1_micro = multilabel_f1_score(
                bucket_preds, bucket_targs, num_labels=num_tags,
                average="micro", threshold=threshold,
            ).item()
            mAP = multilabel_average_precision(
                bucket_preds, bucket_targs, num_labels=num_tags, average="macro",
            ).item()
            mean_support = bucket_targs.sum(dim=0).float().mean().item()

            results[bucket_name] = {
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "mAP": mAP,
                "num_tags": num_tags,
                "mean_support": mean_support,
            }

        return results

    @property
    def bucket_names(self) -> List[str]:
        return list(self._bucket_names)

    @property
    def bucket_tag_counts(self) -> Dict[str, int]:
        return {name: len(indices) for name, indices in self._bucket_assignments.items()}


@dataclass
class ThresholdCalibrator:
    """Calibrate per-tag or per-bucket prediction thresholds by maximizing F1.

    Searches over a range of thresholds to find optimal values, either independently
    per tag or grouped by frequency bucket. Zero training risk — operates on
    accumulated validation predictions post-training.

    Args:
        mode: "per_tag" for individual tag thresholds, "per_bucket" for one per frequency bucket.
        default_threshold: Fallback for tags with zero validation support.
        search_min: Lower bound of threshold search range.
        search_max: Upper bound of threshold search range.
        search_step: Step size for threshold grid search.
    """
    mode: str = "per_bucket"
    default_threshold: float = 0.2653
    search_min: float = 0.1
    search_max: float = 0.9
    search_step: float = 0.02

    def calibrate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        tag_names: List[str],
        skip_indices: Optional[List[int]] = None,
        frequency_bins: Optional[List[float]] = None,
        tag_frequencies: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Find optimal thresholds for each tag or frequency bucket.

        Args:
            predictions: Probability tensor, shape (N, num_labels).
            targets: Binary target tensor, shape (N, num_labels).
            tag_names: Ordered list of tag names matching model output indices.
            skip_indices: Indices to exclude (e.g., PAD/UNK).
            frequency_bins: Required when mode="per_bucket".
            tag_frequencies: Required when mode="per_bucket".

        Returns:
            Dict mapping tag name (per_tag) or bucket name (per_bucket) to optimal threshold.
        """
        preds_np = predictions.detach().float().numpy()
        targs_np = targets.detach().numpy()
        if targs_np.dtype != np.int64:
            targs_np = (targs_np > 0.5).astype(np.int64)

        skip_set = set(skip_indices) if skip_indices else set()
        thresholds = np.arange(self.search_min, self.search_max + self.search_step / 2, self.search_step)

        if self.mode == "per_tag":
            return self._calibrate_per_tag(preds_np, targs_np, tag_names, skip_set, thresholds)
        elif self.mode == "per_bucket":
            if frequency_bins is None or tag_frequencies is None:
                raise ValueError("frequency_bins and tag_frequencies required for per_bucket mode")
            return self._calibrate_per_bucket(
                preds_np, targs_np, tag_names, skip_set, thresholds,
                frequency_bins, tag_frequencies,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _calibrate_per_tag(
        self,
        preds: np.ndarray,
        targs: np.ndarray,
        tag_names: List[str],
        skip_set: set,
        thresholds: np.ndarray,
    ) -> Dict[str, float]:
        result = {}
        for idx in range(preds.shape[1]):
            if idx in skip_set:
                continue
            tag_name = tag_names[idx] if idx < len(tag_names) else str(idx)
            support = targs[:, idx].sum()
            if support == 0:
                result[tag_name] = self.default_threshold
                continue
            best_thresh = self.default_threshold
            best_f1 = -1.0
            for t in thresholds:
                pred_bin = (preds[:, idx] > t).astype(np.int64)
                targ_col = targs[:, idx]
                tp = (pred_bin * targ_col).sum()
                fp = (pred_bin * (1 - targ_col)).sum()
                fn = ((1 - pred_bin) * targ_col).sum()
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(t)
            result[tag_name] = best_thresh
        return result

    def _calibrate_per_bucket(
        self,
        preds: np.ndarray,
        targs: np.ndarray,
        tag_names: List[str],
        skip_set: set,
        thresholds: np.ndarray,
        frequency_bins: List[float],
        tag_frequencies: Dict[str, int],
    ) -> Dict[str, float]:
        # Group tag indices by frequency bucket
        buckets: Dict[str, List[int]] = {}
        for i in range(len(frequency_bins) - 1):
            low = int(frequency_bins[i])
            high = frequency_bins[i + 1]
            name = f"{low}+" if high == float('inf') else f"{low}-{int(high) - 1}"
            buckets[name] = []

        bucket_names = list(buckets.keys())
        for idx, tag_name in enumerate(tag_names):
            if idx in skip_set:
                continue
            freq = tag_frequencies.get(tag_name, 0)
            for i in range(len(frequency_bins) - 1):
                if frequency_bins[i] <= freq < frequency_bins[i + 1]:
                    buckets[bucket_names[i]].append(idx)
                    break

        result = {}
        for bucket_name, indices in buckets.items():
            if not indices:
                result[bucket_name] = self.default_threshold
                continue
            best_thresh = self.default_threshold
            best_f1 = -1.0
            for t in thresholds:
                # Compute macro-F1 across all tags in this bucket
                f1_sum = 0.0
                count = 0
                for idx in indices:
                    support = targs[:, idx].sum()
                    if support == 0:
                        continue
                    pred_bin = (preds[:, idx] > t).astype(np.int64)
                    targ_col = targs[:, idx]
                    tp = (pred_bin * targ_col).sum()
                    fp = (pred_bin * (1 - targ_col)).sum()
                    fn = ((1 - pred_bin) * targ_col).sum()
                    f1_sum += 2 * tp / (2 * tp + fp + fn + 1e-8)
                    count += 1
                macro_f1 = f1_sum / max(count, 1)
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_thresh = float(t)
            result[bucket_name] = best_thresh
        return result

    @staticmethod
    def save(thresholds: Dict[str, float], path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(thresholds, f, indent=2)

    @staticmethod
    def load(path: Union[str, Path]) -> Dict[str, float]:
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def apply_thresholds(
        predictions: torch.Tensor,
        tag_thresholds: Dict[str, float],
        tag_names: List[str],
        default_threshold: float = 0.2653,
    ) -> torch.Tensor:
        """Apply per-tag thresholds to convert probabilities to binary predictions.

        Args:
            predictions: Probability tensor, shape (N, num_labels).
            tag_thresholds: Dict mapping tag name to threshold.
            tag_names: Ordered tag names matching prediction columns.
            default_threshold: Fallback for tags not in tag_thresholds.

        Returns:
            Binary tensor, shape (N, num_labels).
        """
        thresh_tensor = torch.full(
            (predictions.shape[1],), default_threshold,
            dtype=predictions.dtype, device=predictions.device,
        )
        for idx, name in enumerate(tag_names):
            if name in tag_thresholds:
                thresh_tensor[idx] = tag_thresholds[name]
        return (predictions > thresh_tensor.unsqueeze(0)).long()
