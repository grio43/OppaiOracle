#!/usr/bin/env python3
"""
Evaluation Metrics for Anime Image Tagger
Comprehensive metrics for multi-label classification with 200k tags
Improved version with bug fixes and optimizations
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score as skl_average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    multilabel_confusion_matrix
)
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Attempt to import the list of indices corresponding to ignored tags.  If
# ``vocabulary.py`` defines this list it will be used to mask out those
# columns from predictions and targets before metrics are accumulated.  If
# the import fails (for example during unit tests), fall back to an empty
# list which leaves tensors unchanged.
try:
    from vocabulary import IGNORE_TAG_INDICES as _IGNORE_TAG_INDICES  # noqa: F401
    IGNORE_TAG_INDICES: List[int] = list(_IGNORE_TAG_INDICES)
except Exception:
    IGNORE_TAG_INDICES = []

# =============================================================================
# Simple metric functions for backward compatibility
# Migrated from metrics.py
# =============================================================================

def compute_precision_recall_f1(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    pad_index: int = 0,
) -> Dict[str, float]:
    """
    Compute micro‑averaged precision, recall and F1 score for multi‑label
    predictions at a given threshold.

    Args:
        probabilities: Tensor of shape (B, C) with predicted probabilities.
        targets: Tensor of shape (B, C) with binary ground‑truth labels.
        threshold: Threshold above which a probability is considered a positive prediction.
        pad_index: Index of the pad class to ignore.  Both probabilities and
            targets will have this column removed before computation.

    Returns:
        Dictionary with keys ``precision``, ``recall`` and ``f1``.
    """
    if probabilities.dim() != 2 or targets.dim() != 2:
        raise ValueError("probabilities and targets must be 2D tensors")
    # Remove pad column
    probs = probabilities.clone()
    targs = targets.clone()

    # Validate bounds before concatenation
    if pad_index is not None and 0 <= pad_index < probs.size(1):
        if pad_index == 0:
            probs = probs[:, 1:]
            targs = targs[:, 1:]
        elif pad_index == probs.size(1) - 1:
            probs = probs[:, :pad_index]
            targs = targs[:, :pad_index]
        else:
            probs = torch.cat([probs[:, :pad_index], probs[:, pad_index+1:]], dim=1)
            targs = torch.cat([targs[:, :pad_index], targs[:, pad_index+1:]], dim=1)
    preds = (probs >= threshold).float()
    true = targs.float()
    tp = (preds * true).sum()
    fp = ((preds == 1) & (true == 0)).float().sum()
    fn = ((preds == 0) & (true == 1)).float().sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {'precision': precision.item(), 'recall': recall.item(), 'f1': f1.item()}


def simple_average_precision_score(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    pad_index: int = 0,
) -> Dict[str, float]:
    """
    Compute mean average precision (mAP) across classes excluding the pad class.
    Simple implementation for backward compatibility.

    Note: For comprehensive metrics, use MetricComputer.compute_all_metrics() instead.
    
    Args:
        probabilities: Tensor of shape (B, C) with predicted probabilities.
        targets: Tensor of shape (B, C) with binary ground‑truth labels.
        pad_index: Index of the pad class to ignore.

    Returns:
        Dictionary with keys ``map`` (mean AP across classes) and
        ``per_class_ap`` (list of APs for each non‑pad class).
    """
    if probabilities.dim() != 2 or targets.dim() != 2:
        raise ValueError("probabilities and targets must be 2D tensors")
    probs = probabilities.clone()
    targs = targets.clone()

    # Validate bounds before concatenation
    if pad_index is not None and 0 <= pad_index < probs.size(1):
        if pad_index == 0:
            probs = probs[:, 1:]
            targs = targs[:, 1:]
        elif pad_index == probs.size(1) - 1:
            probs = probs[:, :pad_index]
            targs = targs[:, :pad_index]
        else:
            probs = torch.cat([probs[:, :pad_index], probs[:, pad_index+1:]], dim=1)
            targs = torch.cat([targs[:, :pad_index], targs[:, pad_index+1:]], dim=1)
    B, C = probs.shape
    ap_values = []
    for c in range(C):
        scores = probs[:, c]
        labels = targs[:, c]
        # Skip classes with no positive examples
        total_pos = labels.sum().item()
        if total_pos == 0:
            continue
        # Sort by predicted score descending
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        cum_tp = sorted_labels.cumsum(dim=0).float()
        precisions = cum_tp / torch.arange(1, len(cum_tp) + 1, device=probs.device).float()
        recalls = cum_tp / total_pos
        # Integrate precision over recall using step function
        ap = 0.0
        prev_recall = 0.0
        for p, r in zip(precisions, recalls):
            dr = r.item() - prev_recall
            if dr > 0:
                ap += p.item() * dr
                prev_recall = r.item()
        ap_values.append(ap)
    mean_ap = sum(ap_values) / len(ap_values) if ap_values else 0.0
    return {'map': mean_ap, 'per_class_ap': ap_values}

# Alias for backward compatibility with old metrics.py imports
average_precision_score = simple_average_precision_score


@dataclass
class MetricConfig:
    """Configuration for metrics computation"""
    # Thresholds
    prediction_threshold: float = 0.5
    adaptive_threshold: bool = True
    min_predictions: int = 5
    max_predictions: int = 50
    
    # Top-k settings
    top_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    
    # Tag grouping
    num_groups: int = 20
    tags_per_group: int = 10000
    
    # Frequency-based analysis
    frequency_bins: List[int] = field(default_factory=lambda: [10, 100, 1000, 10000])
    
    # Compute expensive metrics
    compute_per_tag_metrics: bool = True
    compute_confusion_matrix: bool = False
    compute_auc: bool = True
    
    # Sampling for large-scale metrics
    sample_size_for_expensive: Optional[int] = 10000
    max_tags_for_detailed: int = 1000  # Maximum tags for detailed analysis
    
    # Visualization
    save_plots: bool = True
    plot_dir: str = "./metric_plots"
    
    # Memory management
    batch_size_for_computation: int = 1000  # Process in batches for memory efficiency


class MetricTracker:
    """Track metrics over time during training"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all tracked values"""
        self.predictions = []
        self.targets = []
        self.losses = []
        self.metadata = []
        
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """Update with batch results"""
        # Validate inputs
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")

        # If there are tag indices to ignore, mask them out from both
        # predictions and targets before storing.  This ensures that
        # ignored tags do not contribute to loss or metrics.  The mask is
        # constructed on the same device as the incoming tensors to avoid
        # unnecessary transfers.  Note that after masking the tensors are
        # detached and moved to CPU for memory efficiency below.
        if IGNORE_TAG_INDICES:
            # Create a boolean mask with True for columns to keep
            device = predictions.device
            num_tags = predictions.size(1)
            mask = torch.ones(num_tags, dtype=torch.bool, device=device)
            # Guard against out of range indices
            valid_ignore = [idx for idx in IGNORE_TAG_INDICES if 0 <= idx < num_tags]
            if valid_ignore:
                mask[valid_ignore] = False
                predictions = predictions[:, mask]
                targets = targets[:, mask]
        
        # Ensure CPU tensors for memory efficiency
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
        
        if loss is not None:
            self.losses.append(float(loss))
            
        if metadata is not None:
            self.metadata.append(metadata)
    
    def compute_metrics(self, config: MetricConfig) -> Dict[str, Any]:
        """Compute all metrics"""
        if not self.predictions:
            raise ValueError("No predictions to compute metrics from")
        
        # Concatenate all batches
        all_predictions = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Create metric computer
        computer = MetricComputer(config)
        
        metrics = computer.compute_all_metrics(all_predictions, all_targets)
        
        # Add loss statistics if available
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
            metrics['std_loss'] = np.std(self.losses)
        
        return metrics


class MetricComputer:
    """Compute various metrics for multi-label classification"""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        tag_names: Optional[List[str]] = None,
        tag_frequencies: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics
        
        Args:
            predictions: (N, num_tags) float predictions (probabilities)
            targets: (N, num_tags) binary targets
            tag_names: Optional tag names for detailed analysis
            tag_frequencies: Optional tag frequencies for weighted metrics
            
        Returns:
            Dictionary of all computed metrics
        
        Raises:
            ValueError: If input validation fails
            RuntimeError: If metric computation fails
        """
        # Validate inputs
        predictions, targets = self._validate_inputs(predictions, targets)
        
        metrics = {}
        failed_metrics = []
        
        # Convert to numpy for sklearn (with memory management)
        logger.info("Converting to numpy arrays...")
        pred_np = self._safe_to_numpy(predictions)
        target_np = self._safe_to_numpy(targets)
        
        # Basic metrics (critical - should always work)
        try:
            logger.info("Computing basic metrics...")
            metrics.update(self._compute_basic_metrics(pred_np, target_np))
        except Exception as e:
            logger.error(f"Failed to compute basic metrics: {e}")
            raise RuntimeError(f"Critical failure in basic metrics computation: {e}") from e
        
        # Top-k metrics (critical for evaluation)
        try:
            logger.info("Computing top-k metrics...")
            metrics.update(self._compute_topk_metrics(predictions, targets))
        except Exception as e:
            logger.error(f"Failed to compute top-k metrics: {e}")
            raise RuntimeError(f"Critical failure in top-k metrics computation: {e}") from e
        
        # Non-critical metrics with individual error handling
        optional_computations = [
            ("threshold", lambda: self._compute_threshold_metrics(pred_np, target_np)),
            ("hierarchical", lambda: self._compute_hierarchical_metrics(predictions, targets) 
            if self._is_hierarchical(predictions) else {}),
            ("per_tag", lambda: self._compute_per_tag_metrics(pred_np, target_np, tag_names)
            if self.config.compute_per_tag_metrics else {}),
            ("frequency", lambda: self._compute_frequency_metrics(pred_np, target_np, tag_frequencies)
            if tag_frequencies is not None else {}),
            ("coverage", lambda: self._compute_coverage_metrics(pred_np, target_np)),
            ("mAP", lambda: {"mAP": self._compute_map(pred_np, target_np)}
            if self.config.compute_auc else {})
        ]
        
        for metric_name, compute_fn in optional_computations:
            try:
                logger.info(f"Computing {metric_name} metrics...")
                result = compute_fn()
                if result:
                    metrics.update(result)
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name} metrics: {e}")
                failed_metrics.append(metric_name)
                # Continue with other metrics instead of failing completely
        
        if failed_metrics:
            metrics['_failed_metrics'] = failed_metrics
            logger.warning(f"Some metrics failed to compute: {failed_metrics}")
        
        return metrics
    
    def _validate_inputs(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Validate input tensors"""
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")
        
        if not (0 <= predictions.min() and predictions.max() <= 1):
            logger.warning("Predictions not in [0,1] range, applying sigmoid")
            predictions = torch.sigmoid(predictions)
        
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("Targets must be binary (0 or 1)")
        return predictions, targets
    
    def _safe_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Safely convert tensor to numpy array"""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    
    def _is_hierarchical(self, predictions: torch.Tensor) -> bool:
        """Check if predictions follow hierarchical structure"""
        return predictions.shape[1] == self.config.num_groups * self.config.tags_per_group
    
    def _compute_basic_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute basic classification metrics"""
        # Apply threshold
        if self.config.adaptive_threshold:
            pred_binary = self._adaptive_threshold(predictions)
        else:
            pred_binary = (predictions > self.config.prediction_threshold).astype(np.int32)
        
        metrics = {}
        
        # Exact match ratio
        metrics['exact_match_ratio'] = np.all(pred_binary == targets, axis=1).mean()
        
        # Hamming loss
        metrics['hamming_loss'] = (pred_binary != targets).mean()
        
        # Vectorized computation for efficiency
        tp = ((pred_binary == 1) & (targets == 1)).sum()
        fp = ((pred_binary == 1) & (targets == 0)).sum()
        fn = ((pred_binary == 0) & (targets == 1)).sum()
        tn = ((pred_binary == 0) & (targets == 0)).sum()
        
        # Micro metrics - use float division with numpy for safety
        metrics['precision_micro'] = float(np.divide(tp, tp + fp, 
                                                     out=np.zeros(1), 
                                                     where=(tp + fp) != 0)[0])
        metrics['recall_micro'] = float(np.divide(tp, tp + fn,
                                                  out=np.zeros(1),
                                                  where=(tp + fn) != 0)[0])
        
        # F1 calculation with safer division
        if metrics['precision_micro'] + metrics['recall_micro'] > 0:
            metrics['f1_micro'] = 2 * metrics['precision_micro'] * metrics['recall_micro'] / \
                                 (metrics['precision_micro'] + metrics['recall_micro'])
        else:
            metrics['f1_micro'] = 0.0
        
        # Macro metrics - handle large number of tags
        if targets.shape[1] <= self.config.max_tags_for_detailed:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    targets.ravel(), pred_binary.ravel(), 
                    average='macro', zero_division=0
                )
        else:
            # Sample tags for macro computation
            sample_indices = np.random.choice(targets.shape[1], 
                                            self.config.max_tags_for_detailed, 
                                            replace=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    targets[:, sample_indices].ravel(), 
                    pred_binary[:, sample_indices].ravel(),
                    average='macro', zero_division=0
                )
        
        metrics['precision_macro'] = float(precision_macro)
        metrics['recall_macro'] = float(recall_macro)
        metrics['f1_macro'] = float(f1_macro)
        
        # Sample metrics
        sample_precisions = []
        sample_recalls = []
        sample_f1s = []
        
        # Process in batches for memory efficiency
        batch_size = self.config.batch_size_for_computation
        for i in range(0, len(pred_binary), batch_size):
            batch_pred = pred_binary[i:i+batch_size]
            batch_target = targets[i:i+batch_size]
            
            for j in range(len(batch_pred)):
                pred_sum = batch_pred[j].sum()
                target_sum = batch_target[j].sum()
                
                if pred_sum > 0 or target_sum > 0:
                    tp_sample = ((batch_pred[j] == 1) & (batch_target[j] == 1)).sum()
                    fp_sample = ((batch_pred[j] == 1) & (batch_target[j] == 0)).sum()
                    fn_sample = ((batch_pred[j] == 0) & (batch_target[j] == 1)).sum()
                    
                    prec = tp_sample / max(tp_sample + fp_sample, 1)
                    rec = tp_sample / max(tp_sample + fn_sample, 1)
                    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                    
                    sample_precisions.append(prec)
                    sample_recalls.append(rec)
                    sample_f1s.append(f1)
        
        metrics['precision_samples'] = float(np.mean(sample_precisions)) if sample_precisions else 0.0
        metrics['recall_samples'] = float(np.mean(sample_recalls)) if sample_recalls else 0.0
        metrics['f1_samples'] = float(np.mean(sample_f1s)) if sample_f1s else 0.0
        
        # Average predictions
        metrics['avg_predictions_per_sample'] = float(pred_binary.sum(axis=1).mean())
        metrics['avg_targets_per_sample'] = float(targets.sum(axis=1).mean())
        
        return metrics
    
    def _compute_topk_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute top-k accuracy metrics"""
        metrics = {}
        
        for k in self.config.top_k_values:
            if k > predictions.shape[1]:
                continue
                
            # Get top-k predictions
            _, top_k_indices = torch.topk(predictions, k=k, dim=1)
            
            # Create binary matrix for top-k predictions
            top_k_preds = torch.zeros_like(predictions)
            top_k_preds.scatter_(1, top_k_indices, 1)
            
            # Compute metrics
            tp = (top_k_preds * targets).sum()
            fp = (top_k_preds * (1 - targets)).sum()
            fn = ((1 - top_k_preds) * targets).sum()
            
            # Use safe division for tensor operations
            precision = torch.div(tp, tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
            recall = torch.div(tp, tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
            
            # F1 with explicit check
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = torch.tensor(0.0)
            
            metrics[f'precision_at_{k}'] = float(precision.item())
            metrics[f'recall_at_{k}'] = float(recall.item())
            metrics[f'f1_at_{k}'] = float(f1.item())
            
            # Top-k accuracy
            correct_in_topk = (top_k_preds * targets).sum(dim=1) > 0
            metrics[f'accuracy_at_{k}'] = float(correct_in_topk.float().mean().item())
        
        return metrics
    
    def _compute_threshold_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Analyze performance at different thresholds"""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        metrics = {}
        
        # Sample for efficiency if dataset is large
        sample_size = min(1000, predictions.shape[0])
        if predictions.shape[0] > sample_size:
            indices = np.random.choice(predictions.shape[0], sample_size, replace=False)
            pred_sample = predictions[indices]
            target_sample = targets[indices]
        else:
            pred_sample = predictions
            target_sample = targets
        
        for thresh in thresholds:
            pred_binary = (pred_sample > thresh).astype(np.int32)
            
            # Vectorized F1 computation
            tp = ((pred_binary == 1) & (target_sample == 1)).sum(axis=1)
            fp = ((pred_binary == 1) & (target_sample == 0)).sum(axis=1)
            fn = ((pred_binary == 0) & (target_sample == 1)).sum(axis=1)
            
            # Avoid division by zero
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), 
                                where=(tp + fp) != 0)
            recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), 
                             where=(tp + fn) != 0)
            f1 = np.divide(2 * precision * recall, precision + recall, 
                          out=np.zeros_like(precision), 
                          where=(precision + recall) != 0)
            
            metrics[f'f1_at_threshold_{thresh}'] = float(f1.mean())
            metrics[f'avg_predictions_at_threshold_{thresh}'] = float(pred_binary.sum(axis=1).mean())
        
        # Find optimal threshold using grid search
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            pred_binary = (pred_sample > thresh).astype(np.int32)
            
            tp = ((pred_binary == 1) & (target_sample == 1)).sum()
            fp = ((pred_binary == 1) & (target_sample == 0)).sum()
            fn = ((pred_binary == 0) & (target_sample == 1)).sum()
            
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        metrics['optimal_threshold'] = float(best_threshold)
        metrics['optimal_threshold_f1'] = float(best_f1)
        
        return metrics
    
    def _compute_hierarchical_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for hierarchical tag structure"""
        batch_size = predictions.shape[0]
        
        # Reshape to hierarchical structure
        pred_hier = predictions.view(batch_size, self.config.num_groups, self.config.tags_per_group)
        target_hier = targets.view(batch_size, self.config.num_groups, self.config.tags_per_group)
        
        metrics = {}
        
        # Per-group metrics
        group_f1_scores = []
        group_precisions = []
        group_recalls = []
        
        for g in range(self.config.num_groups):
            group_preds = pred_hier[:, g, :]
            group_targets = target_hier[:, g, :]
            
            # Convert to numpy for processing
            group_preds_np = group_preds.cpu().numpy()
            group_targets_np = group_targets.cpu().numpy()
            
            # Apply threshold
            if self.config.adaptive_threshold:
                group_binary = self._adaptive_threshold(group_preds_np)
            else:
                group_binary = (group_preds_np > self.config.prediction_threshold).astype(np.int32)
            
            # Compute metrics for this group
            tp = ((group_binary == 1) & (group_targets_np == 1)).sum()
            fp = ((group_binary == 1) & (group_targets_np == 0)).sum()
            fn = ((group_binary == 0) & (group_targets_np == 1)).sum()
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            group_f1_scores.append(f1)
            group_precisions.append(precision)
            group_recalls.append(recall)
        
        # Aggregate group metrics
        metrics['group_f1_mean'] = float(np.mean(group_f1_scores))
        metrics['group_f1_std'] = float(np.std(group_f1_scores))
        metrics['group_precision_mean'] = float(np.mean(group_precisions))
        metrics['group_recall_mean'] = float(np.mean(group_recalls))
        
        # Group coverage
        group_has_pred = (pred_hier > self.config.prediction_threshold).any(dim=2)
        metrics['avg_groups_with_predictions'] = float(group_has_pred.float().mean().item())
        
        # Group balance
        preds_per_group = (pred_hier > self.config.prediction_threshold).sum(dim=2).float()
        metrics['group_prediction_variance'] = float(preds_per_group.var(dim=1).mean().item())
        
        return metrics
    
    def _compute_per_tag_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        tag_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute metrics for individual tags"""
        num_tags = predictions.shape[1]
        
        # Determine which tags to analyze
        if num_tags > self.config.max_tags_for_detailed:
            # Sample tags based on frequency
            tag_freq = targets.sum(axis=0)
            # Get mix of frequent and infrequent tags
            freq_sorted = np.argsort(tag_freq)
            sample_indices = np.concatenate([
                freq_sorted[:self.config.max_tags_for_detailed//2],  # Least frequent
                freq_sorted[-self.config.max_tags_for_detailed//2:]  # Most frequent
            ])
        else:
            sample_indices = np.arange(num_tags)
        
        per_tag_metrics = {}
        tag_f1_scores = []
        tag_precisions = []
        tag_recalls = []
        tag_support = []
        
        for idx, tag_idx in enumerate(sample_indices):
            tag_preds = predictions[:, tag_idx]
            tag_targets = targets[:, tag_idx]
            
            # Skip if tag never appears
            support = tag_targets.sum()
            if support == 0:
                continue
            
            # Binarize predictions for this tag
            if self.config.adaptive_threshold:
                tag_binary = (self._adaptive_threshold(
                    predictions[:, [tag_idx]])[:, 0]).astype(np.int32)
            else:
                tag_binary = (tag_preds > self.config.prediction_threshold).astype(np.int32)
            tp = float(((tag_binary == 1) & (tag_targets == 1)).sum())
            fp = float(((tag_binary == 1) & (tag_targets == 0)).sum())
            fn = float(((tag_binary == 0) & (tag_targets == 1)).sum())
            
            # Use numpy divide for consistent safe division
            precision = np.divide(tp, tp + fp, out=np.zeros(1), where=(tp + fp) > 0)[0]
            recall = np.divide(tp, tp + fn, out=np.zeros(1), where=(tp + fn) > 0)[0]
            
            # F1 with explicit check
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            tag_f1_scores.append(f1)
            tag_precisions.append(precision)
            tag_recalls.append(recall)
            tag_support.append(support)
            
            # Store detailed metrics for specific tags
            if tag_names and tag_idx < len(tag_names):
                tag_name = tag_names[tag_idx]
                per_tag_metrics[tag_name] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'support': int(support)
                }
        
        # Aggregate statistics
        metrics = {
            'per_tag_f1_mean': float(np.mean(tag_f1_scores)) if tag_f1_scores else 0.0,
            'per_tag_f1_std': float(np.std(tag_f1_scores)) if tag_f1_scores else 0.0,
            'per_tag_precision_mean': float(np.mean(tag_precisions)) if tag_precisions else 0.0,
            'per_tag_recall_mean': float(np.mean(tag_recalls)) if tag_recalls else 0.0,
            'tags_with_zero_f1': int(sum(1 for f1 in tag_f1_scores if f1 == 0)),
            'tags_with_perfect_f1': int(sum(1 for f1 in tag_f1_scores if f1 >= 0.99))
        }
        
        # Find worst and best performing tags
        if tag_names and len(tag_f1_scores) > 0:
            sorted_indices = np.argsort(tag_f1_scores)
            
            # Worst 10 tags
            worst_tags = []
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                tag_idx = sample_indices[idx]
                if tag_idx < len(tag_names):
                    worst_tags.append({
                        'tag': tag_names[tag_idx],
                        'f1': float(tag_f1_scores[idx]),
                        'support': int(tag_support[idx])
                    })
            metrics['worst_performing_tags'] = worst_tags
            
            # Best 10 tags
            best_tags = []
            for i in range(max(0, len(sorted_indices)-10), len(sorted_indices)):
                idx = sorted_indices[i]
                tag_idx = sample_indices[idx]
                if tag_idx < len(tag_names):
                    best_tags.append({
                        'tag': tag_names[tag_idx],
                        'f1': float(tag_f1_scores[idx]),
                        'support': int(tag_support[idx])
                    })
            metrics['best_performing_tags'] = best_tags
        
        return metrics
    
    def _compute_frequency_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        tag_frequencies: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics based on tag frequency bins"""
        freq_np = self._safe_to_numpy(tag_frequencies)
        metrics = {}
        
        # Create frequency bins
        bins = [0] + self.config.frequency_bins + [float('inf')]
        
        for i in range(len(bins) - 1):
            bin_name = f"{bins[i]}-{bins[i+1] if bins[i+1] != float('inf') else 'inf'}"
            
            # Find tags in this frequency range
            mask = (freq_np >= bins[i]) & (freq_np < bins[i+1])
            if not mask.any():
                continue
            
            # Compute metrics for tags in this bin
            bin_preds = predictions[:, mask]
            bin_targets = targets[:, mask]
            
            # Binary predictions
            bin_binary = (bin_preds > self.config.prediction_threshold).astype(np.int32)
            
            tp = ((bin_binary == 1) & (bin_targets == 1)).sum()
            fp = ((bin_binary == 1) & (bin_targets == 0)).sum()
            fn = ((bin_binary == 0) & (bin_targets == 1)).sum()
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            metrics[f'precision_freq_{bin_name}'] = float(precision)
            metrics[f'recall_freq_{bin_name}'] = float(recall)
            metrics[f'f1_freq_{bin_name}'] = float(f1)
            metrics[f'num_tags_freq_{bin_name}'] = int(mask.sum())
        
        return metrics
    
    def _compute_coverage_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute tag coverage and diversity metrics"""
        pred_binary = (predictions > self.config.prediction_threshold).astype(np.int32)
        
        metrics = {}
        
        # Tag coverage
        tags_predicted = pred_binary.any(axis=0)
        metrics['tag_coverage'] = float(tags_predicted.mean())
        
        # Tag coverage in ground truth
        tags_in_gt = targets.any(axis=0)
        metrics['tag_coverage_gt'] = float(tags_in_gt.mean())
        
        # Coverage of ground truth tags
        covered_gt_tags = tags_predicted & tags_in_gt
        metrics['gt_tag_coverage'] = float(covered_gt_tags.sum() / max(tags_in_gt.sum(), 1))
        
# Prediction diversity (entropy) - use safe computation
        tag_pred_probs = pred_binary.mean(axis=0)
        tag_pred_probs = tag_pred_probs[tag_pred_probs > 0]
        if len(tag_pred_probs) > 0:
            # Normalize to get proper probability distribution with safety check
            prob_sum = tag_pred_probs.sum()
            if prob_sum > 0:
                tag_pred_probs = tag_pred_probs / prob_sum
                # Use np.where to safely compute log only for positive values
                log_probs = np.where(tag_pred_probs > 0, 
                                     np.log(tag_pred_probs), 
                                     0)
                entropy = -np.sum(tag_pred_probs * log_probs)
            else:
                entropy = 0.0
        else:
            entropy = 0.0
        metrics['prediction_entropy'] = float(entropy)
        
        # Sample similarity - compute on subset for efficiency
        sample_size = min(100, predictions.shape[0])
        if sample_size >= 2:
            indices = np.random.choice(predictions.shape[0], sample_size, replace=False)
            sample_pred = pred_binary[indices]
            
            similarities = []
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    intersection = float((sample_pred[i] & sample_pred[j]).sum())
                    union = float((sample_pred[i] | sample_pred[j]).sum())
                    if union > 0:
                        # Use explicit float division
                        similarity = intersection / union
                        similarities.append(similarity)
            
            if similarities:
                metrics['avg_sample_similarity'] = float(np.mean(similarities))
                metrics['sample_similarity_std'] = float(np.std(similarities))
            else:
                metrics['avg_sample_similarity'] = 0.0
                metrics['sample_similarity_std'] = 0.0
        
        return metrics
    
    def _compute_map(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Compute mean average precision"""
        # Sample if too large
        if predictions.shape[0] > self.config.sample_size_for_expensive:
            indices = np.random.choice(predictions.shape[0], 
                                     self.config.sample_size_for_expensive, 
                                     replace=False)
            predictions = predictions[indices]
            targets = targets[indices]
        
        # Compute AP for each sample
        ap_scores = []
        
        for i in range(predictions.shape[0]):
            # Skip if no positive labels
            if targets[i].sum() == 0:
                continue
            
            try:
                # Only compute for tags that appear in this sample
                mask = targets[i] > 0
                if mask.sum() > 0:
                    ap = skl_average_precision_score(targets[i][mask], predictions[i][mask])
                    ap_scores.append(ap)
            except Exception as e:
                logger.debug(f"Error computing AP for sample {i}: {e}")
                continue
        
        return float(np.mean(ap_scores)) if ap_scores else 0.0
    
    def _adaptive_threshold(self, predictions: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding"""
        binary_preds = np.zeros_like(predictions, dtype=np.int32)
        
        for i in range(predictions.shape[0]):
            sample_preds = predictions[i]
            
            # Sort predictions
            sorted_indices = np.argsort(sample_preds)[::-1]
            sorted_preds = sample_preds[sorted_indices]
            
            # Determine number of predictions
            num_pred = np.sum(sample_preds > self.config.prediction_threshold)
            
            if num_pred < self.config.min_predictions and len(sorted_preds) >= self.config.min_predictions:
                # Take top min_predictions
                threshold_idx = min(self.config.min_predictions, len(sorted_preds))
                top_indices = sorted_indices[:threshold_idx]
                binary_preds[i, top_indices] = 1
            elif num_pred > self.config.max_predictions:
                # Take top max_predictions
                top_indices = sorted_indices[:self.config.max_predictions]
                binary_preds[i, top_indices] = 1
            else:
                # Use standard threshold
                binary_preds[i] = (sample_preds > self.config.prediction_threshold).astype(np.int32)
        
        return binary_preds


class MetricVisualizer:
    """Visualize metrics and create plots"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_pr_curve(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        tag_names: Optional[List[str]] = None,
        num_tags_to_plot: int = 10
    ):
        """Plot precision-recall curves"""
        try:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            # Select tags to plot
            tag_support = targets.sum(axis=0)
            valid_tags = np.where(tag_support > 0)[0]
            
            if len(valid_tags) == 0:
                logger.warning("No tags with positive samples for PR curve")
                return
            
            # Select top tags by support
            top_tags = valid_tags[np.argsort(tag_support[valid_tags])[-num_tags_to_plot:]]
            
            for idx, tag_idx in enumerate(top_tags[:10]):
                ax = axes[idx]
                
                # Compute PR curve
                try:
                    precision, recall, _ = precision_recall_curve(
                        targets[:, tag_idx],
                        predictions[:, tag_idx]
                    )
                    
                    # Plot
                    ax.plot(recall, precision)
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.grid(True, alpha=0.3)
                    
                    # Title with tag name if available
                    if tag_names and tag_idx < len(tag_names):
                        ax.set_title(f'{tag_names[tag_idx][:20]}\n(support: {int(tag_support[tag_idx])})')
                    else:
                        ax.set_title(f'Tag {tag_idx}\n(support: {int(tag_support[tag_idx])})')
                except Exception as e:
                    logger.debug(f"Error plotting PR curve for tag {tag_idx}: {e}")
                    ax.set_title(f"Error for tag {tag_idx}")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'pr_curves.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating PR curve plot: {e}")
            plt.close('all')
    
    def plot_threshold_analysis(self, metrics: Dict[str, Any]):
        """Plot metrics vs threshold"""
        try:
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            f1_scores = [metrics.get(f'f1_at_threshold_{t}', 0) for t in thresholds]
            avg_preds = [metrics.get(f'avg_predictions_at_threshold_{t}', 0) for t in thresholds]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # F1 vs threshold
            ax1.plot(thresholds, f1_scores, 'o-', linewidth=2, markersize=8)
            optimal_thresh = metrics.get('optimal_threshold', 0.5)
            ax1.axvline(optimal_thresh, color='red', linestyle='--', 
                       label=f"Optimal: {optimal_thresh:.2f}")
            ax1.set_xlabel('Threshold')
            ax1.set_ylabel('F1 Score')
            ax1.set_title('F1 Score vs Threshold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Average predictions vs threshold
            ax2.plot(thresholds, avg_preds, 'o-', linewidth=2, markersize=8)
            avg_targets = metrics.get('avg_targets_per_sample', 0)
            ax2.axhline(avg_targets, color='green', linestyle='--',
                       label=f'Avg targets: {avg_targets:.1f}')
            ax2.set_xlabel('Threshold')
            ax2.set_ylabel('Average Predictions')
            ax2.set_title('Average Predictions vs Threshold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating threshold analysis plot: {e}")
            plt.close('all')
    
    def plot_frequency_performance(self, metrics: Dict[str, Any]):
        """Plot performance vs tag frequency"""
        try:
            freq_bins = []
            f1_scores = []
            
            # Extract frequency bin metrics
            for key, value in metrics.items():
                if key.startswith('f1_freq_'):
                    bin_name = key.replace('f1_freq_', '')
                    freq_bins.append(bin_name)
                    f1_scores.append(value)
            
            if not freq_bins:
                logger.warning("No frequency bin metrics to plot")
                return
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(freq_bins))
            bars = plt.bar(x, f1_scores)
            
            # Color bars by performance
            colors = ['red' if f1 < 0.3 else 'yellow' if f1 < 0.6 else 'green' for f1 in f1_scores]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.xticks(x, freq_bins, rotation=45)
            plt.xlabel('Tag Frequency Range')
            plt.ylabel('F1 Score')
            plt.title('Performance vs Tag Frequency')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{f1:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'frequency_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating frequency performance plot: {e}")
            plt.close('all')
    
    def plot_tag_cooccurrence_matrix(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        tag_names: List[str],
        num_tags: int = 20
    ):
        """Plot tag co-occurrence matrix (fixed version)"""
        try:
            # Select top tags by frequency
            tag_freq = targets.sum(axis=0)
            valid_tags = np.where(tag_freq > 0)[0]
            
            if len(valid_tags) == 0:
                logger.warning("No valid tags for co-occurrence matrix")
                return
            
            top_indices = valid_tags[np.argsort(tag_freq[valid_tags])[-num_tags:]]
            
            # Binary predictions
            pred_binary = (predictions > 0.5).astype(np.int32)
            
            # Compute co-occurrence matrices
            pred_cooc = np.zeros((num_tags, num_tags))
            true_cooc = np.zeros((num_tags, num_tags))
            
            for i, idx_i in enumerate(top_indices):
                for j, idx_j in enumerate(top_indices):
                    pred_cooc[i, j] = ((pred_binary[:, idx_i] == 1) & 
                                      (pred_binary[:, idx_j] == 1)).sum()
                    true_cooc[i, j] = ((targets[:, idx_i] == 1) & 
                                      (targets[:, idx_j] == 1)).sum()
            
            # Normalize by diagonal (self-occurrence)
            for i in range(num_tags):
                if true_cooc[i, i] > 0:
                    true_cooc[i, :] /= true_cooc[i, i]
                if pred_cooc[i, i] > 0:
                    pred_cooc[i, :] /= pred_cooc[i, i]
            
            # Compute difference
            diff_matrix = pred_cooc - true_cooc
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Get tag names for axes
            labels = [tag_names[idx][:15] if idx < len(tag_names) else f'Tag {idx}' 
                     for idx in top_indices]
            
            # Ground truth co-occurrence
            sns.heatmap(true_cooc, annot=False, cmap='Blues', 
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Co-occurrence Rate'},
                       ax=axes[0], vmin=0, vmax=1)
            axes[0].set_title('Ground Truth Co-occurrence')
            
            # Predicted co-occurrence
            sns.heatmap(pred_cooc, annot=False, cmap='Oranges',
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Co-occurrence Rate'},
                       ax=axes[1], vmin=0, vmax=1)
            axes[1].set_title('Predicted Co-occurrence')
            
            # Difference
            sns.heatmap(diff_matrix, annot=False, cmap='RdBu_r',
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Prediction - Truth'},
                       ax=axes[2], vmin=-0.5, vmax=0.5, center=0)
            axes[2].set_title('Co-occurrence Difference')
            
            for ax in axes:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'cooccurrence_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating co-occurrence matrix plot: {e}")
            plt.close('all')
    
    def create_metric_report(self, metrics: Dict[str, Any], save_path: Optional[Path] = None):
        """Create a comprehensive metric report"""
        if save_path is None:
            save_path = self.output_dir / 'metric_report.txt'
        
        try:
            with open(save_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("ANIME IMAGE TAGGER - EVALUATION REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # Overall metrics
                f.write("OVERALL PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                f.write(f"Exact Match Ratio: {metrics.get('exact_match_ratio', 0):.4f}\n")
                f.write(f"Hamming Loss: {metrics.get('hamming_loss', 0):.4f}\n")
                f.write(f"Mean Average Precision: {metrics.get('mAP', 0):.4f}\n\n")
                
                # Loss statistics if available
                if 'avg_loss' in metrics:
                    f.write(f"Average Loss: {metrics['avg_loss']:.4f} (±{metrics.get('std_loss', 0):.4f})\n\n")
                
                # Aggregate metrics
                f.write("AGGREGATE METRICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Micro Precision: {metrics.get('precision_micro', 0):.4f}\n")
                f.write(f"Micro Recall: {metrics.get('recall_micro', 0):.4f}\n")
                f.write(f"Micro F1: {metrics.get('f1_micro', 0):.4f}\n")
                f.write(f"Macro Precision: {metrics.get('precision_macro', 0):.4f}\n")
                f.write(f"Macro Recall: {metrics.get('recall_macro', 0):.4f}\n")
                f.write(f"Macro F1: {metrics.get('f1_macro', 0):.4f}\n")
                f.write(f"Sample Precision: {metrics.get('precision_samples', 0):.4f}\n")
                f.write(f"Sample Recall: {metrics.get('recall_samples', 0):.4f}\n")
                f.write(f"Sample F1: {metrics.get('f1_samples', 0):.4f}\n\n")
                
                # Top-k metrics
                f.write("TOP-K PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                for k in [1, 5, 10, 20, 50]:
                    if f'f1_at_{k}' in metrics:
                        f.write(f"@{k:3d}: F1={metrics[f'f1_at_{k}']:.4f} | ")
                        f.write(f"P={metrics[f'precision_at_{k}']:.4f} | ")
                        f.write(f"R={metrics[f'recall_at_{k}']:.4f} | ")
                        f.write(f"Acc={metrics.get(f'accuracy_at_{k}', 0):.4f}\n")
                f.write("\n")
                
                # Threshold analysis
                f.write("THRESHOLD ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Optimal Threshold: {metrics.get('optimal_threshold', 0.5):.3f}\n")
                f.write(f"F1 at Optimal: {metrics.get('optimal_threshold_f1', 0):.4f}\n")
                f.write(f"Avg Predictions per Sample: {metrics.get('avg_predictions_per_sample', 0):.1f}\n")
                f.write(f"Avg Targets per Sample: {metrics.get('avg_targets_per_sample', 0):.1f}\n\n")
                
                # Coverage metrics
                f.write("COVERAGE AND DIVERSITY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Tag Coverage (% predicted): {metrics.get('tag_coverage', 0)*100:.2f}%\n")
                f.write(f"GT Tag Coverage: {metrics.get('gt_tag_coverage', 0)*100:.2f}%\n")
                f.write(f"Prediction Entropy: {metrics.get('prediction_entropy', 0):.4f}\n")
                f.write(f"Avg Sample Similarity: {metrics.get('avg_sample_similarity', 0):.4f}")
                if 'sample_similarity_std' in metrics:
                    f.write(f" (±{metrics['sample_similarity_std']:.4f})")
                f.write("\n\n")
                
                # Hierarchical metrics
                if 'group_f1_mean' in metrics:
                    f.write("HIERARCHICAL GROUP METRICS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Mean Group F1: {metrics['group_f1_mean']:.4f} (±{metrics['group_f1_std']:.4f})\n")
                    f.write(f"Mean Group Precision: {metrics['group_precision_mean']:.4f}\n")
                    f.write(f"Mean Group Recall: {metrics['group_recall_mean']:.4f}\n")
                    f.write(f"Avg Groups with Predictions: {metrics['avg_groups_with_predictions']:.2f}\n")
                    f.write(f"Group Prediction Variance: {metrics['group_prediction_variance']:.4f}\n\n")
                
                # Per-tag statistics
                if 'per_tag_f1_mean' in metrics:
                    f.write("PER-TAG STATISTICS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Mean Tag F1: {metrics['per_tag_f1_mean']:.4f} (±{metrics['per_tag_f1_std']:.4f})\n")
                    f.write(f"Mean Tag Precision: {metrics['per_tag_precision_mean']:.4f}\n")
                    f.write(f"Mean Tag Recall: {metrics['per_tag_recall_mean']:.4f}\n")
                    f.write(f"Tags with Zero F1: {metrics['tags_with_zero_f1']}\n")
                    f.write(f"Tags with Perfect F1: {metrics['tags_with_perfect_f1']}\n\n")
                
                # Worst performing tags
                if 'worst_performing_tags' in metrics and metrics['worst_performing_tags']:
                    f.write("WORST PERFORMING TAGS\n")
                    f.write("-" * 40 + "\n")
                    for tag_info in metrics['worst_performing_tags'][:10]:
                        f.write(f"{tag_info['tag']:30s} F1: {tag_info['f1']:.4f} (support: {tag_info['support']})\n")
                    f.write("\n")
                
                # Best performing tags
                if 'best_performing_tags' in metrics and metrics['best_performing_tags']:
                    f.write("BEST PERFORMING TAGS\n")
                    f.write("-" * 40 + "\n")
                    for tag_info in metrics['best_performing_tags'][:10]:
                        f.write(f"{tag_info['tag']:30s} F1: {tag_info['f1']:.4f} (support: {tag_info['support']})\n")
                    f.write("\n")
                
                # Frequency-based performance
                f.write("FREQUENCY-BASED PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                freq_metrics = [(k, v) for k, v in metrics.items() if k.startswith('f1_freq_')]
                for key, value in sorted(freq_metrics):
                    bin_name = key.replace('f1_freq_', '')
                    f.write(f"Frequency {bin_name:15s}: F1={value:.4f}")
                    if f'num_tags_freq_{bin_name}' in metrics:
                        f.write(f" (tags: {metrics[f'num_tags_freq_{bin_name}']})")
                    f.write("\n")
                
                # Error reporting
                if 'error' in metrics:
                    f.write("\n" + "=" * 40 + "\n")
                    f.write("ERRORS ENCOUNTERED\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{metrics['error']}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
                
            logger.info(f"Metric report saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating metric report: {e}")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: MetricConfig,
    device: torch.device,
    tag_names: Optional[List[str]] = None,
    tag_frequencies: Optional[torch.Tensor] = None,
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Complete evaluation of model on a dataset
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader providing data
        config: Metric configuration
        device: Device to run on
        tag_names: Optional tag names
        tag_frequencies: Optional tag frequencies
        save_plots: Whether to save visualization plots
        
    Returns:
        Dictionary of all metrics
    """
    model.eval()
    metric_tracker = MetricTracker(device=str(device))
    
    # Collect predictions
    logger.info("Collecting model predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                images = batch['image'].to(device)
                targets = batch['labels']['binary'].to(device)
                
                # Get predictions
                outputs = model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Handle hierarchical output
                if logits.dim() == 3:
                    batch_size = logits.shape[0]
                    logits = logits.view(batch_size, -1)
                
                # Convert to probabilities
                predictions = torch.sigmoid(logits)
                
                # Track
                loss = outputs.get('loss', None) if isinstance(outputs, dict) else None
                metric_tracker.update(predictions, targets, loss=loss)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = metric_tracker.compute_metrics(config)
    
    # Add detailed metrics if requested
    if tag_names or tag_frequencies is not None:
        computer = MetricComputer(config)
        all_preds = torch.cat(metric_tracker.predictions, dim=0)
        all_targets = torch.cat(metric_tracker.targets, dim=0)
        
        detailed_metrics = computer.compute_all_metrics(
            all_preds, all_targets, tag_names, tag_frequencies
        )
        metrics.update(detailed_metrics)
    
    # Create visualizations
    if save_plots and config.save_plots:
        logger.info("Creating visualization plots...")
        visualizer = MetricVisualizer(config.plot_dir)
        
        if len(metric_tracker.predictions) > 0:
            all_preds_np = torch.cat(metric_tracker.predictions, dim=0).numpy()
            all_targets_np = torch.cat(metric_tracker.targets, dim=0).numpy()
            
            # Create various plots with error handling
            try:
                visualizer.plot_pr_curve(all_preds_np, all_targets_np, tag_names)
            except Exception as e:
                logger.error(f"Failed to create PR curve: {e}")
            
            try:
                visualizer.plot_threshold_analysis(metrics)
            except Exception as e:
                logger.error(f"Failed to create threshold analysis: {e}")
            
            try:
                visualizer.plot_frequency_performance(metrics)
            except Exception as e:
                logger.error(f"Failed to create frequency performance plot: {e}")
            
            if tag_names and all_preds_np.shape[0] < 10000:
                try:
                    visualizer.plot_tag_cooccurrence_matrix(all_preds_np, all_targets_np, tag_names)
                except Exception as e:
                    logger.error(f"Failed to create co-occurrence matrix: {e}")
        
        # Create report
        visualizer.create_metric_report(metrics)
    
    logger.info("Evaluation complete!")
    return metrics


if __name__ == "__main__":
    from utils.logging_setup import setup_logging
    # Test metrics computation with improved test case
    listener = setup_logging(log_level="INFO")
    logger.info("Testing improved metrics computation...")
    
    # Create more realistic dummy data
    batch_size = 100
    num_tags = 1000
    
    # Simulate predictions and targets
    torch.manual_seed(42)
    predictions = torch.rand(batch_size, num_tags)
    targets = torch.zeros(batch_size, num_tags)
    
    # Add some positive labels with realistic distribution
    for i in range(batch_size):
        num_pos = torch.randint(5, 20, (1,)).item()
        pos_indices = torch.randperm(num_tags)[:num_pos]
        targets[i, pos_indices] = 1
        # Make predictions somewhat correlated with targets
        predictions[i, pos_indices] += torch.rand(num_pos) * 0.5
    
    # Apply sigmoid to get probabilities
    predictions = torch.sigmoid(predictions)
    
    # Create tag names
    tag_names = [f"tag_{i}" for i in range(num_tags)]
    
    # Create tag frequencies (power law distribution)
    tag_frequencies = torch.tensor([1000 / (i + 1) for i in range(num_tags)])
    
    # Test metric computation
    config = MetricConfig(
        compute_per_tag_metrics=True,
        compute_auc=True,
        save_plots=False
    )
    
    computer = MetricComputer(config)
    
    try:
        metrics = computer.compute_all_metrics(
            predictions, targets, tag_names, tag_frequencies
        )
        
        # Print results
        print("\nComputed Metrics:")
        print("-" * 50)
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    print(f"  {key:30s}: {value:.4f}")
                else:
                    print(f"  {key:30s}: {value}")
        
        print("\nMetrics test completed successfully!")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if listener:
            listener.stop()