#!/usr/bin/env python3
"""
Evaluation Metrics for Anime Image Tagger
Comprehensive metrics for multi-label classification with 200k tags
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
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    multilabel_confusion_matrix
)
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
    
    # Visualization
    save_plots: bool = True
    plot_dir: str = "./metric_plots"


class MetricTracker:
    """Track metrics over time during training"""
    
    def __init__(self):
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
        self.predictions.append(predictions.cpu())
        self.targets.append(targets.cpu())
        
        if loss is not None:
            self.losses.append(loss)
            
        if metadata is not None:
            self.metadata.append(metadata)
    
    def compute_metrics(self, config: MetricConfig) -> Dict[str, Any]:
        """Compute all metrics"""
        # Concatenate all batches
        all_predictions = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Create metric computer
        computer = MetricComputer(config)
        
        return computer.compute_all_metrics(all_predictions, all_targets)


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
        """
        metrics = {}
        
        # Convert to numpy for sklearn
        pred_np = predictions.numpy()
        target_np = targets.numpy()
        
        # Basic metrics
        logger.info("Computing basic metrics...")
        metrics.update(self._compute_basic_metrics(pred_np, target_np))
        
        # Top-k metrics
        logger.info("Computing top-k metrics...")
        metrics.update(self._compute_topk_metrics(predictions, targets))
        
        # Threshold analysis
        logger.info("Computing threshold metrics...")
        metrics.update(self._compute_threshold_metrics(pred_np, target_np))
        
        # Hierarchical metrics (for grouped tags)
        if predictions.shape[1] == self.config.num_groups * self.config.tags_per_group:
            logger.info("Computing hierarchical metrics...")
            metrics.update(self._compute_hierarchical_metrics(predictions, targets))
        
        # Per-tag metrics
        if self.config.compute_per_tag_metrics:
            logger.info("Computing per-tag metrics...")
            metrics.update(self._compute_per_tag_metrics(pred_np, target_np, tag_names))
        
        # Frequency-based metrics
        if tag_frequencies is not None:
            logger.info("Computing frequency-based metrics...")
            metrics.update(self._compute_frequency_metrics(pred_np, target_np, tag_frequencies))
        
        # Coverage and diversity
        logger.info("Computing coverage metrics...")
        metrics.update(self._compute_coverage_metrics(pred_np, target_np))
        
        # Mean Average Precision
        if self.config.compute_auc:
            logger.info("Computing mAP...")
            metrics['mAP'] = self._compute_map(pred_np, target_np)
        
        return metrics
    
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
            pred_binary = predictions > self.config.prediction_threshold
        
        # Compute metrics
        metrics = {}
        
        # Exact match ratio
        metrics['exact_match_ratio'] = np.all(pred_binary == targets, axis=1).mean()
        
        # Hamming loss
        metrics['hamming_loss'] = (pred_binary != targets).mean()
        
        # Micro metrics (aggregate)
        tp_micro = ((pred_binary == 1) & (targets == 1)).sum()
        fp_micro = ((pred_binary == 1) & (targets == 0)).sum()
        fn_micro = ((pred_binary == 0) & (targets == 1)).sum()
        tn_micro = ((pred_binary == 0) & (targets == 0)).sum()
        
        metrics['precision_micro'] = tp_micro / (tp_micro + fp_micro + 1e-8)
        metrics['recall_micro'] = tp_micro / (tp_micro + fn_micro + 1e-8)
        metrics['f1_micro'] = 2 * metrics['precision_micro'] * metrics['recall_micro'] / \
                             (metrics['precision_micro'] + metrics['recall_micro'] + 1e-8)
        
        # Macro metrics (average per tag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                targets, pred_binary, average='macro', zero_division=0
            )
        
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        
        # Sample metrics (average per sample)
        precision_sample, recall_sample, f1_sample, _ = precision_recall_fscore_support(
            targets, pred_binary, average='samples', zero_division=0
        )
        
        metrics['precision_samples'] = precision_sample
        metrics['recall_samples'] = recall_sample
        metrics['f1_samples'] = f1_sample
        
        # Average number of predictions per sample
        metrics['avg_predictions_per_sample'] = pred_binary.sum(axis=1).mean()
        metrics['avg_targets_per_sample'] = targets.sum(axis=1).mean()
        
        return metrics
    
    def _compute_topk_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute top-k accuracy metrics"""
        metrics = {}
        
        for k in self.config.top_k_values:
            # Get top-k predictions
            _, top_k_indices = torch.topk(predictions, k=min(k, predictions.shape[1]), dim=1)
            
            # Create binary matrix for top-k predictions
            top_k_preds = torch.zeros_like(predictions)
            top_k_preds.scatter_(1, top_k_indices, 1)
            
            # Compute metrics
            tp = (top_k_preds * targets).sum()
            fp = (top_k_preds * (1 - targets)).sum()
            fn = ((1 - top_k_preds) * targets).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f'precision_at_{k}'] = precision.item()
            metrics[f'recall_at_{k}'] = recall.item()
            metrics[f'f1_at_{k}'] = f1.item()
            
            # Top-k accuracy (at least one correct in top-k)
            correct_in_topk = (top_k_preds * targets).sum(dim=1) > 0
            metrics[f'accuracy_at_{k}'] = correct_in_topk.float().mean().item()
        
        return metrics
    
    def _compute_threshold_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Analyze performance at different thresholds"""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        metrics = {}
        
        for thresh in thresholds:
            pred_binary = predictions > thresh
            
            # F1 score at this threshold
            f1_scores = []
            for i in range(targets.shape[0]):
                if targets[i].sum() > 0:  # Skip samples with no tags
                    tp = ((pred_binary[i] == 1) & (targets[i] == 1)).sum()
                    fp = ((pred_binary[i] == 1) & (targets[i] == 0)).sum()
                    fn = ((pred_binary[i] == 0) & (targets[i] == 1)).sum()
                    
                    prec = tp / (tp + fp + 1e-8)
                    rec = tp / (tp + fn + 1e-8)
                    f1 = 2 * prec * rec / (prec + rec + 1e-8)
                    f1_scores.append(f1)
            
            metrics[f'f1_at_threshold_{thresh}'] = np.mean(f1_scores)
            metrics[f'avg_predictions_at_threshold_{thresh}'] = pred_binary.sum(axis=1).mean()
        
        # Find optimal threshold
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            pred_binary = predictions > thresh
            f1_sample = []
            
            for i in range(min(1000, targets.shape[0])):  # Sample for speed
                if targets[i].sum() > 0:
                    tp = ((pred_binary[i] == 1) & (targets[i] == 1)).sum()
                    fp = ((pred_binary[i] == 1) & (targets[i] == 0)).sum()
                    fn = ((pred_binary[i] == 0) & (targets[i] == 1)).sum()
                    
                    prec = tp / (tp + fp + 1e-8)
                    rec = tp / (tp + fn + 1e-8)
                    f1 = 2 * prec * rec / (prec + rec + 1e-8)
                    f1_sample.append(f1)
            
            avg_f1 = np.mean(f1_sample)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = thresh
        
        metrics['optimal_threshold'] = best_threshold
        metrics['optimal_threshold_f1'] = best_f1
        
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
            
            # Apply threshold
            if self.config.adaptive_threshold:
                group_binary = self._adaptive_threshold(group_preds.numpy())
            else:
                group_binary = group_preds > self.config.prediction_threshold
                group_binary = group_binary.numpy()
            
            # Compute metrics for this group
            tp = ((group_binary == 1) & (group_targets.numpy() == 1)).sum()
            fp = ((group_binary == 1) & (group_targets.numpy() == 0)).sum()
            fn = ((group_binary == 0) & (group_targets.numpy() == 1)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            group_f1_scores.append(f1)
            group_precisions.append(precision)
            group_recalls.append(recall)
        
        # Aggregate group metrics
        metrics['group_f1_mean'] = np.mean(group_f1_scores)
        metrics['group_f1_std'] = np.std(group_f1_scores)
        metrics['group_precision_mean'] = np.mean(group_precisions)
        metrics['group_recall_mean'] = np.mean(group_recalls)
        
        # Group coverage (how many groups have predictions)
        group_has_pred = (pred_hier > self.config.prediction_threshold).any(dim=2)
        metrics['avg_groups_with_predictions'] = group_has_pred.float().mean().item()
        
        # Group balance (variance in predictions across groups)
        preds_per_group = (pred_hier > self.config.prediction_threshold).sum(dim=2).float()
        metrics['group_prediction_variance'] = preds_per_group.var(dim=1).mean().item()
        
        return metrics
    
    def _compute_per_tag_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        tag_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute metrics for individual tags"""
        num_tags = predictions.shape[1]
        
        # Sample tags if too many
        if num_tags > 1000 and self.config.sample_size_for_expensive:
            sample_indices = np.random.choice(num_tags, 1000, replace=False)
        else:
            sample_indices = np.arange(num_tags)
        
        per_tag_metrics = {}
        tag_f1_scores = []
        tag_precisions = []
        tag_recalls = []
        tag_support = []
        
        for tag_idx in sample_indices:
            tag_preds = predictions[:, tag_idx]
            tag_targets = targets[:, tag_idx]
            
            # Skip if tag never appears
            if tag_targets.sum() == 0:
                continue
            
            # Binary predictions for this tag
            tag_binary = tag_preds > self.config.prediction_threshold
            
            tp = ((tag_binary == 1) & (tag_targets == 1)).sum()
            fp = ((tag_binary == 1) & (tag_targets == 0)).sum()
            fn = ((tag_binary == 0) & (tag_targets == 1)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            tag_f1_scores.append(f1)
            tag_precisions.append(precision)
            tag_recalls.append(recall)
            tag_support.append(tag_targets.sum())
            
            # Store detailed metrics for specific tags
            if tag_names and tag_idx < len(tag_names):
                tag_name = tag_names[tag_idx]
                per_tag_metrics[tag_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': int(tag_targets.sum())
                }
        
        # Aggregate statistics
        metrics = {
            'per_tag_f1_mean': np.mean(tag_f1_scores),
            'per_tag_f1_std': np.std(tag_f1_scores),
            'per_tag_precision_mean': np.mean(tag_precisions),
            'per_tag_recall_mean': np.mean(tag_recalls),
            'tags_with_zero_f1': sum(1 for f1 in tag_f1_scores if f1 == 0),
            'tags_with_perfect_f1': sum(1 for f1 in tag_f1_scores if f1 == 1.0)
        }
        
        # Find worst and best performing tags
        if tag_names and len(tag_f1_scores) > 0:
            sorted_indices = np.argsort(tag_f1_scores)
            
            # Worst 10 tags
            worst_tags = []
            for idx in sorted_indices[:10]:
                if sample_indices[idx] < len(tag_names):
                    worst_tags.append({
                        'tag': tag_names[sample_indices[idx]],
                        'f1': tag_f1_scores[idx],
                        'support': int(tag_support[idx])
                    })
            metrics['worst_performing_tags'] = worst_tags
            
            # Best 10 tags
            best_tags = []
            for idx in sorted_indices[-10:]:
                if sample_indices[idx] < len(tag_names):
                    best_tags.append({
                        'tag': tag_names[sample_indices[idx]],
                        'f1': tag_f1_scores[idx],
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
        freq_np = tag_frequencies.numpy()
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
            bin_binary = bin_preds > self.config.prediction_threshold
            
            tp = ((bin_binary == 1) & (bin_targets == 1)).sum()
            fp = ((bin_binary == 1) & (bin_targets == 0)).sum()
            fn = ((bin_binary == 0) & (bin_targets == 1)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f'precision_freq_{bin_name}'] = precision
            metrics[f'recall_freq_{bin_name}'] = recall
            metrics[f'f1_freq_{bin_name}'] = f1
            metrics[f'num_tags_freq_{bin_name}'] = mask.sum()
        
        return metrics
    
    def _compute_coverage_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute tag coverage and diversity metrics"""
        pred_binary = predictions > self.config.prediction_threshold
        
        metrics = {}
        
        # Tag coverage (percentage of tags ever predicted)
        tags_predicted = pred_binary.any(axis=0)
        metrics['tag_coverage'] = tags_predicted.mean()
        
        # Tag coverage in ground truth
        tags_in_gt = targets.any(axis=0)
        metrics['tag_coverage_gt'] = tags_in_gt.mean()
        
        # Coverage of ground truth tags
        covered_gt_tags = tags_predicted & tags_in_gt
        metrics['gt_tag_coverage'] = covered_gt_tags.sum() / (tags_in_gt.sum() + 1e-8)
        
        # Prediction diversity (entropy)
        tag_pred_probs = pred_binary.mean(axis=0)
        tag_pred_probs = tag_pred_probs[tag_pred_probs > 0]  # Remove never-predicted
        entropy = -np.sum(tag_pred_probs * np.log(tag_pred_probs + 1e-8))
        metrics['prediction_entropy'] = entropy
        
        # Jaccard similarity between samples
        sample_similarities = []
        num_samples = min(100, predictions.shape[0])  # Sample for efficiency
        
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                intersection = (pred_binary[i] & pred_binary[j]).sum()
                union = (pred_binary[i] | pred_binary[j]).sum()
                if union > 0:
                    similarity = intersection / union
                    sample_similarities.append(similarity)
        
        if sample_similarities:
            metrics['avg_sample_similarity'] = np.mean(sample_similarities)
            metrics['sample_similarity_std'] = np.std(sample_similarities)
        
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
                ap = average_precision_score(targets[i], predictions[i])
                ap_scores.append(ap)
            except:
                continue
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def _adaptive_threshold(self, predictions: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to ensure reasonable number of predictions"""
        binary_preds = np.zeros_like(predictions, dtype=bool)
        
        for i in range(predictions.shape[0]):
            sample_preds = predictions[i]
            
            # Start with base threshold
            threshold = self.config.prediction_threshold
            num_pred = (sample_preds > threshold).sum()
            
            # Adjust threshold if needed
            if num_pred < self.config.min_predictions:
                # Lower threshold to get more predictions
                sorted_preds = np.sort(sample_preds)[::-1]
                if self.config.min_predictions <= len(sorted_preds):
                    threshold = sorted_preds[self.config.min_predictions - 1] - 1e-6
            elif num_pred > self.config.max_predictions:
                # Raise threshold to get fewer predictions
                sorted_preds = np.sort(sample_preds)[::-1]
                if self.config.max_predictions <= len(sorted_preds):
                    threshold = sorted_preds[self.config.max_predictions] + 1e-6
            
            binary_preds[i] = sample_preds > threshold
        
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
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        # Select tags to plot
        tag_support = targets.sum(axis=0)
        top_tags = np.argsort(tag_support)[-num_tags_to_plot:]
        
        for idx, tag_idx in enumerate(top_tags):
            ax = axes[idx]
            
            # Compute PR curve
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
                ax.set_title(f'{tag_names[tag_idx]}\n(support: {int(tag_support[tag_idx])})')
            else:
                ax.set_title(f'Tag {tag_idx}\n(support: {int(tag_support[tag_idx])})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pr_curves.png', dpi=150)
        plt.close()
    
    def plot_threshold_analysis(self, metrics: Dict[str, Any]):
        """Plot metrics vs threshold"""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        f1_scores = [metrics.get(f'f1_at_threshold_{t}', 0) for t in thresholds]
        avg_preds = [metrics.get(f'avg_predictions_at_threshold_{t}', 0) for t in thresholds]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # F1 vs threshold
        ax1.plot(thresholds, f1_scores, 'o-')
        ax1.axvline(metrics.get('optimal_threshold', 0.5), color='red', linestyle='--', 
                   label=f"Optimal: {metrics.get('optimal_threshold', 0.5):.2f}")
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score vs Threshold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Average predictions vs threshold
        ax2.plot(thresholds, avg_preds, 'o-')
        ax2.axhline(metrics.get('avg_targets_per_sample', 0), color='green', linestyle='--',
                   label='Avg targets per sample')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Average Predictions')
        ax2.set_title('Average Predictions vs Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_analysis.png', dpi=150)
        plt.close()
    
    def plot_frequency_performance(self, metrics: Dict[str, Any]):
        """Plot performance vs tag frequency"""
        freq_bins = []
        f1_scores = []
        
        # Extract frequency bin metrics
        for key, value in metrics.items():
            if key.startswith('f1_freq_'):
                bin_name = key.replace('f1_freq_', '')
                freq_bins.append(bin_name)
                f1_scores.append(value)
        
        if not freq_bins:
            return
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(freq_bins))
        plt.bar(x, f1_scores)
        plt.xticks(x, freq_bins, rotation=45)
        plt.xlabel('Tag Frequency Range')
        plt.ylabel('F1 Score')
        plt.title('Performance vs Tag Frequency')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frequency_performance.png', dpi=150)
        plt.close()
    
    def plot_confusion_examples(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        tag_names: List[str],
        num_examples: int = 20
    ):
        """Plot confusion matrix for top tags"""
        # Select top tags by frequency
        tag_freq = targets.sum(axis=0)
        top_indices = np.argsort(tag_freq)[-num_examples:]
        
        # Binary predictions
        pred_binary = predictions > 0.5
        
        # Compute confusion matrix
        cm = np.zeros((num_examples, num_examples))
        
        for i, idx_i in enumerate(top_indices):
            for j, idx_j in enumerate(top_indices):
                # Co-occurrence in predictions vs targets
                pred_cooc = (pred_binary[:, idx_i] & pred_binary[:, idx_j]).sum()
                true_cooc = (targets[:, idx_i] & targets[:, idx_j]).sum()
                cm[i, j] = pred_cooc / (true_cooc + 1e-8) if i != j else 1.0
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        
        # Get tag names for axes
        labels = [tag_names[idx] if idx < len(tag_names) else f'Tag {idx}' 
                 for idx in top_indices]
        
        sns.heatmap(cm, annot=False, cmap='YlOrRd', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Prediction / Ground Truth Co-occurrence'})
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title('Tag Co-occurrence Confusion')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
    
    def create_metric_report(self, metrics: Dict[str, Any], save_path: Optional[Path] = None):
        """Create a comprehensive metric report"""
        if save_path is None:
            save_path = self.output_dir / 'metric_report.txt'
        
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
            
            # Micro/Macro metrics
            f.write("AGGREGATE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Micro Precision: {metrics.get('precision_micro', 0):.4f}\n")
            f.write(f"Micro Recall: {metrics.get('recall_micro', 0):.4f}\n")
            f.write(f"Micro F1: {metrics.get('f1_micro', 0):.4f}\n")
            f.write(f"Macro Precision: {metrics.get('precision_macro', 0):.4f}\n")
            f.write(f"Macro Recall: {metrics.get('recall_macro', 0):.4f}\n")
            f.write(f"Macro F1: {metrics.get('f1_macro', 0):.4f}\n\n")
            
            # Top-k metrics
            f.write("TOP-K PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            for k in [1, 5, 10, 20, 50]:
                if f'f1_at_{k}' in metrics:
                    f.write(f"F1@{k}: {metrics[f'f1_at_{k}']:.4f} | ")
                    f.write(f"Precision@{k}: {metrics[f'precision_at_{k}']:.4f} | ")
                    f.write(f"Recall@{k}: {metrics[f'recall_at_{k}']:.4f}\n")
            f.write("\n")
            
            # Threshold analysis
            f.write("THRESHOLD ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Optimal Threshold: {metrics.get('optimal_threshold', 0.5):.3f}\n")
            f.write(f"F1 at Optimal: {metrics.get('optimal_threshold_f1', 0):.4f}\n\n")
            
            # Coverage metrics
            f.write("COVERAGE AND DIVERSITY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Tag Coverage: {metrics.get('tag_coverage', 0):.4f}\n")
            f.write(f"GT Tag Coverage: {metrics.get('gt_tag_coverage', 0):.4f}\n")
            f.write(f"Prediction Entropy: {metrics.get('prediction_entropy', 0):.4f}\n")
            f.write(f"Avg Sample Similarity: {metrics.get('avg_sample_similarity', 0):.4f}\n\n")
            
            # Hierarchical metrics
            if 'group_f1_mean' in metrics:
                f.write("HIERARCHICAL GROUP METRICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Group F1: {metrics['group_f1_mean']:.4f} (±{metrics['group_f1_std']:.4f})\n")
                f.write(f"Avg Groups with Predictions: {metrics['avg_groups_with_predictions']:.2f}\n\n")
            
            # Worst performing tags
            if 'worst_performing_tags' in metrics:
                f.write("WORST PERFORMING TAGS\n")
                f.write("-" * 40 + "\n")
                for tag_info in metrics['worst_performing_tags']:
                    f.write(f"{tag_info['tag']:30s} F1: {tag_info['f1']:.4f} (support: {tag_info['support']})\n")
                f.write("\n")
            
            # Best performing tags
            if 'best_performing_tags' in metrics:
                f.write("BEST PERFORMING TAGS\n")
                f.write("-" * 40 + "\n")
                for tag_info in metrics['best_performing_tags']:
                    f.write(f"{tag_info['tag']:30s} F1: {tag_info['f1']:.4f} (support: {tag_info['support']})\n")


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
    metric_tracker = MetricTracker()
    
    # Collect predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
            metric_tracker.update(predictions, targets)
    
    # Compute metrics
    metrics = metric_tracker.compute_metrics(config)
    
    # Add tag names and frequencies if provided
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
        visualizer = MetricVisualizer(config.plot_dir)
        
        # Create various plots
        if len(metric_tracker.predictions) > 0:
            all_preds_np = torch.cat(metric_tracker.predictions, dim=0).numpy()
            all_targets_np = torch.cat(metric_tracker.targets, dim=0).numpy()
            
            visualizer.plot_pr_curve(all_preds_np, all_targets_np, tag_names)
            visualizer.plot_threshold_analysis(metrics)
            visualizer.plot_frequency_performance(metrics)
            
            if tag_names and all_preds_np.shape[0] < 10000:  # Only for smaller datasets
                visualizer.plot_confusion_examples(all_preds_np, all_targets_np, tag_names)
        
        # Create report
        visualizer.create_metric_report(metrics)
    
    return metrics


if __name__ == "__main__":
    # Test metrics computation
    logger.info("Testing metrics computation...")
    
    # Create dummy data
    batch_size = 100
    num_tags = 1000
    
    # Simulate predictions and targets
    predictions = torch.rand(batch_size, num_tags)
    targets = torch.zeros(batch_size, num_tags)
    
    # Add some positive labels
    for i in range(batch_size):
        num_pos = torch.randint(5, 20, (1,)).item()
        pos_indices = torch.randperm(num_tags)[:num_pos]
        targets[i, pos_indices] = 1
        # Make predictions somewhat correlated
        predictions[i, pos_indices] += 0.3
    
    predictions = torch.sigmoid(predictions)
    
    # Test metric computation
    config = MetricConfig()
    computer = MetricComputer(config)
    
    metrics = computer.compute_all_metrics(predictions, targets)
    
    # Print results
    print("\nComputed Metrics:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
    
    print("\nMetrics test completed!")
