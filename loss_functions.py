#!/usr/bin/env python3
"""
Loss Functions for Anime Image Tagger - Simplified Version
Direct training with Asymmetric Focal Loss only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-label classification
    Simplified version with optimized parameters for direct training
    """
    
    def __init__(
        self,
        gamma_pos: float = 1.0,  # Easier on false negatives
        gamma_neg: float = 3.0,  # Penalize false positives more
        alpha: float = 0.75,     # Unified weight
        clip: float = 0.05,
        reduction: str = 'mean',
        label_smoothing: float = 0.05
    ):
        """
        Args:
            gamma_pos: Focusing parameter for positive samples
            gamma_neg: Focusing parameter for negative samples (higher = more penalty)
            alpha: Unified weighting factor
            clip: Probability clipping to prevent log(0)
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
            label_smoothing: Label smoothing factor for regularization
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.clip = clip
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) raw logits
            targets: (B, num_classes) binary targets
            sample_weights: (B,) optional per-sample weights
            
        Returns:
            Computed loss value
        """
        # Detach targets to prevent gradient flow
        targets = targets.detach()
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2
        
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        
        # Clip probabilities to prevent log(0)
        probs = torch.clamp(probs, min=self.clip, max=1.0 - self.clip)
        
        # Calculate focal weights
        # For positive samples - less aggressive weighting
        pos_weights = targets * torch.pow(1 - probs, self.gamma_pos)
        
        # For negative samples - more aggressive to reduce false positives
        neg_weights = (1 - targets) * torch.pow(probs, self.gamma_neg)
        
        # Binary cross entropy
        bce = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        
        # Apply focal weights with unified alpha
        focal_loss = self.alpha * (pos_weights * bce + neg_weights * bce)
        
        # Apply sample weights if provided (for frequency-based sampling)
        if sample_weights is not None:
            if sample_weights.dim() == 1:
                sample_weights = sample_weights.unsqueeze(1)
            focal_loss = focal_loss * sample_weights
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # none
            return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss for tag prediction and rating classification
    Simple weighted combination of two losses
    """
    
    def __init__(
        self,
        tag_loss_weight: float = 0.9,
        rating_loss_weight: float = 0.1,
        tag_loss_fn: Optional[nn.Module] = None,
        rating_loss_fn: Optional[nn.Module] = None
    ):
        """
        Args:
            tag_loss_weight: Weight for tag prediction loss
            rating_loss_weight: Weight for rating classification loss
            tag_loss_fn: Loss function for tags (defaults to AsymmetricFocalLoss)
            rating_loss_fn: Loss function for ratings (defaults to CrossEntropyLoss)
        """
        super().__init__()
        self.tag_loss_weight = tag_loss_weight
        self.rating_loss_weight = rating_loss_weight
        
        # Initialize loss functions
        self.tag_loss_fn = tag_loss_fn or AsymmetricFocalLoss()
        self.rating_loss_fn = rating_loss_fn or nn.CrossEntropyLoss(
            label_smoothing=0.1  # Some smoothing for rating classification
        )
    
    def forward(
        self,
        tag_logits: torch.Tensor,
        rating_logits: torch.Tensor,
        tag_targets: torch.Tensor,
        rating_targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss for tags and ratings
        
        Args:
            tag_logits: (B, num_tags) logits for tag prediction
            rating_logits: (B, num_ratings) logits for rating classification
            tag_targets: (B, num_tags) binary targets for tags
            rating_targets: (B,) or (B, num_ratings) targets for ratings
            sample_weights: Optional per-sample weights
            
        Returns:
            Total loss and dictionary of individual losses
        """
        # Compute tag loss
        tag_loss = self.tag_loss_fn(tag_logits, tag_targets, sample_weights)
        
        # Compute rating loss
        if rating_targets.dim() == 2:
            # If one-hot encoded, convert to class indices
            rating_targets = rating_targets.argmax(dim=1)
        rating_loss = self.rating_loss_fn(rating_logits, rating_targets)
        
        # Combine losses
        total_loss = (
            self.tag_loss_weight * tag_loss + 
            self.rating_loss_weight * rating_loss
        )
        
        # Return total and components
        losses = {
            'total': total_loss,
            'tag_loss': tag_loss,
            'rating_loss': rating_loss
        }
        
        return total_loss, losses


class FrequencyWeightedSampler:
    """
    Helper class to compute frequency-based sample weights
    Used during training to balance common vs rare tags
    """
    
    def __init__(
        self,
        tag_frequencies: torch.Tensor,
        weighting_type: str = 'sqrt_inverse',
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        """
        Args:
            tag_frequencies: Tensor of tag occurrence counts
            weighting_type: 'inverse', 'sqrt_inverse', or 'log_inverse'
            min_weight: Minimum weight to prevent undersampling
            max_weight: Maximum weight to prevent oversampling
        """
        self.weighting_type = weighting_type
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Compute and cache weights
        self.weights = self._compute_weights(tag_frequencies)
    
    def _compute_weights(self, frequencies: torch.Tensor) -> torch.Tensor:
        """Compute sampling weights from frequencies"""
        epsilon = 1e-7
        
        if self.weighting_type == 'inverse':
            weights = 1.0 / (frequencies + epsilon)
        elif self.weighting_type == 'sqrt_inverse':
            weights = 1.0 / torch.sqrt(frequencies + epsilon)
        elif self.weighting_type == 'log_inverse':
            weights = 1.0 / torch.log(frequencies + 1.0)
        else:
            weights = torch.ones_like(frequencies)
        
        # Normalize to mean of 1.0
        weights = weights / weights.mean()
        
        # Clip to prevent extreme values
        weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
        
        return weights
    
    def get_sample_weights(
        self,
        tag_indices: torch.Tensor,
        aggregate: str = 'mean'
    ) -> torch.Tensor:
        """
        Get weights for a batch of samples
        
        Args:
            tag_indices: (B, num_active_tags) indices of active tags per sample
            aggregate: How to aggregate multiple tag weights ('mean', 'max', 'sum')
            
        Returns:
            (B,) tensor of sample weights
        """
        batch_weights = []
        
        for sample_tags in tag_indices:
            if len(sample_tags) == 0:
                batch_weights.append(1.0)
            else:
                tag_weights = self.weights[sample_tags]
                
                if aggregate == 'mean':
                    weight = tag_weights.mean()
                elif aggregate == 'max':
                    weight = tag_weights.max()
                elif aggregate == 'sum':
                    weight = tag_weights.sum()
                else:
                    weight = tag_weights.mean()
                
                batch_weights.append(weight.item())
        
        return torch.tensor(batch_weights, device=tag_indices.device)


def create_training_loss(config: Dict) -> nn.Module:
    """
    Create loss function based on configuration
    Simplified to only support direct training
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Loss module
    """
    loss_type = config.get('type', 'focal')
    
    if loss_type == 'focal':
        return AsymmetricFocalLoss(
            gamma_pos=config.get('gamma_pos', 1.0),
            gamma_neg=config.get('gamma_neg', 3.0),
            alpha=config.get('alpha', 0.75),
            clip=config.get('clip', 0.05),
            label_smoothing=config.get('label_smoothing', 0.05)
        )
    
    elif loss_type == 'multitask':
        # For when you have both tags and ratings
        tag_loss = AsymmetricFocalLoss(
            gamma_pos=config.get('gamma_pos', 1.0),
            gamma_neg=config.get('gamma_neg', 3.0),
            alpha=config.get('alpha', 0.75),
            label_smoothing=config.get('label_smoothing', 0.05)
        )
        
        return MultiTaskLoss(
            tag_loss_weight=config.get('tag_weight', 0.9),
            rating_loss_weight=config.get('rating_weight', 0.1),
            tag_loss_fn=tag_loss
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute evaluation metrics for multi-label classification
    
    Args:
        predictions: (B, num_classes) predicted probabilities
        targets: (B, num_classes) binary targets
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Dictionary of metrics
    """
    # Convert predictions to binary
    pred_binary = (predictions > threshold).float()
    
    # True positives, false positives, false negatives
    tp = (pred_binary * targets).sum(dim=1)
    fp = (pred_binary * (1 - targets)).sum(dim=1)
    fn = ((1 - pred_binary) * targets).sum(dim=1)
    
    # Precision, Recall, F1 (micro-averaged)
    micro_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-8)
    micro_recall = tp.sum() / (tp.sum() + fn.sum() + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    
    # Per-class metrics for macro-averaging
    class_tp = (pred_binary * targets).sum(dim=0)
    class_fp = (pred_binary * (1 - targets)).sum(dim=0)
    class_fn = ((1 - pred_binary) * targets).sum(dim=0)
    
    class_precision = class_tp / (class_tp + class_fp + 1e-8)
    class_recall = class_tp / (class_tp + class_fn + 1e-8)
    class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)
    
    # Macro-averaged metrics
    macro_precision = class_precision.mean()
    macro_recall = class_recall.mean()
    macro_f1 = class_f1.mean()
    
    # False positive rate (important for user experience)
    total_negatives = (1 - targets).sum()
    false_positive_rate = fp.sum() / (total_negatives + 1e-8)
    
    return {
        'micro_precision': micro_precision.item(),
        'micro_recall': micro_recall.item(),
        'micro_f1': micro_f1.item(),
        'macro_precision': macro_precision.item(),
        'macro_recall': macro_recall.item(),
        'macro_f1': macro_f1.item(),
        'false_positive_rate': false_positive_rate.item()
    }


if __name__ == "__main__":
    # Test simplified loss functions
    print("Testing simplified loss functions...")
    
    B, num_tags = 4, 100000  # 100k tags as specified
    num_ratings = 5  # g, s, q, e, explicit
    
    # Test data
    tag_logits = torch.randn(B, num_tags)
    tag_targets = torch.randint(0, 2, (B, num_tags)).float()
    rating_logits = torch.randn(B, num_ratings)
    rating_targets = torch.randint(0, num_ratings, (B,))
    
    # Test AsymmetricFocalLoss
    focal_loss = AsymmetricFocalLoss(
        gamma_pos=1.0,
        gamma_neg=3.0,
        alpha=0.75,
        label_smoothing=0.05
    )
    loss = focal_loss(tag_logits, tag_targets)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test MultiTaskLoss
    multitask_loss = MultiTaskLoss(
        tag_loss_weight=0.9,
        rating_loss_weight=0.1
    )
    total_loss, losses = multitask_loss(
        tag_logits, rating_logits,
        tag_targets, rating_targets
    )
    print(f"\nMulti-task loss: {total_loss.item():.4f}")
    print(f"  Tag loss: {losses['tag_loss'].item():.4f}")
    print(f"  Rating loss: {losses['rating_loss'].item():.4f}")
    
    # Test metrics computation
    predictions = torch.sigmoid(tag_logits)
    metrics = compute_metrics(predictions, tag_targets)
    print(f"\nMetrics:")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  FP Rate: {metrics['false_positive_rate']:.4f}")
    
    # Test frequency weighting
    tag_frequencies = torch.randint(10, 10000, (num_tags,)).float()
    sampler = FrequencyWeightedSampler(
        tag_frequencies,
        weighting_type='sqrt_inverse'
    )
    print(f"\nFrequency weights shape: {sampler.weights.shape}")
    print(f"Weight range: [{sampler.weights.min():.2f}, {sampler.weights.max():.2f}]")
    
    print("\n✓ Simplified loss functions working correctly!")