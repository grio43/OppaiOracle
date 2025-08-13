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
        weights = torch.clamp(weights, min=self.min_weight,