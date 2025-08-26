#!/usr/bin/env python3
"""
Loss functions for the anime image tagger.

This module implements an asymmetric focal loss for multi‑label
classification.  The pad class at index 0 is ignored when computing
the tag loss.  This avoids penalising the model for predicting or
missing the padding index, which is not a valid output class.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi‑label classification.

    The loss ignores the pad class at index 0.  Logits and targets are
    sliced to remove this column before computing the loss.  Reduction
    semantics are unchanged: ``mean`` averages the per‑element losses
    across all remaining classes and samples, ``sum`` sums them, and
    ``none`` returns the unreduced tensor.
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 3.0,
        alpha: float = 0.75,
        clip: float = 0.05,
        reduction: str = 'mean',
        label_smoothing: float = 0.05,
    ):
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
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the asymmetric focal loss.

        Args:
            logits: (B, num_classes) raw logits.
            targets: (B, num_classes) binary targets.
            sample_weights: (B,) optional per‑sample weights.

        Returns:
            Loss value.
        """
        # Detach targets to prevent gradient flow
        targets = targets.detach()

        # Ignore pad class at index 0 by slicing (with bounds checking)
        if logits.size(1) > 1:  # Only slice if we have more than 1 column
            # Validate dimensions before slicing
            if targets.size(1) == logits.size(1):
                logits = logits[:, 1:]
                targets = targets[:, 1:]

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

        # Clamp logits to prevent numerical instability with mixed precision.
        logits = torch.clamp(logits, min=-15.0, max=15.0)

        # Use BCEWithLogitsLoss for numerical stability.
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Calculate probabilities for focal weights.
        probs = torch.sigmoid(logits)

        # Clip probabilities for focal weights to prevent pow(0, gamma) issues.
        probs = torch.clamp(probs, min=self.clip, max=1.0 - self.clip)

        # Calculate focal weights
        pos_weights = targets * torch.pow(1 - probs, self.gamma_pos)
        neg_weights = (1 - targets) * torch.pow(probs, self.gamma_neg)

        # Apply focal weights with unified alpha
        # Note: The original implementation had a slight conceptual error in how it applied
        # weights to the combined BCE loss. We are preserving the original formula's
        # structure but applying it to the more stable BCE loss calculation.
        focal_loss = self.alpha * (pos_weights * bce_loss + neg_weights * bce_loss)

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
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss for tag prediction and rating classification.

    A simple weighted combination of the asymmetric focal loss for tags
    and the cross‑entropy loss for ratings.
    """

    def __init__(
        self,
        tag_loss_weight: float = 0.9,
        rating_loss_weight: float = 0.1,
        tag_loss_fn: Optional[nn.Module] = None,
        rating_loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.tag_loss_weight = tag_loss_weight
        self.rating_loss_weight = rating_loss_weight
        self.tag_loss_fn = tag_loss_fn or AsymmetricFocalLoss()
        self.rating_loss_fn = rating_loss_fn or nn.CrossEntropyLoss(
            label_smoothing=0.1
        )

    def forward(
        self,
        tag_logits: torch.Tensor,
        rating_logits: torch.Tensor,
        tag_targets: torch.Tensor,
        rating_targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss for tags and ratings.

        Args:
            tag_logits: (B, num_tags) logits for tag prediction.
            rating_logits: (B, num_ratings) logits for rating classification.
            tag_targets: (B, num_tags) binary targets for tags.
            rating_targets: (B,) or (B, num_ratings) targets for ratings.
            sample_weights: Optional per‑sample weights.
        """
        tag_loss = self.tag_loss_fn(tag_logits, tag_targets, sample_weights)
        # Compute rating loss
        if rating_targets.dim() == 2:
            rating_targets = rating_targets.argmax(dim=1)
        rating_loss = self.rating_loss_fn(rating_logits, rating_targets)
        total_loss = (
            self.tag_loss_weight * tag_loss +
            self.rating_loss_weight * rating_loss
        )
        losses = {
            'total': total_loss,
            'tag_loss': tag_loss,
            'rating_loss': rating_loss,
        }
        return total_loss, losses


class FrequencyWeightedSampler:
    """
    Helper class to compute frequency‑based sample weights.

    Used during training to balance common vs rare tags.  Tags with high
    frequency receive lower weights, while rare tags receive higher weights.
    """

    def __init__(
        self,
        tag_frequencies: torch.Tensor,
        weighting_type: str = 'sqrt_inverse',
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ):
        self.weighting_type = weighting_type
        self.min_weight = min_weight
        self.max_weight = max_weight
        # Precompute weights per class (excluding pad index)
        self.weights = self._compute_weights(tag_frequencies)

    def _compute_weights(self, freqs: torch.Tensor) -> torch.Tensor:
        """Compute weights inversely proportional to frequencies."""
        # Avoid division by zero
        freqs = freqs.float().clamp(min=1)
        if self.weighting_type == 'inverse':
            weights = 1.0 / freqs
        elif self.weighting_type == 'sqrt_inverse':
            weights = 1.0 / torch.sqrt(freqs)
        elif self.weighting_type == 'log_inverse':
            weights = 1.0 / torch.log(freqs + 1.0)
        else:
            logger.warning(
                "Unknown weighting_type %s, defaulting to sqrt_inverse",
                self.weighting_type,
            )
            weights = 1.0 / torch.sqrt(freqs)
        weights = weights / weights.max()
        weights = torch.clamp(weights, self.min_weight, self.max_weight)
        return weights