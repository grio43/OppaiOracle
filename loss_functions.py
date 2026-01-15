#!/usr/bin/env python3
"""
Loss functions for the anime image tagger.

This module implements an asymmetric focal loss for multi‑label
classification.  The pad class at index 0 is ignored when computing
the tag loss.  This avoids penalising the model for predicting or
missing the padding index, which is not a valid output class.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, ClassVar, Union
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

    # Class-level cache for ignore_indices masks to avoid creating new tensors per forward pass
    # Key: (num_classes, device_index, tuple(sorted_ignore_indices)) -> mask tensor
    _keep_mask_cache: Dict[Tuple[int, int, Tuple[int, ...]], torch.Tensor] = {}

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 3.0,
        alpha: float = 0.75,
        clip: float = 0.05,
        reduction: str = 'mean',
        label_smoothing: float = 0.05,
        ignore_indices: Optional[Union[int, List[int]]] = 0,
    ):
        super().__init__()

        # Validate reduction parameter
        valid_reductions = ('mean', 'sum', 'none')
        if reduction not in valid_reductions:
            raise ValueError(
                f"Invalid reduction '{reduction}'. Must be one of {valid_reductions}"
            )

        # Validate clip parameter (must be in [0, 1))
        if clip < 0 or clip >= 1:
            raise ValueError(
                f"clip must be in range [0, 1), got {clip}. "
                f"Typical values are 0.0 (disabled) or 0.05."
            )

        # Validate gamma values (must be non-negative, reasonable range)
        if gamma_pos < 0:
            raise ValueError(f"gamma_pos must be non-negative, got {gamma_pos}")
        if gamma_neg < 0:
            raise ValueError(f"gamma_neg must be non-negative, got {gamma_neg}")
        if gamma_pos > 10:
            logger.warning(f"gamma_pos={gamma_pos} is unusually high (typical range 0-4)")
        if gamma_neg > 10:
            logger.warning(f"gamma_neg={gamma_neg} is unusually high (typical range 0-4)")

        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.clip = clip
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        # If not None/empty, drop these class indices from both logits and targets.
        # By default we ignore index 0 for TAGS to avoid penalising the <PAD> token.
        # For tags, pass [0, 1] to ignore both <PAD> and <UNK>.
        # For single-label ratings, pass None or [] to keep all classes.
        # Normalize to list for consistent handling
        if ignore_indices is None:
            self.ignore_indices: List[int] = []
        elif isinstance(ignore_indices, int):
            self.ignore_indices = [ignore_indices]
        else:
            self.ignore_indices = list(ignore_indices)

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
            targets: (B, num_classes) binary targets or (B,) class indices.
            sample_weights: (B,) optional per‑sample weights.

        Returns:
            Loss value.

        Raises:
            ValueError: If shapes are incompatible or values are invalid.
        """
        # Validate logits shape
        if logits.dim() != 2:
            raise ValueError(
                f"AsymmetricFocalLoss expects 2D logits (batch, classes), "
                f"got shape {logits.shape}"
            )

        batch_size, num_classes = logits.shape

        # Note: targets are ground truth labels and should not have requires_grad=True.
        # We don't detach here because:
        # 1. Labels shouldn't have gradients in the first place
        # 2. The operations below (one-hot, dtype cast) create new tensors anyway

        # Ensure targets have shape (B, C). If provided as class indices (B,),
        # convert to one-hot suitable for BCE-with-logits.
        if targets.dim() == 1:
            # Validate batch size matches
            if targets.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: logits have {batch_size} samples "
                    f"but targets have {targets.size(0)} samples. "
                    f"Logits shape: {logits.shape}, targets shape: {targets.shape}"
                )

            # Treat as single-label class indices
            if targets.dtype not in (torch.long, torch.int64, torch.int32):
                # Attempt safe cast to long for one-hot encoding
                targets = targets.to(torch.long)

            # Validate indices are in valid range
            if torch.any(targets < 0) or torch.any(targets >= num_classes):
                min_val, max_val = targets.min().item(), targets.max().item()
                raise ValueError(
                    f"AsymmetricFocalLoss: target indices out of range [0, {num_classes-1}]. "
                    f"Got values in range [{min_val}, {max_val}]. "
                    f"This indicates a mismatch between model output size and target labels."
                )

            targets = F.one_hot(targets, num_classes=num_classes).to(dtype=logits.dtype)

        elif targets.dim() == 2:
            # Validate shape matches exactly
            if targets.shape != logits.shape:
                raise ValueError(
                    f"Shape mismatch: logits shape {logits.shape} != targets shape {targets.shape}. "
                    f"For multi-label targets, shapes must match exactly."
                )
            targets = targets.to(dtype=logits.dtype)

        else:
            raise ValueError(
                f"AsymmetricFocalLoss expects targets with 1 or 2 dimensions, "
                f"got {targets.dim()} dimensions (shape: {targets.shape})"
            )

        # Validate sample_weights if provided
        if sample_weights is not None:
            if sample_weights.dim() != 1 or sample_weights.size(0) != batch_size:
                raise ValueError(
                    f"sample_weights must have shape ({batch_size},), "
                    f"got shape {sample_weights.shape}"
                )

        # Optionally ignore specified classes (e.g., PAD at index 0, UNK at index 1)
        if self.ignore_indices:
            c = logits.size(1)
            # Filter to valid indices only
            valid_ignore = tuple(sorted(idx for idx in self.ignore_indices if 0 <= idx < c))
            if valid_ignore:
                # Use cached mask to avoid creating new tensors per forward pass
                # Key: (num_classes, device_index, tuple(sorted_ignore_indices))
                device_idx = logits.device.index if logits.device.type == 'cuda' else -1
                cache_key = (c, device_idx, valid_ignore)

                if cache_key not in AsymmetricFocalLoss._keep_mask_cache:
                    keep = torch.ones(c, dtype=torch.bool, device=logits.device)
                    for idx in valid_ignore:
                        keep[idx] = False
                    AsymmetricFocalLoss._keep_mask_cache[cache_key] = keep

                keep = AsymmetricFocalLoss._keep_mask_cache[cache_key]
                logits = logits[:, keep]
                targets = targets[:, keep]

        # Apply label smoothing if specified
        # Compute in float32 for numerical stability, especially for bfloat16 training
        if self.label_smoothing > 0:
            targets = targets.float() * (1 - self.label_smoothing) + self.label_smoothing / 2
            targets = targets.to(dtype=logits.dtype)

        # Use BCEWithLogitsLoss for numerical stability.
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Calculate probabilities for focal weighting
        probs = torch.sigmoid(logits)

        # Apply probability clipping for asymmetric focusing on negatives.
        # This shifts negative probabilities, reducing the contribution of easy negatives.
        # For negatives, we compute (p + clip).clamp(max=1) which makes the model
        # "think" it's less confident, reducing the focal weight for easy negatives.
        if self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1.0)
        else:
            probs_neg = probs

        # Calculate focal weights using log-space math for numerical stability and
        # gradient preservation (CR-008 fix)
        # For positives: use (1-p)^gamma_pos - down-weight easy positives
        # For negatives: use p_clipped^gamma_neg - down-weight easy negatives (with clipping)
        # Using log-space: exp(gamma * log(x)) = x^gamma
        # This avoids pow(0, gamma) numerically while maintaining gradients everywhere

        # log(1-p) for positive focal weight
        log_one_minus_probs = torch.log((1 - probs).clamp(min=1e-8))
        # log(p_clipped) for negative focal weight
        log_probs_neg = torch.log(probs_neg.clamp(min=1e-8))

        # Clamp the exponent to prevent overflow: exp(88) ≈ 2e38 (near float32 max)
        MAX_EXP = 88.0
        pos_exp = torch.clamp(self.gamma_pos * log_one_minus_probs, min=-MAX_EXP, max=MAX_EXP)
        neg_exp = torch.clamp(self.gamma_neg * log_probs_neg, min=-MAX_EXP, max=MAX_EXP)
        pos_weights = targets * torch.exp(pos_exp)
        neg_weights = (1 - targets) * torch.exp(neg_exp)

        # Apply focal weights with separate positive/negative weighting
        pos_loss = pos_weights * bce_loss
        neg_loss = neg_weights * bce_loss
        focal_loss = self.alpha * pos_loss + (1 - self.alpha) * neg_loss

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

        Raises:
            ValueError: If required inputs are None or invalid type.
        """
        # Validate required inputs
        if tag_logits is None:
            raise ValueError(
                "tag_logits cannot be None. "
                "This indicates a model output issue - check forward() implementation."
            )
        if tag_targets is None:
            raise ValueError(
                "tag_targets cannot be None. "
                "This indicates a dataloader issue - check batch preparation."
            )
        if rating_logits is None:
            raise ValueError(
                "rating_logits cannot be None. "
                "This indicates a model output issue - check forward() implementation."
            )
        if rating_targets is None:
            raise ValueError(
                "rating_targets cannot be None. "
                "This indicates a dataloader issue - check batch preparation."
            )

        # Validate types
        if not isinstance(tag_logits, torch.Tensor):
            raise TypeError(f"tag_logits must be torch.Tensor, got {type(tag_logits)}")
        if not isinstance(tag_targets, torch.Tensor):
            raise TypeError(f"tag_targets must be torch.Tensor, got {type(tag_targets)}")
        if not isinstance(rating_logits, torch.Tensor):
            raise TypeError(f"rating_logits must be torch.Tensor, got {type(rating_logits)}")
        if not isinstance(rating_targets, torch.Tensor):
            raise TypeError(f"rating_targets must be torch.Tensor, got {type(rating_targets)}")

        tag_loss = self.tag_loss_fn(tag_logits, tag_targets, sample_weights)
        # Compute rating loss
        if isinstance(self.rating_loss_fn, AsymmetricFocalLoss):
            rating_loss = self.rating_loss_fn(rating_logits, rating_targets)
        else:
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

    Note: Weights are computed during __init__. For large vocabularies (10K+ tags),
    avoid creating multiple instances with the same frequencies. Instead, create
    the sampler once and reuse it, or compute the weights once and reuse them.
    This design prioritizes simplicity over caching complexity.
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
