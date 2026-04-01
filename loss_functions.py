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
    # Key: (num_classes, device_type, device_index, tuple(sorted_ignore_indices)) -> mask tensor
    # Limited to _KEEP_MASK_CACHE_MAX_SIZE entries with LRU-style eviction
    _keep_mask_cache: Dict[Tuple[int, str, int, Tuple[int, ...]], torch.Tensor] = {}
    _KEEP_MASK_CACHE_MAX_SIZE: ClassVar[int] = 100

    @classmethod
    def clear_keep_mask_cache(cls) -> None:
        """Clear the keep mask cache. Useful for memory management or device changes."""
        cls._keep_mask_cache.clear()

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 3.0,
        alpha: float = 0.75,
        clip: float = 0.05,
        reduction: str = 'mean',
        label_smoothing: float = 0.05,
        ignore_indices: Optional[Union[int, List[int]]] = 0,
        class_weights: Optional[List[float]] = None,
    ):
        """
        Initialize the Asymmetric Focal Loss.

        Args:
            gamma_pos: Focusing parameter for positive examples. Higher values
                down-weight easy positives more aggressively. Default: 1.0.
            gamma_neg: Focusing parameter for negative examples. Higher values
                down-weight easy negatives more aggressively. Default: 3.0.
            alpha: Balance factor between positive and negative loss contributions.
                Controls the relative importance of positive vs negative samples
                in the final loss calculation.

                The loss is computed as:
                    loss = alpha * pos_loss + (1 - alpha) * neg_loss

                Value interpretation:
                - alpha = 0.5: Equal weighting of positive and negative contributions.
                - alpha > 0.5: Favors positives (increases penalty for missing true
                  labels). Use when false negatives are more costly.
                - alpha < 0.5: Favors negatives (increases penalty for false alarms).
                  Use when false positives are more costly.

                For imbalanced multi-label classification (where most labels are
                negative for any given sample), recommended values are in the range
                0.25 to 0.75. Higher values (0.6-0.75) help the model learn rare
                positive labels by weighting their contribution more heavily.
                Default: 0.75 (favors positive contributions).

            clip: Probability margin for hard thresholding on negative samples.
                Shifts negative probabilities by this amount before computing
                focal weights, effectively reducing the contribution of easy
                negatives. Set to 0.0 to disable. Default: 0.05.
            reduction: Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'.
            label_smoothing: Amount of label smoothing to apply. Default: 0.05.
            ignore_indices: Class index or list of indices to exclude from loss
                computation. Default: 0 (ignores the PAD token at index 0).
            class_weights: Optional list of per-class weights for handling class
                imbalance. Applied after focal loss computation.
        """
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

        # Per-class weights for handling class imbalance (e.g., rating distribution)
        # Weights are applied after focal loss computation
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        # Cache for class_weights on specific devices to avoid repeated .to() calls
        # Key: (device_type, device_index) -> cached tensor on that device
        self._class_weights_device_cache: Dict[Tuple[str, int], torch.Tensor] = {}

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
        # Track the keep mask for reuse in class_weights filtering
        ignore_keep_mask = None
        if self.ignore_indices:
            c = logits.size(1)
            # Filter to valid indices only
            valid_ignore = tuple(sorted(idx for idx in self.ignore_indices if 0 <= idx < c))
            if valid_ignore:
                # Use cached mask to avoid creating new tensors per forward pass
                # Key: (num_classes, device_type, device_index, tuple(sorted_ignore_indices))
                # Include device type to avoid cache key collisions between CPU and CUDA tensors
                device_type = logits.device.type
                device_idx = logits.device.index if logits.device.index is not None else -1
                cache_key = (c, device_type, device_idx, valid_ignore)

                if cache_key not in AsymmetricFocalLoss._keep_mask_cache:
                    # Enforce cache size limit with LRU-style eviction (remove oldest entries)
                    if len(AsymmetricFocalLoss._keep_mask_cache) >= AsymmetricFocalLoss._KEEP_MASK_CACHE_MAX_SIZE:
                        # Remove the first (oldest) entry - dict preserves insertion order in Python 3.7+
                        oldest_key = next(iter(AsymmetricFocalLoss._keep_mask_cache))
                        del AsymmetricFocalLoss._keep_mask_cache[oldest_key]
                    keep = torch.ones(c, dtype=torch.bool, device=logits.device)
                    for idx in valid_ignore:
                        keep[idx] = False
                    AsymmetricFocalLoss._keep_mask_cache[cache_key] = keep

                # Mask is already cached per-device (device info is part of cache_key),
                # so no .to() call needed - avoids GPU transfer on every forward pass
                keep = AsymmetricFocalLoss._keep_mask_cache[cache_key]
                logits = logits[:, keep]
                targets = targets[:, keep]
                # Save for reuse in class_weights filtering
                ignore_keep_mask = keep

        # Save binary targets before smoothing for focal weight masks.
        # Focal gating must use clean {0,1} masks so absent tags contribute zero
        # to pos_weights and present tags contribute zero to neg_weights.
        # Smoothed targets go into BCE (where smoothing has its intended effect).
        targets_for_focal = targets

        # Apply label smoothing if specified
        # Compute in float32 for numerical stability, then convert to logits dtype in one step
        # (Avoids two separate dtype conversions: .float() then .to(dtype=logits.dtype))
        if self.label_smoothing > 0:
            targets = (targets.float() * (1 - self.label_smoothing) + self.label_smoothing / 2).to(dtype=logits.dtype)

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

        # Use 1e-6 as minimum to avoid gradient underflow with high gamma values.
        # With 1e-7 and gamma=3: log(1e-7)^3 ≈ (-16.1)^3 ≈ -4160, exp(-4160) underflows to 0.
        # Using 1e-6: log(1e-6)^3 ≈ (-13.8)^3 ≈ -2628, more stable while still preventing log(0).
        _LOG_FLOOR = 1e-6
        # log(1-p) for positive focal weight
        log_one_minus_probs = torch.log((1 - probs).clamp(min=_LOG_FLOOR))
        # log(p_clipped) for negative focal weight
        log_probs_neg = torch.log(probs_neg.clamp(min=_LOG_FLOOR))

        # Clamp the exponent to prevent overflow: exp(88) ≈ 2e38 (near float32 max)
        MAX_EXP = 88.0
        pos_exp = torch.clamp(self.gamma_pos * log_one_minus_probs, min=-MAX_EXP, max=MAX_EXP)
        neg_exp = torch.clamp(self.gamma_neg * log_probs_neg, min=-MAX_EXP, max=MAX_EXP)
        pos_weights = targets_for_focal * torch.exp(pos_exp)
        neg_weights = (1 - targets_for_focal) * torch.exp(neg_exp)

        # Apply focal weights with separate positive/negative weighting
        pos_loss = pos_weights * bce_loss
        neg_loss = neg_weights * bce_loss
        focal_loss = self.alpha * pos_loss + (1 - self.alpha) * neg_loss

        # Apply sample weights if provided (for frequency-based sampling)
        if sample_weights is not None:
            if sample_weights.dim() == 1:
                sample_weights = sample_weights.unsqueeze(1)
            focal_loss = focal_loss * sample_weights

        # Apply class weights if provided (for class imbalance)
        if self.class_weights is not None:
            # class_weights shape: (num_classes,) -> broadcast to (1, num_classes)
            # Use device-specific cache to avoid .to() transfer on every forward pass
            device_type = focal_loss.device.type
            device_idx = focal_loss.device.index if focal_loss.device.index is not None else -1
            device_key = (device_type, device_idx)

            if device_key not in self._class_weights_device_cache:
                # Cache the weights on this device (one-time transfer per device)
                self._class_weights_device_cache[device_key] = self.class_weights.to(focal_loss.device)

            weights = self._class_weights_device_cache[device_key]

            # Filter weights using the same keep mask from logits/targets filtering
            if ignore_keep_mask is not None:
                # Reuse the keep mask from ignore_indices filtering to ensure consistency
                # The mask may need resizing if class_weights has different size than logits
                if self.class_weights.size(0) == ignore_keep_mask.size(0):
                    weights = weights[ignore_keep_mask]
                else:
                    # If class_weights size differs from logits, we cannot safely map indices.
                    # Previous fallback logic was risky and could lead to shape mismatches.
                    raise ValueError(
                        f"Shape mismatch: class_weights size ({self.class_weights.size(0)}) "
                        f"must match logits original size ({ignore_keep_mask.size(0)})."
                    )
            focal_loss = focal_loss * weights.unsqueeze(0)

        # Reduction
        if self.reduction == 'mean':
            # When sample_weights are provided, focal_loss already contains loss * weight
            # from line 320. We take the regular mean to preserve relative weighting
            # without double-dividing by weights.
            # This gives: mean(loss * weight), not sum(loss * weight) / sum(weight)
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Loss for tag prediction (ratings are now part of the tag vocabulary).

    Wraps the asymmetric focal loss for multi-label tag classification.
    """

    def __init__(
        self,
        tag_loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.tag_loss_fn = tag_loss_fn or AsymmetricFocalLoss()

    def forward(
        self,
        tag_logits: torch.Tensor,
        tag_targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss for tags (including rating tags).

        Args:
            tag_logits: (B, num_tags) logits for tag prediction.
            tag_targets: (B, num_tags) binary targets for tags.
            sample_weights: Optional per-sample weights.
        """
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

        tag_loss = self.tag_loss_fn(tag_logits, tag_targets, sample_weights)
        losses = {
            'total': tag_loss,
            'tag_loss': tag_loss,
        }
        return tag_loss, losses
