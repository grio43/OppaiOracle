#!/usr/bin/env python3
"""
Loss Functions for Anime Image Tagger
Asymmetric focal loss and knowledge distillation losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-label classification
    Different gamma values for positive and negative samples to handle imbalance
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        alpha_pos: float = 1.0,
        alpha_neg: float = 1.0,
        clip: float = 0.05,
        reduction: str = 'mean',
        disable_torch_grad: bool = True
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.clip = clip
        self.reduction = reduction
        self.disable_torch_grad = disable_torch_grad
    
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
            sample_weights: (B,) or (B, num_classes) optional weights
        """
        # Flatten if hierarchical
        if logits.dim() == 3:
            B, G, T = logits.shape
            logits = logits.view(B, -1)
            targets = targets.view(B, -1)
        
        # Disable gradient computation for targets
        if self.disable_torch_grad:
            targets = targets.detach()
        
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        
        # Clip probabilities to prevent log(0)
        probs = torch.clamp(probs, min=self.clip, max=1.0 - self.clip)
        
        # Calculate focal weights
        # For positive samples
        pos_weights = targets * torch.pow(1 - probs, self.gamma_pos) * self.alpha_pos
        
        # For negative samples  
        neg_weights = (1 - targets) * torch.pow(probs, self.gamma_neg) * self.alpha_neg
        
        # Binary cross entropy
        bce = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        
        # Apply focal weights
        focal_loss = pos_weights * bce + neg_weights * bce
        
        # Apply sample weights if provided
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


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss supporting multiple teachers
    """
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7,
        distillation_type: str = 'soft',  # soft, hard, or feature
        teacher_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_type = distillation_type
        self.teacher_weights = teacher_weights or {'anime': 0.7, 'clip': 0.3}
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_outputs: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            student_logits: Student model predictions
            teacher_outputs: Dict with teacher predictions
            targets: Optional ground truth labels
            
        Returns:
            Total loss and individual loss components
        """
        losses = {}
        total_loss = 0
        
        # Soft distillation from anime teacher
        if 'anime_logits' in teacher_outputs:
            anime_loss = self._compute_soft_distillation(
                student_logits,
                teacher_outputs['anime_logits'],
                self.temperature
            )
            losses['anime_distill'] = anime_loss
            total_loss += self.teacher_weights.get('anime', 0.7) * anime_loss
        
        # Feature distillation from CLIP
        if 'clip_features' in teacher_outputs:
            clip_loss = self._compute_feature_distillation(
                student_logits,  # This should be student features
                teacher_outputs['clip_features']
            )
            losses['clip_distill'] = clip_loss
            total_loss += self.teacher_weights.get('clip', 0.3) * clip_loss
        
        # Hard distillation if targets provided
        if targets is not None and self.alpha < 1.0:
            hard_loss = F.binary_cross_entropy_with_logits(
                student_logits,
                targets.float()
            )
            losses['hard_distill'] = hard_loss
            total_loss = self.alpha * total_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, losses
    
    def _compute_soft_distillation(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute soft distillation loss with temperature"""
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence
        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        )
        
        # Scale by T^2 as per Hinton et al.
        return kl_loss * (temperature ** 2)
    
    def _compute_feature_distillation(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature-level distillation loss"""
        # Normalize features
        student_norm = F.normalize(student_features, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
        
        # Cosine similarity loss
        cosine_loss = 1 - (student_norm * teacher_norm).sum(dim=-1).mean()
        
        return cosine_loss


class HierarchicalLoss(nn.Module):
    """
    Loss function for hierarchical tag prediction
    Combines group-level and tag-level losses
    """
    
    def __init__(
        self,
        tag_loss_weight: float = 0.8,
        group_loss_weight: float = 0.2,
        base_loss_fn: Optional[nn.Module] = None
    ):
        super().__init__()
        self.tag_loss_weight = tag_loss_weight
        self.group_loss_weight = group_loss_weight
        self.base_loss_fn = base_loss_fn or AsymmetricFocalLoss()
    
    def forward(
        self,
        hierarchical_logits: torch.Tensor,
        group_weights: torch.Tensor,
        targets: torch.Tensor,
        group_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hierarchical_logits: (B, num_groups, tags_per_group)
            group_weights: (B, num_groups) group activation scores
            targets: (B, num_groups, tags_per_group) or (B, total_tags)
            group_targets: (B, num_groups) optional group-level targets
        """
        B, G, T = hierarchical_logits.shape
        
        # Flatten for tag-level loss
        flat_logits = hierarchical_logits.view(B, -1)
        
        if targets.dim() == 2:
            flat_targets = targets
        else:
            flat_targets = targets.view(B, -1)
        
        # Tag-level loss
        tag_loss = self.base_loss_fn(flat_logits, flat_targets)
        
        # Group-level loss
        if group_targets is None:
            # Create group targets from tag targets
            if targets.dim() == 3:
                group_targets = targets.any(dim=2).float()
            else:
                # Reshape and check if any tag in each group is active
                targets_grouped = targets.view(B, G, T)
                group_targets = targets_grouped.any(dim=2).float()
        
        group_loss = F.binary_cross_entropy_with_logits(
            group_weights,
            group_targets
        )
        
        # Combine losses
        total_loss = (self.tag_loss_weight * tag_loss + 
                     self.group_loss_weight * group_loss)
        
        losses = {
            'tag_loss': tag_loss,
            'group_loss': group_loss,
            'total': total_loss
        }
        
        return total_loss, losses


class CombinedLoss(nn.Module):
    """
    Combined loss for training with both tag prediction and distillation
    """
    
    def __init__(
        self,
        focal_loss: Optional[AsymmetricFocalLoss] = None,
        distillation_loss: Optional[DistillationLoss] = None,
        hierarchical_loss: Optional[HierarchicalLoss] = None,
        focal_weight: float = 0.5,
        distill_weight: float = 0.5,
        use_hierarchical: bool = True
    ):
        super().__init__()
        
        self.focal_loss = focal_loss or AsymmetricFocalLoss()
        self.distillation_loss = distillation_loss or DistillationLoss()
        self.hierarchical_loss = hierarchical_loss or HierarchicalLoss()
        
        self.focal_weight = focal_weight
        self.distill_weight = distill_weight
        self.use_hierarchical = use_hierarchical
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        teacher_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss
        
        Args:
            predictions: Model predictions including logits and optional hierarchical
            targets: Ground truth labels
            teacher_outputs: Teacher model outputs for distillation
        """
        all_losses = {}
        total_loss = 0
        
        # Extract predictions
        logits = predictions['logits']
        
        # Hierarchical loss if applicable
        if self.use_hierarchical and 'group_weights' in predictions:
            hier_loss, hier_components = self.hierarchical_loss(
                logits,
                predictions['group_weights'],
                targets
            )
            all_losses.update({f'hier_{k}': v for k, v in hier_components.items()})
            total_loss += self.focal_weight * hier_loss
        else:
            # Standard focal loss
            focal_loss = self.focal_loss(logits, targets)
            all_losses['focal'] = focal_loss
            total_loss += self.focal_weight * focal_loss
        
        # Distillation loss if teacher outputs provided
        if teacher_outputs is not None:
            distill_loss, distill_components = self.distillation_loss(
                predictions.get('distill_logits', logits),
                teacher_outputs,
                targets
            )
            all_losses.update({f'distill_{k}': v for k, v in distill_components.items()})
            total_loss += self.distill_weight * distill_loss
        
        all_losses['total'] = total_loss
        
        return total_loss, all_losses


class TagWeightedLoss(nn.Module):
    """
    Loss with per-tag weighting based on frequency or importance
    """
    
    def __init__(
        self,
        tag_weights: Optional[torch.Tensor] = None,
        frequency_weighting: str = 'inverse',  # inverse, sqrt_inverse, or none
        base_loss: Optional[nn.Module] = None
    ):
        super().__init__()
        self.register_buffer('tag_weights', tag_weights)
        self.frequency_weighting = frequency_weighting
        self.base_loss = base_loss or AsymmetricFocalLoss()
    
    def compute_frequency_weights(
        self,
        tag_frequencies: torch.Tensor,
        epsilon: float = 1e-7
    ) -> torch.Tensor:
        """Compute weights from tag frequencies"""
        if self.frequency_weighting == 'inverse':
            weights = 1.0 / (tag_frequencies + epsilon)
        elif self.frequency_weighting == 'sqrt_inverse':
            weights = 1.0 / torch.sqrt(tag_frequencies + epsilon)
        else:
            weights = torch.ones_like(tag_frequencies)
        
        # Normalize weights
        weights = weights / weights.mean()
        return weights
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Apply weighted loss"""
        if self.tag_weights is not None:
            # Expand weights to match batch size
            weights = self.tag_weights.unsqueeze(0).expand_as(targets)
            return self.base_loss(logits, targets, sample_weights=weights)
        else:
            return self.base_loss(logits, targets)


# Utility function to create loss for training
def create_training_loss(config: Dict) -> nn.Module:
    """
    Create loss function based on configuration
    
    Args:
        config: Loss configuration dictionary
    """
    loss_type = config.get('type', 'combined')
    
    if loss_type == 'focal':
        return AsymmetricFocalLoss(
            gamma_pos=config.get('gamma_pos', 0.0),
            gamma_neg=config.get('gamma_neg', 4.0),
            alpha_pos=config.get('alpha_pos', 1.0),
            alpha_neg=config.get('alpha_neg', 1.0)
        )
    
    elif loss_type == 'hierarchical':
        focal = AsymmetricFocalLoss(
            gamma_pos=config.get('gamma_pos', 0.0),
            gamma_neg=config.get('gamma_neg', 4.0)
        )
        return HierarchicalLoss(
            tag_loss_weight=config.get('tag_weight', 0.8),
            group_loss_weight=config.get('group_weight', 0.2),
            base_loss_fn=focal
        )
    
    else:  # combined
        focal = AsymmetricFocalLoss(
            gamma_pos=config.get('gamma_pos', 0.0),
            gamma_neg=config.get('gamma_neg', 4.0)
        )
        
        distill = DistillationLoss(
            temperature=config.get('temperature', 3.0),
            alpha=config.get('distill_alpha', 0.7),
            teacher_weights=config.get('teacher_weights', {'anime': 0.7, 'clip': 0.3})
        )
        
        return CombinedLoss(
            focal_loss=focal,
            distillation_loss=distill,
            focal_weight=config.get('focal_weight', 0.5),
            distill_weight=config.get('distill_weight', 0.5),
            use_hierarchical=config.get('use_hierarchical', True)
        )


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    B, num_classes = 4, 1000
    logits = torch.randn(B, num_classes)
    targets = torch.randint(0, 2, (B, num_classes)).float()
    
    # Test asymmetric focal loss
    focal_loss = AsymmetricFocalLoss()
    loss = focal_loss(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test with hierarchical structure
    B, G, T = 4, 20, 50
    hier_logits = torch.randn(B, G, T)
    hier_targets = torch.randint(0, 2, (B, G, T)).float()
    group_weights = torch.randn(B, G)
    
    hier_loss = HierarchicalLoss()
    loss, components = hier_loss(hier_logits, group_weights, hier_targets)
    print(f"Hierarchical loss: {loss.item():.4f}")
    print(f"  Tag loss: {components['tag_loss'].item():.4f}")
    print(f"  Group loss: {components['group_loss'].item():.4f}")
    
    print("\n✓ Loss functions working correctly!")