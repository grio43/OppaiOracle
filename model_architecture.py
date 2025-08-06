#!/usr/bin/env python3
"""
Model Architecture for Anime Image Tagger
Vision Transformer with hierarchical tag prediction (1B-3B parameters)
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_, constant_, xavier_uniform_
import torch.utils.checkpoint

logger = logging.getLogger(__name__)


@dataclass
class VisionTransformerConfig:
    """Configuration for Vision Transformer model"""
    # Model architecture
    architecture_type: str = "vit_large_extended"
    hidden_size: int = 1536  # 1B model: 1536, 3B model: 2304
    num_hidden_layers: int = 28  # 1B model: 28, 3B model: 40
    num_attention_heads: int = 24  # 1B model: 24, 3B model: 36
    intermediate_size: int = 6144  # 4x hidden_size
    
    # Vision specific
    image_size: int = 640
    patch_size: int = 14  # Results in 45x45 = 2025 patches for 640x640
    num_channels: int = 3
    qkv_bias: bool = True
    
    # Special tokens for anime understanding
    use_cls_token: bool = True
    use_style_token: bool = True  # For art style
    use_line_token: bool = True   # For line art quality
    use_color_token: bool = True  # For color palette
    num_special_tokens: int = 4
    
    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    drop_path_rate: float = 0.1  # Stochastic depth
    
    # Initialization
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    
    # Attention mechanisms
    use_flash_attention: bool = False  # Use Flash Attention if available
    attention_bias: bool = True
    attention_type: str = "standard"  # standard, efficient, flash
    
    # Tag prediction
    num_labels: int = 200000  # Total number of tags
    num_groups: int = 20  # Hierarchical groups
    tags_per_group: int = 10000
    
    # Efficiency settings
    gradient_checkpointing: bool = False
    use_amp: bool = True
    
    # Distillation
    use_distillation: bool = True
    distillation_alpha: float = 0.7
    distillation_temperature: float = 3.0


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    """Multi-head self-attention with optional Flash Attention"""
    
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(
            config.hidden_size,
            self.all_head_size * 3,
            bias=config.qkv_bias
        )
        
        # Output projection
        self.proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Flash Attention setup if available
        self.use_flash = config.use_flash_attention
        if self.use_flash:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                logger.warning("Flash Attention not available, using standard attention")
                self.use_flash = False
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # QKV transformation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        
        if self.use_flash and mask is None:
            # Use Flash Attention
            x = self.flash_attn_func(q, k, v, dropout_p=self.config.attention_probs_dropout_prob if self.training else 0.0)
            x = x.reshape(B, N, -1)
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation"""
    
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization"""
    
    def __init__(self, config: VisionTransformerConfig, drop_path_prob: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Attention(config)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.img_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        x = self.proj(x)  # B, hidden_size, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, hidden_size
        return x


class AnimeVisionTransformer(nn.Module):
    """Vision Transformer backbone for anime image understanding"""
    
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches
        
        # Special tokens for anime understanding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.style_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_style_token else None
        self.line_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_line_token else None
        self.color_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_color_token else None
        
        # Position embeddings
        total_tokens = num_patches + config.num_special_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, config.hidden_size))
        self.pos_drop = nn.Dropout(p=config.hidden_dropout_prob)
        
        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(config, drop_path_prob=dpr[i])
            for i in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=config.initializer_range)
        trunc_normal_(self.cls_token, std=config.initializer_range)
        if self.style_token is not None:
            trunc_normal_(self.style_token, std=config.initializer_range)
        if self.line_token is not None:
            trunc_normal_(self.line_token, std=config.initializer_range)
        if self.color_token is not None:
            trunc_normal_(self.color_token, std=config.initializer_range)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.config.initializer_range)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_(m.bias, 0)
            constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # B, num_patches, hidden_size
        
        # Add special tokens
        tokens = [self.cls_token.expand(B, -1, -1)]
        if self.config.use_style_token:
            tokens.append(self.style_token.expand(B, -1, -1))
        if self.config.use_line_token:
            tokens.append(self.line_token.expand(B, -1, -1))
        if self.config.use_color_token:
            tokens.append(self.color_token.expand(B, -1, -1))
        
        x = torch.cat(tokens + [x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        if self.config.gradient_checkpointing and self.training:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x)
        else:
            for block in self.blocks:
                x = block(x)
        
        x = self.norm(x)
        
        # Extract different token representations
        outputs = {
            'last_hidden_state': x,
            'cls_token': x[:, 0],
            'patch_tokens': x[:, self.config.num_special_tokens:]
        }
        
        # Extract special tokens if present
        token_idx = 1
        if self.config.use_style_token:
            outputs['style_token'] = x[:, token_idx]
            token_idx += 1
        if self.config.use_line_token:
            outputs['line_token'] = x[:, token_idx]
            token_idx += 1
        if self.config.use_color_token:
            outputs['color_token'] = x[:, token_idx]
            token_idx += 1
        
        return outputs
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_features(x)


class HierarchicalTagHead(nn.Module):
    """Hierarchical prediction head for 200k tags"""
    
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        self.num_groups = config.num_groups
        self.tags_per_group = config.tags_per_group
        
        # Shared MLP for feature extraction
        self.shared_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Group-specific heads
        self.group_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.tags_per_group)
            for _ in range(config.num_groups)
        ])
        
        # Group gating mechanism (to determine which groups are relevant)
        self.group_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.num_groups)
        )
        
        # Style, line, and color specific heads if tokens are used
        if config.use_style_token:
            self.style_head = nn.Linear(config.hidden_size, 100)  # 100 style tags
        if config.use_line_token:
            self.line_head = nn.Linear(config.hidden_size, 50)   # 50 line quality tags
        if config.use_color_token:
            self.color_head = nn.Linear(config.hidden_size, 50)   # 50 color palette tags
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cls_token = features['cls_token']
        B = cls_token.shape[0]
        
        # Shared feature extraction
        shared_features = self.shared_mlp(cls_token)
        
        # Group gating scores
        group_scores = self.group_gate(shared_features)
        group_weights = torch.sigmoid(group_scores)  # B, num_groups
        
        # Compute predictions for each group
        group_logits = []
        for i, head in enumerate(self.group_heads):
            group_logit = head(shared_features)  # B, tags_per_group
            # Apply group gating
            group_logit = group_logit * group_weights[:, i:i+1]
            group_logits.append(group_logit)
        
        # Stack into hierarchical format
        hierarchical_logits = torch.stack(group_logits, dim=1)  # B, num_groups, tags_per_group
        
        outputs = {
            'logits': hierarchical_logits,
            'group_weights': group_weights
        }
        
        # Add special token predictions if available
        if self.config.use_style_token and 'style_token' in features:
            outputs['style_logits'] = self.style_head(features['style_token'])
        if self.config.use_line_token and 'line_token' in features:
            outputs['line_logits'] = self.line_head(features['line_token'])
        if self.config.use_color_token and 'color_token' in features:
            outputs['color_logits'] = self.color_head(features['color_token'])
        
        return outputs


class AnimeTransformerWithHead(nn.Module):
    """Complete model with transformer backbone and hierarchical head"""
    
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        self.transformer = AnimeVisionTransformer(config)
        self.tag_head = HierarchicalTagHead(config)
        
        # For compatibility with some training scripts
        self.num_labels = config.num_labels
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        
        # Extract features
        features = self.transformer(pixel_values)
        
        # Get predictions
        outputs = self.tag_head(features)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = self.compute_loss(outputs['logits'], labels)
            outputs['loss'] = loss
        
        if not return_dict:
            return (outputs['logits'], loss) if loss is not None else (outputs['logits'],)
        
        return outputs
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25
    ) -> torch.Tensor:
        """Compute focal loss for multi-label classification"""
        # Flatten hierarchical predictions
        if logits.dim() == 3:
            B, G, T = logits.shape
            logits = logits.view(B, -1)
        
        # Ensure labels have same shape
        if labels.dim() == 3:
            labels = labels.view(labels.shape[0], -1)
        
        # Binary cross entropy with focal loss
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        
        # Focal loss modulation
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** focal_gamma
        
        # Alpha weighting (optional)
        alpha_t = focal_alpha * labels + (1 - focal_alpha) * (1 - labels)
        
        loss = alpha_t * focal_weight * bce
        
        return loss.mean()
    
    def get_config(self) -> VisionTransformerConfig:
        """Get model configuration"""
        return self.config


class DistillationHead(nn.Module):
    """Distillation head for learning from dual teachers"""
    
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        
        # Projection for anime teacher alignment (70k tags)
        self.anime_teacher_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 70000)  # Anime teacher output size
        )
        
        # Projection for CLIP alignment
        self.clip_proj = nn.Sequential(
            nn.Linear(config.hidden_size, 768),  # CLIP embedding size
            nn.LayerNorm(768)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cls_token = features['cls_token']
        
        outputs = {
            'anime_teacher_logits': self.anime_teacher_proj(cls_token),
            'clip_features': self.clip_proj(cls_token)
        }
        
        return outputs


class AnimeTransformerForDistillation(nn.Module):
    """Model variant for distillation training"""
    
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        self.transformer = AnimeVisionTransformer(config)
        self.tag_head = HierarchicalTagHead(config)
        self.distillation_head = DistillationHead(config)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        
        # Extract features
        features = self.transformer(pixel_values)
        
        # Main predictions
        tag_outputs = self.tag_head(features)
        
        # Distillation predictions
        distill_outputs = self.distillation_head(features)
        
        outputs = {
            'logits': tag_outputs['logits'],
            'group_weights': tag_outputs.get('group_weights'),
            'anime_teacher_logits': distill_outputs['anime_teacher_logits'],
            'clip_features': distill_outputs['clip_features']
        }
        
        # Calculate losses if labels/teachers provided
        total_loss = 0
        losses = {}
        
        if labels is not None:
            tag_loss = self.compute_loss(tag_outputs['logits'], labels)
            losses['tag_loss'] = tag_loss
            total_loss = total_loss + tag_loss
        
        if teacher_logits is not None:
            distill_loss = self.compute_distillation_loss(
                distill_outputs['anime_teacher_logits'],
                teacher_logits,
                temperature=self.config.distillation_temperature
            )
            losses['distill_loss'] = distill_loss
            total_loss = total_loss + self.config.distillation_alpha * distill_loss
        
        if teacher_features is not None:
            feature_loss = F.mse_loss(distill_outputs['clip_features'], teacher_features)
            losses['feature_loss'] = feature_loss
            total_loss = total_loss + (1 - self.config.distillation_alpha) * feature_loss
        
        if total_loss != 0:
            outputs['loss'] = total_loss
            outputs['losses'] = losses
        
        return outputs
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss"""
        if logits.dim() == 3:
            B, G, T = logits.shape
            logits = logits.view(B, -1)
        if labels.dim() == 3:
            labels = labels.view(labels.shape[0], -1)
        
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 3.0
    ) -> torch.Tensor:
        """Compute distillation loss with temperature scaling"""
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        loss = loss * (temperature ** 2)  # Scale by T^2 as per distillation paper
        
        return loss


def create_model(
    config: Optional[Union[VisionTransformerConfig, Dict]] = None,
    pretrained: Optional[str] = None,
    **kwargs
) -> AnimeTransformerWithHead:
    """
    Create model with optional pretrained weights
    
    Args:
        config: Model configuration or dict
        pretrained: Path to pretrained weights
        **kwargs: Override config parameters
        
    Returns:
        Model instance
    """
    # Handle config
    if config is None:
        config = VisionTransformerConfig(**kwargs)
    elif isinstance(config, dict):
        config = VisionTransformerConfig(**{**config, **kwargs})
    else:
        # Update existing config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create model
    model = AnimeTransformerWithHead(config)
    
    # Load pretrained weights if provided
    if pretrained:
        logger.info(f"Loading pretrained weights from {pretrained}")
        checkpoint = torch.load(pretrained, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DDP weights
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        logger.info("Pretrained weights loaded")
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by component
    components = {}
    for name, module in model.named_children():
        components[name] = sum(p.numel() for p in module.parameters())
    
    return {
        'total': total,
        'trainable': trainable,
        'components': components,
        'total_mb': total * 4 / 1024 / 1024,  # Assuming float32
        'total_gb': total * 4 / 1024 / 1024 / 1024
    }


def get_model_configs() -> Dict[str, VisionTransformerConfig]:
    """Get predefined model configurations"""
    configs = {
        '1B': VisionTransformerConfig(
            hidden_size=1536,
            num_hidden_layers=28,
            num_attention_heads=24,
            intermediate_size=6144
        ),
        '1.5B': VisionTransformerConfig(
            hidden_size=1792,
            num_hidden_layers=32,
            num_attention_heads=28,
            intermediate_size=7168
        ),
        '2B': VisionTransformerConfig(
            hidden_size=2048,
            num_hidden_layers=36,
            num_attention_heads=32,
            intermediate_size=8192
        ),
        '3B': VisionTransformerConfig(
            hidden_size=2304,
            num_hidden_layers=40,
            num_attention_heads=36,
            intermediate_size=9216
        )
    }
    
    return configs


if __name__ == "__main__":
    # Test model creation and parameter counting
    print("Testing model architectures...")
    
    configs = get_model_configs()
    
    for name, config in configs.items():
        print(f"\n{name} Model:")
        print("-" * 50)
        
        model = create_model(config)
        params = count_parameters(model)
        
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"Model size: {params['total_gb']:.2f} GB")
        print(f"Components:")
        for comp_name, comp_params in params['components'].items():
            print(f"  {comp_name}: {comp_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            outputs = model(dummy_input)
            print(f"Output shape: {outputs['logits'].shape}")
            print(f"Expected shape: (2, {config.num_groups}, {config.tags_per_group})")
    
    print("\n✓ All models created successfully!")