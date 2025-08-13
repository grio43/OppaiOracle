#!/usr/bin/env python3
"""
Model Architecture for Anime Image Tagger - Direct Training (modified)
Vision Transformer adjusted for orientation-aware tagger
"""

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionTransformerConfig:
    """Configuration for the Vision Transformer used in direct training."""
    image_size: int = 640
    # Ensure the patch size divides the image size evenly (e.g. 16) to avoid losing border information
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 1280
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 5120
    num_tags: int = 100000
    num_ratings: int = 5
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    use_flash_attention: bool = True
    # Enable gradient checkpointing by default to reduce memory usage
    gradient_checkpointing: bool = True


class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class SimplifiedTagger(nn.Module):
    """Vision Transformer based tagger for anime images."""
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        # Position embeddings
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, config.hidden_size)
        )
        self.pos_drop = nn.Dropout(p=config.dropout)
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Classification heads
        self.tag_head = nn.Linear(config.hidden_size, config.num_tags)
        self.rating_head = nn.Linear(config.hidden_size, config.num_ratings)
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = pixel_values.shape[0]
        # Patch embedding
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Add position embeddings
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        # Transformer blocks with optional gradient checkpointing
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        # Final norm
        x = self.norm(x)
        # Use CLS token for classification
        cls_output = x[:, 0]
        # Predictions
        tag_logits = self.tag_head(cls_output)
        rating_logits = self.rating_head(cls_output)
        return {
            'tag_logits': tag_logits,
            'rating_logits': rating_logits,
            'logits': tag_logits
        }


def create_model(**kwargs):
    """Create model from configuration arguments."""
    config = VisionTransformerConfig(**kwargs)
    return SimplifiedTagger(config)