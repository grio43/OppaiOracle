#!/usr/bin/env python3
"""
Model Architecture for Anime Image Tagger - Direct Training (modified)
Vision Transformer adjusted for orientation-aware tagger
"""

import math
import warnings
from dataclasses import dataclass, fields
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mask_utils import ensure_pixel_padding_mask, pixel_to_token_ignore
from custom_drop_path import SafeDropPath


class LayerNormFp32(nn.LayerNorm):
    """
    LayerNorm that casts to float32 before calling the original LayerNorm.
    This is to improve stability when using mixed precision training.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)

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
    num_tags: int = 100000  # This should be overridden with actual vocab size
    num_ratings: int = 5
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    use_flash_attention: bool = True
    # Token ignore threshold: a token is ignored if >= this fraction of its pixels are PAD
    token_ignore_threshold: float = 0.9
    # Enable gradient checkpointing by default to reduce memory usage
    gradient_checkpointing: bool = True
    drop_path_rate: float = 0.0

    def __post_init__(self):
        assert 0.0 <= self.drop_path_rate < 1.0, (
            f"drop_path_rate must be in [0, 1), got {self.drop_path_rate}"
        )
        if self.drop_path_rate > 0.5:
            warnings.warn(
                f"drop_path_rate={self.drop_path_rate} is unusually high; "
                "values between 0.08 and 0.3 are typical for ViT."
            )


class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, config: VisionTransformerConfig, drop_path: float = 0.):
        super().__init__()
        self.config = config
        self.norm1 = LayerNormFp32(config.hidden_size, eps=config.layer_norm_eps)

        # Use flash attention if available and requested
        self.use_flash = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        if self.use_flash:
            # For flash attention, we need separate projection layers
            self.num_heads = config.num_attention_heads
            self.head_dim = config.hidden_size // config.num_attention_heads
            self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
            self.proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.attn_dropout = config.attention_dropout
        else:
            # Standard MultiheadAttention
            self.attn = nn.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_dropout,
                batch_first=True
            )

        self.drop_path = SafeDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = LayerNormFp32(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        normed_x = self.norm1(x)
        
        if self.use_flash:
            # Flash attention path
            B, L, D = normed_x.shape
            qkv = self.qkv(normed_x).reshape(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2)  # (B, num_heads, L, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Use scaled_dot_product_attention with flash backend when available
            attn_out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=key_padding_mask.unsqueeze(1).unsqueeze(2) if key_padding_mask is not None else None,
                dropout_p=self.attn_dropout if self.training else 0.0
            )
            attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
            x = x + self.drop_path(self.proj(attn_out))
        else:
            # Standard attention path
            attn_output, _ = self.attn(
                normed_x, normed_x, normed_x,
               key_padding_mask=key_padding_mask
            )
            x = x + self.drop_path(attn_output)
      
        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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
        rate = getattr(config, 'drop_path_rate', 0.0)
        try:
            dpr = torch.linspace(0.0, rate, config.num_hidden_layers, endpoint=False)
        except TypeError:
            dpr = torch.linspace(0.0, rate, config.num_hidden_layers + 1)[:-1]
        dpr = dpr.clamp_max(1.0 - 1e-6).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(config, drop_path=float(p))
            for p in dpr
        ])
        # Final layer norm
        self.norm = LayerNormFp32(config.hidden_size, eps=config.layer_norm_eps)
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
        elif isinstance(module, (nn.LayerNorm, LayerNormFp32)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,  # (B,H,W) or (B,1,H,W), auto-detected semantics
    ) -> Dict[str, torch.Tensor]:
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
        # Build key-padding mask (CLS + patch tokens) from pixel-level mask.
        # Semantics: True=PAD at pixel-level -> pooled to True=IGNORE at token-level.
        attn_kpm: Optional[torch.Tensor] = None
        if padding_mask is not None:
            pm = ensure_pixel_padding_mask(padding_mask)          # -> (B,1,H,W) bool, True=PAD
            thr = getattr(self.config, "token_ignore_threshold", 0.9)
            token_ignore = pixel_to_token_ignore(pm, patch=self.config.patch_size, threshold=thr)  # (B,Lp)
            # Prepend CLS token (never ignored)
            cls_keep = torch.zeros(B, 1, dtype=torch.bool, device=token_ignore.device)
            attn_kpm = torch.cat([cls_keep, token_ignore], dim=1) # (B, 1+Lp) True=IGNORE

        # Transformer blocks with optional gradient checkpointing
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                # use_reentrant=True is more robust with AMP, though it uses more memory.
                # This is to fix the CheckpointError where recomputed values have different metadata.
                if attn_kpm is not None:
                    # checkpoint requires tensor args; pass mask explicitly
                    x = torch.utils.checkpoint.checkpoint(
                        lambda _x, _m: block(_x, key_padding_mask=_m), x, attn_kpm,
                        use_reentrant=True
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=True)
            else:
                x = block(x, key_padding_mask=attn_kpm)
        # Final norm
        x = self.norm(x)
        # Use CLS token for classification
        cls_output = x[:, 0]
        # Predictions
        tag_logits = self.tag_head(cls_output)
        rating_logits = self.rating_head(cls_output)

        # Clamp logits to prevent numerical instability with mixed precision.
        # This is the PRIMARY FIX for the non-finite error.
        tag_logits = torch.clamp(tag_logits, min=-15.0, max=15.0)
        rating_logits = torch.clamp(rating_logits, min=-15.0, max=15.0)

        return {
            'tag_logits': tag_logits,
            'rating_logits': rating_logits,
            'logits': tag_logits
        }

    # Convenience helpers for instrumentation / TensorBoard
    def forward_for_graph(self, pixel_values: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns a single Tensor (tag_logits) so tools like SummaryWriter.add_graph
        can trace the model without dealing with dict outputs.
        """
        out = self.forward(pixel_values, padding_mask=padding_mask)
        return out['tag_logits']

    def example_inputs(self, batch_size: int = 1) -> torch.Tensor:
        """Create a dummy pixel tensor on the correct device for graph tracing."""
        device = next(self.parameters()).device
        return torch.zeros(batch_size, 3, self.config.image_size, self.config.image_size, device=device)


def create_model(**kwargs):
    """Create model from configuration arguments."""
    # Get the names of the fields in the VisionTransformerConfig dataclass
    config_fields = {f.name for f in fields(VisionTransformerConfig)}

    # Filter kwargs to only include keys that are in the config_fields
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

    config = VisionTransformerConfig(**filtered_kwargs)
    return SimplifiedTagger(config)
