#!/usr/bin/env python3
"""
Model Architecture for Anime Image Tagger - Direct Training (modified)
Vision Transformer adjusted for orientation-aware tagger
"""

import functools
import logging
import math
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from mask_utils import ensure_pixel_padding_mask, pixel_to_token_ignore
from custom_drop_path import SafeDropPath

# Import Flex Attention (PyTorch 2.5+)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

logger.info("Using PyTorch Flex Attention")


def _check_flex_attention_available() -> bool:
    """Check if Flex Attention is available (PyTorch 2.5+)."""
    return hasattr(torch.nn.attention, 'flex_attention')


def _check_triton_available() -> bool:
    """Check if Triton is available for compiled block mask creation."""
    try:
        import triton
        return True
    except ImportError:
        return False


# Cache the Triton availability check at module load time
_TRITON_AVAILABLE = _check_triton_available()


class LayerNormFp32(nn.LayerNorm):
    """
    LayerNorm that optionally casts to float32 before calling the original LayerNorm.
    This is to improve stability when using mixed precision training.

    Args:
        normalized_shape: Input shape from an expected input of size
        eps: A value added to the denominator for numerical stability
        elementwise_affine: A boolean value that when set to True, gives learnable parameters
        use_fp32: If True, cast to float32 before LayerNorm (better stability).
                  If False, use native dtype (faster but potentially less stable).
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, use_fp32=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.use_fp32 = use_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp32:
            return super().forward(x.float()).type_as(x)
        else:
            return super().forward(x)

@dataclass
class VisionTransformerConfig:
    """Configuration for the Vision Transformer used in direct training."""
    image_size: int = 512
    # Patch size must divide image_size evenly to avoid losing border information (validated in __post_init__)
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
    use_flex_attention: bool = True  # Use Flex Attention (PyTorch 2.5+)
    flex_block_size: int = 128  # Block size for Flex Attention sparse computation
    attention_bias: bool = True  # Use bias in attention QKV and projection layers
    # Token ignore threshold: a token is ignored if >= this fraction of its pixels are PAD
    token_ignore_threshold: float = 0.9
    # Enable gradient checkpointing by default to reduce memory usage
    gradient_checkpointing: bool = True
    drop_path_rate: float = 0.0
    # Enable numerical stability checking (for debugging only)
    check_numerical_stability: bool = False
    # Logit clamping for numerical stability (None to disable)
    # exp(15) ≈ 3.3M which is safe for softmax in float16/bfloat16
    logit_clamp_value: Optional[float] = 15.0
    # Precision configuration
    use_fp32_layernorm: bool = False  # Use FP32 for LayerNorm (better stability, slight speed cost). Set to False for full bfloat16.

    def __post_init__(self):
        # Validate drop_path_rate
        assert 0.0 <= self.drop_path_rate < 1.0, (
            f"drop_path_rate must be in [0, 1), got {self.drop_path_rate}"
        )
        if self.drop_path_rate > 0.5:
            warnings.warn(
                f"drop_path_rate={self.drop_path_rate} is unusually high; "
                "values between 0.08 and 0.3 are typical for ViT."
            )

        # Validate image_size and patch_size
        if self.image_size <= 0:
            raise ValueError(
                f"image_size must be positive, got {self.image_size}"
            )

        if self.patch_size <= 0:
            raise ValueError(
                f"patch_size must be positive, got {self.patch_size}"
            )

        if self.image_size % self.patch_size != 0:
            # Calculate valid alternatives
            valid_sizes = [
                s for s in [224, 256, 384, 448, 512, 576, 640, 768, 896, 1024]
                if s % self.patch_size == 0 and abs(s - self.image_size) < 200
            ]

            raise ValueError(
                f"image_size ({self.image_size}) must be evenly divisible by "
                f"patch_size ({self.patch_size}). "
                f"Current: {self.image_size} % {self.patch_size} = {self.image_size % self.patch_size}. "
                f"\n\nSuggested fixes:"
                f"\n  1. Use a standard image size: {valid_sizes if valid_sizes else 'N/A'}"
                f"\n  2. Change patch_size to a factor of {self.image_size}: "
                f"{[d for d in [8, 14, 16, 20, 32] if self.image_size % d == 0]}"
                f"\n  3. Adjust image_size to nearest multiple of {self.patch_size}: "
                f"{(self.image_size // self.patch_size) * self.patch_size} or "
                f"{((self.image_size // self.patch_size) + 1) * self.patch_size}"
            )

        # Validate computed values make sense
        num_patches = (self.image_size // self.patch_size) ** 2
        if num_patches < 4:
            warnings.warn(
                f"Very few patches ({num_patches}) with image_size={self.image_size} "
                f"and patch_size={self.patch_size}. Model may underperform."
            )

        if num_patches > 10000:
            warnings.warn(
                f"Very many patches ({num_patches}) with image_size={self.image_size} "
                f"and patch_size={self.patch_size}. May cause memory issues."
            )


class TransformerBlock(nn.Module):
    """Single transformer block using Flex Attention."""

    def __init__(self, config: VisionTransformerConfig, drop_path: float = 0.):
        super().__init__()
        self.config = config
        self.norm1 = LayerNormFp32(config.hidden_size, eps=config.layer_norm_eps, use_fp32=config.use_fp32_layernorm)

        # Flex Attention setup
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.flex_block_size = config.flex_block_size

        # QKV projection (fused linear for efficiency)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attn_dropout = config.attention_dropout

        # Drop path and MLP
        self.drop_path = SafeDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = LayerNormFp32(config.hidden_size, eps=config.layer_norm_eps, use_fp32=config.use_fp32_layernorm)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor, block_mask: Optional[BlockMask] = None) -> torch.Tensor:
        """Forward pass using Flex Attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            block_mask: Optional pre-computed BlockMask for padding-aware attention.
                       Created once by SimplifiedTagger and shared across all layers.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size)

        Note:
            Uses pre-layer normalization (LayerNorm -> Attention -> Residual).
        """
        normed_x = self.norm1(x)
        B, L, D = normed_x.shape

        # QKV projection and reshape
        qkv = self.qkv(normed_x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, L, num_heads, head_dim)

        # Transpose to (B, num_heads, L, head_dim) for flex_attention
        # IMPORTANT: Make tensors contiguous after transpose to ensure proper memory layout.
        # Non-contiguous tensors from reshape->unbind->transpose cause stride assertion
        # failures when torch.compile traces the graph (inductor expects contiguous strides).
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Flex Attention computation (block_mask is pre-computed and shared across layers)
        attn_out = flex_attention(
            q, k, v,
            block_mask=block_mask,
            scale=self.scale,
        )

        # Reshape back to (B, L, D) - ensure contiguous for torch.compile compatibility
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D).contiguous()

        # Apply attention dropout (post-attention, flex_attention doesn't have built-in dropout)
        if self.training and self.attn_dropout > 0:
            attn_out = F.dropout(attn_out, p=self.attn_dropout, training=True)

        # Residual connections
        x = x + self.drop_path(self.proj(attn_out))
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
        self.norm = LayerNormFp32(config.hidden_size, eps=config.layer_norm_eps, use_fp32=config.use_fp32_layernorm)
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

    def _check_numerical_stability(
        self,
        tag_logits: torch.Tensor,
        rating_logits: torch.Tensor
    ) -> None:
        """Check for NaN/Inf in logits and log statistics.

        This method is only called when config.check_numerical_stability=True.
        It helps diagnose numerical instability issues during training/inference.
        """
        # Check for non-finite values before clamping
        tag_has_nan = torch.isnan(tag_logits).any().item()
        tag_has_inf = torch.isinf(tag_logits).any().item()
        rating_has_nan = torch.isnan(rating_logits).any().item()
        rating_has_inf = torch.isinf(rating_logits).any().item()

        if tag_has_nan or tag_has_inf:
            warnings.warn(
                f"Numerical instability in tag_logits: "
                f"NaN={tag_has_nan}, Inf={tag_has_inf}. "
                f"Stats: min={tag_logits.min():.2f}, max={tag_logits.max():.2f}, "
                f"mean={tag_logits.mean():.2f}, std={tag_logits.std():.2f}"
            )

        if rating_has_nan or rating_has_inf:
            warnings.warn(
                f"Numerical instability in rating_logits: "
                f"NaN={rating_has_nan}, Inf={rating_has_inf}"
            )

        # Check if values exceed clamping thresholds
        clamp_threshold = 15.0
        tag_needs_clamp = (tag_logits.abs() > clamp_threshold).any().item()
        rating_needs_clamp = (rating_logits.abs() > clamp_threshold).any().item()

        if tag_needs_clamp:
            num_clamped = (tag_logits.abs() > clamp_threshold).sum().item()
            warnings.warn(
                f"Clamping {num_clamped} tag logits "
                f"(max abs value: {tag_logits.abs().max():.2f})"
            )

        if rating_needs_clamp:
            num_clamped = (rating_logits.abs() > clamp_threshold).sum().item()
            warnings.warn(
                f"Clamping {num_clamped} rating logits "
                f"(max abs value: {rating_logits.abs().max():.2f})"
            )

    def _create_block_mask(self, key_padding_mask: torch.Tensor, seq_len: int) -> BlockMask:
        """Create BlockMask for padding-aware attention.

        This method creates the mask once per forward pass, which is then shared
        across all transformer layers for efficiency (instead of creating per-layer).

        Args:
            key_padding_mask: (B, L) bool, True=IGNORE (padding tokens)
            seq_len: Sequence length

        Returns:
            BlockMask for use with flex_attention
        """
        B = key_padding_mask.shape[0]
        attend_mask = ~key_padding_mask  # (B, L) True=ATTEND

        def mask_mod(b, h, q_idx, kv_idx):
            # Both query and key must be non-padding for attention
            # This prevents padding tokens from attending (wasteful) and being attended to
            # Use bitwise & instead of logical 'and' for Triton 3.4+ compatibility
            return attend_mask[b, q_idx] & attend_mask[b, kv_idx]

        # Use first block's config for flex_block_size
        flex_block_size = self.config.flex_block_size

        # Use _compile=True for faster mask creation when Triton is available
        return create_block_mask(
            mask_mod,
            B=B,
            H=None,  # Broadcast across heads - mask is head-independent (saves memory)
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=str(key_padding_mask.device),
            BLOCK_SIZE=min(flex_block_size, seq_len),
            _compile=_TRITON_AVAILABLE,  # Requires Triton for compilation
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,  # (B,H,W) or (B,1,H,W), auto-detected semantics
    ) -> Dict[str, torch.Tensor]:
        B = pixel_values.shape[0]
        # Patch embedding (optionally force fp32 for numerical stability under AMP)
        if self.config.use_fp32_layernorm and pixel_values.dtype in (torch.float16, torch.bfloat16):
            # Detect device type dynamically to support cuda/cpu/mps
            device_type = pixel_values.device.type
            # Autocast is only supported on certain device types
            supported_devices = {'cuda', 'cpu', 'mps', 'xpu'}

            if device_type in supported_devices:
                # Use autocast to disable AMP for this operation
                with torch.autocast(device_type=device_type, enabled=False):
                    x = self.patch_embed(pixel_values.float())
            else:
                # Fallback: Just convert to float32 without autocast context
                # This works on all devices but doesn't interact with AMP
                warnings.warn(
                    f"Device type '{device_type}' doesn't support autocast. "
                    f"Using fallback path for patch embedding."
                )
                x = self.patch_embed(pixel_values.float())

            x = x.to(pixel_values.dtype)
        else:
            x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2).contiguous()
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
            pm = ensure_pixel_padding_mask(padding_mask, mask_semantics='pad')  # -> (B,1,H,W) bool, True=PAD
            thr = getattr(self.config, "token_ignore_threshold", 0.9)
            token_ignore = pixel_to_token_ignore(pm, patch=self.config.patch_size, threshold=thr)  # (B,Lp)
            # Prepend CLS token (never ignored)
            cls_keep = torch.zeros(B, 1, dtype=torch.bool, device=token_ignore.device)
            attn_kpm = torch.cat([cls_keep, token_ignore], dim=1) # (B, 1+Lp) True=IGNORE

        if attn_kpm is not None:
            # Only validate in debug mode to avoid expensive GPU-CPU synchronization
            # (skip inside tracing to avoid control flow issues)
            if self.config.check_numerical_stability and not torch.jit.is_tracing():
                if attn_kpm.all(dim=1).any().item():
                    raise RuntimeError("attn_kpm masks all keys for at least one sample.")

        # Create BlockMask once for all layers (optimization: avoids creating per-layer)
        block_mask: Optional[BlockMask] = None
        if attn_kpm is not None and attn_kpm.any():
            block_mask = self._create_block_mask(attn_kpm, x.size(1))

        # Transformer blocks with optional gradient checkpointing
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                # use_reentrant=False is recommended for PyTorch 2.x and works better with
                # complex inputs like attention masks. It's also more memory efficient.
                # See: https://pytorch.org/docs/stable/checkpoint.html

                def create_block_forward(b, mask):
                    """Create a closure that captures block and mask for checkpointing."""
                    def block_forward(hidden_states):
                        return b(hidden_states, block_mask=mask)
                    return block_forward

                x = torch.utils.checkpoint.checkpoint(
                    create_block_forward(block, block_mask),
                    x,
                    use_reentrant=False
                )
            else:
                x = block(x, block_mask=block_mask)
        # Final norm
        x = self.norm(x)
        # Use CLS token for classification
        cls_output = x[:, 0]
        # Predictions
        tag_logits = self.tag_head(cls_output)
        rating_logits = self.rating_head(cls_output)

        # Monitor for numerical issues (optional, controlled by config)
        if self.config.check_numerical_stability:
            self._check_numerical_stability(tag_logits, rating_logits)

        # Clamp logits to prevent numerical instability with mixed precision
        if self.config.logit_clamp_value is not None:
            clamp_val = self.config.logit_clamp_value
            tag_logits = torch.clamp(tag_logits, min=-clamp_val, max=clamp_val)
            rating_logits = torch.clamp(rating_logits, min=-clamp_val, max=clamp_val)

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

    def cleanup(self):
        """Explicitly release GPU memory and clear cached tensors."""
        import gc

        # Move all parameters to CPU to release GPU memory
        self.cpu()

        # Clear any cached gradients
        for param in self.parameters():
            if param.grad is not None:
                param.grad = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            # Synchronize before clearing cache to ensure all GPU operations complete
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup on exit."""
        self.cleanup()
        return False


def create_model(config: Optional[VisionTransformerConfig] = None, **kwargs) -> SimplifiedTagger:
    """Create model from configuration.

    Args:
        config: VisionTransformerConfig instance. If provided, kwargs are ignored.
        **kwargs: Alternative to config - specify config fields as keyword arguments.
                 See VisionTransformerConfig for details on each parameter.

    Returns:
        SimplifiedTagger instance with the specified configuration.

    Examples:
        # Using config object
        config = VisionTransformerConfig(image_size=512, num_tags=10000)
        model = create_model(config=config)

        # Using kwargs
        model = create_model(image_size=512, num_tags=10000)
    """
    if config is None:
        # Get the names of the fields in the VisionTransformerConfig dataclass
        config_fields = {f.name for f in fields(VisionTransformerConfig)}

        # Filter kwargs to only include keys that are in the config_fields
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

        # Warn if kwargs contains invalid keys
        invalid_keys = set(kwargs.keys()) - config_fields
        if invalid_keys:
            warnings.warn(
                f"Ignoring unknown configuration parameters: {sorted(invalid_keys)}. "
                f"Valid parameters: {sorted(config_fields)}"
            )

        config = VisionTransformerConfig(**filtered_kwargs)

    return SimplifiedTagger(config)
