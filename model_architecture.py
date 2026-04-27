#!/usr/bin/env python3
"""
Model Architecture for Anime Image Tagger - Direct Training (modified)
Vision Transformer adjusted for orientation-aware tagger
"""

from abc import ABC, abstractmethod
import functools
import logging
import math
import threading
import types
import warnings
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from mask_utils import ensure_pixel_padding_mask, pixel_to_token_ignore, pixel_to_swin_token_masks
from custom_drop_path import SafeDropPath

# Import Flex Attention (PyTorch 2.5+)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

logger.info("Using PyTorch Flex Attention")


def _check_triton_available() -> bool:
    """Check if Triton is available for compiled block mask creation."""
    try:
        import triton
        return True
    except ImportError:
        return False


# Cache the Triton availability check at module load time
_TRITON_AVAILABLE = _check_triton_available()


def validate_image_size_for_swinv2(
    image_size: int,
    window_size: int = 16,
    num_stages: int = 4,
    initial_patch_size: int = 4,
) -> None:
    """Validate image size for SwinV2 window_size requirements.

    This is a standalone function that can be called early (e.g., during dataset
    creation) to catch image size incompatibilities before training starts.

    SwinV2 uses hierarchical feature maps where each stage halves the spatial resolution.
    For window-based attention to work correctly:
    1. image_size must be divisible by (patch_size * 2^(num_stages-1)) for the final stage
    2. All intermediate feature map sizes must be divisible by window_size

    Args:
        image_size: Input image size (assumes square images)
        window_size: Window size for shifted window attention (default: 16)
        num_stages: Number of hierarchical stages (default: 4, typical for SwinV2)
        initial_patch_size: Initial patch embedding size (default: 4 for all SwinV2 variants)

    Raises:
        ValueError: If image_size is incompatible with the SwinV2 architecture

    Example:
        >>> from model_architecture import validate_image_size_for_swinv2
        >>> validate_image_size_for_swinv2(512, window_size=16)  # OK
        >>> validate_image_size_for_swinv2(500, window_size=16)  # Raises ValueError
    """
    # Calculate the total downsampling factor for the final stage
    # SwinV2 downsamples by patch_size initially, then by 2 at each stage transition
    # Final resolution = image_size / (patch_size * 2^(num_stages-1))
    total_downsample = initial_patch_size * (2 ** (num_stages - 1))

    if image_size % total_downsample != 0:
        final_res = image_size / total_downsample
        valid_sizes = [
            s for s in [192, 224, 256, 384, 448, 512, 640, 768, 896, 1024]
            if s % total_downsample == 0
        ]
        raise ValueError(
            f"image_size ({image_size}) must be divisible by {total_downsample} "
            f"(patch_size={initial_patch_size} * 2^{num_stages-1}) for SwinV2 with {num_stages} stages. "
            f"Current final resolution would be {final_res:.2f} (must be integer). "
            f"Suggested valid image sizes: {valid_sizes}"
        )

    # Check window_size divisibility at each stage
    # Stage 0: feature_size = image_size / patch_size
    # Stage i: feature_size = image_size / (patch_size * 2^i)
    invalid_stages = []
    for stage in range(num_stages):
        downsample_at_stage = initial_patch_size * (2 ** stage)
        feature_size = image_size // downsample_at_stage

        if feature_size % window_size != 0:
            invalid_stages.append(
                f"stage {stage}: feature_size={feature_size}, "
                f"feature_size % window_size = {feature_size % window_size}"
            )

    if invalid_stages:
        # Calculate valid image sizes that work with this window_size
        # Need: image_size / (patch_size * 2^i) divisible by window_size for all stages
        # Simplest: image_size divisible by patch_size * 2^(num_stages-1) * window_size
        required_multiple = total_downsample * window_size
        valid_sizes = [
            s for s in range(required_multiple, 1025, required_multiple)
            if s <= 1024
        ]

        raise ValueError(
            f"image_size ({image_size}) is incompatible with window_size ({window_size}) "
            f"for SwinV2 with {num_stages} stages. "
            f"Feature map sizes must be divisible by window_size at all stages. "
            f"Invalid stages: {'; '.join(invalid_stages)}. "
            f"Suggested valid image sizes (divisible by {required_multiple}): {valid_sizes}"
        )

    # Check that final stage feature size >= window_size
    # The final stage has the smallest feature map (most downsampled)
    final_feature_size = image_size // total_downsample
    if final_feature_size < window_size:
        raise ValueError(
            f"Final stage feature size ({final_feature_size}) is smaller than window_size ({window_size}). "
            f"SwinV2 requires the feature map at each stage to be at least as large as the window size. "
            f"Either increase image_size (current: {image_size}) or decrease window_size. "
            f"Minimum image_size for window_size={window_size} with {num_stages} stages: "
            f"{window_size * total_downsample}"
        )


class BaseTagger(ABC, nn.Module):
    """Abstract base class for image tagging models."""

    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning tag_logits dict."""
        pass

    @abstractmethod
    def set_onnx_mode(self, enabled: bool = True) -> None:
        """Enable/disable ONNX-compatible attention mode."""
        pass

    @property
    @abstractmethod
    def config(self):
        """Return model configuration."""
        pass


def initialize_tag_head_bias(
    model: BaseTagger,
    index_to_tag: Dict[int, str],
    tag_frequencies: Dict[str, int],
    total_samples: int,
    min_prior: float = 1e-5,
    max_prior: float = 0.99,
) -> None:
    """Initialize tag_head bias with log-prior for focal loss (RetinaNet technique).

    Sets each tag's bias to log(prior / (1 - prior)) based on empirical tag frequency.
    This makes the model start by predicting "mostly negative" for sparse labels,
    providing meaningful gradients from the first step instead of random 0.5 predictions.

    Must be called AFTER model creation (which zeros the bias) and BEFORE checkpoint
    loading (which will overwrite the bias if resuming).
    """
    with torch.no_grad():
        bias = model.tag_head.bias
        num_tags = bias.shape[0]
        for idx in range(num_tags):
            tag = index_to_tag.get(idx, "")
            freq = tag_frequencies.get(tag, 0)
            prior = max(min_prior, min(max_prior, freq / max(1, total_samples)))
            bias[idx] = math.log(prior / (1 - prior))

        bias_vals = bias.tolist()
        logger.info(
            "Tag head bias initialized with log-prior: min=%.2f, max=%.2f, mean=%.2f",
            min(bias_vals), max(bias_vals), sum(bias_vals) / len(bias_vals),
        )


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
    image_size: int = 448
    # Patch size must divide image_size evenly to avoid losing border information (validated in __post_init__)
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 1024
    num_hidden_layers: int = 17
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    num_tags: int = 100000  # This should be overridden with actual vocab size (includes rating tags)
    dropout: float = 0.1
    pos_dropout: float = 0.0  # Position embedding dropout (0.0 = modern standard; drop_path handles regularization)
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    use_flex_attention: bool = True  # Use Flex Attention (PyTorch 2.5+)
    flex_block_size: int = 128  # Block size for Flex Attention sparse computation
    attention_bias: bool = True  # Use bias in attention QKV and projection layers
    # Token ignore threshold: a token is ignored if >= this fraction of its pixels are PAD
    token_ignore_threshold: float = 0.9
    # Enable gradient checkpointing by default to reduce memory usage
    gradient_checkpointing: bool = True
    # Checkpoint every N-th layer (1=all layers, 2=every 2nd, 4=every 4th, etc.)
    checkpoint_every_n_layers: int = 1
    drop_path_rate: float = 0.0
    # Enable numerical stability checking (for debugging only)
    check_numerical_stability: bool = False
    # Logit clamping for numerical stability (None to disable)
    # exp(15) ~ 3.3M which is safe for softmax in float16/bfloat16
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

        # Validate hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads}). "
                f"Current remainder: {self.hidden_size % self.num_attention_heads}. "
                f"head_dim would be {self.hidden_size / self.num_attention_heads:.2f} (must be integer)."
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

        # flex_attention has no native attention-weight dropout, so apply V-dropout
        # before the call. Zeroing random value tokens removes their contribution to
        # all queries — a documented approximation of attention dropout used in T5
        # and several BERT variants. Replaces the previous post-attention output
        # dropout, which did not regularize attention weights as the config name
        # suggested.
        if self.training and self.attn_dropout > 0:
            v = F.dropout(v, p=self.attn_dropout, training=True)

        # Flex Attention computation (block_mask is pre-computed and shared across layers)
        attn_out = flex_attention(
            q, k, v,
            block_mask=block_mask,
            scale=self.scale,
        )

        # Reshape back to (B, L, D) - ensure contiguous for torch.compile compatibility
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D).contiguous()

        # Residual connections
        x = x + self.drop_path(self.proj(attn_out))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward_sdpa(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using scaled_dot_product_attention (ONNX-compatible).

        This method provides an ONNX-exportable alternative to the flex_attention path.
        It uses F.scaled_dot_product_attention which is supported by ONNX opset 14+.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attn_mask: Optional boolean attention mask (B, L) with True=ATTEND, False=IGNORE.
                      Passed directly to SDPA as a boolean key-padding mask — avoids the
                      additive float mask path that forces PyTorch off the Flash/efficient
                      backends.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size)

        Note:
            Uses pre-layer normalization (LayerNorm -> Attention -> Residual).
            For ONNX export, this path is used instead of flex_attention.
        """
        normed_x = self.norm1(x)
        B, L, D = normed_x.shape

        # QKV projection and reshape
        qkv = self.qkv(normed_x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, L, num_heads, head_dim)

        # Transpose to (B, num_heads, L, head_dim) for scaled_dot_product_attention
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Pass the boolean mask straight through. SDPA accepts a bool key-padding mask
        # broadcastable to (B, num_heads, L_q, L_k); we broadcast (B, L) -> (B, 1, 1, L).
        # Bool semantics for SDPA: True=ATTEND, False=IGNORE — already matches `attn_mask`.
        sdpa_mask: Optional[torch.Tensor] = None
        if attn_mask is not None:
            if attn_mask.ndim != 2 or attn_mask.shape[0] != B or attn_mask.shape[1] != L:
                raise ValueError(
                    f"attn_mask shape {attn_mask.shape} mismatch. "
                    f"Expected ({B}, {L}) for input shape ({B}, {L}, {D})"
                )
            sdpa_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L) bool

        # Scaled dot-product attention (ONNX-compatible)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape back to (B, L, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D).contiguous()

        # Residual connections
        x = x + self.drop_path(self.proj(attn_out))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SimplifiedTagger(BaseTagger):
    """Vision Transformer based tagger for anime images."""
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self._config = config
        self._onnx_mode = False  # When True, use SDPA instead of flex_attention
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
        # Initialize pos_embed and cls_token with truncated normal (std=0.02)
        # These are nn.Parameter objects, not nn.Module, so _init_weights() doesn't handle them.
        # Standard ViT initialization follows timm/HuggingFace convention.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(p=config.pos_dropout)
        # Transformer blocks
        rate = getattr(config, 'drop_path_rate', 0.0)
        dpr = torch.linspace(0.0, rate, config.num_hidden_layers)
        dpr = dpr.clamp_max(1.0 - 1e-6).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(config, drop_path=float(p))
            for p in dpr
        ])
        # Final layer norm
        self.norm = LayerNormFp32(config.hidden_size, eps=config.layer_norm_eps, use_fp32=config.use_fp32_layernorm)
        # Classification head (ratings are now part of the tag vocabulary)
        self.tag_head = nn.Linear(config.hidden_size, config.num_tags)
        # Weight initialization
        self.apply(self._init_weights)
        # Override patch_embed init: standard ViT uses trunc_normal(std=0.02).
        # The Conv2d branch in _init_weights sets std = sqrt(2/(k*k*out_channels))
        # ≈ 0.0028 for our k=16, out_channels=1024 — ~7x smaller than ViT convention,
        # which produces vanishing patch-embed activations early in from-scratch
        # training. Re-init here so the apply() pass above doesn't overwrite us.
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

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

    def set_onnx_mode(self, enabled: bool = True) -> None:
        """Enable or disable ONNX-compatible mode.

        When enabled, the model uses F.scaled_dot_product_attention instead of
        flex_attention. This is required for ONNX export since flex_attention
        is a PyTorch 2.5+ CUDA kernel that cannot be traced by ONNX.

        Args:
            enabled: If True, use SDPA (ONNX-compatible). If False, use flex_attention.

        Note:
            This only affects the attention computation path. Model weights are unchanged.
            The SDPA path produces numerically equivalent results to flex_attention.
        """
        self._onnx_mode = enabled
        logger.info(f"ONNX mode {'enabled' if enabled else 'disabled'} - using {'SDPA' if enabled else 'flex_attention'}")

    @property
    def config(self):
        """Return model configuration."""
        return self._config

    def _check_numerical_stability(
        self,
        tag_logits: torch.Tensor,
    ) -> None:
        """Check for NaN/Inf in logits and log statistics.

        This method is only called when config.check_numerical_stability=True.
        It helps diagnose numerical instability issues during training/inference.
        """
        tag_has_nan = torch.isnan(tag_logits).any().item()
        tag_has_inf = torch.isinf(tag_logits).any().item()

        if tag_has_nan or tag_has_inf:
            warnings.warn(
                f"Numerical instability in tag_logits: "
                f"NaN={tag_has_nan}, Inf={tag_has_inf}. "
                f"Stats: min={tag_logits.min():.2f}, max={tag_logits.max():.2f}, "
                f"mean={tag_logits.mean():.2f}, std={tag_logits.std():.2f}"
            )

        clamp_threshold = 15.0
        tag_needs_clamp = (tag_logits.abs() > clamp_threshold).any().item()
        if tag_needs_clamp:
            num_clamped = (tag_logits.abs() > clamp_threshold).sum().item()
            warnings.warn(
                f"Clamping {num_clamped} tag logits "
                f"(max abs value: {tag_logits.abs().max():.2f})"
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
        flex_block_size = self._config.flex_block_size

        # Use _compile=True for faster mask creation when Triton is available
        return create_block_mask(
            mask_mod,
            B=B,
            H=None,  # Broadcast across heads - mask is head-independent (saves memory)
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=key_padding_mask.device,
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
        if self._config.use_fp32_layernorm and pixel_values.dtype in (torch.float16, torch.bfloat16):
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
        # =======================================================================
        # MASK SEMANTICS DOCUMENTATION
        # =======================================================================
        # This section handles padding masks with multiple transformations.
        # Understanding the semantics at each step is crucial for correctness.
        #
        # STEP 1: Input padding_mask (pixel-level)
        #   - Shape: (B, H, W) or (B, 1, H, W)
        #   - Semantics: True = PADDING pixel (should be ignored)
        #
        # STEP 2: attn_kpm (token-level key padding mask)
        #   - Shape: (B, 1+Lp) where Lp = number of patch tokens
        #   - Semantics: True = IGNORE this token (it's mostly padding)
        #   - CLS token is always False (never ignored)
        #
        # STEP 3a: For ONNX/SDPA mode -> sdpa_attn_mask
        #   - We invert: sdpa_attn_mask = ~attn_kpm
        #   - Semantics: True = ATTEND to this token
        #   - forward_sdpa() passes this directly to F.scaled_dot_product_attention as
        #     a boolean key-padding mask (broadcast to (B, 1, 1, L)). SDPA's bool-mask
        #     contract matches: True=attend, False=ignore — so no additive conversion
        #     is needed, and PyTorch can keep using the Flash/efficient backends.
        #
        # STEP 3b: For flex_attention mode -> block_mask
        #   - _create_block_mask inverts internally: attend_mask = ~key_padding_mask
        #   - BlockMask uses: True = ATTEND (allow attention between these positions)
        #   - flex_attention natively handles this without further inversion
        # =======================================================================

        # Build key-padding mask (CLS + patch tokens) from pixel-level mask.
        # Output: attn_kpm with semantics True=IGNORE (padding tokens to exclude)
        attn_kpm: Optional[torch.Tensor] = None
        if padding_mask is not None:
            # Convert pixel mask to standard format: (B,1,H,W) bool, True=PAD
            pm = ensure_pixel_padding_mask(padding_mask, mask_semantics='pad')
            thr = getattr(self._config, "token_ignore_threshold", 0.9)
            # Pool pixel mask to token-level: token is IGNORE if >= threshold fraction is padding
            token_ignore = pixel_to_token_ignore(pm, patch=self._config.patch_size, threshold=thr)  # (B,Lp)
            # Prepend CLS token (always attend, never ignored)
            cls_keep = torch.zeros(B, 1, dtype=torch.bool, device=token_ignore.device)  # False = don't ignore
            attn_kpm = torch.cat([cls_keep, token_ignore], dim=1)  # (B, 1+Lp), True=IGNORE

        if attn_kpm is not None:
            # Only validate in debug mode to avoid expensive GPU-CPU synchronization
            # (skip inside tracing to avoid control flow issues)
            if self._config.check_numerical_stability and not torch.jit.is_tracing() and not self._onnx_mode:
                if attn_kpm.all(dim=1).any().item():
                    raise RuntimeError("attn_kpm masks all keys for at least one sample.")

        # Create attention mask based on mode (ONNX vs flex_attention)
        # Each path converts attn_kpm (True=IGNORE) to the format expected by the attention mechanism
        block_mask: Optional[BlockMask] = None
        sdpa_attn_mask: Optional[torch.Tensor] = None

        if self._onnx_mode:
            # ONNX mode: use SDPA with simple boolean mask
            # For ONNX export, images are padded to square by InferenceWrapper,
            # so typically no masking is needed. But support it for completeness.
            if attn_kpm is not None:
                # Invert semantics: attn_kpm True=IGNORE -> sdpa_attn_mask True=ATTEND
                # forward_sdpa() will then do: masked_fill(~attn_mask, -inf)
                # which gives: True=ATTEND -> 0, False=IGNORE -> -inf (correct SDPA format)
                # Always apply (no .any() guard) so the mask branch is captured during export.
                # An all-False attn_kpm produces an all-True sdpa_attn_mask (all-attend),
                # which is mathematically identical to no mask.
                sdpa_attn_mask = ~attn_kpm
        else:
            # Normal mode: use flex_attention with BlockMask
            # _create_block_mask handles the inversion internally:
            #   attend_mask = ~key_padding_mask (True=ATTEND)
            #   BlockMask allows attention where attend_mask[q] & attend_mask[kv] is True
            if attn_kpm is not None and attn_kpm.any():
                block_mask = self._create_block_mask(attn_kpm, x.size(1))

        # Transformer blocks with optional selective gradient checkpointing
        # checkpoint_every_n_layers controls granularity: 1=all, 2=every 2nd, 4=every 4th
        checkpoint_interval = getattr(self._config, 'checkpoint_every_n_layers', 1)
        for idx, block in enumerate(self.blocks):
            should_checkpoint = (
                self._config.gradient_checkpointing
                and self.training
                and (idx % checkpoint_interval == 0)
            )

            if self._onnx_mode:
                # ONNX mode: use SDPA path (no checkpointing during export)
                x = block.forward_sdpa(x, attn_mask=sdpa_attn_mask)
            elif should_checkpoint:
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
        # Predictions (ratings are included as tags in the vocabulary)
        tag_logits = self.tag_head(cls_output)

        # Monitor for numerical issues (optional, controlled by config)
        if self._config.check_numerical_stability:
            self._check_numerical_stability(tag_logits)

        # Clamp logits to prevent numerical instability with mixed precision
        if self._config.logit_clamp_value is not None:
            clamp_val = self._config.logit_clamp_value
            tag_logits = torch.clamp(tag_logits, min=-clamp_val, max=clamp_val)

        return {
            'tag_logits': tag_logits,
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
        return torch.zeros(batch_size, 3, self._config.image_size, self._config.image_size, device=device)

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


class SwinV2Tagger(BaseTagger):
    """SwinV2-based tagger for anime images.

    Uses timm's SwinV2 backbone with classification head
    for tags (including rating tags).
    """

    # Known SwinV2 model configurations from timm
    # Maps model_name -> (embed_dim, depths, num_heads, default_window_size)
    KNOWN_SWINV2_CONFIGS = {
        'swinv2_tiny_window8_256': (96, (2, 2, 6, 2), (3, 6, 12, 24), 8),
        'swinv2_tiny_window16_256': (96, (2, 2, 6, 2), (3, 6, 12, 24), 16),
        'swinv2_small_window8_256': (96, (2, 2, 18, 2), (3, 6, 12, 24), 8),
        'swinv2_small_window16_256': (96, (2, 2, 18, 2), (3, 6, 12, 24), 16),
        'swinv2_base_window8_256': (128, (2, 2, 18, 2), (4, 8, 16, 32), 8),
        'swinv2_base_window12_192': (128, (2, 2, 18, 2), (4, 8, 16, 32), 12),
        'swinv2_base_window12to16_192to256_22kft1k': (128, (2, 2, 18, 2), (4, 8, 16, 32), 16),
        'swinv2_base_window12to24_192to384_22kft1k': (128, (2, 2, 18, 2), (4, 8, 16, 32), 24),
        'swinv2_base_window16_256': (128, (2, 2, 18, 2), (4, 8, 16, 32), 16),
        'swinv2_large_window12_192': (192, (2, 2, 18, 2), (6, 12, 24, 48), 12),
        'swinv2_large_window12to16_192to256_22kft1k': (192, (2, 2, 18, 2), (6, 12, 24, 48), 16),
        'swinv2_large_window12to24_192to384_22kft1k': (192, (2, 2, 18, 2), (6, 12, 24, 48), 24),
    }

    # Default config values (used to detect custom params)
    _DEFAULT_EMBED_DIM = 256
    _DEFAULT_DEPTHS = (2, 2, 18, 2)
    _DEFAULT_NUM_HEADS = (8, 16, 32, 64)
    _DEFAULT_WINDOW_SIZE = 16

    @staticmethod
    def _has_custom_architecture_params(swin_config) -> bool:
        """Check if swin_config specifies custom architecture parameters.

        Returns True if any architecture params differ from defaults, indicating
        the user wants a custom architecture rather than a standard pretrained model.
        """
        if swin_config is None:
            return False

        # Check each architecture param against defaults
        if hasattr(swin_config, 'embed_dim') and swin_config.embed_dim != SwinV2Tagger._DEFAULT_EMBED_DIM:
            return True
        if hasattr(swin_config, 'depths') and tuple(swin_config.depths) != SwinV2Tagger._DEFAULT_DEPTHS:
            return True
        if hasattr(swin_config, 'num_heads') and tuple(swin_config.num_heads) != SwinV2Tagger._DEFAULT_NUM_HEADS:
            return True
        if hasattr(swin_config, 'window_size') and swin_config.window_size != SwinV2Tagger._DEFAULT_WINDOW_SIZE:
            return True

        return False

    @classmethod
    def _select_model_for_config(
        cls,
        embed_dim: int,
        depths: Tuple[int, ...],
        num_heads: Tuple[int, ...],
        window_size: int,
    ) -> str:
        """Select the best matching timm model name for the given config.

        Tries to find a model that exactly matches the config, otherwise returns
        the closest match (preferring models with matching embed_dim and depths).

        Args:
            embed_dim: Embedding dimension
            depths: Tuple of depths per stage
            num_heads: Tuple of attention heads per stage
            window_size: Window size for attention

        Returns:
            Model name string suitable for timm.create_model()
        """
        # First, try to find exact match
        for model_name, (m_embed, m_depths, m_heads, m_window) in cls.KNOWN_SWINV2_CONFIGS.items():
            if (embed_dim == m_embed and depths == m_depths and
                num_heads == m_heads and window_size == m_window):
                return model_name

        # No exact match - find closest by embed_dim and depths
        candidates = []
        for model_name, (m_embed, m_depths, m_heads, m_window) in cls.KNOWN_SWINV2_CONFIGS.items():
            # Score by how close the config matches
            score = 0
            if embed_dim == m_embed:
                score += 4  # embed_dim match is most important
            if depths == m_depths:
                score += 2  # depths match is second
            if num_heads == m_heads:
                score += 1  # heads match is third
            if window_size == m_window:
                score += 1  # window match is fourth
            candidates.append((score, model_name))

        if candidates:
            candidates.sort(reverse=True)
            best_model = candidates[0][1]
            logger.info(
                f"No exact SwinV2 config match. Selected '{best_model}' as closest match for "
                f"embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}, window_size={window_size}. "
                f"Note: Custom params are NOT passed to timm - the model uses its default architecture."
            )
            return best_model

        # Fallback to default
        return 'swinv2_base_window12_192'

    def __init__(self, config: VisionTransformerConfig, swin_config=None):
        super().__init__()
        self._config = config
        self._swin_config = swin_config
        self._onnx_mode = False

        import timm

        # Get config values with defaults
        embed_dim = swin_config.embed_dim if swin_config else 256
        depths = tuple(swin_config.depths) if swin_config else (2, 2, 18, 2)
        num_heads = tuple(swin_config.num_heads) if swin_config else (8, 16, 32, 64)
        window_size = swin_config.window_size if swin_config else 16
        mlp_ratio = swin_config.mlp_ratio if swin_config else 4.0
        drop_path_rate = swin_config.drop_path_rate if swin_config else 0.2

        # Determine if using pretrained weights or custom architecture
        use_pretrained = False
        use_custom = False
        has_custom_architecture = self._has_custom_architecture_params(swin_config)
        custom_name = None
        if swin_config is not None:
            custom_name = getattr(swin_config, 'custom_name', None)
            if custom_name is not None:
                custom_name = str(custom_name).strip() or None

        if custom_name and swin_config is not None:
            if getattr(swin_config, 'pretrained_name', None):
                raise ValueError(
                    "custom_name cannot be combined with pretrained_name. "
                    "Custom SwinV2 architectures do not support pretrained weights."
                )
            if getattr(swin_config, 'model_name', None):
                raise ValueError(
                    "custom_name cannot be combined with model_name. "
                    "Use a timm preset (model_name) or a custom architecture (custom_name)."
                )

        if swin_config is not None and swin_config.pretrained_name:
            # User specified pretrained_name - use that model with pretrained weights
            if has_custom_architecture:
                # Warn: pretrained weights + custom params is contradictory
                warnings.warn(
                    f"SwinV2 config specifies both pretrained_name='{swin_config.pretrained_name}' "
                    f"and custom architecture parameters (embed_dim={embed_dim}, depths={depths}, "
                    f"num_heads={num_heads}, window_size={window_size}). "
                    f"When using pretrained weights, custom architecture params are IGNORED. "
                    f"The model will use the architecture defined by '{swin_config.pretrained_name}'. "
                    f"To use custom architecture, set pretrained_name=None and provide custom_name.",
                    UserWarning
                )
            model_name = swin_config.pretrained_name
            use_pretrained = True
        elif hasattr(swin_config, 'model_name') and swin_config.model_name if swin_config else False:
            # User specified model_name without pretrained_name - use as template for from-scratch training
            model_name = swin_config.model_name
        elif custom_name:
            # Explicit custom architecture request
            model_name = custom_name
            use_custom = True
        else:
            # Default: select appropriate model based on config params for from-scratch training
            model_name = self._select_model_for_config(embed_dim, depths, num_heads, window_size)

        # Validate config parameters against the model template (only warn, don't fail)
        if not use_pretrained and not use_custom:
            self._validate_config_compatibility(
                model_name, embed_dim, depths, num_heads, window_size
            )

        # Validate image size for SwinV2 window_size requirements
        # For timm presets, validate against the template values (actual model)
        effective_window_size = window_size
        effective_depths = depths
        if not use_custom and model_name in self.KNOWN_SWINV2_CONFIGS:
            _, effective_depths, _, effective_window_size = self.KNOWN_SWINV2_CONFIGS[model_name]
        self._validate_image_size_for_swinv2(
            config.image_size, effective_window_size, len(effective_depths)
        )

        if use_custom:
            # Build custom SwinV2 backbone with explicit architecture params
            from timm.models.swin_transformer_v2 import SwinTransformerV2
            self.backbone = SwinTransformerV2(
                img_size=config.image_size,
                patch_size=4,  # Fixed for SwinV2
                in_chans=config.num_channels,
                num_classes=0,  # Remove classification head
                global_pool='',  # Disable internal pooling for manual masked pooling
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path_rate=drop_path_rate,
            )
            logger.info(
                f"Using custom SwinV2 backbone '{model_name}': "
                f"embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}, "
                f"window_size={window_size}, mlp_ratio={mlp_ratio}, drop_path_rate={drop_path_rate}"
            )
        else:
            # Build SwinV2 backbone from timm preset
            # IMPORTANT: timm.create_model() only accepts specific kwargs for SwinV2.
            # Valid kwargs: pretrained, num_classes, global_pool, img_size
            # Invalid kwargs that will cause errors: embed_dim, depths, num_heads, window_size, mlp_ratio, drop_path_rate
            # When using pretrained weights, we MUST use the architecture defined by model_name.
            # When training from scratch, we still use the model's default architecture.
            self.backbone = timm.create_model(
                model_name,
                pretrained=use_pretrained,
                num_classes=0,  # Remove classification head
                global_pool='',  # Disable internal pooling for manual masked pooling
                img_size=config.image_size,
            )

        # Get feature dimension from backbone - ALWAYS use backbone.num_features
        # CRITICAL: timm ignores custom config values (embed_dim, depths, etc.) so we must
        # always trust the backbone's actual feature dimension, not calculate from config.
        if not hasattr(self.backbone, 'num_features'):
            raise RuntimeError(
                f"Backbone '{model_name}' does not have num_features attribute. "
                f"Cannot determine feature dimension for classification heads."
            )
        self.feature_dim = self.backbone.num_features

        # Warn if user-provided config values differ from what timm actually uses
        # This helps users understand that their config values are being ignored
        if not use_custom:
            self._warn_if_config_ignored(model_name, embed_dim, depths, num_heads, window_size)

        # Classification head (ratings are now part of the tag vocabulary)
        self.tag_head = nn.Linear(self.feature_dim, config.num_tags)

        # Initialize classification heads
        self._init_weights()

        # Setup gradient checkpointing if requested
        # Priority: swin_config.use_checkpoint > config.gradient_checkpointing (for compatibility)
        # This allows users to set either gradient_checkpointing=True (unified approach) or
        # swin_config.use_checkpoint=True (SwinV2-specific approach)
        enable_checkpointing = False
        if swin_config and swin_config.use_checkpoint:
            enable_checkpointing = True
        elif config.gradient_checkpointing:
            # Fallback: if gradient_checkpointing is set on the main config, use it
            enable_checkpointing = True
            logger.info(
                "SwinV2: gradient_checkpointing=True from main config applied. "
                "Note: For SwinV2 models, the canonical setting is swin_config.use_checkpoint."
            )

        checkpoint_interval = self._apply_swinv2_checkpoint_interval(
            getattr(config, "checkpoint_every_n_layers", 1)
        )

        if enable_checkpointing:
            self.backbone.set_grad_checkpointing(enable=True)

        logger.info(f"Created SwinV2Tagger: model_name={model_name}, pretrained={use_pretrained}, "
                    f"custom={use_custom}, feature_dim={self.feature_dim}, num_tags={config.num_tags}, "
                    f"gradient_checkpointing={enable_checkpointing}, checkpoint_interval={checkpoint_interval}")

    @staticmethod
    def _patch_swinv2_stage_forward(stage: nn.Module) -> None:
        """Patch a SwinV2 stage to support interval checkpointing without changing state_dict keys."""
        if getattr(stage, "_checkpoint_interval_patched", False):
            return

        def _stage_forward(self, x: torch.Tensor) -> torch.Tensor:
            if getattr(self, "downsample", None) is not None:
                x = self.downsample(x)

            if self.grad_checkpointing and not torch.jit.is_scripting():
                interval = getattr(self, "_checkpoint_interval", 1)
                if interval <= 1:
                    # Fallback to standard checkpointing if interval is 1 but patching occurred
                    for blk in self.blocks:
                        x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    for idx, blk in enumerate(self.blocks):
                        if idx % interval == 0:
                            x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                        else:
                            x = blk(x)
            else:
                for blk in self.blocks:
                    x = blk(x)
            return x

        stage.forward = types.MethodType(_stage_forward, stage)
        stage._checkpoint_interval_patched = True

    def _apply_swinv2_checkpoint_interval(self, interval: Any) -> int:
        """Apply checkpoint interval to SwinV2 stages (per-stage block index)."""
        try:
            interval = int(interval)
        except (TypeError, ValueError):
            logger.warning(
                "SwinV2 checkpoint interval must be an integer; got %r. Using 1.",
                interval,
            )
            interval = 1

        if interval < 1:
            logger.warning(
                "SwinV2 checkpoint interval must be >= 1; got %s. Using 1.",
                interval,
            )
            interval = 1

        # Optimization: If interval is 1, use timm's native checkpointing implementation
        # verified to work with SwinTransformerV2Stage.
        if interval == 1:
            logger.info("SwinV2: Using native timm gradient checkpointing (interval=1)")
            return 1

        stages = getattr(self.backbone, "layers", None)
        if stages is None:
            return interval

        logger.info(f"SwinV2: Patching stages for gradient checkpointing interval={interval}")
        for stage in stages:
            if not hasattr(stage, "blocks"):
                continue
            stage._checkpoint_interval = interval
            self._patch_swinv2_stage_forward(stage)

        return interval

    def _validate_config_compatibility(
        self,
        model_name: str,
        embed_dim: int,
        depths: Tuple[int, ...],
        num_heads: Tuple[int, ...],
        window_size: int,
    ) -> None:
        """Validate that config parameters are compatible with the model template.

        Logs warnings if parameters differ significantly from the template defaults,
        as this may indicate a configuration error or suboptimal setup.
        """
        if model_name not in self.KNOWN_SWINV2_CONFIGS:
            logger.warning(
                f"Unknown SwinV2 model '{model_name}'. "
                f"Known models: {list(self.KNOWN_SWINV2_CONFIGS.keys())}. "
                f"Using custom config: embed_dim={embed_dim}, depths={depths}, "
                f"num_heads={num_heads}, window_size={window_size}"
            )
            return

        template_embed, template_depths, template_heads, template_window = \
            self.KNOWN_SWINV2_CONFIGS[model_name]

        mismatches = []

        # Note: We intentionally allow overriding these values - just warn about potential issues
        if embed_dim != template_embed:
            mismatches.append(
                f"embed_dim={embed_dim} (template: {template_embed})"
            )

        if depths != template_depths:
            mismatches.append(
                f"depths={depths} (template: {template_depths})"
            )

        if num_heads != template_heads:
            mismatches.append(
                f"num_heads={num_heads} (template: {template_heads})"
            )

        # Window size can legitimately differ (fine-tuning at different resolutions)
        # but still worth noting
        if window_size != template_window:
            mismatches.append(
                f"window_size={window_size} (template: {template_window})"
            )

        if mismatches:
            logger.warning(
                f"SwinV2 config parameters differ from '{model_name}' template defaults. "
                f"This is expected when training from scratch with a custom architecture. "
                f"Note: timm.create_model() uses the model's built-in architecture, "
                f"custom params in swin_config are for documentation only. "
                f"Differences: {'; '.join(mismatches)}."
            )

    def _warn_if_config_ignored(
        self,
        model_name: str,
        embed_dim: int,
        depths: Tuple[int, ...],
        num_heads: Tuple[int, ...],
        window_size: int,
    ) -> None:
        """Warn if user-provided config values differ from what timm actually uses.

        CRITICAL: timm.create_model() ignores custom architecture parameters like
        embed_dim, depths, num_heads, window_size for SwinV2 models. It always uses
        the architecture defined by the model_name. This method warns users when
        their config values don't match what the model actually uses.
        """
        if model_name not in self.KNOWN_SWINV2_CONFIGS:
            # Unknown model - can't validate, but still warn about potential issues
            if hasattr(self.backbone, 'embed_dim'):
                actual_embed = self.backbone.embed_dim
                if actual_embed != embed_dim:
                    warnings.warn(
                        f"Config embed_dim={embed_dim} is IGNORED by timm. "
                        f"Model '{model_name}' actually uses embed_dim={actual_embed}. "
                        f"Update your config to match or remove custom embed_dim.",
                        UserWarning
                    )
            return

        template_embed, template_depths, template_heads, template_window = \
            self.KNOWN_SWINV2_CONFIGS[model_name]

        ignored_params = []

        if embed_dim != template_embed:
            ignored_params.append(
                f"embed_dim={embed_dim} (actual: {template_embed})"
            )

        if depths != template_depths:
            ignored_params.append(
                f"depths={depths} (actual: {template_depths})"
            )

        if num_heads != template_heads:
            ignored_params.append(
                f"num_heads={num_heads} (actual: {template_heads})"
            )

        if window_size != template_window:
            ignored_params.append(
                f"window_size={window_size} (actual: {template_window})"
            )

        if ignored_params:
            warnings.warn(
                f"CONFIG VALUES IGNORED BY TIMM: {'; '.join(ignored_params)}. "
                f"Model '{model_name}' uses its built-in architecture, not your config values. "
                f"This can cause feature dimension mismatches. "
                f"Either update your config to match the model's actual values, "
                f"or choose a different model_name that matches your desired architecture.",
                UserWarning
            )

    def _validate_image_size_for_swinv2(
        self,
        image_size: int,
        window_size: int,
        num_stages: int,
        initial_patch_size: int = 4,
    ) -> None:
        """Validate image size for SwinV2 window_size requirements.

        SwinV2 uses hierarchical feature maps where each stage halves the spatial resolution.
        For window-based attention to work correctly:
        1. image_size must be divisible by (patch_size * 2^(num_stages-1)) for the final stage
        2. All intermediate feature map sizes must be divisible by window_size

        Args:
            image_size: Input image size (assumes square images)
            window_size: Window size for shifted window attention
            num_stages: Number of hierarchical stages (typically 4)
            initial_patch_size: Initial patch embedding size (4 for all SwinV2 variants)

        Raises:
            ValueError: If image_size is incompatible with the SwinV2 architecture
        """
        # Calculate the total downsampling factor for the final stage
        # SwinV2 downsamples by patch_size initially, then by 2 at each stage transition
        # Final resolution = image_size / (patch_size * 2^(num_stages-1))
        total_downsample = initial_patch_size * (2 ** (num_stages - 1))

        if image_size % total_downsample != 0:
            final_res = image_size / total_downsample
            valid_sizes = [
                s for s in [192, 224, 256, 384, 448, 512, 640, 768, 896, 1024]
                if s % total_downsample == 0
            ]
            raise ValueError(
                f"image_size ({image_size}) must be divisible by {total_downsample} "
                f"(patch_size={initial_patch_size} * 2^{num_stages-1}) for SwinV2 with {num_stages} stages. "
                f"Current final resolution would be {final_res:.2f} (must be integer). "
                f"Suggested valid image sizes: {valid_sizes}"
            )

        # Check window_size divisibility at each stage
        # Stage 0: feature_size = image_size / patch_size
        # Stage i: feature_size = image_size / (patch_size * 2^i)
        invalid_stages = []
        for stage in range(num_stages):
            downsample_at_stage = initial_patch_size * (2 ** stage)
            feature_size = image_size // downsample_at_stage

            if feature_size % window_size != 0:
                invalid_stages.append(
                    f"stage {stage}: feature_size={feature_size}, "
                    f"feature_size % window_size = {feature_size % window_size}"
                )

        if invalid_stages:
            # Calculate valid image sizes that work with this window_size
            # Need: image_size / (patch_size * 2^i) divisible by window_size for all stages
            # Simplest: image_size divisible by patch_size * 2^(num_stages-1) * window_size
            required_multiple = total_downsample * window_size
            valid_sizes = [
                s for s in range(required_multiple, 1025, required_multiple)
                if s <= 1024
            ]

            raise ValueError(
                f"image_size ({image_size}) is incompatible with window_size ({window_size}) "
                f"for SwinV2 with {num_stages} stages. "
                f"Feature map sizes must be divisible by window_size at all stages. "
                f"Invalid stages: {'; '.join(invalid_stages)}. "
                f"Suggested valid image sizes (divisible by {required_multiple}): {valid_sizes}"
            )

        # Check that final stage feature size >= window_size
        # The final stage has the smallest feature map (most downsampled)
        final_feature_size = image_size // total_downsample
        if final_feature_size < window_size:
            raise ValueError(
                f"Final stage feature size ({final_feature_size}) is smaller than window_size ({window_size}). "
                f"SwinV2 requires the feature map at each stage to be at least as large as the window size. "
                f"Either increase image_size (current: {image_size}) or decrease window_size. "
                f"Minimum image_size for window_size={window_size} with {num_stages} stages: "
                f"{window_size * total_downsample}"
            )

    def _determine_feature_dim(
        self,
        embed_dim: int,
        depths: Tuple[int, ...],
        model_name: str,
    ) -> int:
        """DEPRECATED: Always use self.backbone.num_features instead.

        This method previously calculated feature_dim from config values, but that
        approach is unreliable because timm ignores custom config values. The backbone
        always uses its built-in architecture, so we must query it directly.

        Args:
            embed_dim: Ignored (kept for backward compatibility)
            depths: Ignored (kept for backward compatibility)
            model_name: Model name for error messages

        Returns:
            Feature dimension from backbone.num_features

        Raises:
            RuntimeError: If backbone lacks num_features attribute
        """
        warnings.warn(
            "_determine_feature_dim is deprecated. Use self.backbone.num_features directly.",
            DeprecationWarning,
            stacklevel=2
        )

        if not hasattr(self.backbone, 'num_features'):
            raise RuntimeError(
                f"Backbone '{model_name}' does not have num_features attribute. "
                f"Cannot determine feature dimension."
            )

        return self.backbone.num_features

    def _init_weights(self):
        """Initialize classification head with truncated normal."""
        nn.init.trunc_normal_(self.tag_head.weight, std=0.02)
        nn.init.zeros_(self.tag_head.bias)

    def set_onnx_mode(self, enabled: bool = True) -> None:
        """Enable ONNX-compatible mode.

        Note: SwinV2 ONNX export requires Opset 18+ to support the cyclic shift
        operations (torch.roll) used in shifted window attention.

        For ONNX deployment, ensure you export with --opset 18 and --force-swinv2.

        Args:
            enabled: If True, logs a status message.
                    SwinV2 uses the same forward path regardless of this flag
                    as modern PyTorch/ONNX versions handle torch.roll natively.
        """
        self._onnx_mode = enabled
        if enabled:
            logger.info(
                "SwinV2 ONNX mode enabled. Ensure you export with Opset 18+ "
                "to support shifted window attention (torch.roll)."
            )
        logger.info(
            f"SwinV2Tagger ONNX mode set to {enabled}"
        )

    @property
    def config(self):
        """Return model configuration."""
        return self._config

    def _check_numerical_stability(
        self,
        tag_logits: torch.Tensor,
    ) -> None:
        """Check for NaN/Inf in logits and log statistics."""
        tag_has_nan = torch.isnan(tag_logits).any().item()
        tag_has_inf = torch.isinf(tag_logits).any().item()

        if tag_has_nan or tag_has_inf:
            warnings.warn(
                f"[SwinV2] Numerical instability in tag_logits: "
                f"NaN={tag_has_nan}, Inf={tag_has_inf}. "
                f"Stats: min={tag_logits.min():.2f}, max={tag_logits.max():.2f}, "
                f"mean={tag_logits.mean():.2f}, std={tag_logits.std():.2f}"
            )

        clamp_threshold = getattr(self._config, 'logit_clamp_value', 15.0) or 15.0
        tag_needs_clamp = (tag_logits.abs() > clamp_threshold).any().item()
        if tag_needs_clamp:
            num_clamped = (tag_logits.abs() > clamp_threshold).sum().item()
            warnings.warn(
                f"[SwinV2] Clamping {num_clamped} tag logits "
                f"(max abs value: {tag_logits.abs().max():.2f}, threshold: {clamp_threshold})"
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Extract features from backbone
        # With global_pool='', backbone returns (B, H', W', C) spatial features
        features = self.backbone(pixel_values)

        # features shape: (B, H', W', C) where H', W' are reduced spatial dims
        B = features.shape[0]

        # Apply masked global average pooling
        if padding_mask is not None:
            # Ensure mask is in standard format: (B, 1, H, W), True=PAD
            pm = ensure_pixel_padding_mask(padding_mask, mask_semantics='pad')

            # Get final stage token mask using pixel_to_swin_token_masks
            # SwinV2 with initial_patch_size=4 and 4 stages gives H/32 x W/32 final resolution
            #
            # initial_patch_size=4 is fixed for all SwinV2 architectures in timm:
            # SwinV2 uses a PatchEmbed layer with kernel_size=4, stride=4 that converts
            # (B, 3, H, W) -> (B, H/4, W/4, embed_dim). This is part of the architecture spec.
            #
            # num_stages is derived from the backbone's actual layer structure to ensure
            # consistency. We query the backbone directly when possible, falling back to
            # swin_config, then to the default depths tuple (2, 2, 18, 2) which has 4 stages.
            if hasattr(self.backbone, 'layers'):
                # timm's SwinV2 stores stages in self.layers
                num_stages = len(self.backbone.layers)
            elif self._swin_config is not None and hasattr(self._swin_config, 'depths'):
                num_stages = len(self._swin_config.depths)
            else:
                # Fallback: use default depths tuple length (matches default in __init__)
                # Default depths = (2, 2, 18, 2) which has 4 stages
                num_stages = 4
            stage_masks = pixel_to_swin_token_masks(
                pm,
                initial_patch_size=4,  # Fixed for all SwinV2 architectures in timm
                num_stages=num_stages,
                threshold=getattr(self._config, 'token_ignore_threshold', 0.9)
            )
            # Use final stage mask (B, num_tokens) where True=IGNORE
            final_mask = stage_masks[-1]  # (B, H'*W')

            # Reshape features for pooling: (B, H', W', C) -> (B, H'*W', C)
            H_feat, W_feat = features.shape[1], features.shape[2]
            features_flat = features.reshape(B, H_feat * W_feat, -1)  # (B, L, C)

            # Create attention weights: valid tokens get 1, padding gets 0
            # final_mask: True=IGNORE, so we want ~final_mask for valid tokens
            valid_mask = ~final_mask  # (B, L), True=VALID
            valid_mask_float = valid_mask.to(features_flat.dtype)  # (B, L)

            # Masked average pooling: sum(features * mask) / sum(mask)
            # Add small epsilon to avoid division by zero when all tokens are masked
            mask_sum = valid_mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
            features_weighted = features_flat * valid_mask_float.unsqueeze(-1)  # (B, L, C)
            features = features_weighted.sum(dim=1) / mask_sum  # (B, C)
        else:
            # No padding mask: simple global average pooling
            # features: (B, H', W', C) -> (B, C)
            features = features.mean(dim=(1, 2))

        # Classification head (ratings are included as tags)
        tag_logits = self.tag_head(features)

        # Monitor for numerical issues (optional, controlled by config)
        if getattr(self._config, 'check_numerical_stability', False):
            self._check_numerical_stability(tag_logits)

        # Logit clamping for numerical stability
        if hasattr(self._config, 'logit_clamp_value') and self._config.logit_clamp_value:
            clamp_val = self._config.logit_clamp_value
            tag_logits = torch.clamp(tag_logits, min=-clamp_val, max=clamp_val)

        return {
            'tag_logits': tag_logits,
            'logits': tag_logits,  # Alias for backward compatibility
        }


def create_model(config: Optional[VisionTransformerConfig] = None, **kwargs) -> BaseTagger:
    """Create model from configuration.

    Supports both ViT (SimplifiedTagger) and SwinV2 (SwinV2Tagger) architectures.

    Args:
        config: VisionTransformerConfig instance. If provided, config fields in kwargs are ignored.
        **kwargs: Alternative to config - specify config fields as keyword arguments.
                 See VisionTransformerConfig for details on each parameter.
                 Additional kwargs:
                   - architecture_type: 'vit' (default) or 'swinv2'
                   - swin_config: SwinV2 configuration object (required if architecture_type='swinv2')

    Returns:
        BaseTagger instance (SimplifiedTagger or SwinV2Tagger) with the specified configuration.

    Examples:
        # Using config object (ViT architecture)
        config = VisionTransformerConfig(image_size=512, num_tags=10000)
        model = create_model(config=config)

        # Using kwargs (ViT architecture)
        model = create_model(image_size=512, num_tags=10000)

        # SwinV2 architecture
        model = create_model(config=config, architecture_type='swinv2', swin_config=swin_cfg)
    """
    # Extract architecture-specific kwargs before filtering
    architecture = kwargs.pop('architecture_type', 'vit')
    swin_config = kwargs.pop('swin_config', None)

    if config is None:
        # Get the names of the fields in the VisionTransformerConfig dataclass
        config_fields = {f.name for f in fields(VisionTransformerConfig)}

        # Filter kwargs to only include keys that are in the config_fields
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

        # Warn if kwargs contains invalid keys (excluding already-extracted keys)
        invalid_keys = set(kwargs.keys()) - config_fields
        if invalid_keys:
            warnings.warn(
                f"Ignoring unknown configuration parameters: {sorted(invalid_keys)}. "
                f"Valid parameters: {sorted(config_fields)}"
            )

        config = VisionTransformerConfig(**filtered_kwargs)

    # Dispatch based on architecture type
    if architecture == 'swinv2':
        return SwinV2Tagger(config, swin_config)
    else:
        # Default: Original ViT architecture
        return SimplifiedTagger(config)
