#!/usr/bin/env python3
"""
Model Architecture for Anime Image Tagger - Direct Training (modified)
Vision Transformer for anime image tagging
"""

from abc import ABC, abstractmethod
import logging
import math
import warnings
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


def _check_triton_available() -> bool:
    """Check if Triton is available for compiled block mask creation."""
    try:
        import triton
        return True
    except ImportError:
        return False


# Cache the Triton availability check at module load time
_TRITON_AVAILABLE = _check_triton_available()


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
    num_hidden_layers: int = 18
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
    # Logit clamping for inference/export outputs (None to disable)
    # exp(15) ~ 3.3M which is safe for softmax in float16/bfloat16
    # Applied only in eval mode: training logits are left unclamped because
    # clamp has zero gradient outside the bounds (see forward()).
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

    def _get_pos_embed(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Return position embeddings matched to the runtime token grid.

        Fast path (the common case, and the ONNX-export resolution): the runtime
        grid equals the stored square grid, so the stored parameter is returned
        unchanged. Otherwise the 2D patch position embeddings are bicubically
        interpolated to (grid_h, grid_w) — same approach as the checkpoint-resume
        interpolation in training_utils.py — keeping the CLS embedding as-is.
        Without this, inputs at a different resolution would silently add the
        raster-order *prefix* of the stored grid.
        """
        stored_len = self.pos_embed.shape[1]  # 1 (CLS) + stored_grid**2
        stored_grid = int(math.isqrt(stored_len - 1))
        if grid_h == stored_grid and grid_w == stored_grid:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        hidden_size = patch_pos.shape[-1]
        # Reshape to 2D spatial grid, interpolate, flatten back
        patch_pos = patch_pos.reshape(1, stored_grid, stored_grid, hidden_size).permute(0, 3, 1, 2).float()
        patch_pos = F.interpolate(
            patch_pos, size=(grid_h, grid_w), mode='bicubic', align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, hidden_size)
        patch_pos = patch_pos.to(self.pos_embed.dtype)
        return torch.cat([cls_pos, patch_pos], dim=1)

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

        # Diagnostic only: report how many logits exceed the *configured* clamp
        # value (the real clamp happens later in forward() at logit_clamp_value,
        # eval mode only — training logits are left unclamped).
        clamp_threshold = self._config.logit_clamp_value
        if clamp_threshold is not None:
            tag_over_clamp = (tag_logits.abs() > clamp_threshold).any().item()
            if tag_over_clamp:
                num_over = (tag_logits.abs() > clamp_threshold).sum().item()
                warnings.warn(
                    f"{num_over} tag logits exceed logit_clamp_value={clamp_threshold} "
                    f"(clamped downstream in eval mode only; max abs value: {tag_logits.abs().max():.2f})"
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
        # Capture the runtime token grid before flattening (for pos-embed matching)
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2).contiguous()
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Add position embeddings (bicubically interpolated to the runtime grid
        # if it differs from the stored grid; no-op at the trained resolution)
        x = x + self._get_pos_embed(grid_h, grid_w)
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
                # forward_sdpa() passes this boolean mask straight to SDPA (True=ATTEND,
                # False=IGNORE); no additive -inf conversion, so the Flash/efficient backend stays in use.
                # Always apply (no .any() guard) so the mask branch is captured during export.
                # An all-False attn_kpm produces an all-True sdpa_attn_mask (all-attend),
                # which is mathematically identical to no mask.
                sdpa_attn_mask = ~attn_kpm
        else:
            # Normal mode: use flex_attention with BlockMask
            # _create_block_mask handles the inversion internally:
            #   attend_mask = ~key_padding_mask (True=ATTEND)
            #   BlockMask allows attention where attend_mask[q] & attend_mask[kv] is True
            # No .any() guard: it forces a GPU->CPU sync every forward and with
            # letterboxed data the mask is almost always non-empty anyway. An
            # all-False attn_kpm yields an all-attend BlockMask (same result).
            if attn_kpm is not None:
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

        # Clamp logits for inference/export outputs only. During training the
        # clamp is skipped: torch.clamp has zero gradient outside the bounds,
        # which would kill the learning signal exactly for confidently-wrong
        # predictions. The loss (loss_functions.py) computes in fp32 with its
        # own log/exp clamps, so unclamped training logits are numerically safe.
        if self._config.logit_clamp_value is not None and not self.training:
            clamp_val = self._config.logit_clamp_value
            tag_logits = torch.clamp(tag_logits, min=-clamp_val, max=clamp_val)

        return {
            'tag_logits': tag_logits,
            'logits': tag_logits
        }

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


def create_model(config: Optional[VisionTransformerConfig] = None, **kwargs) -> BaseTagger:
    """Create the ViT tagger model (SimplifiedTagger) from configuration.

    Args:
        config: VisionTransformerConfig instance. If provided, config fields in kwargs are ignored.
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
    # Pop the legacy architecture selector so it doesn't reach the config-field filter
    # below. Only the ViT path (SimplifiedTagger) is supported now; the value is ignored.
    kwargs.pop('architecture_type', None)

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

    return SimplifiedTagger(config)
