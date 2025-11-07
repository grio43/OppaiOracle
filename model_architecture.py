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


def _check_flash_attention_available() -> bool:
    """Check if flash attention (SDPA) is available and properly supported.

    Returns:
        True if scaled_dot_product_attention is available and the PyTorch version
        is recent enough to support it reliably.
    """
    if not hasattr(F, 'scaled_dot_product_attention'):
        return False

    # Check PyTorch version - SDPA was added in 2.0, stabilized in 2.1
    try:
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    except (ValueError, AttributeError):
        # Cannot parse version, assume not supported
        return False

    if torch_version < (2, 0):
        return False

    # Warn if using older version with potential issues
    if torch_version < (2, 1):
        warnings.warn(
            f"PyTorch {torch.__version__} has scaled_dot_product_attention "
            "but it may not be fully stable. Consider upgrading to 2.1+."
        )

    return True


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
    # Enable numerical stability checking (for debugging only)
    check_numerical_stability: bool = False
    # Logit clamping for numerical stability (None to disable)
    # exp(15) ≈ 3.3M which is safe for softmax in float16/bfloat16
    logit_clamp_value: Optional[float] = 15.0

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
        self.use_flash = config.use_flash_attention and _check_flash_attention_available()

        if config.use_flash_attention and not self.use_flash:
            warnings.warn(
                "Flash attention was requested but is not available. "
                "Falling back to standard attention. "
                f"PyTorch version: {torch.__version__}"
            )
        
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
            
            # Build attention mask with SDPA semantics:
            # PyTorch SDPA expects boolean attn_mask where True = KEEP (participate),
            # False = MASK (disallow attention). Our key_padding_mask uses True = IGNORE.
            # Invert it for SDPA.
            attn_mask = None
            if key_padding_mask is not None:
                # If any sample masks all positions, fail fast
                # Note: During tracing, this check is skipped to avoid control flow issues.
                # The model can still handle this case since CLS token is never masked.
                if not torch.jit.is_tracing():
                    if key_padding_mask.all(dim=1).any().item():
                        raise RuntimeError("key_padding_mask masks all keys for at least one sample.")
                # Invert for SDPA: True=keep, False=mask
                attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)

            # Use scaled_dot_product_attention with flash backend when available
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
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

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,  # (B,H,W) or (B,1,H,W), auto-detected semantics
    ) -> Dict[str, torch.Tensor]:
        B = pixel_values.shape[0]
        # Patch embedding (force fp32 for numerical stability under AMP)
        if pixel_values.dtype in (torch.float16, torch.bfloat16):
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

        if attn_kpm is not None:
            # If any sample masks all positions, fail fast (skip inside tracing)
            if not torch.jit.is_tracing():
                if attn_kpm.all(dim=1).any().item():
                    raise RuntimeError("attn_kpm masks all keys for at least one sample.")

        # Transformer blocks with optional gradient checkpointing
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                # use_reentrant=True is more robust with AMP, though it uses more memory.
                # This is to fix the CheckpointError where recomputed values have different metadata.
                if attn_kpm is not None:
                    # checkpoint requires tensor args; pass mask explicitly
                    # Lambda creates new function object each call, but overhead is negligible
                    # compared to the checkpoint recomputation cost
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
        config = VisionTransformerConfig(image_size=640, num_tags=10000)
        model = create_model(config=config)

        # Using kwargs
        model = create_model(image_size=640, num_tags=10000)
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
