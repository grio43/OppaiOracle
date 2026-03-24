#!/usr/bin/env python3
"""
Mask migration shims.
Standardize pixel padding masks to (B,1,H,W) bool with True=PAD,
and pool to token-level ignore masks (B,L) with True=IGNORE.
"""
from typing import List, Optional
import torch
import torch.nn.functional as F

@torch.no_grad()
def ensure_pixel_padding_mask(
    mask: torch.Tensor,
    mask_semantics: str = 'auto'
) -> Optional[torch.Tensor]:
    """
    (mask) -> (B,1,H,W,bool, True=PAD)
    Accepts (B,H,W) or (B,1,H,W) of {bool,uint8,int,float}.

    Args:
        mask: Input mask tensor
        mask_semantics: Interpretation of True values:
            - 'auto': Heuristic detection (legacy, unreliable)
            - 'pad': True means padding (no inversion needed)
            - 'valid': True means valid pixels (will be inverted)

    Returns:
        Standardized mask with True=PAD semantics
    """
    if mask is None:
        return None

    # Use tensor-based assertions that work during tracing
    original_dims = mask.dim()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)

    # Trace-friendly validation
    if not torch.jit.is_tracing():
        assert mask.dim() == 4, \
            f"padding_mask must be (B,H,W) or (B,1,H,W) after expansion; got {original_dims}D -> {mask.dim()}D with shape {tuple(mask.shape)}"

    # Binarize + cast
    if mask.dtype != torch.bool:
        mask = (mask > 0.5)

    # Apply semantics
    if mask_semantics == 'valid':
        # Invert: True=valid -> True=pad
        mask = ~mask
    elif mask_semantics == 'pad':
        # Already correct semantics
        pass
    elif mask_semantics == 'auto':
        # Legacy heuristic (unreliable, deprecated) - per-sample decision
        frac_true = mask.flatten(2).to(torch.bfloat16).mean(dim=2)  # (B, 1)
        should_invert = frac_true > 0.5  # (B, 1)
        mask = torch.where(should_invert.unsqueeze(-1).unsqueeze(-1), ~mask, mask)
        # Log warning in non-tracing mode
        if not torch.jit.is_tracing():
            import logging
            logging.getLogger(__name__).warning(
                "Using 'auto' mask semantics detection is unreliable. "
                "Please specify mask_semantics='pad' or 'valid' explicitly."
            )
    else:
        raise ValueError(f"Invalid mask_semantics: {mask_semantics}. Use 'auto', 'pad', or 'valid'.")

    return mask

@torch.no_grad()
def pixel_to_token_ignore(mask_b1hw: torch.Tensor, patch: int, threshold: float = 0.9) -> torch.Tensor:
    """
    (B,1,H,W,bool True=PAD) -> (B,L,bool True=IGNORE)
    Pools pad-fraction per patch via avg_pool2d(kernel=stride=patch) and marks
    tokens IGNORE when pad_fraction >= threshold.
    """
    B, C, H, W = mask_b1hw.shape

    # Trace-friendly validation
    if not torch.jit.is_tracing():
        assert mask_b1hw.dim() == 4, \
            f"Expected 4D tensor (B,1,H,W), got {mask_b1hw.dim()}D with shape {tuple(mask_b1hw.shape)}"
        assert C == 1, \
            f"Expected channel dimension of 1, got {C} with shape {tuple(mask_b1hw.shape)}"
        assert mask_b1hw.dtype == torch.bool, \
            f"mask_b1hw must be bool (True=PAD), got dtype {mask_b1hw.dtype}"
        assert H % patch == 0, \
            f"Image height {H} must be divisible by patch size {patch}"
        assert W % patch == 0, \
            f"Image width {W} must be divisible by patch size {patch}"

    # pad_fraction per patch
    pad_frac = F.avg_pool2d(mask_b1hw.to(torch.bfloat16), kernel_size=patch, stride=patch)  # (B,1,GH,GW)
    ignore = (pad_frac.squeeze(1) >= threshold)                                   # (B,GH,GW)
    return ignore.flatten(1)                                                      # (B,L)

@torch.no_grad()
def pixel_to_swin_token_masks(
    mask_b1hw: torch.Tensor,
    initial_patch_size: int = 4,
    num_stages: int = 4,
    threshold: float = 0.9
) -> List[torch.Tensor]:
    """Create token-level ignore masks for each SwinV2 stage.

    SwinV2 has hierarchical structure with patch merging between stages:
    - Stage 1: patch_size=4, resolution H/4 x W/4
    - Stage 2: resolution H/8 x W/8 (2x2 merge)
    - Stage 3: resolution H/16 x W/16
    - Stage 4: resolution H/32 x W/32

    Args:
        mask_b1hw: (B, 1, H, W) bool, True=PAD
        initial_patch_size: Initial patch embedding size (typically 4 for Swin)
        num_stages: Number of hierarchical stages (typically 4)
        threshold: Fraction of padding to consider token as ignore

    Returns:
        List of (B, num_tokens_at_stage) bool tensors, True=IGNORE
    """
    B, C, H, W = mask_b1hw.shape

    # Trace-friendly validation
    if not torch.jit.is_tracing():
        assert mask_b1hw.dim() == 4, \
            f"Expected 4D tensor (B,1,H,W), got {mask_b1hw.dim()}D with shape {tuple(mask_b1hw.shape)}"
        assert C == 1, \
            f"Expected channel dimension of 1, got {C} with shape {tuple(mask_b1hw.shape)}"
        assert mask_b1hw.dtype == torch.bool, \
            f"mask_b1hw must be bool (True=PAD), got dtype {mask_b1hw.dtype}"

    stage_masks: List[torch.Tensor] = []

    # Convert to float for pooling operations
    mask_float = mask_b1hw.to(torch.bfloat16)

    for stage_idx in range(num_stages):
        # Calculate effective downscale factor for this stage
        # Stage 0: initial_patch_size (e.g., 4 -> H/4 x W/4)
        # Stage 1: initial_patch_size * 2 (e.g., 8 -> H/8 x W/8)
        # Stage 2: initial_patch_size * 4 (e.g., 16 -> H/16 x W/16)
        # Stage 3: initial_patch_size * 8 (e.g., 32 -> H/32 x W/32)
        downscale = initial_patch_size * (2 ** stage_idx)

        # Trace-friendly validation
        if not torch.jit.is_tracing():
            assert H % downscale == 0, \
                f"Image height {H} must be divisible by downscale factor {downscale} at stage {stage_idx}"
            assert W % downscale == 0, \
                f"Image width {W} must be divisible by downscale factor {downscale} at stage {stage_idx}"

        # Pool pixel mask to token resolution for this stage
        # Using avg_pool2d to get fraction of padding in each token region
        pad_frac = F.avg_pool2d(mask_float, kernel_size=downscale, stride=downscale)  # (B,1,GH,GW)

        # Mark token as IGNORE if padding fraction >= threshold
        ignore = (pad_frac.squeeze(1) >= threshold)  # (B, GH, GW)
        stage_masks.append(ignore.flatten(1))        # (B, num_tokens_at_stage)

    return stage_masks
