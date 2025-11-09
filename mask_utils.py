#!/usr/bin/env python3
"""
Mask migration shims.
Standardize pixel padding masks to (B,1,H,W) bool with True=PAD,
and pool to token-level ignore masks (B,L) with True=IGNORE.
"""
from typing import Optional
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

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)
    if mask.dim() != 4:
        raise ValueError(f"padding_mask must be (B,H,W) or (B,1,H,W); got {tuple(mask.shape)}")

    # Binarize + cast
    if mask.dtype is not torch.bool:
        mask = (mask > 0.5)

    # Apply semantics
    if mask_semantics == 'valid':
        # Invert: True=valid -> True=pad
        mask = ~mask
    elif mask_semantics == 'pad':
        # Already correct semantics
        pass
    elif mask_semantics == 'auto':
        # Legacy heuristic (unreliable, deprecated)
        frac_true = mask.flatten(1).float().mean(dim=1)
        should_invert = torch.median(frac_true) > 0.5
        mask = torch.where(should_invert, ~mask, mask)
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
    # These checks are not tracer-friendly, so we disable them during tracing.
    if not torch.jit.is_tracing():
        if mask_b1hw.dim() != 4 or mask_b1hw.size(1) != 1:
            raise ValueError(f"Expected (B,1,H,W) bool, got {tuple(mask_b1hw.shape)}")
        if mask_b1hw.dtype is not torch.bool:
            raise ValueError("mask_b1hw must be bool with True=PAD")

        B, _, H, W = mask_b1hw.shape
        if H % patch != 0 or W % patch != 0:
            raise AssertionError(
                f"Image size (H={H},W={W}) must be divisible by patch={patch} before pooling."
            )
    # pad_fraction per patch
    pad_frac = F.avg_pool2d(mask_b1hw.float(), kernel_size=patch, stride=patch)  # (B,1,GH,GW)
    ignore = (pad_frac.squeeze(1) >= threshold)                                   # (B,GH,GW)
    return ignore.flatten(1)                                                      # (B,L)
