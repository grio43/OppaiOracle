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

    # Use tensor-based assertions that work during tracing
    original_dims = mask.dim()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)

    # Trace-friendly validation: use torch assertions instead of Python conditionals
    torch._assert(
        mask.dim() == 4,
        f"padding_mask must be (B,H,W) or (B,1,H,W) after expansion; got {original_dims}D -> {mask.dim()}D with shape {tuple(mask.shape)}"
    )

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
    B, C, H, W = mask_b1hw.shape

    # Use trace-friendly torch assertions instead of Python conditionals
    torch._assert(
        mask_b1hw.dim() == 4,
        f"Expected 4D tensor (B,1,H,W), got {mask_b1hw.dim()}D with shape {tuple(mask_b1hw.shape)}"
    )
    torch._assert(
        C == 1,
        f"Expected channel dimension of 1, got {C} with shape {tuple(mask_b1hw.shape)}"
    )
    torch._assert(
        mask_b1hw.dtype == torch.bool,
        f"mask_b1hw must be bool (True=PAD), got dtype {mask_b1hw.dtype}"
    )
    torch._assert(
        H % patch == 0,
        f"Image height {H} must be divisible by patch size {patch}"
    )
    torch._assert(
        W % patch == 0,
        f"Image width {W} must be divisible by patch size {patch}"
    )

    # pad_fraction per patch
    pad_frac = F.avg_pool2d(mask_b1hw.float(), kernel_size=patch, stride=patch)  # (B,1,GH,GW)
    ignore = (pad_frac.squeeze(1) >= threshold)                                   # (B,GH,GW)
    return ignore.flatten(1)                                                      # (B,L)
