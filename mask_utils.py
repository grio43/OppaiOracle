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
def ensure_pixel_padding_mask(mask: torch.Tensor) -> Optional[torch.Tensor]:
    """
    (mask) -> (B,1,H,W,bool, True=PAD)
    Accepts (B,H,W) or (B,1,H,W) of {bool,uint8,int,float}.
    Heuristic to flip legacy 'True=valid' masks: if the fraction of True
    exceeds 0.5 on median sample, assume it's a valid-mask and invert.
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

    # Detect legacy semantics (True=valid) and invert once here
    B = mask.shape[0]
    frac_true = mask.flatten(1).float().mean(dim=1)  # per-sample fraction
    # If most samples are mostly True, it's almost surely "valid" -> invert
    # Use torch.where to make this tracer-friendly
    should_invert = torch.median(frac_true) > 0.5
    mask = torch.where(should_invert, ~mask, mask)
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
