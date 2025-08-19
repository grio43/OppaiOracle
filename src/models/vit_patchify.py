#!/usr/bin/env python3
"""
Patch embedding utilities for Vision Transformer models.

This module provides simple functions to partition images into non‑overlapping
patches and, optionally, generate padding masks.  When images have been
letterbox‑resized to preserve aspect ratio, the padded regions should not
contribute to the attention mechanism of a Vision Transformer.  The
``compute_padding_mask`` function uses the padding information returned by
``letterbox_resize`` to mark patches that fall entirely within padded areas.

Functions
---------
    patchify(images: Tensor, patch_size: int) -> Tensor
        Convert a batch of images into a batch of patch embeddings.

    compute_padding_mask(pads: Iterable[Tuple[int, int, int, int]],
                         out_sizes: Iterable[Tuple[int, int]],
                         patch_size: int) -> torch.Tensor
        Generate a boolean mask indicating which patches correspond to
        padding for each image in a batch.

Example
-------
>>> from src.data.HDF5_loader import letterbox_resize
>>> image = torch.rand(3, 480, 640)
>>> img_padded, info = letterbox_resize(image, 640, (114,114,114), 16)
>>> patches = patchify(img_padded.unsqueeze(0), 16)
>>> mask = compute_padding_mask([info['pad']], [info['out_size']], 16)
>>> patches.shape, mask.shape
((1, 1600, 768), (1, 1600))
"""

from __future__ import annotations
from typing import Iterable, List, Tuple
import torch

def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert a batch of images into a batch of flattened patches.

    Args:
        images: Tensor of shape (B, C, H, W).
        patch_size: Size of each square patch.  Both H and W must be
            divisible by ``patch_size``.

    Returns:
        Tensor of shape (B, num_patches, C * patch_size * patch_size)
        containing flattened patches.
    """
    if images.dim() != 4:
        raise ValueError(f"images must be 4D (B,C,H,W), got shape {images.shape}")
    B, C, H, W = images.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Image dimensions ({H},{W}) must be divisible by patch_size ({patch_size})")
    # Number of patches along height and width
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    # Use unfold to extract sliding blocks without overlap
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches shape: (B, C, n_patches_h, n_patches_w, patch_size, patch_size)
    # Rearrange to (B, n_patches_h * n_patches_w, C, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(B, n_patches_h * n_patches_w, C * patch_size * patch_size)
    return patches


def compute_padding_mask(
    pads: Iterable[Tuple[int, int, int, int]],
    out_sizes: Iterable[Tuple[int, int]],
    patch_size: int,
) -> torch.Tensor:
    """Generate a padding mask for patchified images.

    Given per‑image padding information (left, top, right, bottom) and the
    output image sizes, this function computes a boolean mask of shape
    (B, num_patches) where ``True`` indicates that a patch lies completely
    within the padded region and should therefore be ignored by attention.

    Args:
        pads: Iterable of tuples ``(pad_left, pad_top, pad_right, pad_bottom)``.
        out_sizes: Iterable of tuples ``(H_out, W_out)`` corresponding to the
            sizes of the padded images.
        patch_size: Size of each patch used during patchification.

    Returns:
        Boolean tensor of shape (B, num_patches) where ``True`` marks padded
        patches.
    """
    pad_list = list(pads)
    size_list = list(out_sizes)
    if len(pad_list) != len(size_list):
        raise ValueError("pads and out_sizes must have the same length")
    batch = len(pad_list)
    masks: List[torch.Tensor] = []
    for i in range(batch):
        pad_left, pad_top, pad_right, pad_bottom = pad_list[i]
        H_out, W_out = size_list[i]
        n_h = H_out // patch_size
        n_w = W_out // patch_size
        mask = torch.zeros(n_h, n_w, dtype=torch.bool)
        # Compute valid region boundaries (in patch coordinates)
        # left boundary: first patch whose left edge is >= pad_left
        valid_start_x = pad_left // patch_size
        # top boundary: first patch whose top edge is >= pad_top
        valid_start_y = pad_top // patch_size
        # right boundary: last valid patch index (exclusive)
        valid_end_x = (W_out - pad_right) // patch_size
        valid_end_y = (H_out - pad_bottom) // patch_size
        # Fill mask with True (padded) then set valid region to False
        mask[:, :] = True
        mask[valid_start_y:valid_end_y, valid_start_x:valid_end_x] = False
        masks.append(mask.reshape(-1))
    return torch.stack(masks, dim=0)


__all__ = ["patchify", "compute_padding_mask"]