"""Model utilities for the anime tagger.

This package exposes helper functions for patchification and optional
padding masks used by Vision Transformer models.  Additional model
definitions can be added here in the future.
"""

from .vit_patchify import patchify, compute_padding_mask  # noqa: F401

__all__ = ["patchify", "compute_padding_mask"]