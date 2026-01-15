"""
VRAM Profiling Tools for OppaiOracle

This module provides utilities for measuring VRAM usage across
different model configurations (batch sizes, layer counts, checkpointing).

Usage:
    python -m profiling.vram_profiler --quick
    python -m profiling.vram_profiler --batch-sizes 16,24,32,36 --layers 24,28
"""

from .vram_profiler import VRAMProfiler, ProfileResult

__all__ = ['VRAMProfiler', 'ProfileResult']
