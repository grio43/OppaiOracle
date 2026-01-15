"""
Cache warmup module for OppaiOracle training pipeline.

This module provides GPU-accelerated and CPU multi-process cache warmup
functionality for pre-populating the sidecar cache before training.

Usage:
    # As a script:
    python -m cache_warmup --config configs/unified_config.yaml --use-gpu

    # Or directly:
    python cache_warmup/l2_cache_warmup.py --config configs/unified_config.yaml --use-gpu

    # Programmatic:
    from cache_warmup import CacheWarmup, CacheWarmupConfig, GPUBatchProcessor
"""

from cache_warmup.l2_cache_warmup import (
    CacheWarmup,
    CacheWarmupConfig,
    main,
)
from cache_warmup.gpu_batch_processor import (
    GPUBatchProcessor,
    GPUBatchPreprocessor,
    VRAMMonitor,
    DynamicBatchSizer,
    AsyncImagePreloader,
)

__all__ = [
    # Main warmup classes
    "CacheWarmup",
    "CacheWarmupConfig",
    "main",
    # GPU processing
    "GPUBatchProcessor",
    "GPUBatchPreprocessor",
    "VRAMMonitor",
    "DynamicBatchSizer",
    "AsyncImagePreloader",
]
