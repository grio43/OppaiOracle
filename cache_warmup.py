#!/usr/bin/env python3
"""
Optional cache warmup utilities for attention mask cache.

Use this to pre-populate the attention mask cache before training starts,
eliminating cold-start cache misses.
"""

import logging
from typing import Optional, List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def warmup_attention_cache(
    model: torch.nn.Module,
    dataloaders: List[DataLoader],
    num_batches_per_loader: int = 10,
    device: Optional[torch.device] = None,
    show_progress: bool = True
) -> dict:
    """
    Warm the attention mask cache by sampling batches from dataloaders.

    This pre-computes and caches attention masks for different batch sizes
    and sequence lengths, eliminating cache misses during actual training.

    Args:
        model: The model containing TransformerBlocks with mask cache
        dataloaders: List of dataloaders to sample from (train, val, etc.)
        num_batches_per_loader: Number of batches to sample per loader
        device: Device to use (defaults to model's device)
        show_progress: Show progress bar

    Returns:
        dict: Warmup statistics including patterns found and cache entries created

    Example:
        >>> model = create_model(config)
        >>> model.to('cuda')
        >>> warmup_attention_cache(
        ...     model,
        ...     [train_loader, val_loader],
        ...     num_batches_per_loader=5
        ... )
        Warming attention cache...
        Found 2 unique mask patterns
        Cache entries: 2
        Time: 1.2s
    """
    from model_architecture import TransformerBlock

    if device is None:
        device = next(model.parameters()).device

    # Track unique patterns found
    patterns_seen = set()
    initial_entries = len(TransformerBlock._mask_cache)

    model.eval()

    total_batches = len(dataloaders) * num_batches_per_loader
    iterator = range(total_batches)

    if show_progress:
        iterator = tqdm(iterator, desc="Warming attention cache", unit="batch")

    with torch.no_grad():
        for loader_idx, dataloader in enumerate(dataloaders):
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches_per_loader:
                    break

                # Extract padding mask (if present)
                padding_mask = batch.get('padding_mask')

                if padding_mask is None:
                    # No padding masks in this dataset - cache not needed
                    continue

                padding_mask = padding_mask.to(device, non_blocking=True)

                # Record pattern signature
                pattern_key = (
                    padding_mask.shape[0],  # batch_size
                    padding_mask.shape[1],  # seq_len
                    str(padding_mask.device)  # device
                )
                patterns_seen.add(pattern_key)

                # Trigger cache population by computing attention mask
                # We don't need to run full model - just trigger mask computation
                # in first transformer block
                try:
                    # Create dummy input with correct sequence length
                    batch_size, seq_len = padding_mask.shape
                    hidden_size = model.config.hidden_size if hasattr(model, 'config') else 1280

                    dummy_x = torch.zeros(
                        batch_size, seq_len, hidden_size,
                        device=device,
                        dtype=next(model.parameters()).dtype
                    )

                    # Forward through first block to populate cache
                    _ = model.blocks[0](dummy_x, key_padding_mask=padding_mask)

                except Exception as e:
                    logger.warning(f"Cache warmup failed for batch {batch_idx}: {e}")
                    continue

                if show_progress:
                    iterator.update(1)

    final_entries = len(TransformerBlock._mask_cache)
    new_entries = final_entries - initial_entries

    stats = {
        'unique_patterns': len(patterns_seen),
        'cache_entries_before': initial_entries,
        'cache_entries_after': final_entries,
        'new_entries': new_entries,
        'patterns': list(patterns_seen),
    }

    logger.info(
        f"Cache warmup complete: "
        f"{len(patterns_seen)} unique patterns found, "
        f"{new_entries} cache entries created"
    )

    return stats


def estimate_cache_coverage(
    dataloaders: List[DataLoader],
    num_batches_per_loader: int = 100,
    show_progress: bool = True
) -> dict:
    """
    Estimate cache coverage by scanning dataloaders for unique mask patterns.

    This doesn't populate the cache, but tells you how many unique patterns
    exist in your dataset (useful for sizing _max_cache_entries).

    Args:
        dataloaders: List of dataloaders to scan
        num_batches_per_loader: Number of batches to sample
        show_progress: Show progress bar

    Returns:
        dict: Statistics about mask pattern diversity

    Example:
        >>> stats = estimate_cache_coverage([train_loader, val_loader])
        >>> print(f"Need cache size of at least {stats['unique_patterns']} entries")
    """
    patterns_seen = set()
    pattern_frequencies = {}

    total_batches = len(dataloaders) * num_batches_per_loader
    iterator = range(total_batches)

    if show_progress:
        iterator = tqdm(iterator, desc="Analyzing mask patterns", unit="batch")

    for loader_idx, dataloader in enumerate(dataloaders):
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches_per_loader:
                break

            padding_mask = batch.get('padding_mask')

            if padding_mask is None:
                continue

            # Pattern signature
            pattern_key = (
                padding_mask.shape[0],  # batch_size
                padding_mask.shape[1],  # seq_len
            )

            patterns_seen.add(pattern_key)
            pattern_frequencies[pattern_key] = pattern_frequencies.get(pattern_key, 0) + 1

            if show_progress:
                iterator.update(1)

    # Calculate statistics
    total_batches_seen = sum(pattern_frequencies.values())
    most_common = max(pattern_frequencies.items(), key=lambda x: x[1]) if pattern_frequencies else None

    stats = {
        'unique_patterns': len(patterns_seen),
        'total_batches_sampled': total_batches_seen,
        'patterns': list(patterns_seen),
        'pattern_frequencies': pattern_frequencies,
        'most_common_pattern': most_common[0] if most_common else None,
        'most_common_frequency': most_common[1] if most_common else 0,
        'most_common_percentage': (most_common[1] / total_batches_seen * 100) if most_common and total_batches_seen > 0 else 0,
    }

    logger.info(
        f"Pattern analysis: "
        f"{stats['unique_patterns']} unique patterns across {total_batches_seen} batches. "
        f"Most common pattern appears {stats['most_common_percentage']:.1f}% of the time."
    )

    return stats
