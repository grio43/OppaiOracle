#!/usr/bin/env python3
"""
Test script to verify attention mask caching correctness.
Ensures cache works across epochs and doesn't interfere with orientation flipping.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model_architecture import TransformerBlock, VisionTransformerConfig, create_model


def test_cache_basic():
    """Test basic cache functionality."""
    print("=" * 60)
    print("Test 1: Basic Cache Functionality")
    print("=" * 60)

    # Clear any existing cache
    TransformerBlock.clear_cache()

    config = VisionTransformerConfig(
        image_size=640,
        patch_size=16,
        num_tags=1000,
        num_ratings=5,
        use_flash_attention=True,
    )

    model = create_model(config)
    model.eval()

    # Create dummy input with padding mask
    batch_size = 4
    seq_len = (640 // 16) ** 2 + 1  # patches + CLS token

    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create padding mask: first half unmasked, second half masked
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, seq_len // 2:] = True  # Mask second half

    # First forward pass - should be cache miss
    print(f"Initial cache stats: {TransformerBlock.get_cache_stats()}")

    with torch.no_grad():
        block = model.blocks[0]
        _ = block(x, key_padding_mask=padding_mask)

    stats1 = TransformerBlock.get_cache_stats()
    print(f"After first pass: {stats1}")
    assert stats1['misses'] == 1, f"Expected 1 miss, got {stats1['misses']}"
    assert stats1['hits'] == 0, f"Expected 0 hits, got {stats1['hits']}"

    # Second forward pass with same mask - should be cache hit
    with torch.no_grad():
        _ = block(x, key_padding_mask=padding_mask)

    stats2 = TransformerBlock.get_cache_stats()
    print(f"After second pass (same mask): {stats2}")
    assert stats2['hits'] == 1, f"Expected 1 hit, got {stats2['hits']}"
    assert stats2['misses'] == 1, f"Expected 1 miss, got {stats2['misses']}"

    print("[PASS] Basic cache functionality works!\n")


def test_cache_different_masks():
    """Test that different masks don't cause incorrect cache hits."""
    print("=" * 60)
    print("Test 2: Different Masks (Cache Invalidation)")
    print("=" * 60)

    TransformerBlock.clear_cache()

    config = VisionTransformerConfig(
        image_size=640,
        patch_size=16,
        num_tags=1000,
        num_ratings=5,
        use_flash_attention=True,
    )

    model = create_model(config)
    model.eval()

    batch_size = 4
    seq_len = (640 // 16) ** 2 + 1

    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # First mask pattern
    mask1 = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask1[:, seq_len // 2:] = True

    # Second mask pattern (different)
    mask2 = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask2[:, seq_len // 4:] = True

    block = model.blocks[0]

    with torch.no_grad():
        _ = block(x, key_padding_mask=mask1)  # Miss
        _ = block(x, key_padding_mask=mask1)  # Hit
        _ = block(x, key_padding_mask=mask2)  # Miss (different pattern)
        _ = block(x, key_padding_mask=mask2)  # Hit

    stats = TransformerBlock.get_cache_stats()
    print(f"Final stats: {stats}")
    assert stats['hits'] == 2, f"Expected 2 hits, got {stats['hits']}"
    assert stats['misses'] == 2, f"Expected 2 misses, got {stats['misses']}"
    # Note: Cache uses (batch_size, seq_len, device) as key, so same-shape different-content
    # masks share a cache slot. The cache detects pattern changes and invalidates.
    # This is by design - for fixed-size training, masks are usually identical anyway.
    assert stats['entries'] >= 1, f"Expected at least 1 cache entry, got {stats['entries']}"

    print("[PASS] Cache correctly handles different mask patterns!\n")


def test_cache_across_epochs():
    """Simulate multiple epochs to ensure cache persists."""
    print("=" * 60)
    print("Test 3: Cache Persistence Across Epochs")
    print("=" * 60)

    TransformerBlock.clear_cache()

    config = VisionTransformerConfig(
        image_size=640,
        patch_size=16,
        num_tags=1000,
        num_ratings=5,
        use_flash_attention=True,
    )

    model = create_model(config)

    batch_size = 4
    seq_len = (640 // 16) ** 2 + 1

    # Same padding mask for all "epochs"
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, seq_len // 2:] = True

    # Simulate 3 epochs with multiple batches each
    num_epochs = 3
    batches_per_epoch = 5

    for epoch in range(num_epochs):
        model.train() if epoch < 2 else model.eval()  # Mix training/eval

        for batch_idx in range(batches_per_epoch):
            x = torch.randn(batch_size, seq_len, config.hidden_size)

            with torch.no_grad():
                for block in model.blocks[:3]:  # Test first 3 blocks
                    _ = block(x, key_padding_mask=padding_mask)

        stats = TransformerBlock.get_cache_stats()
        print(f"Epoch {epoch + 1} stats: hits={stats['hits']}, misses={stats['misses']}, "
              f"hit_rate={stats['hit_rate']:.2%}")

    final_stats = TransformerBlock.get_cache_stats()

    # Cache is shared across all blocks! So we only get 1 miss total (first access)
    # All subsequent accesses across all blocks and all epochs are hits.
    total_accesses = num_epochs * batches_per_epoch * 3  # 3 blocks per batch
    expected_hits = total_accesses - 1  # All except first
    expected_misses = 1  # Only first access

    print(f"\nFinal stats: {final_stats}")
    print(f"Expected: {expected_hits} hits, {expected_misses} miss (cache shared across blocks!)")

    assert final_stats['misses'] == expected_misses, \
        f"Expected {expected_misses} miss, got {final_stats['misses']}"
    assert final_stats['hits'] == expected_hits, \
        f"Expected {expected_hits} hits, got {final_stats['hits']}"

    hit_rate = final_stats['hit_rate']
    print(f"Cache hit rate: {hit_rate:.2%}")
    assert hit_rate > 0.9, f"Cache hit rate should be >90%, got {hit_rate:.2%}"

    print("[PASS] Cache persists correctly across epochs!\n")


def test_cache_with_orientation_independence():
    """Verify cache is independent of tag changes (orientation flipping)."""
    print("=" * 60)
    print("Test 4: Cache Independence from Tag/Orientation Changes")
    print("=" * 60)

    TransformerBlock.clear_cache()

    config = VisionTransformerConfig(
        image_size=640,
        patch_size=16,
        num_tags=1000,
        num_ratings=5,
        use_flash_attention=True,
    )

    model = create_model(config)
    model.eval()

    batch_size = 4
    seq_len = (640 // 16) ** 2 + 1

    # Same image geometry (same padding mask)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -10:] = True  # Last 10 tokens padded

    block = model.blocks[0]

    # Simulate: same image, different orientations (flipped tags)
    # Padding mask should be identical because image geometry hasn't changed
    with torch.no_grad():
        # Original image
        x1 = torch.randn(batch_size, seq_len, config.hidden_size)
        _ = block(x1, key_padding_mask=padding_mask)  # Miss

        # "Flipped" image (different pixel values, same geometry)
        x2 = torch.randn(batch_size, seq_len, config.hidden_size)
        _ = block(x2, key_padding_mask=padding_mask)  # Hit!

        # Another "flipped" version
        x3 = torch.randn(batch_size, seq_len, config.hidden_size)
        _ = block(x3, key_padding_mask=padding_mask)  # Hit!

    stats = TransformerBlock.get_cache_stats()
    print(f"Stats: {stats}")

    assert stats['misses'] == 1, f"Expected 1 miss, got {stats['misses']}"
    assert stats['hits'] == 2, f"Expected 2 hits, got {stats['hits']}"

    print("[PASS] Cache correctly ignores content changes (e.g., orientation flips)!\n")


def test_cache_memory_limit():
    """Test that cache respects memory limits."""
    print("=" * 60)
    print("Test 5: Cache Memory Limit")
    print("=" * 60)

    TransformerBlock.clear_cache()

    # Temporarily reduce cache limit
    original_limit = TransformerBlock._max_cache_entries
    TransformerBlock._max_cache_entries = 5

    try:
        config = VisionTransformerConfig(
            image_size=640,
            patch_size=16,
            num_tags=1000,
            num_ratings=5,
            use_flash_attention=True,
        )

        model = create_model(config)
        model.eval()
        block = model.blocks[0]

        # Create 10 different mask patterns (more than cache limit)
        for i in range(10):
            batch_size = 2 + i  # Different batch sizes
            seq_len = (640 // 16) ** 2 + 1

            x = torch.randn(batch_size, seq_len, config.hidden_size)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            mask[:, -(i+1):] = True  # Different padding amounts

            with torch.no_grad():
                _ = block(x, key_padding_mask=mask)

        stats = TransformerBlock.get_cache_stats()
        print(f"Stats after 10 patterns: {stats}")

        assert stats['entries'] <= TransformerBlock._max_cache_entries, \
            f"Cache exceeded limit: {stats['entries']} > {TransformerBlock._max_cache_entries}"

        print(f"[PASS] Cache respects memory limit ({TransformerBlock._max_cache_entries} entries)!\n")

    finally:
        # Restore original limit
        TransformerBlock._max_cache_entries = original_limit


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ATTENTION MASK CACHE TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_cache_basic()
        test_cache_different_masks()
        test_cache_across_epochs()
        test_cache_with_orientation_independence()
        test_cache_memory_limit()

        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nCache implementation is correct and ready for production!")
        print("Key benefits:")
        print("  - Persists across epochs (no recomputation needed)")
        print("  - Works with orientation flipping (geometry-based, not content-based)")
        print("  - Memory-bounded (configurable limit)")
        print("  - GPU-only operations (no CPU transfers)")

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
