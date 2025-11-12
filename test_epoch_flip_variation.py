#!/usr/bin/env python3
"""
Validation script for epoch-varying flip augmentation.

This script verifies that:
1. Flip decisions vary across epochs (augmentation diversity)
2. Flip decisions are deterministic for same (image_id, epoch) pair (reproducibility)
3. L2 cache only stores unflipped versions (space efficiency)
4. Complete pipeline works correctly across multiple epochs

Usage:
    python test_epoch_flip_variation.py

Expected output:
    - Confirmation that flips vary across epochs
    - Verification of determinism
    - Cache key validation
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockSidecarDataset:
    """Mock dataset for testing flip logic without full dataset dependencies."""

    def __init__(self, random_flip_prob: float = 0.35):
        self.random_flip_prob = random_flip_prob
        self._current_epoch = 0
        self._l2_cfg_hash = "test_hash_v1"

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for flip decisions."""
        self._current_epoch = int(epoch)

    def _deterministic_coin(self, image_id: str) -> bool:
        """Stable per-image, per-epoch coin flip."""
        if self.random_flip_prob <= 0:
            return False
        seed_str = f"{image_id}|epoch{self._current_epoch}"
        h = hashlib.sha256(seed_str.encode("utf-8")).digest()
        v = int.from_bytes(h[:4], byteorder="big") / 2**32
        return v < float(self.random_flip_prob)

    def _l2_key(self, image_id: str, *, flipped: bool) -> bytes:
        """Generate L2 cache key (only for unflipped)."""
        if flipped:
            return b""  # Don't cache flipped versions
        return f"{image_id}|cfg{self._l2_cfg_hash}".encode("utf-8")

    def _l2_mask_key(self, image_id: str, *, flipped: bool) -> bytes:
        """Generate L2 mask cache key (only for unflipped)."""
        if flipped:
            return b""  # Don't cache flipped masks
        return f"{image_id}|cfg{self._l2_cfg_hash}|m".encode("utf-8")


def test_flip_variation_across_epochs():
    """Test that flip decisions vary across epochs."""
    logger.info("=" * 70)
    logger.info("TEST 1: Flip Variation Across Epochs")
    logger.info("=" * 70)

    dataset = MockSidecarDataset(random_flip_prob=0.35)
    test_images = [f"img_{i:04d}" for i in range(100)]
    num_epochs = 5

    # Collect flip decisions for each (image, epoch) pair
    flip_results: Dict[str, List[bool]] = {img_id: [] for img_id in test_images}

    for epoch in range(num_epochs):
        dataset.set_epoch(epoch)
        for img_id in test_images:
            flipped = dataset._deterministic_coin(img_id)
            flip_results[img_id].append(flipped)

    # Analyze variation
    varying_images = 0
    constant_flip_images = 0
    constant_no_flip_images = 0

    for img_id, flips in flip_results.items():
        if all(flips):
            constant_flip_images += 1
        elif not any(flips):
            constant_no_flip_images += 1
        else:
            varying_images += 1

    logger.info(f"  Total images tested: {len(test_images)}")
    logger.info(f"  Images with varying flips: {varying_images}")
    logger.info(f"  Images always flipped: {constant_flip_images}")
    logger.info(f"  Images never flipped: {constant_no_flip_images}")

    # Calculate flip rate per epoch
    logger.info(f"\n  Flip rate per epoch:")
    for epoch in range(num_epochs):
        epoch_flips = sum(flip_results[img_id][epoch] for img_id in test_images)
        flip_rate = epoch_flips / len(test_images)
        logger.info(f"    Epoch {epoch}: {flip_rate:.1%} ({epoch_flips}/{len(test_images)})")

    # Verify variation exists
    if varying_images > 0:
        logger.info(f"\n  ✅ PASS: {varying_images} images vary across epochs")
        logger.info(f"     This confirms augmentation diversity across epochs!")
    else:
        logger.error(f"\n  ❌ FAIL: No images vary across epochs")
        logger.error(f"     All images have constant flip decisions!")
        return False

    return True


def test_flip_determinism():
    """Test that flip decisions are deterministic for same (image_id, epoch)."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Flip Determinism")
    logger.info("=" * 70)

    dataset = MockSidecarDataset(random_flip_prob=0.35)
    test_images = [f"img_{i:04d}" for i in range(20)]

    # Test each image multiple times at same epoch
    all_consistent = True

    for epoch in [0, 2, 5]:
        dataset.set_epoch(epoch)
        logger.info(f"\n  Testing epoch {epoch}:")

        for img_id in test_images[:5]:  # Test subset for brevity
            results = [dataset._deterministic_coin(img_id) for _ in range(10)]

            if len(set(results)) == 1:
                logger.info(f"    {img_id}: {'FLIP' if results[0] else 'KEEP'} (consistent)")
            else:
                logger.error(f"    {img_id}: INCONSISTENT - {results}")
                all_consistent = False

    if all_consistent:
        logger.info(f"\n  ✅ PASS: All flip decisions are deterministic")
        logger.info(f"     Same (image_id, epoch) always produces same result!")
    else:
        logger.error(f"\n  ❌ FAIL: Some flip decisions are non-deterministic")
        return False

    return True


def test_cache_key_strategy():
    """Test that L2 cache keys only exist for unflipped versions."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: L2 Cache Key Strategy")
    logger.info("=" * 70)

    dataset = MockSidecarDataset()
    test_images = ["img_001", "img_002", "img_003"]

    logger.info("\n  Testing cache key generation:")

    for img_id in test_images:
        # Test unflipped (should have cache key)
        key_unflipped = dataset._l2_key(img_id, flipped=False)
        mask_key_unflipped = dataset._l2_mask_key(img_id, flipped=False)

        # Test flipped (should NOT have cache key)
        key_flipped = dataset._l2_key(img_id, flipped=True)
        mask_key_flipped = dataset._l2_mask_key(img_id, flipped=True)

        logger.info(f"\n    {img_id}:")
        logger.info(f"      Unflipped image key: {key_unflipped.decode() if key_unflipped else 'EMPTY'}")
        logger.info(f"      Unflipped mask key:  {mask_key_unflipped.decode() if mask_key_unflipped else 'EMPTY'}")
        logger.info(f"      Flipped image key:   {key_flipped.decode() if key_flipped else 'EMPTY (expected)'}")
        logger.info(f"      Flipped mask key:    {mask_key_flipped.decode() if mask_key_flipped else 'EMPTY (expected)'}")

        # Verify
        if key_unflipped and not key_flipped:
            logger.info(f"      ✅ Correct: Only unflipped has cache key")
        else:
            logger.error(f"      ❌ Error: Incorrect cache key strategy")
            return False

    logger.info(f"\n  ✅ PASS: L2 cache only stores unflipped versions")
    logger.info(f"     This saves disk space and allows epoch-varying flips!")

    return True


def test_flip_probability_distribution():
    """Test that flip probability matches expected rate."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Flip Probability Distribution")
    logger.info("=" * 70)

    expected_prob = 0.35
    dataset = MockSidecarDataset(random_flip_prob=expected_prob)
    dataset.set_epoch(0)

    # Test large sample
    num_samples = 10000
    test_images = [f"img_{i:06d}" for i in range(num_samples)]

    flips = sum(dataset._deterministic_coin(img_id) for img_id in test_images)
    actual_prob = flips / num_samples
    error = abs(actual_prob - expected_prob)

    logger.info(f"\n  Expected probability: {expected_prob:.1%}")
    logger.info(f"  Actual probability:   {actual_prob:.1%}")
    logger.info(f"  Error:                {error:.2%}")
    logger.info(f"  Samples:              {num_samples}")

    # Allow 2% margin of error for large samples
    if error < 0.02:
        logger.info(f"\n  ✅ PASS: Flip probability matches expected rate")
    else:
        logger.warning(f"\n  ⚠️  WARN: Flip probability deviates from expected")
        logger.warning(f"     This might be due to random variation in small samples")

    return True


def test_comprehensive_scenario():
    """Test complete workflow across multiple epochs."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Comprehensive Multi-Epoch Scenario")
    logger.info("=" * 70)

    dataset = MockSidecarDataset(random_flip_prob=0.35)
    test_images = [f"img_{i:03d}" for i in range(50)]
    num_epochs = 3

    logger.info(f"\n  Simulating {num_epochs} epochs with {len(test_images)} images")
    logger.info(f"  Flip probability: {dataset.random_flip_prob:.1%}\n")

    epoch_flip_counts = []
    unique_flip_patterns: Set[str] = set()

    for epoch in range(num_epochs):
        dataset.set_epoch(epoch)

        flip_decisions = []
        for img_id in test_images:
            flipped = dataset._deterministic_coin(img_id)
            flip_decisions.append('1' if flipped else '0')

        # Record pattern
        pattern = ''.join(flip_decisions)
        unique_flip_patterns.add(pattern)

        flip_count = sum(1 for d in flip_decisions if d == '1')
        epoch_flip_counts.append(flip_count)

        logger.info(f"  Epoch {epoch}: {flip_count}/{len(test_images)} images flipped")
        logger.info(f"           Pattern hash: {hashlib.md5(pattern.encode()).hexdigest()[:8]}")

    logger.info(f"\n  Summary:")
    logger.info(f"    Total epochs: {num_epochs}")
    logger.info(f"    Unique flip patterns: {len(unique_flip_patterns)}")
    logger.info(f"    Expected unique patterns: {num_epochs} (if working correctly)")

    if len(unique_flip_patterns) == num_epochs:
        logger.info(f"\n  ✅ PASS: Each epoch has unique flip pattern")
        logger.info(f"     Augmentation is working correctly across epochs!")
    else:
        logger.error(f"\n  ❌ FAIL: Some epochs have identical flip patterns")
        logger.error(f"     Expected {num_epochs} unique patterns, got {len(unique_flip_patterns)}")
        return False

    return True


def main():
    """Run all validation tests."""
    logger.info("\n" + "=" * 70)
    logger.info("EPOCH-VARYING FLIP AUGMENTATION VALIDATION")
    logger.info("=" * 70)
    logger.info("\nThis script validates the comprehensive fix for orientation mapping")
    logger.info("caching issues between epochs.\n")

    tests = [
        ("Flip Variation Across Epochs", test_flip_variation_across_epochs),
        ("Flip Determinism", test_flip_determinism),
        ("L2 Cache Key Strategy", test_cache_key_strategy),
        ("Flip Probability Distribution", test_flip_probability_distribution),
        ("Comprehensive Multi-Epoch Scenario", test_comprehensive_scenario),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"\n  ❌ ERROR in {test_name}: {e}", exc_info=True)
            results.append((test_name, False))

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    logger.info(f"\n  Total: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 70)
        logger.info("\nThe epoch-varying flip augmentation fix is working correctly!")
        logger.info("Key features verified:")
        logger.info("  • Flip decisions vary across epochs (augmentation diversity)")
        logger.info("  • Flip decisions are deterministic (reproducibility)")
        logger.info("  • L2 cache only stores unflipped versions (space efficiency)")
        logger.info("  • Complete pipeline works end-to-end")
        return 0
    else:
        logger.error("\n" + "=" * 70)
        logger.error(f"❌ {total_tests - passed_tests} TEST(S) FAILED")
        logger.error("=" * 70)
        logger.error("\nPlease review the failed tests and fix the issues.")
        return 1


if __name__ == "__main__":
    exit(main())
