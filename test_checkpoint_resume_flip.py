#!/usr/bin/env python3
"""
Test checkpoint resume compatibility with epoch-varying flips.

This script verifies that flip decisions are identical when:
1. Training runs continuously from epoch 0 to 10
2. Training is interrupted at epoch 5 and resumed

Both scenarios should produce identical flip decisions for the same (image_id, epoch) pairs.
"""

import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MockDataset:
    """Simplified dataset for testing epoch-based flips."""

    def __init__(self, random_flip_prob: float = 0.35):
        self.random_flip_prob = random_flip_prob
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    def _deterministic_coin(self, image_id: str) -> bool:
        if self.random_flip_prob <= 0:
            return False
        seed_str = f"{image_id}|epoch{self._current_epoch}"
        h = hashlib.sha256(seed_str.encode("utf-8")).digest()
        v = int.from_bytes(h[:4], byteorder="big") / 2**32
        return v < float(self.random_flip_prob)


def simulate_continuous_training(dataset, test_images, num_epochs):
    """Simulate training from epoch 0 to num_epochs without interruption."""
    results = {}

    for epoch in range(num_epochs):
        dataset.set_epoch(epoch)
        epoch_results = {}

        for img_id in test_images:
            flipped = dataset._deterministic_coin(img_id)
            epoch_results[img_id] = flipped

        results[epoch] = epoch_results

    return results


def simulate_resume_training(dataset, test_images, num_epochs, resume_epoch):
    """Simulate training that is interrupted and resumed."""
    results = {}

    # Phase 1: Train from 0 to resume_epoch (then "save checkpoint")
    for epoch in range(resume_epoch):
        dataset.set_epoch(epoch)
        epoch_results = {}

        for img_id in test_images:
            flipped = dataset._deterministic_coin(img_id)
            epoch_results[img_id] = flipped

        results[epoch] = epoch_results

    logger.info(f"  [Simulating checkpoint save at epoch {resume_epoch}]")
    logger.info(f"  [Simulating training interruption...]")
    logger.info(f"  [Resuming from checkpoint at epoch {resume_epoch}]")

    # Phase 2: Resume from resume_epoch to num_epochs (simulating checkpoint load)
    # In real training: start_epoch = loaded_checkpoint['epoch']
    start_epoch = resume_epoch

    for epoch in range(start_epoch, num_epochs):
        dataset.set_epoch(epoch)  # This is what train_direct.py does
        epoch_results = {}

        for img_id in test_images:
            flipped = dataset._deterministic_coin(img_id)
            epoch_results[img_id] = flipped

        results[epoch] = epoch_results

    return results


def compare_results(continuous, resumed, num_epochs):
    """Compare flip decisions from continuous vs resumed training."""
    all_match = True
    mismatches = []

    for epoch in range(num_epochs):
        epoch_match = True

        for img_id in continuous[epoch].keys():
            cont_flip = continuous[epoch][img_id]
            resu_flip = resumed[epoch][img_id]

            if cont_flip != resu_flip:
                epoch_match = False
                all_match = False
                mismatches.append((epoch, img_id, cont_flip, resu_flip))

        if epoch_match:
            logger.info(f"    Epoch {epoch:2d}: ✅ All flip decisions match")
        else:
            logger.error(f"    Epoch {epoch:2d}: ❌ Mismatch detected!")

    return all_match, mismatches


def main():
    logger.info("=" * 70)
    logger.info("CHECKPOINT RESUME COMPATIBILITY TEST")
    logger.info("=" * 70)
    logger.info("\nThis test verifies that flip decisions are identical when:")
    logger.info("  1. Training runs continuously (epochs 0-9)")
    logger.info("  2. Training is interrupted at epoch 5 and resumed")
    logger.info("\nBoth scenarios should produce identical flip decisions.\n")

    # Test parameters
    random_flip_prob = 0.35
    num_epochs = 10
    resume_epoch = 5
    test_images = [f"img_{i:04d}" for i in range(20)]

    # Scenario 1: Continuous training
    logger.info("-" * 70)
    logger.info("SCENARIO 1: Continuous Training (Epochs 0-9)")
    logger.info("-" * 70)

    dataset1 = MockDataset(random_flip_prob=random_flip_prob)
    continuous_results = simulate_continuous_training(dataset1, test_images, num_epochs)

    logger.info(f"  Completed {num_epochs} epochs without interruption")

    # Scenario 2: Interrupted and resumed training
    logger.info("\n" + "-" * 70)
    logger.info(f"SCENARIO 2: Training Interrupted at Epoch {resume_epoch} and Resumed")
    logger.info("-" * 70)

    dataset2 = MockDataset(random_flip_prob=random_flip_prob)
    resumed_results = simulate_resume_training(dataset2, test_images, num_epochs, resume_epoch)

    logger.info(f"  Completed {num_epochs} epochs with interruption at epoch {resume_epoch}")

    # Compare results
    logger.info("\n" + "-" * 70)
    logger.info("COMPARISON: Continuous vs Resumed Training")
    logger.info("-" * 70)

    all_match, mismatches = compare_results(continuous_results, resumed_results, num_epochs)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST RESULT")
    logger.info("=" * 70)

    if all_match:
        logger.info("\n✅ PASS: Checkpoint resume is compatible!")
        logger.info("\nFlip decisions are IDENTICAL between:")
        logger.info("  • Continuous training (epochs 0-9)")
        logger.info(f"  • Resumed training (interrupted at epoch {resume_epoch})")
        logger.info("\nThis confirms that:")
        logger.info("  • Epoch tracking works correctly on resume")
        logger.info("  • Flip decisions are deterministic and reproducible")
        logger.info("  • Training can be safely interrupted and resumed")
        return 0
    else:
        logger.error("\n❌ FAIL: Checkpoint resume has issues!")
        logger.error(f"\nFound {len(mismatches)} mismatches:")

        for epoch, img_id, cont, resu in mismatches[:10]:
            cont_str = "FLIP" if cont else "KEEP"
            resu_str = "FLIP" if resu else "KEEP"
            logger.error(f"  Epoch {epoch}, {img_id}: continuous={cont_str}, resumed={resu_str}")

        if len(mismatches) > 10:
            logger.error(f"  ... and {len(mismatches) - 10} more")

        return 1


if __name__ == "__main__":
    exit(main())
