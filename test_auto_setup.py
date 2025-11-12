#!/usr/bin/env python3
"""
Test script to demonstrate automatic training setup.
Run this to see how the auto-configuration works without starting actual training.
"""

import sys
from pathlib import Path

# Minimal mock objects for testing without full dataset
class MockDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class MockDataLoader:
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MockConfig:
    def __init__(self, dataset_size):
        self.data = type('obj', (object,), {
            'batch_size': None,  # Let auto-setup determine
            'image_size': type('obj', (object,), {'height': 512})(),
        })()

        self.training = type('obj', (object,), {
            'num_epochs': 50,
            'world_size': 1,
            'gradient_accumulation_steps': None,  # Let auto-setup determine
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
        })()


class MockModel:
    """Mock model for testing"""
    def __init__(self, num_params=125_000_000):
        self.num_params = num_params

    def parameters(self):
        # Return empty list for testing
        import torch
        return [torch.randn(100, 100)]


def test_auto_setup(dataset_size):
    """Test the automatic setup with a given dataset size."""
    print(f"\n{'='*80}")
    print(f"Testing Auto-Setup with {dataset_size:,} samples")
    print(f"{'='*80}\n")

    # Create mock objects
    dataset = MockDataset(dataset_size)
    train_loader = MockDataLoader(dataset)
    config = MockConfig(dataset_size)
    model = MockModel()

    # Run automatic setup
    try:
        from auto_training_setup import setup_training_automatically, print_recommendation_summary

        optimizer, scheduler, info = setup_training_automatically(
            train_loader=train_loader,
            model=model,
            config=config
        )

        # Print summary
        print_recommendation_summary(info)

        # Show what code you'd use
        print("=" * 80)
        print("   📝 CODE TO USE IN YOUR TRAINING SCRIPT")
        print("=" * 80)
        print()
        print("```python")
        print("from auto_training_setup import auto_setup")
        print()
        print("# One line to set everything up:")
        print("optimizer, scheduler, info = auto_setup(train_loader, model, config)")
        print()
        print("# Then use in your training loop:")
        print("for epoch in range(num_epochs):")
        print("    for batch in train_loader:")
        print("        # ... forward pass ...")
        print("        loss.backward()")
        print("        optimizer.step()")
        print("        scheduler.step()  # Step-based scheduler")
        print("        optimizer.zero_grad()")
        print("```")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests for different dataset sizes."""
    print("\n" + "=" * 80)
    print("   AUTOMATIC TRAINING SETUP - DEMONSTRATION")
    print("=" * 80)

    test_cases = [
        ("Tiny dataset", 5_000),
        ("Small dataset", 25_000),
        ("Medium dataset", 50_000),
        ("Large dataset", 150_000),
        ("Very large dataset", 500_000),
    ]

    print("\nThis demo shows how auto-setup adapts to different dataset sizes.")
    print("Watch how batch size, learning rate, weight decay, and scheduler change!\n")

    success_count = 0
    for name, size in test_cases:
        if test_auto_setup(size):
            success_count += 1

        print("\nPress Enter to continue to next test (or Ctrl+C to quit)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user.")
            break

    print("\n" + "=" * 80)
    print(f"   DEMO COMPLETE: {success_count}/{len(test_cases)} tests successful")
    print("=" * 80)
    print("\nReady to integrate into your training script?")
    print("See AUTOMATIC_SETUP_INTEGRATION.md for details!")
    print()


if __name__ == "__main__":
    main()
