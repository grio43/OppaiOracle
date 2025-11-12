#!/usr/bin/env python3
"""
Demo script showing what automatic configuration recommends for different dataset sizes.
Doesn't require bitsandbytes - just shows the configuration decisions.
"""

import sys


def demo_config_for_dataset(dataset_size, num_epochs=50):
    """Show what configuration would be chosen for a dataset."""
    print(f"\n{'='*70}")
    print(f" Dataset: {dataset_size:,} samples, {num_epochs} epochs")
    print(f"{'='*70}\n")

    try:
        from optimizer_config import get_recommended_batch_size, get_adamw8bit_config
        from scheduler_config import recommend_scheduler, get_scheduler_config

        # Step 1: Get batch size recommendation
        print("Step 1: Determining batch size...")
        batch_size, grad_accum = get_recommended_batch_size(
            dataset_size=dataset_size,
            target_steps_per_epoch=500,
            image_size=512,
            model_size="medium"
        )

        # Step 2: Get optimizer configuration
        print("\nStep 2: Configuring optimizer...")
        lr, optim_kwargs, warmup = get_adamw8bit_config(
            dataset_size=dataset_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            gradient_accumulation_steps=grad_accum,
        )

        # Step 3: Determine scheduler
        print("\nStep 3: Selecting scheduler...")
        effective_batch = batch_size * grad_accum

        # Determine training goal
        if dataset_size < 30000:
            training_goal = "exploration"
        elif dataset_size < 100000 and num_epochs >= 50:
            training_goal = "exploration"
        else:
            training_goal = "best_convergence"

        sched_type, explanation = recommend_scheduler(
            dataset_size=dataset_size,
            num_epochs=num_epochs,
            effective_batch_size=effective_batch,
            training_goal=training_goal
        )

        print(f"\n  Recommended: {sched_type.value}")
        print(f"  Reason: {explanation}")

        # Summary
        print(f"\n{'='*70}")
        print(" CONFIGURATION SUMMARY")
        print(f"{'='*70}")
        print(f"\n  Batch Configuration:")
        print(f"    Per-device batch size:      {batch_size}")
        print(f"    Gradient accumulation:      {grad_accum}")
        print(f"    Effective batch size:       {effective_batch}")

        print(f"\n  Optimizer (AdamW8bit):")
        print(f"    Learning rate:              {lr:.6f}")
        print(f"    Weight decay:               {optim_kwargs['weight_decay']:.6f}")
        print(f"    Beta1:                      {optim_kwargs['betas'][0]}")
        print(f"    Beta2:                      {optim_kwargs['betas'][1]:.6f}")

        print(f"\n  Scheduler:")
        print(f"    Type:                       {sched_type.value}")
        print(f"    Training goal:              {training_goal}")
        print(f"    Warmup steps:               {warmup:,}")

        print(f"\n{'='*70}\n")

        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Demo configurations for different dataset sizes."""
    print("\n" + "="*70)
    print(" AUTOMATIC CONFIGURATION DEMO")
    print("="*70)
    print("\nThis shows what settings would be chosen for different dataset sizes.")
    print("Notice how batch size, learning rate, weight decay, and scheduler adapt!\n")

    test_cases = [
        ("Tiny", 5_000, 100),
        ("Small", 25_000, 100),
        ("Medium", 50_000, 50),
        ("Large", 150_000, 30),
        ("Very Large", 500_000, 20),
    ]

    for name, size, epochs in test_cases:
        print(f"\n{name} Dataset Example:")
        demo_config_for_dataset(size, epochs)
        print("\n" + "-"*70)

    print("\n" + "="*70)
    print(" DEMO COMPLETE")
    print("="*70)
    print("\nReady to integrate? See AUTO_CONFIG_SUMMARY.md")
    print("Or run: python train_direct.py with auto_setup() added\n")


if __name__ == "__main__":
    main()
