#!/usr/bin/env python3
"""
Example configurations for AdamW8bit optimizer across different dataset sizes.
Run this script to see how the optimizer parameters scale for your specific use case.
"""

from training_config import (
    get_adamw8bit_config,
    get_recommended_batch_size,
    AdamW8bitConfig
)


def print_separator(title=""):
    """Print a nice separator"""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)
    print()


def example_small_dataset():
    """Example for small dataset (5k samples)"""
    print_separator("Small Dataset Example (5,000 samples)")

    dataset_size = 5000
    batch_size = 16
    num_epochs = 100
    grad_accum = 4

    lr, optim_kwargs, warmup_steps = get_adamw8bit_config(
        dataset_size=dataset_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gradient_accumulation_steps=grad_accum,
    )

    print("\nUsage example:")
    print("```python")
    print("import bitsandbytes as bnb")
    print(f"optimizer = bnb.optim.AdamW8bit(")
    print(f"    model.parameters(),")
    print(f"    lr={lr:.6f},")
    print(f"    betas={optim_kwargs['betas']},")
    print(f"    eps={optim_kwargs['eps']},")
    print(f"    weight_decay={optim_kwargs['weight_decay']:.6f}")
    print(f")")
    print("```")


def example_medium_dataset():
    """Example for medium dataset (50k samples)"""
    print_separator("Medium Dataset Example (50,000 samples)")

    dataset_size = 50000
    batch_size = 32
    num_epochs = 50
    grad_accum = 2

    lr, optim_kwargs, warmup_steps = get_adamw8bit_config(
        dataset_size=dataset_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gradient_accumulation_steps=grad_accum,
    )

    print("\nUsage example:")
    print("```python")
    print("import bitsandbytes as bnb")
    print(f"optimizer = bnb.optim.AdamW8bit(")
    print(f"    model.parameters(),")
    print(f"    lr={lr:.6f},")
    print(f"    **{optim_kwargs}")
    print(f")")
    print("```")


def example_large_dataset():
    """Example for large dataset (500k samples)"""
    print_separator("Large Dataset Example (500,000 samples)")

    dataset_size = 500000
    batch_size = 64
    num_epochs = 20
    grad_accum = 1

    lr, optim_kwargs, warmup_steps = get_adamw8bit_config(
        dataset_size=dataset_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gradient_accumulation_steps=grad_accum,
    )

    print("\nUsage example:")
    print("```python")
    print("import bitsandbytes as bnb")
    print(f"optimizer = bnb.optim.AdamW8bit(")
    print(f"    model.parameters(),")
    print(f"    lr={lr:.6f},")
    print(f"    **{optim_kwargs}")
    print(f")")
    print("```")


def example_multi_gpu():
    """Example for multi-GPU training"""
    print_separator("Multi-GPU Training Example (4 GPUs)")

    dataset_size = 100000
    batch_size = 32  # Per GPU
    num_epochs = 30
    grad_accum = 2
    num_gpus = 4

    lr, optim_kwargs, warmup_steps = get_adamw8bit_config(
        dataset_size=dataset_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gradient_accumulation_steps=grad_accum,
        num_gpus=num_gpus,
    )

    print("\nNote: With 4 GPUs, effective batch size is multiplied by 4")
    print(f"Effective batch: {batch_size * grad_accum * num_gpus}")

    print("\nUsage example:")
    print("```python")
    print("import bitsandbytes as bnb")
    print(f"optimizer = bnb.optim.AdamW8bit(")
    print(f"    model.parameters(),")
    print(f"    lr={lr:.6f},  # Scaled for 4 GPUs")
    print(f"    **{optim_kwargs}")
    print(f")")
    print("```")


def example_custom_config():
    """Example with custom configuration"""
    print_separator("Custom Configuration Example")

    # Create custom config with higher learning rate and linear scaling
    custom_config = AdamW8bitConfig(
        base_lr=2e-4,
        base_batch_size=512,
        lr_scaling_mode="linear",  # Use linear instead of sqrt
        weight_decay=0.02,
        wd_scaling_mode="fixed",   # Don't scale weight decay
        warmup_ratio=0.1,          # 10% warmup
    )

    dataset_size = 50000
    batch_size = 32
    num_epochs = 50

    lr, optim_kwargs, warmup_steps = get_adamw8bit_config(
        dataset_size=dataset_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        base_config=custom_config
    )

    print("\nCustom settings applied:")
    print(f"  - Base LR: 2e-4 (higher than default 1e-4)")
    print(f"  - Scaling mode: linear (not sqrt)")
    print(f"  - Weight decay: fixed at 0.02 (not scaled)")
    print(f"  - Warmup ratio: 10% (not 5%)")


def example_batch_size_recommendation():
    """Example of getting batch size recommendations"""
    print_separator("Batch Size Recommendations")

    test_cases = [
        (5000, 12, "Small dataset, RTX 3060"),
        (50000, 24, "Medium dataset, RTX 3090/4090"),
        (500000, 40, "Large dataset, A100 40GB"),
    ]

    for dataset_size, gpu_memory, description in test_cases:
        print(f"\n{description}:")
        print(f"  Dataset size: {dataset_size:,}")
        print(f"  GPU memory: {gpu_memory}GB")

        batch_size, grad_accum = get_recommended_batch_size(
            dataset_size=dataset_size,
            available_memory_gb=gpu_memory,
            target_steps_per_epoch=500
        )

        effective_batch = batch_size * grad_accum
        steps_per_epoch = dataset_size // effective_batch

        print(f"  Recommended batch_size: {batch_size}")
        print(f"  Recommended grad_accum: {grad_accum}")
        print(f"  Effective batch size: {effective_batch}")
        print(f"  Steps per epoch: ~{steps_per_epoch}")


def example_comparison_table():
    """Show comparison across different dataset sizes"""
    print_separator("Scaling Comparison Across Dataset Sizes")

    dataset_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    batch_size = 32
    num_epochs = 50

    print(f"\n{'Dataset Size':>15} | {'LR':>12} | {'Weight Decay':>12} | {'Warmup Steps':>12}")
    print("-" * 65)

    for size in dataset_sizes:
        lr, optim_kwargs, warmup = get_adamw8bit_config(
            dataset_size=size,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

        print(
            f"{size:>15,} | "
            f"{lr:>12.6f} | "
            f"{optim_kwargs['weight_decay']:>12.6f} | "
            f"{warmup:>12,}"
        )


def example_your_dataset():
    """Interactive example for user's specific dataset"""
    print_separator("Configure for YOUR Dataset")

    try:
        print("\nEnter your dataset characteristics:")
        dataset_size = int(input("  Dataset size (number of samples): "))
        batch_size = int(input("  Batch size per GPU: "))
        num_epochs = int(input("  Number of epochs: "))
        grad_accum = int(input("  Gradient accumulation steps (default 1): ") or "1")
        num_gpus = int(input("  Number of GPUs (default 1): ") or "1")

        print("\n" + "-" * 80)

        lr, optim_kwargs, warmup_steps = get_adamw8bit_config(
            dataset_size=dataset_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            gradient_accumulation_steps=grad_accum,
            num_gpus=num_gpus,
        )

        print("\n✅ Your optimized configuration:")
        print(f"\n```python")
        print(f"import bitsandbytes as bnb")
        print(f"")
        print(f"optimizer = bnb.optim.AdamW8bit(")
        print(f"    model.parameters(),")
        print(f"    lr={lr:.6f},")
        print(f"    betas={optim_kwargs['betas']},")
        print(f"    eps={optim_kwargs['eps']},")
        print(f"    weight_decay={optim_kwargs['weight_decay']:.6f}")
        print(f")")
        print(f"```")
        print(f"\nWarmup steps: {warmup_steps:,}")
        print(f"Effective batch size: {batch_size * grad_accum * num_gpus}")

    except (ValueError, KeyboardInterrupt):
        print("\nSkipped interactive configuration.")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print(" AdamW8bit Optimizer Configuration Examples")
    print(" For OppaiOracle Anime Image Tagger")
    print("=" * 80)

    # Run preset examples
    example_small_dataset()
    example_medium_dataset()
    example_large_dataset()
    example_multi_gpu()
    example_custom_config()
    example_batch_size_recommendation()
    example_comparison_table()

    # Interactive example
    example_your_dataset()

    print("\n" + "=" * 80)
    print(" See OPTIMIZER_SETUP_GUIDE.md for detailed documentation")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
