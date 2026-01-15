#!/usr/bin/env python3
"""
VRAM Profiler for OppaiOracle Model Configurations

Tests multiple batch sizes, layer counts, and checkpointing settings
to measure actual VRAM usage and training speed.

Usage:
    python -m profiling.vram_profiler --baseline     # RECOMMENDED FIRST STEP
    python -m profiling.vram_profiler --layer-sweep  # Find max layers (checkpointing OFF)
    python -m profiling.vram_profiler --quick        # Quick matrix test
    python -m profiling.vram_profiler                # Full matrix test

Examples:
    # Step 1: Get baseline measurements (model, optimizer, per-sample costs)
    python -m profiling.vram_profiler --baseline
    python -m profiling.vram_profiler --baseline --layers 24  # Test different layer count

    # Layer sweep: find max layers with checkpointing OFF (fixed costs only)
    python -m profiling.vram_profiler --layer-sweep                    # 16-48 layers, step 4
    python -m profiling.vram_profiler --layer-sweep --layer-range 20,60  # Custom range
    python -m profiling.vram_profiler --layer-sweep --layer-step 2     # Smaller steps
    python -m profiling.vram_profiler --layer-sweep --vram-budget 32   # Explicit budget
    python -m profiling.vram_profiler --layer-sweep --output sweep.csv # Save results

    # Step 2: Run matrix tests once baseline is understood
    python -m profiling.vram_profiler --quick  # 4 configs, ~1 min

    # Full test matrix (~56 configs, ~15 min)
    python -m profiling.vram_profiler

    # Custom test range
    python -m profiling.vram_profiler --batch-sizes 24,32,36,40,44 --layers 24,28

    # Test only checkpointing impact at current config
    python -m profiling.vram_profiler --batch-sizes 36 --layers 28
"""

import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.amp import autocast

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_architecture import SimplifiedTagger, VisionTransformerConfig
from loss_functions import MultiTaskLoss, AsymmetricFocalLoss
from torchmetrics.classification import MultilabelF1Score


@dataclass
class ProfileResult:
    """Result of profiling a single configuration."""
    batch_size: int
    num_layers: int
    checkpointing: bool
    with_optimizer: bool
    peak_allocated_gb: float
    peak_reserved_gb: float
    forward_ms: float
    backward_ms: float
    total_ms: float
    overflow_detected: bool
    samples_per_sec: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        if self.total_ms > 0 and self.batch_size > 0:
            self.samples_per_sec = (self.batch_size / self.total_ms) * 1000


class VRAMProfiler:
    """Profiles VRAM usage across model configurations."""

    BASE_CONFIG = {
        "hidden_size": 1536,
        "num_attention_heads": 24,
        "intermediate_size": 6144,
        "image_size": 512,
        "patch_size": 16,
        "num_tags": 100002,
        "num_ratings": 5,
        "dropout": 0.1,  # Match real training config
        "attention_dropout": 0.1,  # Match real training config
        "use_flex_attention": True,
        "drop_path_rate": 0.1,  # Match real training config
    }

    def __init__(self, warmup_iterations: int = 2, measure_iterations: int = 3):
        self.warmup_iterations = warmup_iterations
        self.measure_iterations = measure_iterations
        self.results: List[ProfileResult] = []
        self.gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"
        self.total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    def profile_config(
        self,
        batch_size: int,
        num_layers: int,
        checkpointing: bool,
        with_optimizer: bool = True,
    ) -> ProfileResult:
        """Profile a single configuration."""

        # Clean slate
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = None
        optimizer = None

        try:
            # Build config
            config = VisionTransformerConfig(
                **self.BASE_CONFIG,
                num_hidden_layers=num_layers,
                gradient_checkpointing=checkpointing,
            )

            # Create model
            model = SimplifiedTagger(config).cuda().bfloat16()
            model.train()

            # Real loss function (matches actual training setup)
            criterion = MultiTaskLoss(
                tag_loss_weight=0.9,
                rating_loss_weight=0.1,
                tag_loss_fn=AsymmetricFocalLoss(
                    gamma_pos=1.0, gamma_neg=1.0, alpha=0.5, clip=0.05
                ),
                rating_loss_fn=AsymmetricFocalLoss(
                    gamma_pos=1.0, gamma_neg=1.0, alpha=0.5, clip=0.05, ignore_index=None
                ),
            )

            # 8-bit AdamW optimizer (matches actual training setup)
            if with_optimizer:
                try:
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
                except ImportError:
                    # Fallback to standard AdamW if bitsandbytes not available
                    print("    Warning: bitsandbytes not found, using standard AdamW")
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            # Dummy input (matches actual training)
            x = torch.randn(
                batch_size, 3, 512, 512,
                device='cuda', dtype=torch.bfloat16
            )

            # Create realistic dummy labels (matches actual training)
            tag_labels = torch.zeros(
                batch_size, self.BASE_CONFIG['num_tags'],
                device='cuda', dtype=torch.float32
            )
            # Set ~50 random positive tags per sample (realistic density)
            for i in range(batch_size):
                pos_indices = torch.randint(1, self.BASE_CONFIG['num_tags'], (50,), device='cuda')
                tag_labels[i, pos_indices] = 1.0

            rating_labels = torch.zeros(
                batch_size, self.BASE_CONFIG['num_ratings'],
                device='cuda', dtype=torch.float32
            )
            rating_labels.scatter_(1, torch.randint(0, 5, (batch_size, 1), device='cuda'), 1.0)

            # Warmup runs (JIT compilation, CUDA kernels, etc.)
            for _ in range(self.warmup_iterations):
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(x)
                    loss, _ = criterion(out['tag_logits'], out['rating_logits'], tag_labels, rating_labels)
                loss.backward()
                if optimizer:
                    optimizer.step()
                    optimizer.zero_grad()

            # Reset stats after warmup
            torch.cuda.reset_peak_memory_stats()

            # Measurement runs
            forward_times = []
            backward_times = []

            for _ in range(self.measure_iterations):
                torch.cuda.synchronize()

                # Forward pass timing (with autocast to match real training)
                t0 = time.perf_counter()
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(x)
                    loss, _ = criterion(out['tag_logits'], out['rating_logits'], tag_labels, rating_labels)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                forward_times.append((t1 - t0) * 1000)

                # Backward pass + optimizer step timing
                t0 = time.perf_counter()
                loss.backward()
                if optimizer:
                    optimizer.step()
                    optimizer.zero_grad()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                backward_times.append((t1 - t0) * 1000)

            # Collect memory stats
            peak_alloc = torch.cuda.max_memory_allocated() / 1e9
            peak_reserved = torch.cuda.max_memory_reserved() / 1e9

            # Detect overflow to system RAM
            # If reserved >> allocated, it suggests fragmentation or unified memory spillover
            overflow = (peak_reserved - peak_alloc) > 2.0

            avg_forward = sum(forward_times) / len(forward_times)
            avg_backward = sum(backward_times) / len(backward_times)
            avg_total = avg_forward + avg_backward

            return ProfileResult(
                batch_size=batch_size,
                num_layers=num_layers,
                checkpointing=checkpointing,
                with_optimizer=with_optimizer,
                peak_allocated_gb=peak_alloc,
                peak_reserved_gb=peak_reserved,
                forward_ms=avg_forward,
                backward_ms=avg_backward,
                total_ms=avg_total,
                overflow_detected=overflow,
            )

        except torch.cuda.OutOfMemoryError:
            return ProfileResult(
                batch_size=batch_size,
                num_layers=num_layers,
                checkpointing=checkpointing,
                with_optimizer=with_optimizer,
                peak_allocated_gb=-1,
                peak_reserved_gb=-1,
                forward_ms=-1,
                backward_ms=-1,
                total_ms=-1,
                overflow_detected=False,
                error="OOM",
            )
        except Exception as e:
            return ProfileResult(
                batch_size=batch_size,
                num_layers=num_layers,
                checkpointing=checkpointing,
                with_optimizer=with_optimizer,
                peak_allocated_gb=-1,
                peak_reserved_gb=-1,
                forward_ms=-1,
                backward_ms=-1,
                total_ms=-1,
                overflow_detected=False,
                error=str(e)[:80],
            )
        finally:
            # Aggressive cleanup
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if 'criterion' in locals():
                del criterion
            if 'x' in locals():
                del x
            if 'tag_labels' in locals():
                del tag_labels
            if 'rating_labels' in locals():
                del rating_labels
            if 'out' in locals():
                del out
            if 'loss' in locals():
                del loss
            gc.collect()
            torch.cuda.empty_cache()

    def measure_model_only(self, num_layers: int) -> dict:
        """Measure VRAM for model parameters only (no optimizer, no forward pass)."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        config = VisionTransformerConfig(
            **self.BASE_CONFIG,
            num_hidden_layers=num_layers,
            gradient_checkpointing=False,
        )

        model = SimplifiedTagger(config).cuda().bfloat16()
        torch.cuda.synchronize()

        model_vram = torch.cuda.max_memory_allocated() / 1e9
        param_count = sum(p.numel() for p in model.parameters())

        del model
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "model_vram_gb": model_vram,
            "param_count": param_count,
            "param_count_m": param_count / 1e6,
        }

    def measure_fixed_overhead(self, num_layers: int) -> dict:
        """
        Measure all fixed VRAM costs that don't scale with batch size:
        - Model parameters (bf16)
        - Gradient buffers (bf16, same size as params)
        - Optimizer states (8-bit AdamW)
        - Validation metrics (MultilabelF1Score buffers)

        This excludes activation memory which scales with batch size.
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        config = VisionTransformerConfig(
            **self.BASE_CONFIG,
            num_hidden_layers=num_layers,
            gradient_checkpointing=False,
        )

        # 1. Model parameters only
        model = SimplifiedTagger(config).cuda().bfloat16()
        model.train()
        torch.cuda.synchronize()
        model_only_vram = torch.cuda.memory_allocated() / 1e9

        # 2. Add optimizer (states not yet initialized)
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
            optimizer_type = "AdamW8bit"
        except ImportError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer_type = "AdamW (32-bit fallback)"

        # 3. Do one forward+backward to create gradient buffers
        #    Use minimal batch=1 to minimize activation memory during this step
        x = torch.randn(1, 3, 512, 512, device='cuda', dtype=torch.bfloat16)
        out = model(x)
        loss = out['tag_logits'].sum()
        loss.backward()

        # Now gradients exist in .grad attributes
        torch.cuda.synchronize()
        model_plus_gradients_vram = torch.cuda.memory_allocated() / 1e9

        # Clear activations (they're released after backward anyway)
        del x, out, loss
        gc.collect()
        torch.cuda.empty_cache()

        # 4. Initialize optimizer states with one step
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)  # Keep gradient buffers allocated

        torch.cuda.synchronize()
        full_fixed_vram = torch.cuda.memory_allocated() / 1e9

        # 5. Add validation metrics (these persist on GPU during training)
        num_tags = self.BASE_CONFIG['num_tags']
        threshold = 0.5
        val_metrics = {
            'f1_macro': MultilabelF1Score(num_labels=num_tags, average="macro", threshold=threshold).to('cuda'),
            'f1_micro': MultilabelF1Score(num_labels=num_tags, average="micro", threshold=threshold).to('cuda'),
        }
        torch.cuda.synchronize()
        full_fixed_with_metrics_vram = torch.cuda.memory_allocated() / 1e9

        # Calculate component sizes
        gradient_vram = model_plus_gradients_vram - model_only_vram
        optimizer_state_vram = full_fixed_vram - model_plus_gradients_vram
        metrics_vram = full_fixed_with_metrics_vram - full_fixed_vram

        del model, optimizer, val_metrics
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "model_params_gb": model_only_vram,
            "gradient_buffers_gb": gradient_vram,
            "optimizer_states_gb": optimizer_state_vram,
            "validation_metrics_gb": metrics_vram,
            "total_fixed_gb": full_fixed_with_metrics_vram,
            "optimizer_type": optimizer_type,
        }

    def run_baseline(self, num_layers: int = 28) -> dict:
        """
        Run baseline measurements to understand fixed costs vs batch-scaling costs.

        Fixed costs (don't scale with batch):
        - Model parameters
        - Gradient buffers
        - Optimizer states
        - Validation metrics (MultilabelF1Score)

        Batch-scaling costs:
        - Activation memory (scales linearly with batch size)
        """
        print(f"\n{'='*70}")
        print(f"BASELINE MEASUREMENTS ({num_layers} layers)")
        print(f"{'='*70}")
        print(f"GPU: {self.gpu_name}")
        print(f"Total VRAM: {self.total_vram:.1f} GB")
        print(f"{'='*70}\n")

        results = {"num_layers": num_layers, "total_vram_gb": self.total_vram}

        # 1. Get parameter count
        print("1. Counting model parameters...")
        model_info = self.measure_model_only(num_layers)
        results["param_count"] = model_info["param_count"]
        results["param_count_m"] = model_info["param_count_m"]
        print(f"   Parameters: {model_info['param_count_m']:.1f}M")

        # 2. Measure all fixed overhead (model + gradients + optimizer + metrics)
        print("\n2. Measuring fixed VRAM costs (excludes batch-dependent activations)...")
        fixed_info = self.measure_fixed_overhead(num_layers)
        results.update(fixed_info)
        print(f"   Model parameters:    {fixed_info['model_params_gb']:.3f} GB")
        print(f"   Gradient buffers:    {fixed_info['gradient_buffers_gb']:.3f} GB")
        print(f"   Optimizer states:    {fixed_info['optimizer_states_gb']:.3f} GB ({fixed_info['optimizer_type']})")
        print(f"   Validation metrics:  {fixed_info['validation_metrics_gb']:.3f} GB")
        print(f"   ─────────────────────────────────")
        print(f"   TOTAL FIXED:         {fixed_info['total_fixed_gb']:.3f} GB")

        # 3. Calculate activation memory per sample
        print("\n3. Measuring activation memory per sample...")
        print("   (Comparing batch=1 vs batch=8 to isolate per-sample cost)")

        # Test with checkpointing ON
        result_b1_ckpt = self.profile_config(batch_size=1, num_layers=num_layers, checkpointing=True)
        result_b8_ckpt = self.profile_config(batch_size=8, num_layers=num_layers, checkpointing=True)

        if not result_b1_ckpt.error and not result_b8_ckpt.error:
            # Per-sample activation = (peak_b8 - peak_b1) / 7
            activation_per_sample_ckpt = (result_b8_ckpt.peak_allocated_gb - result_b1_ckpt.peak_allocated_gb) / 7
            results["activation_per_sample_ckpt_on_gb"] = activation_per_sample_ckpt
            results["activation_per_sample_ckpt_on_mb"] = activation_per_sample_ckpt * 1024
            print(f"   Checkpointing ON:  {activation_per_sample_ckpt*1024:.1f} MB/sample")
        else:
            print(f"   Checkpointing ON:  ERROR - {result_b1_ckpt.error or result_b8_ckpt.error}")

        # Test with checkpointing OFF
        result_b1_no_ckpt = self.profile_config(batch_size=1, num_layers=num_layers, checkpointing=False)
        result_b8_no_ckpt = self.profile_config(batch_size=8, num_layers=num_layers, checkpointing=False)

        if not result_b1_no_ckpt.error and not result_b8_no_ckpt.error:
            activation_per_sample_no_ckpt = (result_b8_no_ckpt.peak_allocated_gb - result_b1_no_ckpt.peak_allocated_gb) / 7
            results["activation_per_sample_ckpt_off_gb"] = activation_per_sample_no_ckpt
            results["activation_per_sample_ckpt_off_mb"] = activation_per_sample_no_ckpt * 1024
            print(f"   Checkpointing OFF: {activation_per_sample_no_ckpt*1024:.1f} MB/sample")
        else:
            print(f"   Checkpointing OFF: ERROR - {result_b1_no_ckpt.error or result_b8_no_ckpt.error}")

        # 4. Calculate checkpointing savings
        if "activation_per_sample_ckpt_on_gb" in results and "activation_per_sample_ckpt_off_gb" in results:
            savings_pct = (1 - results["activation_per_sample_ckpt_on_gb"] / results["activation_per_sample_ckpt_off_gb"]) * 100
            results["checkpointing_savings_pct"] = savings_pct
            print(f"\n   Checkpointing saves {savings_pct:.0f}% activation memory per sample")

        # 5. Calculate max batch sizes
        print("\n" + "-" * 70)
        print("PREDICTED MAX BATCH SIZES (with 1 GB headroom)")
        print("-" * 70)

        available_vram = self.total_vram - 1.0  # Leave 1GB headroom
        vram_for_activations = available_vram - fixed_info["total_fixed_gb"]
        results["vram_for_activations_gb"] = vram_for_activations

        print(f"   Available for activations: {vram_for_activations:.2f} GB")
        print()

        if "activation_per_sample_ckpt_on_gb" in results:
            max_batch_on = int(vram_for_activations / results["activation_per_sample_ckpt_on_gb"])
            results["predicted_max_batch_ckpt_on"] = max_batch_on
            predicted_vram_on = fixed_info["total_fixed_gb"] + (max_batch_on * results["activation_per_sample_ckpt_on_gb"])
            print(f"   Checkpointing ON:  max batch ~{max_batch_on} → ~{predicted_vram_on:.1f} GB")

        if "activation_per_sample_ckpt_off_gb" in results:
            max_batch_off = int(vram_for_activations / results["activation_per_sample_ckpt_off_gb"])
            results["predicted_max_batch_ckpt_off"] = max_batch_off
            predicted_vram_off = fixed_info["total_fixed_gb"] + (max_batch_off * results["activation_per_sample_ckpt_off_gb"])
            print(f"   Checkpointing OFF: max batch ~{max_batch_off} → ~{predicted_vram_off:.1f} GB")

        # 6. Print final summary
        print("\n" + "=" * 70)
        print("BASELINE SUMMARY")
        print("=" * 70)
        print(f"\nFIXED COSTS (batch-independent):")
        print(f"   Model parameters:    {fixed_info['model_params_gb']:.2f} GB  ({results['param_count_m']:.0f}M params)")
        print(f"   Gradient buffers:    {fixed_info['gradient_buffers_gb']:.2f} GB")
        print(f"   Optimizer states:    {fixed_info['optimizer_states_gb']:.2f} GB  ({fixed_info['optimizer_type']})")
        print(f"   Validation metrics:  {fixed_info['validation_metrics_gb']:.2f} GB")
        print(f"   ─────────────────────────────────────")
        print(f"   TOTAL FIXED:         {fixed_info['total_fixed_gb']:.2f} GB")

        print(f"\nBATCH-SCALING COSTS (per sample):")
        if "activation_per_sample_ckpt_on_mb" in results:
            print(f"   Checkpointing ON:  {results['activation_per_sample_ckpt_on_mb']:.1f} MB/sample")
        if "activation_per_sample_ckpt_off_mb" in results:
            print(f"   Checkpointing OFF: {results['activation_per_sample_ckpt_off_mb']:.1f} MB/sample")
        if "checkpointing_savings_pct" in results:
            print(f"   Checkpointing saves {results['checkpointing_savings_pct']:.0f}% memory")

        print(f"\nVRAM BUDGET ({self.total_vram:.1f} GB total, 1 GB headroom):")
        print(f"   Fixed overhead:    {fixed_info['total_fixed_gb']:.2f} GB")
        print(f"   For activations:   {vram_for_activations:.2f} GB")

        if "predicted_max_batch_ckpt_on" in results:
            print(f"\n   → Max batch (ckpt ON):  {results['predicted_max_batch_ckpt_on']}")
        if "predicted_max_batch_ckpt_off" in results:
            print(f"   → Max batch (ckpt OFF): {results['predicted_max_batch_ckpt_off']}")

        return results

    def sweep_layers(
        self,
        layer_range: tuple = (16, 48),
        step: int = 4,
        vram_budget_gb: float = None,
        headroom_gb: float = 1.0,
    ) -> dict:
        """
        Sweep layer counts to find maximum viable layers with checkpointing OFF.

        Tests fixed costs (model + gradients + optimizer states) across layer counts
        to determine the maximum layers that fit within VRAM budget.

        Args:
            layer_range: (min_layers, max_layers) tuple
            step: Step size for layer sweep
            vram_budget_gb: VRAM budget (default: auto-detect total GPU VRAM)
            headroom_gb: Reserved headroom for safety

        Returns:
            dict with layer_results, max_viable_layers, scaling analysis
        """
        min_layers, max_layers = layer_range
        vram_budget = vram_budget_gb if vram_budget_gb else self.total_vram
        effective_budget = vram_budget - headroom_gb

        print(f"\n{'='*70}")
        print("LAYER SWEEP (Checkpointing OFF)")
        print(f"{'='*70}")
        print(f"GPU: {self.gpu_name}")
        print(f"Total VRAM: {vram_budget:.1f} GB | Budget: {effective_budget:.1f} GB ({headroom_gb:.1f} GB headroom)")
        print(f"Layer range: {min_layers}-{max_layers} (step {step})")
        print(f"{'='*70}\n")

        # Generate layer counts to test
        layer_counts = list(range(min_layers, max_layers + 1, step))
        if max_layers not in layer_counts:
            layer_counts.append(max_layers)

        results = {
            "layer_range": layer_range,
            "step": step,
            "vram_budget_gb": vram_budget,
            "headroom_gb": headroom_gb,
            "effective_budget_gb": effective_budget,
            "layer_results": [],
            "max_viable_layers": None,
            "scaling_gb_per_layer": None,
        }

        # Print header
        print(f"{'Layers':>6} | {'Params (M)':>10} | {'Model':>7} | {'Grads':>7} | {'Optim':>7} | {'Total':>7} | Status")
        print("-" * 70)

        max_viable = None
        last_successful = None

        for num_layers in layer_counts:
            try:
                # Get parameter count
                model_info = self.measure_model_only(num_layers)
                param_count_m = model_info["param_count_m"]

                # Get fixed overhead breakdown
                fixed_info = self.measure_fixed_overhead(num_layers)

                layer_result = {
                    "num_layers": num_layers,
                    "param_count_m": param_count_m,
                    "model_params_gb": fixed_info["model_params_gb"],
                    "gradient_buffers_gb": fixed_info["gradient_buffers_gb"],
                    "optimizer_states_gb": fixed_info["optimizer_states_gb"],
                    "total_fixed_gb": fixed_info["total_fixed_gb"],
                    "optimizer_type": fixed_info["optimizer_type"],
                    "success": True,
                    "error": None,
                }
                results["layer_results"].append(layer_result)

                # Check if within budget
                if fixed_info["total_fixed_gb"] <= effective_budget:
                    status = "OK"
                    max_viable = num_layers
                    last_successful = layer_result
                else:
                    status = "OVER BUDGET"

                print(f"{num_layers:>6} | {param_count_m:>10.1f} | "
                      f"{fixed_info['model_params_gb']:>6.2f} | "
                      f"{fixed_info['gradient_buffers_gb']:>6.2f} | "
                      f"{fixed_info['optimizer_states_gb']:>6.2f} | "
                      f"{fixed_info['total_fixed_gb']:>6.2f} | {status}")

            except torch.cuda.OutOfMemoryError:
                layer_result = {
                    "num_layers": num_layers,
                    "param_count_m": -1,
                    "model_params_gb": -1,
                    "gradient_buffers_gb": -1,
                    "optimizer_states_gb": -1,
                    "total_fixed_gb": -1,
                    "optimizer_type": "N/A",
                    "success": False,
                    "error": "OOM",
                }
                results["layer_results"].append(layer_result)
                print(f"{num_layers:>6} | {'---':>10} | {'---':>7} | {'---':>7} | {'---':>7} | {'---':>7} | OOM")

                # Cleanup after OOM
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                layer_result = {
                    "num_layers": num_layers,
                    "param_count_m": -1,
                    "model_params_gb": -1,
                    "gradient_buffers_gb": -1,
                    "optimizer_states_gb": -1,
                    "total_fixed_gb": -1,
                    "optimizer_type": "N/A",
                    "success": False,
                    "error": str(e)[:50],
                }
                results["layer_results"].append(layer_result)
                print(f"{num_layers:>6} | {'---':>10} | {'---':>7} | {'---':>7} | {'---':>7} | {'---':>7} | ERROR: {str(e)[:30]}")

                gc.collect()
                torch.cuda.empty_cache()

        results["max_viable_layers"] = max_viable

        # Calculate scaling (VRAM per layer) using linear regression
        successful_results = [r for r in results["layer_results"] if r["success"]]
        if len(successful_results) >= 2:
            layers = [r["num_layers"] for r in successful_results]
            vrams = [r["total_fixed_gb"] for r in successful_results]

            # Simple linear regression: y = mx + b
            n = len(layers)
            sum_x = sum(layers)
            sum_y = sum(vrams)
            sum_xy = sum(x * y for x, y in zip(layers, vrams))
            sum_xx = sum(x * x for x in layers)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n

            results["scaling_gb_per_layer"] = slope
            results["intercept_gb"] = intercept

            # Predict max layers based on budget
            if slope > 0:
                predicted_max = int((effective_budget - intercept) / slope)
                results["predicted_max_layers"] = predicted_max

        # Print summary
        print("\n" + "-" * 70)
        print("SCALING ANALYSIS")
        print("-" * 70)

        if results.get("scaling_gb_per_layer"):
            print(f"VRAM per layer: ~{results['scaling_gb_per_layer']*1000:.1f} MB ({results['scaling_gb_per_layer']:.4f} GB)")
            print(f"Linear fit: VRAM = {results['scaling_gb_per_layer']:.4f} * layers + {results.get('intercept_gb', 0):.2f} GB")
            if results.get("predicted_max_layers"):
                print(f"Predicted max layers at {effective_budget:.1f} GB: ~{results['predicted_max_layers']}")

        print("\n" + "-" * 70)
        print("RESULTS")
        print("-" * 70)

        if max_viable:
            print(f"Maximum viable layers: {max_viable}")
            if last_successful:
                print(f"Peak VRAM at {max_viable} layers: {last_successful['total_fixed_gb']:.2f} GB")
                remaining = effective_budget - last_successful['total_fixed_gb']
                print(f"Remaining VRAM budget: {remaining:.2f} GB (for activations/batches)")
        else:
            print("No viable layer count found within budget!")

        # Recommendation
        print("\n" + "-" * 70)
        print("RECOMMENDATION")
        print("-" * 70)

        if max_viable and last_successful:
            remaining = effective_budget - last_successful['total_fixed_gb']
            if remaining > 10:
                print(f"{max_viable} layers fits comfortably with {remaining:.1f} GB remaining.")
                print("Consider testing higher layer counts or using this headroom for larger batches.")
            elif remaining > 5:
                print(f"{max_viable} layers is a good fit with {remaining:.1f} GB for activations.")
            else:
                print(f"{max_viable} layers is tight. Consider reducing layers for batch flexibility.")

        print("\nNote: These are FIXED costs only (model + gradients + optimizer).")
        print("      Activation memory during training with larger batches will add more VRAM.")

        print("=" * 70)

        return results

    def validate_against_training(self, num_layers: int = 28, batch_size: int = 8) -> dict:
        """
        Run a minimal real training step and compare VRAM with profiler estimate.

        This validates the profiler's accuracy by:
        1. Getting the profiler's fixed cost estimate
        2. Running an actual training step with full setup
        3. Comparing measured peak VRAM with predicted VRAM

        Args:
            num_layers: Number of transformer layers
            batch_size: Batch size for validation step

        Returns:
            dict with profiler_estimate, actual_vram, difference, and accuracy_pct
        """
        print(f"\n{'='*70}")
        print(f"PROFILER VALIDATION ({num_layers} layers, batch {batch_size})")
        print(f"{'='*70}")
        print(f"GPU: {self.gpu_name}")
        print(f"Total VRAM: {self.total_vram:.1f} GB")
        print(f"{'='*70}\n")

        results = {}

        # 1. Get profiler's fixed cost estimate
        print("1. Getting profiler's fixed cost estimate...")
        fixed_info = self.measure_fixed_overhead(num_layers)
        results["fixed_cost_estimate_gb"] = fixed_info["total_fixed_gb"]
        print(f"   Fixed costs: {fixed_info['total_fixed_gb']:.3f} GB")

        # 2. Measure activation per sample
        print("\n2. Measuring activation memory per sample...")
        result_b1 = self.profile_config(batch_size=1, num_layers=num_layers, checkpointing=True)
        result_b8 = self.profile_config(batch_size=8, num_layers=num_layers, checkpointing=True)

        if result_b1.error or result_b8.error:
            print(f"   ERROR: {result_b1.error or result_b8.error}")
            return results

        activation_per_sample = (result_b8.peak_allocated_gb - result_b1.peak_allocated_gb) / 7
        results["activation_per_sample_gb"] = activation_per_sample
        print(f"   Activation per sample: {activation_per_sample * 1024:.1f} MB")

        # 3. Calculate predicted VRAM for batch_size
        predicted_vram = fixed_info["total_fixed_gb"] + (batch_size * activation_per_sample)
        results["predicted_vram_gb"] = predicted_vram
        print(f"\n3. Predicted VRAM for batch {batch_size}:")
        print(f"   Fixed: {fixed_info['total_fixed_gb']:.3f} GB + Activations: {batch_size * activation_per_sample:.3f} GB")
        print(f"   = {predicted_vram:.3f} GB predicted")

        # 4. Run actual training step and measure
        print(f"\n4. Running actual training step with batch={batch_size}...")
        actual_result = self.profile_config(
            batch_size=batch_size,
            num_layers=num_layers,
            checkpointing=True,
            with_optimizer=True
        )

        if actual_result.error:
            print(f"   ERROR: {actual_result.error}")
            return results

        actual_vram = actual_result.peak_allocated_gb
        results["actual_vram_gb"] = actual_vram
        print(f"   Actual peak VRAM: {actual_vram:.3f} GB")

        # 5. Calculate accuracy
        difference = actual_vram - predicted_vram
        accuracy_pct = (1 - abs(difference) / actual_vram) * 100
        results["difference_gb"] = difference
        results["accuracy_pct"] = accuracy_pct

        print(f"\n" + "-" * 70)
        print("VALIDATION RESULTS")
        print("-" * 70)
        print(f"   Predicted:   {predicted_vram:.3f} GB")
        print(f"   Actual:      {actual_vram:.3f} GB")
        print(f"   Difference:  {difference:+.3f} GB ({difference/actual_vram*100:+.1f}%)")
        print(f"   Accuracy:    {accuracy_pct:.1f}%")

        if accuracy_pct >= 90:
            print(f"\n   ✓ Profiler accuracy is GOOD (>= 90%)")
        elif accuracy_pct >= 80:
            print(f"\n   ~ Profiler accuracy is ACCEPTABLE (80-90%)")
        else:
            print(f"\n   ✗ Profiler accuracy needs improvement (< 80%)")

        print("=" * 70)

        return results

    def run_matrix(
        self,
        batch_sizes: List[int],
        layer_counts: List[int],
        test_checkpointing: bool = True,
    ) -> List[ProfileResult]:
        """Run full test matrix."""

        checkpoint_options = [True, False] if test_checkpointing else [True]
        total_tests = len(batch_sizes) * len(layer_counts) * len(checkpoint_options)
        current = 0

        print(f"\n{'='*70}")
        print("VRAM PROFILING")
        print(f"{'='*70}")
        print(f"GPU: {self.gpu_name}")
        print(f"Total VRAM: {self.total_vram:.1f} GB")
        print(f"Test matrix: {len(batch_sizes)} batch sizes x {len(layer_counts)} layer configs x {len(checkpoint_options)} checkpoint modes")
        print(f"Total configurations: {total_tests}")
        print(f"{'='*70}\n")

        for num_layers in layer_counts:
            print(f"\n--- Testing {num_layers} layers ---")
            for batch_size in batch_sizes:
                for checkpointing in checkpoint_options:
                    current += 1
                    ckpt_str = "ON " if checkpointing else "OFF"
                    print(f"[{current:3d}/{total_tests}] layers={num_layers:2d}, "
                          f"batch={batch_size:2d}, checkpoint={ckpt_str}", end="  ")

                    result = self.profile_config(
                        batch_size=batch_size,
                        num_layers=num_layers,
                        checkpointing=checkpointing,
                    )
                    self.results.append(result)

                    if result.error:
                        print(f"ERROR: {result.error}")
                    else:
                        overflow_str = " [OVERFLOW!]" if result.overflow_detected else ""
                        print(f"VRAM: {result.peak_allocated_gb:5.2f} GB, "
                              f"Time: {result.total_ms:6.1f} ms, "
                              f"Speed: {result.samples_per_sec:5.1f} img/s{overflow_str}")

        return self.results

    def save_csv(self, path: Path):
        """Save results to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'batch_size', 'num_layers', 'checkpointing', 'with_optimizer',
                'peak_allocated_gb', 'peak_reserved_gb',
                'forward_ms', 'backward_ms', 'total_ms',
                'samples_per_sec', 'overflow_detected', 'error'
            ])
            for r in self.results:
                writer.writerow([
                    r.batch_size, r.num_layers, r.checkpointing, r.with_optimizer,
                    f"{r.peak_allocated_gb:.3f}" if r.peak_allocated_gb >= 0 else "N/A",
                    f"{r.peak_reserved_gb:.3f}" if r.peak_reserved_gb >= 0 else "N/A",
                    f"{r.forward_ms:.2f}" if r.forward_ms >= 0 else "N/A",
                    f"{r.backward_ms:.2f}" if r.backward_ms >= 0 else "N/A",
                    f"{r.total_ms:.2f}" if r.total_ms >= 0 else "N/A",
                    f"{r.samples_per_sec:.2f}" if r.samples_per_sec > 0 else "N/A",
                    r.overflow_detected, r.error or ""
                ])
        print(f"\nResults saved to: {path}")

    def print_summary(self):
        """Print comprehensive summary analysis."""
        print("\n" + "=" * 70)
        print("VRAM PROFILING SUMMARY")
        print("=" * 70)

        # Find baseline (28 layers, batch 36, checkpointing ON)
        baseline = next(
            (r for r in self.results
             if r.num_layers == 28 and r.batch_size == 36 and r.checkpointing and not r.error),
            None
        )

        if baseline:
            print(f"\nBaseline (28 layers, batch 36, checkpoint ON):")
            print(f"  VRAM: {baseline.peak_allocated_gb:.2f} GB")
            print(f"  Time: {baseline.total_ms:.1f} ms/step")
            print(f"  Speed: {baseline.samples_per_sec:.1f} samples/sec")

        # Checkpointing impact analysis
        print("\n" + "-" * 70)
        print("CHECKPOINTING IMPACT (same batch size, same layers)")
        print("-" * 70)
        print(f"{'Layers':>6} {'Batch':>6} {'Ckpt ON':>12} {'Ckpt OFF':>12} {'VRAM Diff':>12} {'Speedup':>10}")

        unique_configs = set((r.num_layers, r.batch_size) for r in self.results)
        for num_layers, batch_size in sorted(unique_configs):
            on = next((r for r in self.results
                       if r.batch_size == batch_size and r.num_layers == num_layers
                       and r.checkpointing and not r.error), None)
            off = next((r for r in self.results
                        if r.batch_size == batch_size and r.num_layers == num_layers
                        and not r.checkpointing and not r.error), None)

            if on and off:
                vram_diff = off.peak_allocated_gb - on.peak_allocated_gb
                if on.total_ms > 0:
                    speedup = (on.total_ms - off.total_ms) / on.total_ms * 100
                else:
                    speedup = 0
                print(f"{num_layers:>6} {batch_size:>6} {on.peak_allocated_gb:>10.2f}GB "
                      f"{off.peak_allocated_gb:>10.2f}GB {vram_diff:>+10.2f}GB {speedup:>+9.1f}%")
            elif on:
                print(f"{num_layers:>6} {batch_size:>6} {on.peak_allocated_gb:>10.2f}GB {'OOM':>12} {'N/A':>12} {'N/A':>10}")
            elif off:
                print(f"{num_layers:>6} {batch_size:>6} {'OOM':>12} {off.peak_allocated_gb:>10.2f}GB {'N/A':>12} {'N/A':>10}")

        # Layer count impact
        print("\n" + "-" * 70)
        print("LAYER COUNT IMPACT (batch 36, checkpoint ON)")
        print("-" * 70)

        layer_results = [r for r in self.results
                        if r.batch_size == 36 and r.checkpointing and not r.error]
        layer_results.sort(key=lambda x: x.num_layers)

        if layer_results:
            print(f"{'Layers':>8} {'VRAM':>10} {'Time':>10} {'Speed':>12}")
            for r in layer_results:
                print(f"{r.num_layers:>8} {r.peak_allocated_gb:>8.2f}GB "
                      f"{r.total_ms:>8.1f}ms {r.samples_per_sec:>10.1f}img/s")

        # Maximum batch sizes
        print("\n" + "-" * 70)
        print(f"MAXIMUM BATCH SIZES (within {self.total_vram:.0f}GB VRAM)")
        print("-" * 70)

        for num_layers in sorted(set(r.num_layers for r in self.results)):
            for ckpt in [True, False]:
                valid = [r for r in self.results
                        if r.num_layers == num_layers and r.checkpointing == ckpt
                        and not r.error and r.peak_allocated_gb <= self.total_vram]
                if valid:
                    max_batch = max(r.batch_size for r in valid)
                    max_result = next(r for r in valid if r.batch_size == max_batch)
                    ckpt_str = "ON" if ckpt else "OFF"
                    print(f"  {num_layers} layers, ckpt {ckpt_str}: max batch={max_batch}, "
                          f"VRAM={max_result.peak_allocated_gb:.2f}GB, "
                          f"speed={max_result.samples_per_sec:.1f}img/s")

        # Recommendations
        print("\n" + "-" * 70)
        print("RECOMMENDATIONS")
        print("-" * 70)

        # Find best throughput config within VRAM limit
        valid_results = [r for r in self.results
                        if not r.error and r.peak_allocated_gb <= self.total_vram]
        if valid_results:
            best = max(valid_results, key=lambda x: x.samples_per_sec)
            ckpt_str = "ON" if best.checkpointing else "OFF"
            print(f"\nHighest throughput within VRAM limit:")
            print(f"  Layers: {best.num_layers}")
            print(f"  Batch size: {best.batch_size}")
            print(f"  Checkpointing: {ckpt_str}")
            print(f"  VRAM: {best.peak_allocated_gb:.2f} GB")
            print(f"  Speed: {best.samples_per_sec:.1f} samples/sec")


def main():
    parser = argparse.ArgumentParser(
        description="VRAM Profiler for OppaiOracle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m profiling.vram_profiler --baseline              # Measure fixed costs first
  python -m profiling.vram_profiler --baseline --layers 24  # Baseline for 24 layers
  python -m profiling.vram_profiler --layer-sweep           # Find max layers (ckpt OFF)
  python -m profiling.vram_profiler --layer-sweep --layer-range 20,60  # Custom range
  python -m profiling.vram_profiler --validate              # Validate profiler accuracy
  python -m profiling.vram_profiler --validate --layers 28 --validate-batch 16  # Custom validation
  python -m profiling.vram_profiler --quick                 # Quick batch/layer matrix
  python -m profiling.vram_profiler --batch-sizes 24,32,36,40 --layers 24,28
        """
    )
    parser.add_argument(
        '--baseline', action='store_true',
        help='Run baseline measurements to understand fixed costs (model, optimizer, per-sample activation)'
    )
    parser.add_argument(
        '--batch-sizes', type=str, default='8,16,24,28,32,36,40',
        help='Comma-separated batch sizes to test (default: 8,16,24,28,32,36,40)'
    )
    parser.add_argument(
        '--layers', type=str, default='16,20,24,28',
        help='Comma-separated layer counts to test (default: 16,20,24,28)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test with minimal configs (batch 24,36 x layers 24,28)'
    )
    parser.add_argument(
        '--no-checkpoint-test', action='store_true',
        help='Only test with checkpointing ON (skip OFF tests)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path (default: profiling/results/vram_profile_TIMESTAMP.csv)'
    )
    parser.add_argument(
        '--warmup', type=int, default=2,
        help='Number of warmup iterations (default: 2)'
    )
    parser.add_argument(
        '--measure', type=int, default=3,
        help='Number of measurement iterations (default: 3)'
    )
    parser.add_argument(
        '--layer-sweep', action='store_true',
        help='Sweep layer counts to find max viable layers (checkpointing OFF, fixed costs only)'
    )
    parser.add_argument(
        '--layer-range', type=str, default='16,48',
        help='Layer range for sweep: min,max (default: 16,48)'
    )
    parser.add_argument(
        '--layer-step', type=int, default=4,
        help='Step size for layer sweep (default: 4)'
    )
    parser.add_argument(
        '--vram-budget', type=float, default=None,
        help='VRAM budget in GB for layer sweep (default: auto-detect GPU total)'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Validate profiler accuracy by comparing estimate with actual training step'
    )
    parser.add_argument(
        '--validate-batch', type=int, default=8,
        help='Batch size for validation mode (default: 8)'
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This profiler requires a GPU.")
        sys.exit(1)

    profiler = VRAMProfiler(
        warmup_iterations=args.warmup,
        measure_iterations=args.measure
    )

    # Baseline mode: measure fixed costs
    if args.baseline:
        # Use first layer count if specified, else default to 28
        layer_counts = [int(x.strip()) for x in args.layers.split(',')]
        num_layers = layer_counts[0] if layer_counts else 28
        profiler.run_baseline(num_layers=num_layers)
        return

    # Layer sweep mode: find max layers with checkpointing OFF
    if args.layer_sweep:
        min_layers, max_layers = [int(x.strip()) for x in args.layer_range.split(',')]
        results = profiler.sweep_layers(
            layer_range=(min_layers, max_layers),
            step=args.layer_step,
            vram_budget_gb=args.vram_budget,
        )

        # Optionally save results to CSV
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'num_layers', 'param_count_m', 'model_params_gb',
                    'gradient_buffers_gb', 'optimizer_states_gb', 'total_fixed_gb',
                    'optimizer_type', 'success', 'error'
                ])
                for r in results["layer_results"]:
                    writer.writerow([
                        r["num_layers"],
                        f"{r['param_count_m']:.1f}" if r["param_count_m"] >= 0 else "N/A",
                        f"{r['model_params_gb']:.3f}" if r["model_params_gb"] >= 0 else "N/A",
                        f"{r['gradient_buffers_gb']:.3f}" if r["gradient_buffers_gb"] >= 0 else "N/A",
                        f"{r['optimizer_states_gb']:.3f}" if r["optimizer_states_gb"] >= 0 else "N/A",
                        f"{r['total_fixed_gb']:.3f}" if r["total_fixed_gb"] >= 0 else "N/A",
                        r["optimizer_type"],
                        r["success"],
                        r["error"] or ""
                    ])
            print(f"\nResults saved to: {output_path}")
        return

    # Validate mode: compare profiler estimate with actual training
    if args.validate:
        layer_counts = [int(x.strip()) for x in args.layers.split(',')]
        num_layers = layer_counts[0] if layer_counts else 28
        profiler.validate_against_training(num_layers=num_layers, batch_size=args.validate_batch)
        return

    # Matrix mode: test multiple configurations
    if args.quick:
        batch_sizes = [24, 36]
        layer_counts = [24, 28]
    else:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
        layer_counts = [int(x.strip()) for x in args.layers.split(',')]

    print(f"Batch sizes to test: {batch_sizes}")
    print(f"Layer counts to test: {layer_counts}")
    print(f"Checkpointing: {'ON only' if args.no_checkpoint_test else 'ON and OFF'}")

    profiler.run_matrix(
        batch_sizes,
        layer_counts,
        test_checkpointing=not args.no_checkpoint_test
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "results" / f"vram_profile_{timestamp}.csv"

    profiler.save_csv(output_path)
    profiler.print_summary()


if __name__ == "__main__":
    main()
