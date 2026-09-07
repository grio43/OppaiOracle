"""Bounded CUDA training probe using synthetic images; never reads the dataset.

On Windows, run Start_AI_Training.ps1 -CheckTrainingStack to set up MSVC/SDK.
Uses the configured patch grid, width, heads, MLP, precision and compile mode,
but only two transformer blocks, two images and 1,024 synthetic labels.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from Configuration_System import load_config
from loss_functions import AsymmetricFocalLoss
from model_architecture import VisionTransformerConfig, create_model
from training_utils import TrainingUtils


def check_training_stack(config_path: str, steps: int = 5) -> dict:
    if steps < 2:
        raise ValueError("Use at least two steps; the first step includes compilation.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The training probe requires a CUDA GPU with bfloat16 support.")
    import triton
    import torchvision  # Verify that the companion wheel can actually load.

    cfg = load_config(config_path)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(cfg.training.benchmark)
    torch._dynamo.config.suppress_errors = False

    model_cfg = VisionTransformerConfig(
        image_size=cfg.data.image_size,
        patch_size=cfg.model.patch_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=2,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        num_tags=1024,
        qk_norm=cfg.model.qk_norm,
        layer_scale_init=cfg.model.layer_scale_init,
        gradient_checkpointing=True,
        checkpoint_every_n_layers=cfg.model.checkpoint_every_n_layers,
        # Disable randomness so SDPA and compiled Flex can be compared.
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    model = create_model(model_cfg).cuda().train()
    optimizer = TrainingUtils.get_optimizer(
        model, "adamw8bit", 5.4e-4, .05, eps=1e-7, fp32_head_optimizer=True,
    )
    criterion = AsymmetricFocalLoss(
        gamma_pos=0., gamma_neg=7., alpha=1., clip=.05,
        label_smoothing=0., ignore_indices=[0, 1], detach_focal_weight=True,
    )
    images = torch.randn(2, 3, model_cfg.image_size, model_cfg.image_size,
                         device="cuda", dtype=torch.bfloat16)
    masks = torch.zeros(2, model_cfg.image_size, model_cfg.image_size,
                        device="cuda", dtype=torch.bool)
    masks[0, :3 * model_cfg.patch_size] = True
    masks[1, :, -2 * model_cfg.patch_size:] = True
    targets = torch.zeros(2, model_cfg.num_tags, device="cuda")
    targets[:, 2:40] = 1
    targets[0, -4:] = -1  # Exercise the unobserved-rating loss path.

    # Compare actual model logits/loss/gradients to its existing SDPA path.
    model.set_onnx_mode(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_logits = model(images, padding_mask=masks)["tag_logits"]
        reference_loss = criterion(reference_logits, targets)
    reference_loss.backward()
    checked_parameters = {
        "qkv": model.blocks[0].qkv.weight,
        "mlp": model.blocks[0].mlp[0].weight,
        "tag_head": model.tag_head.weight,
    }
    reference_grads = {name: param.grad.detach().clone()
                       for name, param in checked_parameters.items()}
    reference_logits = reference_logits.detach()
    reference_loss = reference_loss.detach()
    optimizer.zero_grad(set_to_none=True)
    model.set_onnx_mode(False)

    compiled = torch.compile(model, mode=cfg.training.compile_mode,
                             dynamic=cfg.training.compile_dynamic,
                             fullgraph=cfg.training.compile_fullgraph)
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "triton": triton.__version__,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compile_mode": cfg.training.compile_mode,
        "shape": {"batch": 2, "image_size": model_cfg.image_size,
                  "width": model_cfg.hidden_size, "layers": 2, "tags": 1024},
        "steps": [],
    }
    print(json.dumps({"environment": report}), flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = compiled(images, padding_mask=masks)["tag_logits"]
            loss = criterion(logits, targets)
        loss.backward()
        if step == 0:
            torch.testing.assert_close(logits, reference_logits, rtol=.03, atol=.02)
            torch.testing.assert_close(loss, reference_loss, rtol=.02, atol=1e-5)
            relative_errors = {}
            for name, param in checked_parameters.items():
                reference = reference_grads[name].float()
                error = (param.grad.float() - reference).norm() / reference.norm().clamp_min(1e-12)
                if not torch.isfinite(error) or error > .05:
                    raise AssertionError(f"{name} gradient differs from SDPA: {error.item():.4g}")
                relative_errors[name] = error.item()
            report["gradient_relative_errors"] = relative_errors
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite loss")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        row = {"step": step, "seconds": time.perf_counter() - started,
               "loss": loss.item(), "grad_norm": norm.item()}
        report["steps"].append(row)
        print(json.dumps(row), flush=True)

    if optimizer.state[model.tag_head.weight]["state1"].dtype != torch.float32:
        raise AssertionError("Tag head optimizer state must stay fp32")
    if optimizer.state[model.blocks[0].mlp[0].weight]["state1"].dtype != torch.uint8:
        raise AssertionError("Backbone optimizer state must use 8-bit quantization")
    compiled.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in (2, 1):
            logits = compiled(images[:batch], padding_mask=masks[:batch])["tag_logits"]
            if not torch.isfinite(logits).all():
                raise AssertionError(f"Nonfinite validation logits for batch size {batch}")
    torch.cuda.synchronize()
    report["median_warm_step_seconds"] = statistics.median(
        row["seconds"] for row in report["steps"][1:])
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["status"] = "passed"
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified_config.yaml")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    result = check_training_stack(args.config, args.steps)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
