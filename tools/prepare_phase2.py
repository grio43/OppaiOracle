"""Prepare a Phase 1 checkpoint for Phase 2 progressive resolution training.

Resets optimizer/scheduler state and updates embedded config image_size to prevent
validation failures.
Position embedding interpolation is handled automatically by the training resume logic.
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Phase 1 checkpoint into a Phase 2-ready checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to Phase 1 checkpoint (e.g. best_model.pt)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for the Phase 2-ready checkpoint"
    )
    parser.add_argument(
        "--image-size", type=int, required=True,
        help="Target image size for Phase 2 (e.g. 448)"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    out_path = Path(args.output)
    if out_path.exists():
        print(f"ERROR: Output already exists: {out_path}")
        print("       Remove it or choose a different path to avoid accidental overwrites.")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        print("ERROR: Checkpoint has no 'state_dict' key.")
        sys.exit(1)

    # --- 1. Strip optimizer, scheduler, scaler, RNG, sampler state ---
    stripped = []
    for key in [
        "optimizer_state_dict", "optimizer_class",
        "scheduler_state_dict", "scheduler_class", "scheduler_params",
        "scaler_state_dict", "rng_states", "sampler_state",
    ]:
        if key in ckpt:
            del ckpt[key]
            stripped.append(key)
    if stripped:
        print(f"Stripped: {', '.join(stripped)}")

    # --- 2. Reset training counters ---
    ckpt["epoch"] = 0
    ckpt["step"] = 0
    ckpt["training_state"] = {}  # TrainingState.from_dict({}) gives fresh defaults
    ckpt["metrics"] = {}
    ckpt["is_best"] = False
    print("Reset epoch, step, training_state, metrics.")

    # --- 3. Update embedded config image_size ---
    cfg = ckpt.get("config")
    if isinstance(cfg, dict):
        updated = []
        for section in ("data", "model"):
            if isinstance(cfg.get(section), dict) and "image_size" in cfg[section]:
                cfg[section]["image_size"] = args.image_size
                updated.append(f"{section}.image_size")
        val_pre = cfg.get("validation", {})
        if isinstance(val_pre, dict):
            pre = val_pre.get("preprocessing", {})
            if isinstance(pre, dict) and "image_size" in pre:
                pre["image_size"] = args.image_size
                updated.append("validation.preprocessing.image_size")
        if updated:
            print(f"Updated config: {', '.join(updated)} -> {args.image_size}")
    else:
        print("WARNING: No embedded config dict found. You may hit validation errors on resume.")

    # --- 3b. Update top-level preprocessing_params (self-describing checkpoint) ---
    # This block is SEPARATE from ckpt["config"] above. Inference_Engine /
    # validation_loop / ONNX_Export PREFER preprocessing_params over the embedded
    # config, so leaving it at the stale Phase 1 resolution would silently feed the
    # old image_size to those consumers when run directly on this checkpoint.
    pp = ckpt.get("preprocessing_params")
    if isinstance(pp, dict) and "image_size" in pp:
        pp["image_size"] = args.image_size
        print(f"Updated preprocessing_params.image_size -> {args.image_size}")
    elif pp is None:
        print("Note: no top-level preprocessing_params block found (older checkpoint).")
    # Legacy top-level image_size key (extract_preprocessing_params fallback path)
    if "image_size" in ckpt:
        ckpt["image_size"] = args.image_size
        print(f"Updated legacy top-level image_size -> {args.image_size}")

    # --- 4. Save ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nPhase 2 checkpoint saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  - Point resume_from at this file")
    print(f"  - Set image_size={args.image_size} in your config")
    print(f"  - Position embeddings will be interpolated automatically on load")


if __name__ == "__main__":
    main()
