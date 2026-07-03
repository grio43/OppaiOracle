"""V1.1 release helper.

Builds the safetensors (bfloat16) artifact and copies supporting files for
the huggingface_release/V1.1_safetensors directory.

Run from the project root with the project's Python environment.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def strip_state_dict_prefixes(sd: dict) -> dict:
    """Drop torch.compile / DDP prefixes; drop legacy rating_head keys."""
    out = {}
    for k, v in sd.items():
        if "rating_head" in k:
            continue
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        out[nk] = v
    return out


def cast_state_dict(sd: dict, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in sd.items():
        if v.is_floating_point():
            out[k] = v.detach().to(dtype).contiguous()
        else:
            out[k] = v.detach().contiguous()
    return out


def write_selected_tags_csv(vocab: dict, csv_path: Path) -> int:
    index_to_tag = {int(k): v for k, v in vocab["index_to_tag"].items()}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag_id", "name", "category"])
        for idx in sorted(index_to_tag.keys()):
            w.writerow([idx, index_to_tag[idx], 0])
    return len(index_to_tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=r"L:\Dab\OppaiOracle\experiments\run1_vit\checkpoints\last.pt",
    )
    ap.add_argument(
        "--out_dir",
        default=r"L:\Dab\OppaiOracle\huggingface_release\V1.1_safetensors",
    )
    ap.add_argument(
        "--pr_thresholds",
        default=r"L:\Dab\OppaiOracle\experiments\run1_vit\checkpoints\pr_threshold_last.json",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    epoch = int(ckpt.get("epoch", -1))
    step = int(ckpt.get("step", -1))
    cfg = ckpt.get("config", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    pre_params = ckpt.get("preprocessing_params", {}) or {}

    image_size = int(pre_params.get("image_size") or data_cfg.get("image_size") or model_cfg.get("image_size") or 448)
    patch_size = int(pre_params.get("patch_size") or model_cfg.get("patch_size") or 16)
    norm_mean = list(pre_params.get("normalize_mean") or data_cfg.get("normalize_mean") or [0.5, 0.5, 0.5])
    norm_std = list(pre_params.get("normalize_std") or data_cfg.get("normalize_std") or [0.5, 0.5, 0.5])
    pad_color = list(data_cfg.get("pad_color") or [114, 114, 114])
    # color_order: prefer checkpoint's recorded order, then data config, default
    # to RGB (the project default and what V1.x was trained with). Historical
    # artifacts already on disk are NOT touched.
    color_order = (
        pre_params.get("color_order")
        or data_cfg.get("color_order")
        or "RGB"
    )
    color_order = str(color_order).upper()
    if color_order not in ("RGB", "BGR"):
        print(f"WARNING: unknown color_order '{color_order}'; defaulting to RGB")
        color_order = "RGB"

    state_dict_raw = ckpt["state_dict"]
    state_dict = strip_state_dict_prefixes(state_dict_raw)

    head_w = state_dict["tag_head.weight"]
    num_labels = int(head_w.shape[0])
    print(f"  epoch={epoch} step={step} image_size={image_size} num_labels={num_labels}")

    # Cast to bfloat16
    sd_bf16 = cast_state_dict(state_dict, torch.bfloat16)
    safetensors_path = out_dir / "model.safetensors"
    print(f"Writing bfloat16 safetensors -> {safetensors_path}")
    save_file(
        sd_bf16,
        str(safetensors_path),
        metadata={
            "format": "pt",
            "checkpoint_source": "experiments/run1_vit/checkpoints/last.pt",
            "epoch": str(epoch),
            "step": str(step),
            "dtype": "bfloat16",
            "num_labels": str(num_labels),
            "image_size": str(image_size),
            "patch_size": str(patch_size),
            "vocab_format_version": str(ckpt.get("vocab_format_version", "1")),
        },
    )
    print(f"  size: {safetensors_path.stat().st_size / (1024*1024):.1f} MB")

    # Extract embedded vocabulary -> vocabulary.json
    vocab_b64 = ckpt.get("vocab_b64_gzip")
    if not vocab_b64:
        raise RuntimeError("No embedded vocabulary in checkpoint")
    vocab_bytes = gzip.decompress(base64.b64decode(vocab_b64))
    vocab_sha = hashlib.sha256(vocab_bytes).hexdigest()
    vocab = json.loads(vocab_bytes.decode("utf-8"))

    vocab_path = out_dir / "vocabulary.json"
    print(f"Writing vocabulary.json (SHA256: {vocab_sha[:16]}...) -> {vocab_path}")
    with open(vocab_path, "w", encoding="utf-8") as f:
        # Persist canonical JSON (no extra reformatting) so SHA stays stable
        f.write(vocab_bytes.decode("utf-8"))
    assert hashlib.sha256(open(vocab_path, "rb").read()).hexdigest() == vocab_sha

    # selected_tags.csv (SmilingWolf-style)
    csv_path = out_dir / "selected_tags.csv"
    n_csv = write_selected_tags_csv(vocab, csv_path)
    print(f"Wrote {n_csv} rows -> {csv_path}")

    # pr_thresholds.json -- copy fresh thresholds for V1.1 (epoch 7 / step 85517)
    pr_src = Path(args.pr_thresholds)
    pr_dst = out_dir / "pr_thresholds.json"
    if pr_src.exists():
        shutil.copy2(pr_src, pr_dst)
        print(f"Copied pr_thresholds.json -> {pr_dst}")
    else:
        print(f"WARNING: pr_thresholds source not found: {pr_src}")

    # config.json
    config = {
        "architecture_type": str(model_cfg.get("architecture_type", "vit")),
        "num_labels": num_labels,
        "num_channels": int(model_cfg.get("num_channels", 3)),
        "image_size": image_size,
        "patch_size": patch_size,
        "hidden_size": int(model_cfg.get("hidden_size", 1024)),
        "num_hidden_layers": int(model_cfg.get("num_hidden_layers", 18)),
        "num_attention_heads": int(model_cfg.get("num_attention_heads", 16)),
        "intermediate_size": int(model_cfg.get("intermediate_size", 4096)),
        "hidden_dropout_prob": float(model_cfg.get("hidden_dropout_prob", 0.10)),
        "pos_dropout": float(model_cfg.get("pos_dropout", 0.0)),
        "attention_dropout": float(model_cfg.get("attention_dropout", 0.05)),
        "drop_path_rate": float(model_cfg.get("drop_path_rate", 0.20)),
        "initializer_range": float(model_cfg.get("initializer_range", 0.02)),
        "layer_norm_eps": float(model_cfg.get("layer_norm_eps", 1e-6)),
        "use_fp32_layernorm": bool(model_cfg.get("use_fp32_layernorm", False)),
        "attention_bias": bool(model_cfg.get("attention_bias", True)),
        "num_groups": int(model_cfg.get("num_groups", 20)),
        "tags_per_group": int(model_cfg.get("tags_per_group", 10000)),
        "training_epoch": epoch,
        "training_step": step,
        "vocab_format_version": int(ckpt.get("vocab_format_version", 1)),
        "vocab_sha256": vocab_sha,
        "state_dict_keys_format": "plain (no _orig_mod./module. prefixes)",
        "state_dict_dtype": "bfloat16",
        "checkpoint_source": f"experiments/run1_vit/checkpoints/last.pt (epoch {epoch}, step {step})",
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote config.json")

    # preprocessing.json (PyTorch / safetensors flavor)
    preproc = {
        "image_size": image_size,
        "patch_size": patch_size,
        "num_channels": int(model_cfg.get("num_channels", 3)),
        "color_order": color_order,
        "resize_mode": "letterbox",
        # Pad color values are interpreted in the channel order declared above.
        # Renamed from 'pad_color_rgb' since the field is no longer RGB-specific
        # (and the values used in practice are symmetric grays).
        "pad_color": pad_color,
        "normalize_mean": norm_mean,
        "normalize_std": norm_std,
        "input_dtype": "float32",
        "input_layout": "BCHW",
        "padding_mask": {
            "required_for_pytorch_forward": True,
            "shape": "(B, H, W)",
            "dtype": "bool",
            "convention": "True = padded pixel, False = valid pixel",
            "all_false_equivalent_to": "no masking",
        },
        "output": {
            "name": "tag_logits",
            "shape": f"(B, {num_labels})",
            "activation": "apply sigmoid for probabilities",
        },
        "notes": [
            f"Letterbox resize keeps aspect ratio; pad with the pad_color (in {color_order} order) above to reach {image_size}x{image_size}.",
            f"Channel order is {color_order}: load via PIL.convert('RGB') then flip channels to BGR before normalize if color_order=='BGR'.",
            "Normalize per-channel: (x/255 - mean) / std after letterboxing.",
            "Built-in recommended thresholds are in pr_thresholds.json (per-tag and global).",
        ],
    }
    with open(out_dir / "preprocessing.json", "w", encoding="utf-8") as f:
        json.dump(preproc, f, indent=2)
    print(f"Wrote preprocessing.json")

    print("\n[OK] V1.1 safetensors release prepared.")


if __name__ == "__main__":
    main()
