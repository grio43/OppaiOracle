"""Diagnose the F1 scoring system end-to-end.

Runs the model on N validation batches, then asks five questions:

  1. Are the probabilities being computed and passed to the metric the way we
     think they are? (Logit/sigmoid/threshold pipeline.)
  2. Does the in-training F1 number match an independent recomputation
     (sklearn) on the same arrays?
  3. What threshold maximizes F1, and how far off is the configured threshold?
  4. What does precision/recall actually look like at the thresholds we care
     about (configured 0.2653, the screenshot's ~0.65, optimal)?

Usage:
    l:/Dab/payton_env/Scripts/python.exe tools/diagnose_f1.py \\
        --checkpoint experiments/run1_vit/checkpoints/checkpoint_epoch_5_step_30000.pt \\
        --batches 20

Prints a single report block. Does not modify config or write to TensorBoard.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Configuration_System import load_config
from dataset_loader import create_dataloaders
from model_architecture import create_model
from torchmetrics.functional.classification import multilabel_f1_score
from sklearn.metrics import f1_score as sk_f1_score


SKIP_COLS = 2  # PAD=0, UNK=1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "unified_config.yaml")
    p.add_argument("--batches", type=int, default=20,
                   help="Number of val batches to score (more = tighter numbers, slower).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _resolve_active_data_path(config) -> Path:
    """Mirror train_direct.py's first-enabled-storage-location selection."""
    active = next((loc for loc in config.data.storage_locations if loc.get("enabled")), None)
    if not active:
        raise ValueError("No enabled storage location in config.data.storage_locations")
    p = Path(active["path"])
    if not p.is_dir():
        raise NotADirectoryError(f"Active data path missing or not a directory: {p}")
    return p


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove `_orig_mod.` prefix added by torch.compile and `module.` from DDP."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def load_model_and_val_loader(args):
    config = load_config(str(args.config))
    active_data_path = _resolve_active_data_path(config)

    # ---- Val loader & vocab ------------------------------------------------
    train_loader, val_loader, vocab = create_dataloaders(
        data_config=config.data,
        validation_config=config.validation,
        vocab_path=Path(config.vocab_path),
        active_data_path=active_data_path,
        seed=getattr(config.training, "seed", 42),
        debug_config=config.debug,
        architecture_type=config.model.architecture_type,
        patch_size=getattr(config.model, "patch_size", None),
    )
    # Drop training workers — we don't need them.
    try:
        from train_direct import _shutdown_dataloader_workers
        _shutdown_dataloader_workers(train_loader)
    except Exception:
        pass

    num_tags = len(vocab.tag_to_index)

    # ---- Build model the same way train_direct does -----------------------
    model_config = config.model.to_dict()
    model_config["num_tags"] = num_tags

    unused = {
        "architecture_type", "hidden_dropout_prob", "initializer_range",
        "num_groups", "num_labels", "tags_per_group", "swin_config",
    }
    filtered = {k: v for k, v in model_config.items() if k not in unused}
    model = create_model(**filtered)

    # ---- Load checkpoint weights ------------------------------------------
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict") or raw.get("state_dict") or raw
    sd = _strip_compile_prefix(sd)
    # Strip rating_head keys that no longer exist on this architecture.
    sd = {k: v for k, v in sd.items() if "rating_head" not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    if missing and len(missing) > 50:
        raise RuntimeError("Too many missing keys — checkpoint architecture likely mismatched.")
    model = model.to(args.device).eval()
    return model, val_loader, vocab, config


@torch.no_grad()
def collect_predictions(model, val_loader, n_batches: int, device: str):
    """Run the model on n_batches and return (probs, targets) on CPU."""
    probs_chunks, targ_chunks = [], []
    started = time.time()
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        images = batch["images"].to(device, non_blocking=True).float()
        targets = batch["tag_labels"]  # CPU long/float tensor
        outputs = model(images)
        logits = outputs["tag_logits"].float()
        probs = torch.sigmoid(logits).cpu()
        probs_chunks.append(probs)
        targ_chunks.append(targets.cpu())
        if (i + 1) % 5 == 0:
            print(f"  batch {i + 1}/{n_batches}  elapsed={time.time() - started:.1f}s", flush=True)
    probs = torch.cat(probs_chunks, dim=0)
    targs = torch.cat(targ_chunks, dim=0)
    if targs.dtype.is_floating_point:
        targs = (targs > 0.5).long()
    else:
        targs = targs.long()
    return probs, targs


def f1_at_threshold(probs: torch.Tensor, targs: torch.Tensor, thr: float):
    """Macro/micro F1 averaged only over classes with positive support — matches
    train_direct.py's `keep_classes = (val_pos_counts > 0)` filter."""
    pos = targs.sum(dim=0)
    keep = pos > 0
    p = probs[:, keep]
    t = targs[:, keep]
    n_labels = int(keep.sum().item())
    if n_labels == 0:
        return float("nan"), float("nan")
    f_macro = multilabel_f1_score(p, t, num_labels=n_labels, average="macro", threshold=thr).item()
    f_micro = multilabel_f1_score(p, t, num_labels=n_labels, average="micro", threshold=thr).item()
    return f_macro, f_micro


def sklearn_cross_check(probs: torch.Tensor, targs: torch.Tensor, thr: float):
    """Recompute macro/micro F1 with sklearn over supported classes only."""
    pos = targs.sum(dim=0).numpy()
    keep = pos > 0
    p_bin = (probs[:, keep].numpy() > thr).astype(np.int8)
    t = targs[:, keep].numpy().astype(np.int8)
    macro = sk_f1_score(t, p_bin, average="macro", zero_division=0)
    micro = sk_f1_score(t, p_bin, average="micro", zero_division=0)
    return float(macro), float(micro)


def per_image_pr(probs: torch.Tensor, targs: torch.Tensor, thr: float):
    """Mean precision/recall computed *per image*, then averaged across images.

    This is what a human looking at one image's tag list (the screenshot view)
    actually experiences — distinct from class-averaged macro F1.
    """
    pred = (probs > thr)
    targ = targs.bool()
    tp = (pred & targ).sum(dim=1).float()
    fp = (pred & ~targ).sum(dim=1).float()
    fn = (~pred & targ).sum(dim=1).float()
    prec = tp / (tp + fp).clamp_min(1e-9)
    rec = tp / (tp + fn).clamp_min(1e-9)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    return prec.mean().item(), rec.mean().item(), f1.mean().item(), pred.sum(dim=1).float().mean().item()


def main():
    args = parse_args()
    print(f"[setup] checkpoint = {args.checkpoint}")
    print(f"[setup] device     = {args.device}")
    print(f"[setup] batches    = {args.batches}")
    model, val_loader, vocab, config = load_model_and_val_loader(args)
    cfg_thr = float(config.inference.prediction_threshold)
    print(f"[setup] num_tags   = {len(vocab.tag_to_index)}")
    print(f"[setup] cfg thr    = {cfg_thr}")

    print("\n[run] collecting predictions...")
    probs_full, targs_full = collect_predictions(model, val_loader, args.batches, args.device)
    n_samples, n_tags = probs_full.shape
    print(f"[run] collected probs={tuple(probs_full.shape)}  targets={tuple(targs_full.shape)}")

    # Strip PAD/UNK exactly like training does.
    probs = probs_full[:, SKIP_COLS:]
    targs = targs_full[:, SKIP_COLS:]
    n_tags_eff = probs.shape[1]
    print(f"[run] after skip: probs={tuple(probs.shape)}")

    # ------------------------------------------------------------------------
    # 1. Probability distribution sanity
    # ------------------------------------------------------------------------
    flat = probs.flatten()
    qs = torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999])
    quantiles = torch.quantile(flat[:: max(1, flat.numel() // 200_000)], qs).tolist()
    print("\n========== 1. Probability distribution ==========")
    print(f"  min/max:        {flat.min():.4f} / {flat.max():.4f}")
    print(f"  mean / median:  {flat.mean():.4f} / {flat.median():.4f}")
    print(f"  quantiles 50/90/99/99.9/99.99:")
    for q, v in zip(qs.tolist(), quantiles):
        print(f"    p{q:>7.4f} = {v:.4f}")
    above = (probs > cfg_thr).float().sum(dim=1).mean().item()
    print(f"  mean tags above cfg threshold ({cfg_thr}) per image = {above:.1f}")
    above_065 = (probs > 0.65).float().sum(dim=1).mean().item()
    print(f"  mean tags above 0.65 per image                      = {above_065:.1f}")
    above_05 = (probs > 0.5).float().sum(dim=1).mean().item()
    print(f"  mean tags above 0.50 per image                      = {above_05:.1f}")
    true_per_img = targs.sum(dim=1).float().mean().item()
    print(f"  mean TRUE tags per image                            = {true_per_img:.1f}")

    # ------------------------------------------------------------------------
    # 2. F1 at config threshold + sklearn cross-check (catches metric bugs)
    # ------------------------------------------------------------------------
    print("\n========== 2. F1 at config threshold (with sklearn cross-check) ==========")
    tm_macro, tm_micro = f1_at_threshold(probs, targs, cfg_thr)
    sk_macro, sk_micro = sklearn_cross_check(probs, targs, cfg_thr)
    print(f"  threshold = {cfg_thr}")
    print(f"  torchmetrics  macro={tm_macro:.4f}  micro={tm_micro:.4f}")
    print(f"  sklearn       macro={sk_macro:.4f}  micro={sk_micro:.4f}")
    drift_macro = abs(tm_macro - sk_macro)
    drift_micro = abs(tm_micro - sk_micro)
    print(f"  |drift|       macro={drift_macro:.5f}  micro={drift_micro:.5f}")
    if drift_macro > 0.01 or drift_micro > 0.01:
        print("  >>> METRIC MISMATCH — torchmetrics vs sklearn disagree by >0.01.")
        print("  >>> The scoring code itself may have a bug. Investigate before trusting any F1.")
    else:
        print("  -> Metrics agree. F1 calculation itself is correct.")

    # ------------------------------------------------------------------------
    # 3. Threshold sweep
    # ------------------------------------------------------------------------
    print("\n========== 3. Threshold sweep ==========")
    thresholds = [round(x, 3) for x in np.arange(0.05, 0.96, 0.05)]
    print(f"  {'thr':>6} {'mean_active':>12} {'macroF1':>9} {'microF1':>9}")
    sweep = []
    for thr in thresholds:
        f_macro, f_micro = f1_at_threshold(probs, targs, thr)
        ma = (probs > thr).float().sum(dim=1).mean().item()
        sweep.append((thr, ma, f_macro, f_micro))
        print(f"  {thr:>6.3f} {ma:>12.1f} {f_macro:>9.4f} {f_micro:>9.4f}")
    best_macro = max(sweep, key=lambda x: x[2])
    best_micro = max(sweep, key=lambda x: x[3])
    print(f"\n  best macro-F1: thr={best_macro[0]:.3f}  F1={best_macro[2]:.4f}  mean_active={best_macro[1]:.1f}")
    print(f"  best micro-F1: thr={best_micro[0]:.3f}  F1={best_micro[3]:.4f}  mean_active={best_micro[1]:.1f}")

    # ------------------------------------------------------------------------
    # 4. Per-image precision/recall (what the human-in-the-loop sees)
    # ------------------------------------------------------------------------
    print("\n========== 4. Per-image precision/recall (the screenshot view) ==========")
    probe_thrs = sorted({cfg_thr, 0.5, 0.65, best_macro[0]})
    print(f"  {'thr':>6} {'pred/img':>10} {'precision':>11} {'recall':>9} {'F1':>8}")
    for thr in probe_thrs:
        pr, rc, f1m, np_per = per_image_pr(probs, targs, thr)
        print(f"  {thr:>6.3f} {np_per:>10.1f} {pr:>11.3f} {rc:>9.3f} {f1m:>8.3f}")

    # ------------------------------------------------------------------------
    # 5. Mass distribution: where do the FPs live?
    # ------------------------------------------------------------------------
    print("\n========== 5. Where are the false positives concentrated? ==========")
    bands = [(0.05, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.01)]
    targ_bool = targs.bool()
    for lo, hi in bands:
        mask = (probs >= lo) & (probs < hi)
        n_total = mask.sum().item()
        n_tp = (mask & targ_bool).sum().item()
        n_fp = n_total - n_tp
        prec = n_tp / max(1, n_total)
        per_img = n_total / probs.shape[0]
        print(f"  [{lo:.2f}, {hi:.2f})  total/img={per_img:>7.1f}  TPs={n_tp:>6}  FPs={n_fp:>7}  band-precision={prec:.3f}")

    print("\n[done] If macro-F1 jumps from ~0.01 at thr=0.27 to >0.10 at thr=0.55-0.70,")
    print("[done] the F1 system is correct; the deployed threshold is the entire issue.")


if __name__ == "__main__":
    main()
