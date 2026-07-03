#!/usr/bin/env python3
"""Find Precision = Recall (P=R) thresholds against the validation set.

Runs a single forward pass over the configured validation split for a
checkpoint, collects sigmoid probabilities + binary targets, then sweeps
thresholds to locate operating points where precision equals recall.

Reports four operating points (PAD/UNK excluded throughout):

  - Micro P=R / F1-opt:
      Single global threshold; precision/recall computed across all
      (sample, tag) pairs aggregated. The standard headline number.

  - Macro P=R / F1-opt (single global threshold, WD14-comparable):
      One threshold for every tag; macro = mean of per-tag F1
      (sklearn / torchmetrics convention). Break-even is the threshold
      where mean(per-tag P) == mean(per-tag R). This is the apples-to-
      apples number against SmilingWolf's "F1 @ P=R threshold" reports.

  - Per-tag P=R / F1-opt (each tag tuned independently):
      Each tag picks its own break-even / argmax-F1 threshold. Strictly
      >= the single-global-threshold macro number above; useful for
      threshold calibration but NOT comparable to single-threshold
      reports from other models.

  - Reported at three support cutoffs (0 / 1 / configurable --min-support):
      Macro is sensitive to how many low-support tags get included.
      WD reports macro over the full trained vocab (no support filter);
      a small val sample inflates that with structural zeros.

Defaults match training: pulls dataset path / vocab / val subsample
(30K) from configs/unified_config.yaml; picks the most recent
checkpoint under experiments/run1_vit/checkpoints/. Pass --full to
evaluate over the entire 5% val split.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

from Configuration_System import load_config
from dataset_loader import create_dataloaders
from model_architecture import create_model


SKIP_INDICES = (0, 1)  # PAD, UNK — match training loss ignore_indices.
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / 'experiments' / 'run1_vit' / 'checkpoints'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=str(PROJECT_ROOT / 'configs' / 'unified_config.yaml'),
                   help='Unified config (used for data path, vocab, val subset, AMP dtype).')
    p.add_argument('--checkpoint', default=None,
                   help='Path to .pt checkpoint. Defaults to most recently modified .pt under '
                        f'{DEFAULT_CHECKPOINT_DIR.relative_to(PROJECT_ROOT)} (excluding best_model.pt). '
                        "Pass 'best' to use best_model.pt or 'last' for last.pt.")
    p.add_argument('--max-samples', type=int, default=None,
                   help='Override validation.max_samples (smaller = faster, less precise). '
                        'Default: value in config (typically 30000).')
    p.add_argument('--full', action='store_true',
                   help='Use the full validation split (~280K-300K samples) for a high-precision '
                        'sweep. Overrides --max-samples and config max-samples.')
    p.add_argument('--threshold-step', type=float, default=0.001,
                   help='Granularity of the threshold sweep (default: 0.001 -> 999 thresholds).')
    p.add_argument('--threshold-min', type=float, default=0.001,
                   help='Lower bound of threshold sweep (default: 0.001).')
    p.add_argument('--threshold-max', type=float, default=0.999,
                   help='Upper bound of threshold sweep (default: 0.999).')
    p.add_argument('--min-support', type=int, default=5,
                   help='Support cutoff for per-tag operating points (default: 5). '
                        'Macro single-threshold numbers are reported at three cutoffs '
                        '(0 / 1 / this value) for transparency.')
    p.add_argument('--no-per-tag', action='store_true',
                   help='Skip per-tag operating-point search (saves a few hundred MB and ~1s).')
    p.add_argument('--save-per-tag-table', action='store_true',
                   help='Embed full per-tag table in the output JSON (large file).')
    p.add_argument('--sample-chunk', type=int, default=8192,
                   help='Sample-axis chunk size for the GPU sweep accumulator (default: 8192). '
                        'Lower if you OOM on the bincount step.')
    p.add_argument('--output', default=None,
                   help='Where to write the JSON report. Default: '
                        '<checkpoint_dir>/pr_threshold_<ckpt_stem>.json')
    p.add_argument('--save-curve', action='store_true',
                   help='Also save the full micro P/R sweep (CSV) for plotting.')
    p.add_argument('--cache-preds', action='store_true', default=True,
                   help='Cache the (probs, targets) tensors to disk after the forward pass '
                        'and reuse them on subsequent runs with the same checkpoint+val-size '
                        '(default: on).')
    p.add_argument('--no-cache-preds', dest='cache_preds', action='store_false',
                   help='Disable prediction caching.')
    return p.parse_args()


def predictions_cache_path(checkpoint_path: Path, n_samples_expected) -> Path:
    """Cache file alongside the checkpoint, keyed by name + expected val size."""
    tag = 'full' if n_samples_expected is None else str(int(n_samples_expected))
    return checkpoint_path.parent / f'_preds_cache_{checkpoint_path.stem}_{tag}.pt'


def resolve_checkpoint(arg) -> Path:
    """Resolve the --checkpoint argument to a concrete .pt path."""
    if arg in (None, 'auto', 'latest'):
        candidates = sorted(
            (p for p in DEFAULT_CHECKPOINT_DIR.glob('*.pt') if p.name != 'best_model.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f'No checkpoint .pt files found under {DEFAULT_CHECKPOINT_DIR}. '
                'Pass --checkpoint <path> explicitly.'
            )
        return candidates[0]
    if arg in ('best', 'last'):
        path = DEFAULT_CHECKPOINT_DIR / f'{arg}_model.pt' if arg == 'best' else DEFAULT_CHECKPOINT_DIR / 'last.pt'
        if not path.exists():
            raise FileNotFoundError(f'{path.name} not found at {path}')
        return path
    path = Path(arg)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return path


def strip_compile_prefix(state_dict):
    if any(k.startswith('_orig_mod.') for k in state_dict):
        return {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def build_model(config, num_tags):
    model_config = config.model.to_dict()
    model_config['num_tags'] = num_tags
    drop_keys = {'architecture_type', 'hidden_dropout_prob', 'initializer_range',
                 'num_groups', 'num_labels', 'tags_per_group', 'swin_config'}
    model_config = {k: v for k, v in model_config.items() if k not in drop_keys}
    return create_model(**model_config)


def collect_predictions(model, val_loader, device, amp_ctx):
    """Run forward pass over val_loader, return (probs fp16 cpu, targets uint8 cpu)."""
    model.eval()
    all_probs, all_targs = [], []
    total = 0
    t1 = time.time()
    log_every = max(25, len(val_loader) // 40)
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            err = batch.get('error')
            if err is not None and isinstance(err, torch.Tensor) and err.any():
                mask = ~err
                if mask.sum() == 0:
                    continue
                batch = {
                    k: v[mask] if isinstance(v, torch.Tensor) and v.size(0) == len(err) else v
                    for k, v in batch.items()
                }
            images = batch['images'].to(device, non_blocking=True)
            tag_labels = batch['tag_labels'].to(device, non_blocking=True)
            pmask = batch.get('padding_mask', None)
            if pmask is not None:
                pmask = pmask.to(device=device, dtype=torch.bool, non_blocking=True)
            total += images.size(0)
            with amp_ctx():
                outputs = model(images, padding_mask=pmask)
                logits = outputs['tag_logits'] if isinstance(outputs, dict) else outputs
            probs = torch.sigmoid(logits.float()).to(torch.float16)
            all_probs.append(probs.cpu())
            all_targs.append(tag_labels.to(torch.uint8).cpu())
            if step % log_every == 0:
                elapsed = time.time() - t1
                rate = total / max(elapsed, 1e-6)
                eta = (len(val_loader) - step - 1) * (elapsed / max(step + 1, 1))
                print(f'  val step {step}/{len(val_loader)} '
                      f'({total} samples, {rate:.0f} img/s, ETA {eta/60:.1f}m)', flush=True)
    print(f'[info] forward pass: {time.time() - t1:.1f}s ({total} samples)', flush=True)
    return torch.cat(all_probs, dim=0), torch.cat(all_targs, dim=0)


def accumulate_bin_counts(probs_cpu, targs_cpu, num_bins, device, sample_chunk):
    """Build (num_bins, L) bin counts of (sample, tag) prob bins.

    Returns (bin_pred, bin_pos) on `device`, both float32.
      bin_pred[b, c] = # samples where probs[c] falls in bin b (regardless of label)
      bin_pos[b, c]  = # samples where probs[c] falls in bin b AND target == 1
    """
    N, L = probs_cpu.shape
    bin_pred = torch.zeros(num_bins, L, device=device, dtype=torch.float32)
    bin_pos = torch.zeros(num_bins, L, device=device, dtype=torch.float32)
    for start in range(0, N, sample_chunk):
        end = min(start + sample_chunk, N)
        probs_g = probs_cpu[start:end].to(device, non_blocking=True).float()
        targs_g = targs_cpu[start:end].to(device, non_blocking=True).float()
        idx = (probs_g.clamp(0, 1) * (num_bins - 1)).round().to(torch.int64)
        bin_pred.scatter_add_(0, idx, torch.ones_like(probs_g))
        bin_pos.scatter_add_(0, idx, targs_g)
    return bin_pred, bin_pos


def build_sweep_tensors(bin_pred, bin_pos, thresholds_gpu):
    """From bincounts, derive (T, L) TP/FP/FN/precision/recall/F1 at each threshold."""
    num_bins = bin_pred.shape[0]
    cum_pred = torch.flip(torch.cumsum(torch.flip(bin_pred, [0]), 0), [0])  # (B, L)
    cum_pos = torch.flip(torch.cumsum(torch.flip(bin_pos, [0]), 0), [0])    # (B, L)

    bin_idx = torch.clamp(
        torch.ceil(thresholds_gpu * (num_bins - 1) + 1e-9).to(torch.int64),
        min=0, max=num_bins - 1,
    )  # (T,)

    tp = cum_pos[bin_idx]            # (T, L)
    pred_pos = cum_pred[bin_idx]     # (T, L)
    fp = pred_pos - tp
    pos_per_tag = bin_pos.sum(dim=0)  # (L,)
    fn = pos_per_tag.unsqueeze(0) - tp

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'pos_per_tag': pos_per_tag,
    }


def micro_at_each_threshold(swept):
    """Aggregate TP/FP/FN across all tags at each threshold; return P/R/F1 (T,) numpy."""
    tp = swept['tp'].sum(dim=1)
    fp = swept['fp'].sum(dim=1)
    fn = swept['fn'].sum(dim=1)
    eps = 1e-12
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return p.cpu().numpy(), r.cpu().numpy(), f1.cpu().numpy()


def macro_at_each_threshold(swept, support_mask):
    """Mean per-tag P/R/F1 over `support_mask`-true tags; return (T,) numpy each.

    macro_F1 := mean(F1_per_tag), the sklearn/torchmetrics convention.
    """
    if not bool(support_mask.any()):
        T = swept['precision'].shape[0]
        nan = np.full(T, np.nan)
        return nan, nan, nan, 0
    p = swept['precision'][:, support_mask].mean(dim=1)
    r = swept['recall'][:, support_mask].mean(dim=1)
    f1 = swept['f1'][:, support_mask].mean(dim=1)
    return p.cpu().numpy(), r.cpu().numpy(), f1.cpu().numpy(), int(support_mask.sum().item())


def find_breakeven(thresholds, precision, recall):
    """Linear-interpolated threshold where precision crosses recall."""
    diff = precision - recall
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        idx = int(np.argmin(np.abs(diff)))
        p, r = float(precision[idx]), float(recall[idx])
        return {
            'threshold': float(thresholds[idx]),
            'precision': p, 'recall': r,
            'f1': float(2 * p * r / (p + r + 1e-12)),
            'interpolated': False,
            'note': 'no exact P=R crossing in sweep range; reporting closest point.',
        }
    i = int(sign_change[0])
    t0, t1 = thresholds[i], thresholds[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    alpha = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.0
    t_star = float(t0 + alpha * (t1 - t0))
    p_star = float(precision[i] + alpha * (precision[i + 1] - precision[i]))
    r_star = float(recall[i] + alpha * (recall[i + 1] - recall[i]))
    return {
        'threshold': t_star,
        'precision': p_star, 'recall': r_star,
        'f1': float(2 * p_star * r_star / (p_star + r_star + 1e-12)),
        'interpolated': True,
    }


def find_argmax_f1(thresholds, precision, recall, f1):
    idx = int(np.argmax(f1))
    return {
        'threshold': float(thresholds[idx]),
        'precision': float(precision[idx]),
        'recall': float(recall[idx]),
        'f1': float(f1[idx]),
    }


def per_tag_operating_points(swept, thresholds_np, support_mask):
    """For each tag (where support_mask is True), find the per-tag P=R and F1-optimal
    operating points. Returns numpy arrays of length L (NaN for unsupported tags).
    """
    L = swept['precision'].shape[1]
    pr_threshold = np.full(L, np.nan)
    pr_precision = np.full(L, np.nan)
    pr_recall = np.full(L, np.nan)
    f1_threshold = np.full(L, np.nan)
    f1_precision = np.full(L, np.nan)
    f1_recall = np.full(L, np.nan)
    f1_value = np.full(L, np.nan)

    valid_idx = torch.nonzero(support_mask, as_tuple=False).flatten().cpu().numpy()
    if len(valid_idx) == 0:
        return {'pr_threshold': pr_threshold, 'pr_precision': pr_precision, 'pr_recall': pr_recall,
                'f1_threshold': f1_threshold, 'f1_precision': f1_precision, 'f1_recall': f1_recall,
                'f1_value': f1_value}

    p_np = swept['precision'][:, support_mask].cpu().numpy()
    r_np = swept['recall'][:, support_mask].cpu().numpy()
    f_np = swept['f1'][:, support_mask].cpu().numpy()

    diff = p_np - r_np
    for j, global_idx in enumerate(valid_idx):
        d = diff[:, j]
        sign_change = np.where(np.diff(np.sign(d)) != 0)[0]
        if len(sign_change) == 0:
            k = int(np.argmin(np.abs(d)))
            t_star = float(thresholds_np[k])
            p_star = float(p_np[k, j])
            r_star = float(r_np[k, j])
        else:
            k = int(sign_change[0])
            t0, t1 = thresholds_np[k], thresholds_np[k + 1]
            d0, d1 = d[k], d[k + 1]
            alpha = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.0
            t_star = float(t0 + alpha * (t1 - t0))
            p_star = float(p_np[k, j] + alpha * (p_np[k + 1, j] - p_np[k, j]))
            r_star = float(r_np[k, j] + alpha * (r_np[k + 1, j] - r_np[k, j]))
        pr_threshold[global_idx] = t_star
        pr_precision[global_idx] = p_star
        pr_recall[global_idx] = r_star

        kf = int(np.argmax(f_np[:, j]))
        f1_threshold[global_idx] = float(thresholds_np[kf])
        f1_precision[global_idx] = float(p_np[kf, j])
        f1_recall[global_idx] = float(r_np[kf, j])
        f1_value[global_idx] = float(f_np[kf, j])

    return {'pr_threshold': pr_threshold, 'pr_precision': pr_precision, 'pr_recall': pr_recall,
            'f1_threshold': f1_threshold, 'f1_precision': f1_precision, 'f1_recall': f1_recall,
            'f1_value': f1_value}


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.full:
        # Override both data.max_val_samples (split-time) and validation.max_samples (post-load).
        config.data.max_val_samples = None
        config.validation.max_samples = None
    elif args.max_samples is not None:
        config.data.max_val_samples = args.max_samples
        config.validation.max_samples = args.max_samples

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    output_path = Path(args.output) if args.output else (
        checkpoint_path.parent / f'pr_threshold_{checkpoint_path.stem}.json'
    )

    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(getattr(config.training, 'benchmark', True))

    active_location = next((loc for loc in config.data.storage_locations if loc.get('enabled')), None)
    if active_location is None:
        raise RuntimeError('No enabled storage_location in config.')
    active_data_path = Path(active_location['path'])

    print(f'[info] config         : {args.config}', flush=True)
    print(f'[info] checkpoint     : {checkpoint_path}', flush=True)
    print(f'[info] output         : {output_path}', flush=True)
    print(f'[info] data path      : {active_data_path}', flush=True)
    print(f'[info] device         : {device}', flush=True)
    print(f'[info] mode           : {"FULL val split" if args.full else f"max_samples={config.validation.max_samples}"}', flush=True)
    print(f'[info] threshold sweep: [{args.threshold_min}, {args.threshold_max}] step {args.threshold_step}', flush=True)

    t0 = time.time()
    train_loader, val_loader, vocab = create_dataloaders(
        data_config=config.data,
        validation_config=config.validation,
        vocab_path=Path(config.vocab_path),
        active_data_path=active_data_path,
        seed=getattr(config.training, 'seed', 42),
        debug_config=config.debug,
        architecture_type=config.model.architecture_type,
        patch_size=getattr(config.model, 'patch_size', None),
    )
    if val_loader is None:
        raise RuntimeError('create_dataloaders returned no validation loader.')

    # Apply post-load subsample if validation.max_samples < dataset (matches train_direct).
    val_max_samples = getattr(config.validation, 'max_samples', None)
    if val_max_samples and val_max_samples < len(val_loader.dataset):
        val_subset_seed = getattr(config.training, 'seed', 42)
        rng = np.random.RandomState(val_subset_seed)
        idx = np.sort(rng.choice(len(val_loader.dataset), val_max_samples, replace=False))
        val_dl_cfg = getattr(config.validation, 'dataloader', None)
        val_batch = getattr(val_dl_cfg, 'batch_size', None) or config.data.batch_size
        val_workers = getattr(val_dl_cfg, 'num_workers', None) or 0
        val_prefetch = getattr(val_dl_cfg, 'prefetch_factor', None) or getattr(config.data, 'prefetch_factor', 2)
        val_pin = getattr(val_dl_cfg, 'pin_memory', None)
        if val_pin is None:
            val_pin = getattr(config.data, 'pin_memory', True)
        val_persistent = getattr(val_dl_cfg, 'persistent_workers', None)
        if val_persistent is None:
            val_persistent = getattr(config.data, 'persistent_workers', False)
        try:
            if hasattr(val_loader, '_iterator') and val_loader._iterator is not None:
                val_loader._iterator._shutdown_workers()
        except Exception:
            pass
        kw = dict(batch_size=val_batch, shuffle=False, num_workers=val_workers, pin_memory=val_pin)
        if val_workers > 0:
            kw['prefetch_factor'] = val_prefetch
            kw['persistent_workers'] = val_persistent
        val_loader = DataLoader(Subset(val_loader.dataset, idx.tolist()), **kw)

    try:
        if train_loader is not None and hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
            train_loader._iterator._shutdown_workers()
    except Exception:
        pass
    del train_loader

    if hasattr(val_loader.dataset, 'set_epoch'):
        val_loader.dataset.set_epoch(0)

    num_tags = len(vocab.tag_to_index)
    if config.model.num_labels == 0:
        config.model.num_labels = num_tags

    print(f'[info] num_tags       : {num_tags}', flush=True)
    print(f'[info] val_loader     : {len(val_loader.dataset)} samples, batch={val_loader.batch_size}', flush=True)

    model = build_model(config, num_tags).to(device)

    expected_n = None if args.full else getattr(config.validation, 'max_samples', None)
    cache_path = predictions_cache_path(checkpoint_path, expected_n)
    cached = None
    if args.cache_preds and cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location='cpu', weights_only=True)
            if (cached.get('checkpoint_stem') == checkpoint_path.stem
                    and cached['probs'].shape[0] == len(val_loader.dataset)
                    and cached['probs'].shape[1] == num_tags):
                print(f'[info] reusing cached predictions: {cache_path} '
                      f'(shape={tuple(cached["probs"].shape)})', flush=True)
            else:
                print(f'[info] cache mismatch at {cache_path}; recomputing', flush=True)
                cached = None
        except Exception as e:
            print(f'[warn] failed to load cache ({e}); recomputing', flush=True)
            cached = None

    if cached is None:
        print(f'[info] dataloader+model setup: {time.time() - t0:.1f}s', flush=True)
        print('[info] loading checkpoint...', flush=True)
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = strip_compile_prefix(ckpt['state_dict'])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[warn] {len(missing)} missing keys (showing first 3): {missing[:3]}', flush=True)
        if unexpected:
            print(f'[warn] {len(unexpected)} unexpected keys (showing first 3): {unexpected[:3]}', flush=True)
        ckpt_epoch = ckpt.get('epoch')
        ckpt_step = ckpt.get('step')
        print(f'[info] ckpt epoch={ckpt_epoch} step={ckpt_step}', flush=True)

        amp_dtype_cfg = str(getattr(config.training, 'amp_dtype', 'bfloat16')).lower()
        amp_dtype = (torch.bfloat16 if amp_dtype_cfg in ('bfloat16', 'bf16')
                     else torch.float16 if amp_dtype_cfg == 'float16'
                     else torch.float32)
        use_amp = bool(getattr(config.training, 'use_amp', True))
        amp_ctx = (lambda: autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp))

        probs_cpu, targs_cpu = collect_predictions(model, val_loader, device, amp_ctx)
        del model
        torch.cuda.empty_cache()

        if args.cache_preds:
            try:
                torch.save({
                    'checkpoint_stem': checkpoint_path.stem,
                    'checkpoint_epoch': ckpt_epoch,
                    'checkpoint_step': ckpt_step,
                    'probs': probs_cpu,
                    'targets': targs_cpu,
                }, cache_path)
                print(f'[info] cached predictions -> {cache_path}', flush=True)
            except Exception as e:
                print(f'[warn] failed to write cache: {e}', flush=True)
    else:
        probs_cpu = cached['probs']
        targs_cpu = cached['targets']
        ckpt_epoch = cached.get('checkpoint_epoch')
        ckpt_step = cached.get('checkpoint_step')
        del model
        torch.cuda.empty_cache()

    # Drop PAD/UNK columns up-front.
    keep = torch.ones(num_tags, dtype=torch.bool)
    for i in SKIP_INDICES:
        if 0 <= i < num_tags:
            keep[i] = False
    probs_cpu = probs_cpu[:, keep]
    targs_cpu = targs_cpu[:, keep]
    kept_indices = np.flatnonzero(keep.numpy())
    tag_names = [vocab.index_to_tag[i] for i in kept_indices.tolist()]
    L = probs_cpu.shape[1]
    N = probs_cpu.shape[0]
    print(f'[info] collected probs: ({N}, {L}), targets: ({N}, {L})', flush=True)

    # ---- Sweep -----------------------------------------------------------------------
    thresholds_np = np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2,
                              args.threshold_step, dtype=np.float64)
    thresholds_gpu = torch.from_numpy(thresholds_np).to(device, dtype=torch.float32)
    T = len(thresholds_np)
    print(f'[info] sweeping {T} thresholds (sample_chunk={args.sample_chunk})...', flush=True)

    t1 = time.time()
    num_bins = 1001
    bin_pred, bin_pos = accumulate_bin_counts(probs_cpu, targs_cpu, num_bins, device, args.sample_chunk)
    swept = build_sweep_tensors(bin_pred, bin_pos, thresholds_gpu)
    del bin_pred, bin_pos
    torch.cuda.empty_cache()
    print(f'[info] sweep tensors built in {time.time() - t1:.1f}s '
          f'(per-tag P/R/F1 cached on GPU at ({T}, {L}) fp32 ~ {3 * T * L * 4 / 1e9:.2f} GB)', flush=True)

    # ---- Operating points ------------------------------------------------------------
    pos_per_tag_np = swept['pos_per_tag'].cpu().numpy().astype(np.int64)

    # Micro
    p_micro, r_micro, f1_micro = micro_at_each_threshold(swept)
    micro_breakeven = find_breakeven(thresholds_np, p_micro, r_micro)
    micro_f1opt = find_argmax_f1(thresholds_np, p_micro, r_micro, f1_micro)

    # Macro at three support cutoffs (single global threshold)
    macro_results = {}
    cutoff_set = sorted({0, 1, int(args.min_support)})
    for cutoff in cutoff_set:
        mask = swept['pos_per_tag'] >= cutoff
        p_mac, r_mac, f1_mac, n_valid = macro_at_each_threshold(swept, mask)
        if n_valid == 0:
            continue
        be = find_breakeven(thresholds_np, p_mac, r_mac)
        f1opt = find_argmax_f1(thresholds_np, p_mac, r_mac, f1_mac)
        macro_results[f'support_ge_{cutoff}'] = {
            'tags_evaluated': n_valid,
            'tags_total': L,
            'pr_breakeven': be,
            'f1_optimal': f1opt,
        }

    print('', flush=True)
    print('=' * 70, flush=True)
    print(f'CHECKPOINT: {checkpoint_path}', flush=True)
    print(f'CKPT epoch={ckpt_epoch} step={ckpt_step}   val_samples={N}   tags={L}', flush=True)
    print('=' * 70, flush=True)
    print('Micro (single global threshold, support-weighted):', flush=True)
    print(f'  P=R         : t={micro_breakeven["threshold"]:.4f}  P={micro_breakeven["precision"]:.4f}  '
          f'R={micro_breakeven["recall"]:.4f}  F1={micro_breakeven["f1"]:.4f}', flush=True)
    print(f'  F1-optimal  : t={micro_f1opt["threshold"]:.4f}  P={micro_f1opt["precision"]:.4f}  '
          f'R={micro_f1opt["recall"]:.4f}  F1={micro_f1opt["f1"]:.4f}', flush=True)
    print('', flush=True)
    print('Macro (single global threshold, mean of per-tag F1 -- WD14-comparable):', flush=True)
    for cutoff in cutoff_set:
        key = f'support_ge_{cutoff}'
        if key not in macro_results:
            continue
        m = macro_results[key]
        be, fo = m['pr_breakeven'], m['f1_optimal']
        label = f'support>={cutoff} ({m["tags_evaluated"]}/{m["tags_total"]} tags)'
        print(f'  [{label}]', flush=True)
        print(f'    P=R       : t={be["threshold"]:.4f}  P={be["precision"]:.4f}  '
              f'R={be["recall"]:.4f}  F1={be["f1"]:.4f}', flush=True)
        print(f'    F1-optimal: t={fo["threshold"]:.4f}  P={fo["precision"]:.4f}  '
              f'R={fo["recall"]:.4f}  F1={fo["f1"]:.4f}', flush=True)

    report = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_epoch': ckpt_epoch,
        'checkpoint_step': ckpt_step,
        'val_samples': int(N),
        'num_tags_evaluated': int(L),
        'skip_indices': list(SKIP_INDICES),
        'sweep': {
            'min': float(args.threshold_min),
            'max': float(args.threshold_max),
            'step': float(args.threshold_step),
            'count': int(T),
        },
        'micro': {
            'pr_breakeven': micro_breakeven,
            'f1_optimal': micro_f1opt,
        },
        'macro_single_threshold': macro_results,
    }

    if not args.no_per_tag:
        print('', flush=True)
        print(f'[info] per-tag operating points (support>={args.min_support})...', flush=True)
        t2 = time.time()
        mask_pt = swept['pos_per_tag'] >= int(args.min_support)
        per_tag = per_tag_operating_points(swept, thresholds_np, mask_pt)
        n_pt_valid = int(mask_pt.sum().item())
        print(f'[info] per-tag op-points done in {time.time() - t2:.1f}s', flush=True)

        valid = ~np.isnan(per_tag['pr_threshold'])
        n_valid = int(valid.sum())
        if n_valid > 0:
            macro_pr_perTag = {
                'tags_with_support': n_valid,
                'mean_threshold': float(np.mean(per_tag['pr_threshold'][valid])),
                'median_threshold': float(np.median(per_tag['pr_threshold'][valid])),
                'p25_threshold': float(np.percentile(per_tag['pr_threshold'][valid], 25)),
                'p75_threshold': float(np.percentile(per_tag['pr_threshold'][valid], 75)),
                'mean_precision': float(np.mean(per_tag['pr_precision'][valid])),
                'mean_recall': float(np.mean(per_tag['pr_recall'][valid])),
            }
            f1_valid = ~np.isnan(per_tag['f1_threshold'])
            macro_f1_perTag = {
                'mean_threshold': float(np.mean(per_tag['f1_threshold'][f1_valid])),
                'median_threshold': float(np.median(per_tag['f1_threshold'][f1_valid])),
                'mean_f1': float(np.mean(per_tag['f1_value'][f1_valid])),
            }
            print('', flush=True)
            print(f'Per-tag operating points (each tag at its own threshold, support>={args.min_support}):', flush=True)
            print(f'  P=R   : mean t={macro_pr_perTag["mean_threshold"]:.4f}  '
                  f'IQR=[{macro_pr_perTag["p25_threshold"]:.4f}, {macro_pr_perTag["p75_threshold"]:.4f}]  '
                  f'mean P=R={macro_pr_perTag["mean_precision"]:.4f}  '
                  f'(n={macro_pr_perTag["tags_with_support"]})', flush=True)
            print(f'  F1-opt: mean t={macro_f1_perTag["mean_threshold"]:.4f}  '
                  f'mean F1={macro_f1_perTag["mean_f1"]:.4f}', flush=True)

            per_tag_section = {
                'min_support': int(args.min_support),
                'pr_summary': macro_pr_perTag,
                'f1_summary': macro_f1_perTag,
            }
            if args.save_per_tag_table:
                per_tag_table = {}
                for j, name in enumerate(tag_names):
                    if not valid[j]:
                        continue
                    per_tag_table[name] = {
                        'support': int(pos_per_tag_np[j]),
                        'pr_threshold': float(per_tag['pr_threshold'][j]),
                        'pr_precision': float(per_tag['pr_precision'][j]),
                        'pr_recall': float(per_tag['pr_recall'][j]),
                        'f1_threshold': float(per_tag['f1_threshold'][j]) if not np.isnan(per_tag['f1_threshold'][j]) else None,
                        'f1_value': float(per_tag['f1_value'][j]) if not np.isnan(per_tag['f1_value'][j]) else None,
                    }
                per_tag_section['per_tag'] = per_tag_table
            report['per_tag'] = per_tag_section
        else:
            print(f'[warn] no tags reached min_support={args.min_support}', flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('', flush=True)
    print(f'[info] wrote report -> {output_path}', flush=True)

    if args.save_curve:
        curve_path = output_path.with_suffix('.curve.csv')
        with open(curve_path, 'w', encoding='utf-8') as f:
            f.write('threshold,micro_p,micro_r,micro_f1\n')
            for t, p, r, f1 in zip(thresholds_np, p_micro, r_micro, f1_micro):
                f.write(f'{t:.6f},{p:.6f},{r:.6f},{f1:.6f}\n')
        print(f'[info] wrote micro curve -> {curve_path}', flush=True)


if __name__ == '__main__':
    main()
