#!/usr/bin/env python3
"""Run validation against a single checkpoint and emit a TRAINING_HEALTH_TRACKER row.

Replicates train_direct.py's validation logic (same metrics, same threshold,
same skip indices, same 30K val subsample seed) without spinning up an
optimizer, scheduler, torch.compile, train loader, or TB monitor — so it
can score a saved checkpoint quickly when the in-loop validation event
was missed.
"""

import argparse
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
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision

from Configuration_System import load_config
from dataset_loader import create_dataloaders
from evaluation_metrics import FrequencyBucketMetrics
from loss_functions import AsymmetricFocalLoss, MultiTaskLoss
from model_architecture import create_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(PROJECT_ROOT / 'configs' / 'unified_config.yaml'))
    p.add_argument('--checkpoint', default=str(PROJECT_ROOT / 'experiments' / 'run1_vit' / 'checkpoints' / 'last.pt'))
    p.add_argument('--epoch-label', type=int, default=None,
                   help='0-based tracker epoch number for the output row. Defaults to (checkpoint.epoch - 1).')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Override validation.max_samples (smaller = faster, less precise).')
    return p.parse_args()


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


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.max_samples is not None:
        config.validation.max_samples = args.max_samples

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
    print(f'[info] checkpoint     : {args.checkpoint}', flush=True)
    print(f'[info] data path      : {active_data_path}', flush=True)
    print(f'[info] device         : {device}', flush=True)
    print(f'[info] val max_samples: {config.validation.max_samples}', flush=True)

    # Build dataloaders. We discard the train loader immediately to free its workers.
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

    # Apply the same deterministic 30K subset the training loop uses, so the
    # number we report is comparable to historical val/* tracker rows.
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

    # Drop train loader workers right away; we won't be needing them.
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

    print(f'[info] dataloader+model setup: {time.time() - t0:.1f}s', flush=True)

    # Load checkpoint weights only — no optimizer/scheduler/scaler.
    print(f'[info] loading checkpoint...', flush=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = strip_compile_prefix(ckpt['state_dict'])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[warn] {len(missing)} missing keys (showing first 3): {missing[:3]}', flush=True)
    if unexpected:
        print(f'[warn] {len(unexpected)} unexpected keys (showing first 3): {unexpected[:3]}', flush=True)
    ckpt_epoch_1based = ckpt.get('epoch')
    ckpt_step = ckpt.get('step')
    print(f'[info] ckpt epoch={ckpt_epoch_1based} (1-based), step={ckpt_step}', flush=True)

    epoch_label = args.epoch_label
    if epoch_label is None and ckpt_epoch_1based is not None:
        epoch_label = ckpt_epoch_1based - 1  # 1-based -> 0-based tracker epoch

    # Loss — same configuration as training, used so val/loss is comparable.
    tag_loss_cfg = config.training.tag_loss
    criterion = MultiTaskLoss(
        tag_loss_fn=AsymmetricFocalLoss(
            alpha=tag_loss_cfg.alpha,
            clip=tag_loss_cfg.clip,
            gamma_neg=tag_loss_cfg.gamma_neg,
            gamma_pos=tag_loss_cfg.gamma_pos,
            label_smoothing=tag_loss_cfg.label_smoothing,
            ignore_indices=[0, 1],
            class_weights=None,
        ),
    ).to(device)

    # Metrics — mirror train_direct.py:1374-1388
    tc = getattr(config, 'threshold_calibration', None)
    threshold = getattr(tc, 'default_threshold', 0.2653) if tc is not None else 0.2653
    skip_metric_cols = 2  # PAD=0, UNK=1
    num_metric_labels = num_tags - skip_metric_cols
    val_metrics = {
        'f1_macro_per_class': MultilabelF1Score(num_labels=num_metric_labels, average=None, threshold=threshold).to(device),
        'f1_micro': MultilabelF1Score(num_labels=num_metric_labels, average='micro', threshold=threshold).to(device),
        'map_per_class': MultilabelAveragePrecision(num_labels=num_metric_labels, average=None).to(device),
    }
    val_pos_counts = torch.zeros(num_metric_labels, dtype=torch.long, device=device)

    # AMP autocast (same as training: bfloat16 by default)
    amp_dtype_cfg = str(getattr(config.training, 'amp_dtype', 'bfloat16')).lower()
    amp_dtype = (torch.bfloat16 if amp_dtype_cfg in ('bfloat16', 'bf16')
                 else torch.float16 if amp_dtype_cfg == 'float16'
                 else torch.float32)
    use_amp = bool(getattr(config.training, 'use_amp', True))
    amp_ctx = (lambda: autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp))

    model.eval()
    val_loss = torch.tensor(0.0, device=device)
    total_val_samples = 0
    all_probs, all_targs = [], []

    print('[info] running validation...', flush=True)
    t1 = time.time()
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
            total_val_samples += images.size(0)
            with amp_ctx():
                outputs = model(images, padding_mask=pmask)
                loss, _ = criterion(outputs['tag_logits'], tag_labels)
            val_loss = val_loss + loss.detach() * images.size(0)
            probs = torch.sigmoid(outputs['tag_logits']).float()
            targs = tag_labels.long()
            mp_ = probs[:, skip_metric_cols:]
            mt_ = targs[:, skip_metric_cols:]
            val_metrics['f1_macro_per_class'].update(mp_, mt_)
            val_metrics['f1_micro'].update(mp_, mt_)
            val_metrics['map_per_class'].update(mp_, mt_)
            val_pos_counts += mt_.sum(dim=0)
            all_probs.append(probs.to('cpu', non_blocking=True))
            all_targs.append(targs.to('cpu', non_blocking=True))
            if step % 25 == 0:
                print(f'  val step {step}/{len(val_loader)} (samples seen: {total_val_samples})', flush=True)

    val_loss_avg = (val_loss / max(1, total_val_samples)).item()
    per_class_f1 = val_metrics['f1_macro_per_class'].compute()
    per_class_ap = val_metrics['map_per_class'].compute()
    keep = (val_pos_counts > 0)
    n_supp = int(keep.sum().item())
    if n_supp > 0:
        val_f1_macro = per_class_f1[keep].float().mean().item()
        val_mAP = per_class_ap[keep].float().mean().item()
    else:
        val_f1_macro = 0.0
        val_mAP = 0.0
    val_f1_micro = val_metrics['f1_micro'].compute().item()

    cat_probs = torch.cat(all_probs, dim=0).float()
    cat_targs = torch.cat(all_targs, dim=0)
    pred_thr = float(config.inference.prediction_threshold)
    mean_active = (cat_probs[:, skip_metric_cols:] > pred_thr).float().sum(dim=1).mean().item()

    freq_bins = getattr(config.validation, 'frequency_bins', None) or [300, 500, 1000, 5000, 10000, float('inf')]
    tag_names = [vocab.index_to_tag[i] for i in range(len(vocab.index_to_tag))]
    bucketer = FrequencyBucketMetrics(
        tag_frequencies=vocab.tag_frequencies,
        frequency_bins=freq_bins,
        tag_names=tag_names,
        skip_indices=[0, 1],
    )
    bucketed = bucketer.compute_bucketed_metrics(
        cat_probs, cat_targs, threshold=pred_thr,
    )

    print(f'[info] validation done in {time.time() - t1:.1f}s', flush=True)
    print('', flush=True)
    print('=' * 70, flush=True)
    print(f'CHECKPOINT       : {args.checkpoint}', flush=True)
    print(f'CKPT EPOCH (1-based): {ckpt_epoch_1based}   STEP: {ckpt_step}', flush=True)
    if epoch_label is not None:
        print(f'TRACKER EPOCH (0-based): {epoch_label}', flush=True)
    print('=' * 70, flush=True)
    print(f'val/loss         : {val_loss_avg:.6f}', flush=True)
    print(f'val/mAP          : {val_mAP:.6f}', flush=True)
    print(f'val/f1_micro     : {val_f1_micro:.6f}', flush=True)
    print(f'val/f1_macro     : {val_f1_macro:.6f}    (avg over {n_supp}/{num_metric_labels} supported classes)', flush=True)
    print(f'val/mean_active  : {mean_active:.4f}', flush=True)
    train_loss_cached = ckpt.get('training_state', {}).get('train_loss', None)
    print(f'train_loss(epoch end, from checkpoint training_state): {train_loss_cached!r}', flush=True)
    print('', flush=True)
    print('Bucketed mAP:', flush=True)
    bucket_order = ['300-499', '500-999', '1000-4999', '5000-9999', '10000+']
    bucket_map = {k: bucketed.get(k, {}).get('mAP', float('nan')) for k in bucket_order}
    for b in bucket_order:
        m = bucketed.get(b, {})
        print(f'  {b:>10s}: mAP={m.get("mAP", float("nan")):.6f}  '
              f'f1_macro={m.get("f1_macro", float("nan")):.6f}  '
              f'num_tags={int(m.get("num_tags", 0))}', flush=True)

    # Tracker row (markdown). Step is the actual checkpoint step (matches the
    # training loop's "step at which validation fired" convention used in
    # historical rows like E22 and E24).
    # Use ASCII '-' as the placeholder so the row prints on cp932/Windows
    # consoles. The tracker uses an em-dash (—) by convention; copy the
    # printed row and swap '-' for '—' if you care about the typography.
    DASH = '-'
    train_loss_str = f'{train_loss_cached:.6f}' if isinstance(train_loss_cached, (int, float)) else DASH
    print('', flush=True)
    print('Tracker row (paste into TRAINING_HEALTH_TRACKER.md; replace ASCII - with em-dash if desired):', flush=True)
    cells = [
        str(epoch_label) if epoch_label is not None else DASH,
        str(ckpt_step) if ckpt_step is not None else DASH,
        train_loss_str,
        f'{val_loss_avg:.6f}',
        f'{val_mAP:.6f}',
        DASH,
        f'{bucket_map["500-999"]:.6f}',
        f'{bucket_map["1000-4999"]:.6f}',
        f'{bucket_map["5000-9999"]:.6f}',
        f'{bucket_map["10000+"]:.6f}',
        f'{val_f1_micro:.6f}',
        f'{val_f1_macro:.6f}',
        f'{mean_active:.2f}',
        DASH,
        '0',
        '(filled by run_validation_for_epoch.py against ' + Path(args.checkpoint).name + ')',
    ]
    print('| ' + ' | '.join(cells) + ' |', flush=True)


if __name__ == '__main__':
    main()
