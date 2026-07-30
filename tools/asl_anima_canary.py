#!/usr/bin/env python3
"""Anima recall canary (todos/ASL_plan.md SS5) -- STANDALONE, EVALUATION-ONLY.

Scores a checkpoint's recall on the prompt-controlled synthetic golden set
(known-positive tag per image). This is the only label-clean recall signal
under Option A, licensed strictly as a PROBE, never an anchor:

- the golden images are NEVER used for training or model selection;
- only the single known-positive prompted tag per image is measured (the set
  is not exhaustively tagged, so nothing else about it is trustworthy);
- run this MANDATORILY before and after each manual gamma_neg step: each
  step must HOLD per-bucket recall. A drop means the descent is crushing
  real positives (missing-positive-style collapse) -- step gamma back up.

Usage:
  python tools/asl_anima_canary.py --checkpoint experiments/<run>/checkpoints/last.pt \
      --images-root <dir containing hair_color/...> [--labels .research/golden_gen/labels.jsonl]

Compare two runs with a JSON diff of the emitted report files.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.amp import autocast

from Configuration_System import load_config
from model_architecture import create_model
from vocabulary import load_vocabulary_for_training


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=str(PROJECT_ROOT / 'configs' / 'unified_config.yaml'))
    p.add_argument('--checkpoint', required=True, help='Checkpoint .pt to score')
    p.add_argument('--labels', default=str(PROJECT_ROOT / '.research' / 'golden_gen' / 'labels.jsonl'),
                   help='JSONL with {label, path, bucket, axis} per golden image')
    p.add_argument('--images-root', default=None,
                   help='Directory the label "path" fields are relative to '
                        '(default: the labels file\'s parent directory)')
    p.add_argument('--threshold', type=float, default=None,
                   help='Recall threshold (default: config.inference.prediction_threshold)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-images', type=int, default=None, help='Cap for a quick smoke run')
    p.add_argument('--output', default=None,
                   help='JSON report path (default: reports/anima_canary_<ckpt>_<step>.json)')
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


def load_and_preprocess(path: Path, image_size: int, mean, std, pad_color):
    """Mirror the inference pipeline: EXIF transpose, alpha -> gray composite,
    RGB, letterbox downscale-only with pad_color, normalize."""
    with Image.open(path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, tuple(pad_color))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            img = bg
        else:
            img = img.convert("RGB")

        w, h = img.size
        scale = min(1.0, min(image_size / float(w), image_size / float(h)))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        # LANCZOS to match dataset_loader's training-time RESAMPLE_LANCZOS. This
        # feeds the ASL recall canary, so a preprocessing mismatch here would
        # show up as a gate signal rather than as the skew it actually is.
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), tuple(pad_color))
        canvas.paste(resized, ((image_size - nw) // 2, (image_size - nh) // 2))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    threshold = args.threshold if args.threshold is not None else float(config.inference.prediction_threshold)
    image_size = int(config.data.image_size)
    mean = list(config.data.normalize_mean)
    std = list(config.data.normalize_std)
    pad_color = [int(c) for c in config.data.pad_color]
    if str(getattr(config.data, 'color_order', 'RGB')).upper() != 'RGB':
        raise SystemExit('This canary assumes color_order=RGB (matches current config).')

    labels_path = Path(args.labels)
    images_root = Path(args.images_root) if args.images_root else labels_path.parent

    # --- Load golden labels ---
    entries = []
    missing_files = 0
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_path = images_root / rec['path']
            if not img_path.exists():
                missing_files += 1
                continue
            entries.append((img_path, rec['label'], rec.get('bucket', rec['label']), rec.get('axis', '?')))
    if args.max_images:
        entries = entries[:args.max_images]
    if missing_files:
        print(f'[warn] {missing_files} label rows skipped: image file not found under {images_root}', flush=True)
    if not entries:
        raise SystemExit(f'No golden images found (labels={labels_path}, images_root={images_root}). '
                         f'Pass --images-root pointing at the generated image tree.')

    # --- Vocab + model ---
    vocab = load_vocabulary_for_training(Path(config.vocab_path))
    num_tags = len(vocab.tag_to_index)
    unknown = sorted({lab for _, lab, _, _ in entries if lab not in vocab.tag_to_index})
    if unknown:
        print(f'[warn] {len(unknown)} golden labels not in vocabulary, their rows are skipped: {unknown}', flush=True)
        entries = [e for e in entries if e[1] in vocab.tag_to_index]

    model = build_model(config, num_tags).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(strip_compile_prefix(ckpt['state_dict']), strict=False)
    if missing:
        print(f'[warn] {len(missing)} missing state_dict keys (first 3): {missing[:3]}', flush=True)
    if unexpected:
        print(f'[warn] {len(unexpected)} unexpected state_dict keys (first 3): {unexpected[:3]}', flush=True)
    ckpt_epoch = ckpt.get('epoch')
    ckpt_step = ckpt.get('step')
    gamma_neg = ((ckpt.get('training_state') or {}).get('loss_state') or {}).get('gamma_neg')
    del ckpt
    model.eval()

    amp_dtype = torch.bfloat16 if str(getattr(config.training, 'amp_dtype', 'bfloat16')).lower() in ('bfloat16', 'bf16') else torch.float16
    use_amp = bool(getattr(config.training, 'use_amp', True)) and device.type == 'cuda'

    print(f'[info] checkpoint : {args.checkpoint} (epoch={ckpt_epoch}, step={ckpt_step}, gamma_neg={gamma_neg})', flush=True)
    print(f'[info] images     : {len(entries)} golden samples @ {image_size}px, threshold={threshold}', flush=True)

    # --- Score ---
    hits = defaultdict(int)
    counts = defaultdict(int)
    prob_sums = defaultdict(float)
    bucket_axis = {}
    overall_hits = 0
    overall_probs = 0.0
    failures = 0
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(entries), args.batch_size):
            batch = entries[i:i + args.batch_size]
            tensors, metas = [], []
            for img_path, label, bucket, axis in batch:
                try:
                    tensors.append(load_and_preprocess(img_path, image_size, mean, std, pad_color))
                    metas.append((label, bucket, axis))
                except Exception as e:
                    failures += 1
                    print(f'[warn] failed to load {img_path}: {e}', flush=True)
            if not tensors:
                continue
            images = torch.stack(tensors).to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(images, padding_mask=None)
            probs = torch.sigmoid(outputs['tag_logits'].float())
            label_cols = torch.tensor([vocab.tag_to_index[m[0]] for m in metas], device=probs.device)
            p_label = probs[torch.arange(probs.size(0), device=probs.device), label_cols].cpu().tolist()
            for (label, bucket, axis), p in zip(metas, p_label):
                counts[bucket] += 1
                prob_sums[bucket] += p
                bucket_axis[bucket] = axis
                overall_probs += p
                if p >= threshold:
                    hits[bucket] += 1
                    overall_hits += 1
            if (i // args.batch_size) % 20 == 0:
                print(f'  {i + len(batch)}/{len(entries)} scored', flush=True)

    total = sum(counts.values())
    if total == 0:
        raise SystemExit('No images scored.')
    print(f'[info] scored {total} images in {time.time() - t0:.1f}s ({failures} load failures)', flush=True)

    # --- Report ---
    per_bucket = {}
    print('', flush=True)
    print(f'{"axis":<14} {"bucket":<28} {"n":>5} {"recall":>8} {"mean_p":>8}', flush=True)
    print('-' * 68, flush=True)
    for bucket in sorted(counts, key=lambda b: (bucket_axis[b], b)):
        n = counts[bucket]
        rec = hits[bucket] / n
        mp = prob_sums[bucket] / n
        per_bucket[bucket] = {'axis': bucket_axis[bucket], 'n': n, 'recall': rec, 'mean_prob': mp}
        print(f'{bucket_axis[bucket]:<14} {bucket:<28} {n:>5} {rec:>8.4f} {mp:>8.4f}', flush=True)
    macro_recall = sum(v['recall'] for v in per_bucket.values()) / len(per_bucket)
    print('-' * 68, flush=True)
    print(f'{"OVERALL":<43} {total:>5} {overall_hits / total:>8.4f} {overall_probs / total:>8.4f}', flush=True)
    print(f'{"MACRO (per-bucket mean)":<43} {"":>5} {macro_recall:>8.4f}', flush=True)

    out_path = Path(args.output) if args.output else (
        PROJECT_ROOT / 'reports' /
        f'anima_canary_{Path(args.checkpoint).stem}_step{ckpt_step or "NA"}.json'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        'checkpoint': str(args.checkpoint),
        'ckpt_epoch': ckpt_epoch,
        'ckpt_step': ckpt_step,
        'gamma_neg': gamma_neg,
        'threshold': threshold,
        'image_size': image_size,
        'total_images': total,
        'micro_recall': overall_hits / total,
        'macro_recall': macro_recall,
        'mean_prob': overall_probs / total,
        'per_bucket': per_bucket,
    }, indent=2), encoding='utf-8')
    print(f'\n[info] report written: {out_path}', flush=True)
    print('[note] ASL_plan SS5 gate: each gamma_neg step must HOLD per-bucket recall '
          'vs the pre-step report. A drop = positives being crushed -> step gamma back up.', flush=True)


if __name__ == '__main__':
    main()
