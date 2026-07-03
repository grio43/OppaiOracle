#!/usr/bin/env python3
"""Measure FP8 vs FP32 accuracy: true mAP against ground-truth labels + drift.

Runs both models over a labeled sample from the dataset shards (each .json has a
comma-separated `tags` field), builds ground-truth multi-hot vectors via the
vocabulary, and reports micro/macro mAP for each model, the FP8-vs-FP32 drift,
and tag-flip behaviour at the deployed operating threshold.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_fp8 import load_rgb, preprocess

SKIP = {0, 1}  # <PAD>, <UNK>


def load_vocab(path):
    d = json.load(open(path, encoding="utf-8"))
    t2i = d["tag_to_index"]
    return t2i, len(t2i)


def sample_labeled(root, limit, seed):
    root = Path(root)
    jsons = []
    for p in root.rglob("*.json"):
        jsons.append(p)
        if len(jsons) >= max(limit * 30, 3000):
            break
    random.Random(seed).shuffle(jsons)
    out = []
    for j in jsons:
        try:
            d = json.load(open(j, encoding="utf-8"))
        except Exception:
            continue
        img = j.with_suffix(".jpg")
        if not img.exists() or "tags" not in d:
            continue
        out.append((img, [t.strip() for t in d["tags"].split(",") if t.strip()]))
        if len(out) >= limit:
            break
    return out


def infer_all(sess, items, has_mask, num_tags):
    P = np.zeros((len(items), num_tags), dtype=np.float32)
    for i, (img, _) in enumerate(items):
        x, mask = preprocess(load_rgb(img))
        feed = {"pixel_values": x}
        if has_mask:
            feed["padding_mask"] = mask
        P[i] = sess.run(None, feed)[0][0]
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(items)}", flush=True)
    return P


def maps(y, p):
    """micro mAP (flattened) and macro mAP (mean AP over tags with >=1 positive)."""
    cols = np.where(y.sum(0) > 0)[0]
    cols = np.array([c for c in cols if c not in SKIP])
    micro = average_precision_score(y[:, cols].ravel(), p[:, cols].ravel())
    macro = np.mean([average_precision_score(y[:, c], p[:, c]) for c in cols])
    return micro, macro, len(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", default="huggingface_release/V1.1_fp8_onnx/model.onnx")
    ap.add_argument("--fp32", default="huggingface_release/V1.1_onnx/model.onnx")
    ap.add_argument("--vocab", default="huggingface_release/V1.1_fp8_onnx/vocabulary.json")
    ap.add_argument("--data-root", default="L:/Dab/Dab")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--thr", type=float, default=0.805)  # f1_optimal from pr_thresholds.json
    args = ap.parse_args()

    t2i, num_tags = load_vocab(args.vocab)
    print(f"vocab: {num_tags} tags")
    items = sample_labeled(args.data_root, args.limit, args.seed)
    print(f"sampled {len(items)} labeled images")

    # ground truth multi-hot
    Y = np.zeros((len(items), num_tags), dtype=np.int8)
    for i, (_, tags) in enumerate(items):
        for t in tags:
            j = t2i.get(t)
            if j is not None:
                Y[i, j] = 1
    print(f"mean positives/image: {Y.sum(1).mean():.1f}")

    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    s32 = ort.InferenceSession(args.fp32, providers=prov)
    s8 = ort.InferenceSession(args.fp8, providers=prov)
    has_mask = "padding_mask" in {i.name for i in s32.get_inputs()}

    print("running FP32 ..."); t = time.time(); P32 = infer_all(s32, items, has_mask, num_tags)
    print(f"  {time.time()-t:.0f}s")
    print("running FP8 ...");  t = time.time(); P8 = infer_all(s8, items, has_mask, num_tags)
    print(f"  {time.time()-t:.0f}s")

    mi32, ma32, ncols = maps(Y, P32)
    mi8, ma8, _ = maps(Y, P8)
    print("\n================ mAP vs ground truth ================")
    print(f"tags scored (>=1 positive): {ncols}")
    print(f"               micro-mAP    macro-mAP")
    print(f"  FP32   :     {mi32:.4f}      {ma32:.4f}")
    print(f"  FP8    :     {mi8:.4f}      {ma8:.4f}")
    print(f"  drift  :     {(mi8-mi32)*100:+.3f} pp   {(ma8-ma32)*100:+.3f} pp")

    # probability agreement
    diff = np.abs(P32 - P8)
    r = np.corrcoef(P32.ravel(), P8.ravel())[0, 1]
    print("\n================ FP8 vs FP32 probability agreement ================")
    print(f"  Pearson r: {r:.5f}   MAE: {diff.mean():.5f}   max|Δ|: {diff.max():.4f}")

    # tag flips at the deployed threshold
    b32 = P32 >= args.thr; b8 = P8 >= args.thr
    for c in SKIP:
        b32[:, c] = False; b8[:, c] = False
    n32 = b32.sum(1).mean(); n8 = b8.sum(1).mean()
    added = (b8 & ~b32).sum(1).mean()
    removed = (~b8 & b32).sum(1).mean()
    exact = np.mean((b32 == b8).all(1))
    print(f"\n================ tag set at threshold {args.thr} ================")
    print(f"  mean tags/image  FP32={n32:.2f}  FP8={n8:.2f}")
    print(f"  mean flips/image  added={added:.3f}  removed={removed:.3f}  "
          f"(net {n8-n32:+.3f})")
    print(f"  images with identical tag set: {exact*100:.1f}%")


if __name__ == "__main__":
    main()
