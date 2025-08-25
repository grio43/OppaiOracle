#!/usr/bin/env python3
"""
Per-tag threshold calibration.
Sweeps thresholds on a held-out validation set and writes artifacts/thresholds.json.
Usage:
  python tools/calibrate_thresholds.py --val_dir <images> --labels <val_labels.json> --out artifacts/thresholds.json
"""
import argparse, json, os
from pathlib import Path
from typing import Dict, List
import numpy as np


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    For each tag (column), choose threshold that maximizes F-beta.
    """
    T = y_true.shape[1]
    thresholds = np.zeros(T, dtype=np.float32)
    for j in range(T):
        scores = y_prob[:, j]
        truth = y_true[:, j] > 0.5
        # candidate thresholds from unique scores
        cands = np.unique(scores)
        best_f, best_t = -1.0, 0.5
        for t in cands:
            pred = scores >= t
            tp = np.sum(pred & truth)
            fp = np.sum(pred & ~truth)
            fn = np.sum(~pred & truth)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            if precision + recall == 0:
                f = 0.0
            else:
                b2 = beta * beta
                f = (1 + b2) * precision * recall / max(b2 * precision + recall, 1e-8)
            if f > best_f:
                best_f, best_t = f, float(t)
        thresholds[j] = best_t
    return thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=str, required=True, help="Numpy .npz with probs and tag_names")
    ap.add_argument("--labels", type=str, required=True, help="Numpy .npz with binary labels")
    ap.add_argument("--beta", type=float, default=1.0, help="F-beta to optimize")
    ap.add_argument("--out", type=str, default="artifacts/thresholds.json")
    args = ap.parse_args()

    probs = np.load(args.probs)
    labels = np.load(args.labels)
    y_prob = probs["probs"].astype(np.float32)
    y_true = labels["labels"].astype(np.float32)
    tag_names = probs["tag_names"].tolist()

    th = sweep_thresholds(y_true, y_prob, beta=args.beta)
    out = {t: float(v) for t, v in zip(tag_names, th)}
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} with {len(out)} thresholds.")


if __name__ == "__main__":
    main()

