#!/usr/bin/env python3
"""Benchmark FP32 / FP16 / FP8(weight-only) ONNX on the CUDA EP."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort


def info(path):
    m = onnx.load(path, load_external_data=False)
    inp = {i.name: i.type.tensor_type.elem_type for i in m.graph.input}
    out0 = m.graph.output[0]
    odims = [d.dim_param or d.dim_value for d in out0.type.tensor_type.shape.dim]
    return inp, odims


def bench(path, batch, runs, warmup, image_size=448):
    inp, odims = info(path)
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    prov = sess.get_providers()[0]
    # input dtype: 1=float32, 10=float16
    px_dtype = np.float16 if inp.get("pixel_values") == 10 else np.float32
    feed = {"pixel_values": np.random.randn(batch, 3, image_size, image_size).astype(px_dtype)}
    if "padding_mask" in inp:
        feed["padding_mask"] = np.zeros((batch, image_size, image_size), dtype=bool)
    for _ in range(warmup):
        sess.run(None, feed)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, feed)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {
        "provider": prov, "px_dtype": px_dtype.__name__, "out": odims,
        "mean_ms": ts.mean(), "median_ms": np.median(ts), "p90_ms": np.percentile(ts, 90),
        "img_per_s": batch * 1000 / ts.mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "FP32=huggingface_release/V1.1_onnx/model.onnx",
        "FP16=exported_model_fp16.onnx",
        "FP8wo=huggingface_release/V1.1_fp8_onnx/model.onnx",
    ])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    parsed = [m.split("=", 1) for m in args.models]
    for name, path in parsed:
        if not Path(path).exists():
            print(f"  ! {name}: missing {path}")
    print(f"{'model':8} {'batch':>5} {'dtype':>8} {'mean ms':>9} {'median':>8} {'p90':>7} {'img/s':>9}")
    results = {}
    for name, path in parsed:
        if not Path(path).exists():
            continue
        for b in args.batches:
            r = bench(path, b, args.runs, args.warmup)
            results[(name, b)] = r
            print(f"{name:8} {b:5d} {r['px_dtype']:>8} {r['mean_ms']:9.2f} "
                  f"{r['median_ms']:8.2f} {r['p90_ms']:7.2f} {r['img_per_s']:9.1f}")

    # speed ratios vs FP16 at each batch
    print("\nrelative throughput (img/s), FP16 = 1.00x:")
    for b in args.batches:
        if ("FP16", b) in results:
            base = results[("FP16", b)]["img_per_s"]
            row = "  batch %d: " % b + "  ".join(
                f"{n}={results[(n,b)]['img_per_s']/base:.2f}x"
                for n, _ in parsed if (n, b) in results)
            print(row)


if __name__ == "__main__":
    main()
