#!/usr/bin/env python3
"""Validate an FP8 ONNX export: structure, runtime, and agreement vs FP32."""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_fp8 import load_rgb, preprocess  # reuse identical preprocessing

FLOAT8E4M3FN = 17  # TensorProto.FLOAT8E4M3FN


def describe(model_path: Path):
    m = onnx.load(str(model_path), load_external_data=False)
    ops = Counter(n.op_type for n in m.graph.node)
    # count float8 initializers
    f8_init = sum(1 for init in m.graph.initializer if init.data_type == FLOAT8E4M3FN)
    print(f"  nodes={len(m.graph.node)} QuantizeLinear={ops.get('QuantizeLinear',0)} "
          f"DequantizeLinear={ops.get('DequantizeLinear',0)} MatMul={ops.get('MatMul',0)} "
          f"Conv={ops.get('Conv',0)} Gemm={ops.get('Gemm',0)}")
    print(f"  float8e4m3fn initializers: {f8_init}")
    meta = {p.key for p in m.metadata_props}
    print(f"  metadata: vocab_embedded={'vocab_b64_gzip' in meta} "
          f"quantization={[p.value for p in m.metadata_props if p.key=='quantization']}")
    return ops, f8_init


def run(session, img_path, has_mask):
    x, mask = preprocess(load_rgb(Path(img_path)))
    feed = {"pixel_values": x}
    if has_mask:
        feed["padding_mask"] = mask
    return session.run(None, feed)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", default="huggingface_release/V1.1_fp8_onnx/model.onnx")
    ap.add_argument("--fp32", default="huggingface_release/V1.1_onnx/model.onnx")
    ap.add_argument("--images", nargs="+",
                    default=sorted(str(p) for p in Path("exported_model/test").glob("*.jpg"))[:5])
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    print(f"== FP8 model: {args.fp8} ({Path(args.fp8).stat().st_size/1e6:.1f} MB) ==")
    describe(Path(args.fp8))
    print(f"== FP32 model: {args.fp32} ({Path(args.fp32).stat().st_size/1e6:.1f} MB) ==")

    providers = ort.get_available_providers()
    print(f"available providers: {providers}")

    # Numerical-correctness check: load FP8 with all graph optimizations DISABLED
    # so QDQ is NOT fused into QLinear* ops. DequantizeLinear casts fp8->float and
    # compute stays in float, which runs on the CPU EP everywhere and isolates the
    # quantization error from any EP-specific kernel behaviour.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_fp8 = ort.InferenceSession(args.fp8, so, providers=["CPUExecutionProvider"])
    has_mask = "padding_mask" in {i.name for i in sess_fp8.get_inputs()}
    sess_fp32 = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])

    print(f"\n[CPU EP, fake-quant / opt=DISABLE_ALL] FP8 loaded. has_mask={has_mask}")
    jaccards, max_diffs = [], []
    for img in args.images:
        p8 = run(sess_fp8, img, has_mask)
        p32 = run(sess_fp32, img, has_mask)
        top8 = set(np.argsort(p8)[::-1][:args.topk].tolist())
        top32 = set(np.argsort(p32)[::-1][:args.topk].tolist())
        inter = len(top8 & top32)
        jac = inter / len(top8 | top32)
        md = float(np.max(np.abs(p8.astype(np.float32) - p32.astype(np.float32))))
        jaccards.append(jac); max_diffs.append(md)
        print(f"  {Path(img).name}: top{args.topk} overlap={inter}/{args.topk} "
              f"Jaccard={jac:.3f} max|Δprob|={md:.4f} "
              f"range8=[{p8.min():.3f},{p8.max():.3f}]")
    print(f"\nmean top-{args.topk} Jaccard={np.mean(jaccards):.3f} "
          f"mean max|Δprob|={np.mean(max_diffs):.4f}")

    # Real FP8 execution target: CUDA EP (RTX 5090 / sm_120) with full optimization.
    if "CUDAExecutionProvider" in providers:
        try:
            s = ort.InferenceSession(args.fp8, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            cuda_jac = []
            for img in args.images:
                pc = run(s, img, has_mask)
                p32 = run(sess_fp32, img, has_mask)
                tc = set(np.argsort(pc)[::-1][:args.topk].tolist())
                t32 = set(np.argsort(p32)[::-1][:args.topk].tolist())
                cuda_jac.append(len(tc & t32) / len(tc | t32))
            print(f"\n[CUDA EP, full opt] FP8 ran. providers={s.get_providers()} "
                  f"mean top-{args.topk} Jaccard vs FP32={np.mean(cuda_jac):.3f}")
        except Exception as e:
            print(f"\n[CUDA EP] FP8 run failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
