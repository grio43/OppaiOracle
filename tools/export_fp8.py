#!/usr/bin/env python3
"""FP8 (E4M3) static quantization of the V1.1 ONNX tagger -> "FP8 OppaiOracle".

ONNX Runtime only supports float8 via *static* quantization (QDQ format) with a
calibration data reader and CalibrationMethod.Distribution. There is no float8
dynamic-quantization path. This script:

  1. Takes a clean FP32 ONNX graph (the released V1.1 model).
  2. Calibrates activation ranges on real images preprocessed exactly like the
     model expects (letterbox 448, RGB, mean/std 0.5, pad 114).
  3. Runs quantize_static with weight+activation type QFLOAT8E4M3FN.
  4. Re-attaches the embedded vocabulary + preprocessing metadata that
     quantize_static strips, and stamps an fp8 marker.

Run with the project interpreter, e.g.:
  L:\\Dab\\payton_env\\Scripts\\python.exe tools\\export_fp8.py --limit 8 --smoke
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    CalibrationMethod,
    QuantType,
    QuantFormat,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_fp8")


def install_onnx_float8_shims():
    """Bridge an onnxruntime 1.23 <-> onnx 1.20 float8 incompatibility.

    ORT's `compute_scale_zp_float8` imports two symbols that onnx 1.20 moved/
    removed: `onnx.numpy_helper.float8e4m3_to_float32` and the module
    `onnx.reference.custom_element_types` (for the `float8e4m3fn` numpy dtype).
    onnx 1.20 represents float8 natively via `ml_dtypes`, so we reconstruct both
    from `ml_dtypes` without touching site-packages or package versions.
    """
    import types
    import importlib
    import ml_dtypes
    import onnx.numpy_helper as nh

    if not hasattr(nh, "float8e4m3_to_float32"):
        def float8e4m3_to_float32(i):
            # i is a raw uint8 byte (0..255); reinterpret as an E4M3FN value.
            return float(np.array(i, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32))
        nh.float8e4m3_to_float32 = float8e4m3_to_float32

    try:
        importlib.import_module("onnx.reference.custom_element_types")
    except ModuleNotFoundError:
        mod = types.ModuleType("onnx.reference.custom_element_types")
        mod.float8e4m3fn = ml_dtypes.float8_e4m3fn
        mod.float8e5m2 = ml_dtypes.float8_e5m2
        mod.bfloat16 = ml_dtypes.bfloat16
        sys.modules["onnx.reference.custom_element_types"] = mod
    logger.info("Installed onnx float8 compatibility shims (ml_dtypes-backed)")

# ── V1.1 preprocessing constants (from huggingface_release/V1.1_*/preprocessing.json) ──
IMAGE_SIZE = 448
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]
PAD_COLOR = (114, 114, 114)


def load_rgb(path: Path, pad_color=PAD_COLOR) -> np.ndarray:
    """Load image as (H,W,3) uint8 RGB, compositing transparency onto pad_color."""
    with Image.open(path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or "transparency" in img.info:
            bg = Image.new("RGB", img.size, pad_color)
            rgba = img.convert("RGBA")
            bg.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
            img = bg
        else:
            img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def preprocess(image: np.ndarray):
    """Letterbox + normalize -> ((1,3,H,W) f32, (1,H,W) bool mask). Matches onnx_infer."""
    h, w = image.shape[:2]
    target = IMAGE_SIZE
    scale = min(target / w, target / h, 1.0)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    if (new_w, new_h) != (w, h):
        image = np.asarray(
            Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8
        )
    canvas = np.full((target, target, 3), PAD_COLOR, dtype=np.uint8)
    top, left = (target - new_h) // 2, (target - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = image
    mask = np.ones((target, target), dtype=bool)
    mask[top:top + new_h, left:left + new_w] = False
    x = canvas.astype(np.float32) / 255.0
    x = (x - np.array(NORMALIZE_MEAN, np.float32)) / np.array(NORMALIZE_STD, np.float32)
    x = x.transpose(2, 0, 1)
    return np.expand_dims(x, 0), np.expand_dims(mask, 0)


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed (pixel_values, padding_mask) pairs to the ORT calibrator."""

    def __init__(self, image_paths, has_mask: bool):
        self.image_paths = list(image_paths)
        self.has_mask = has_mask
        self._iter = None
        self._used = 0

    def _gen(self):
        for p in self.image_paths:
            try:
                x, mask = preprocess(load_rgb(Path(p)))
            except Exception as e:  # skip unreadable images
                logger.warning(f"  skip calibration image {p}: {e}")
                continue
            self._used += 1
            feed = {"pixel_values": x}
            if self.has_mask:
                feed["padding_mask"] = mask
            yield feed

    def get_next(self):
        if self._iter is None:
            self._iter = self._gen()
        return next(self._iter, None)

    def rewind(self):
        self._iter = None


def sample_images(roots, limit: int, seed: int = 0):
    """Collect up to `limit` image paths, randomly sampled, from the given roots."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    found = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in exts:
                found.append(p)
            # cap the crawl so we don't walk the whole 296k-image dataset
            if len(found) >= max(limit * 40, 4000):
                break
        if len(found) >= max(limit * 40, 4000):
            break
    random.Random(seed).shuffle(found)
    return found[:limit]


def carry_metadata(src_path: Path, dst_path: Path):
    """Copy metadata_props from src ONNX onto dst ONNX and add fp8 markers."""
    src = onnx.load(str(src_path), load_external_data=False)
    src_meta = {p.key: p.value for p in src.metadata_props}

    dst_data = Path(str(dst_path) + ".data")
    has_ext = dst_data.exists()
    dst = onnx.load(str(dst_path), load_external_data=has_ext)
    del dst.metadata_props[:]
    src_meta["quantization"] = "fp8_e4m3fn_static_qdq"
    src_meta["model_description"] = "Anime Image Tagger (FP8 OppaiOracle, E4M3 static QDQ)"
    for k, v in src_meta.items():
        m = dst.metadata_props.add()
        m.key, m.value = k, str(v)
    if has_ext:
        dst_data.unlink()
        dst_path.unlink()
        onnx.save(dst, str(dst_path), save_as_external_data=True,
                  all_tensors_to_one_file=True, location=dst_data.name, size_threshold=1024)
    else:
        onnx.save(dst, str(dst_path))
    logger.info(f"Re-attached {len(src_meta)} metadata entries (vocab embedded: {'vocab_b64_gzip' in src_meta})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="huggingface_release/V1.1_onnx/model.onnx",
                    help="Source FP32 ONNX to quantize")
    ap.add_argument("--out", default="huggingface_release/V1.1_fp8_onnx/model.onnx",
                    help="Destination FP8 ONNX path")
    ap.add_argument("--data-root", default="L:/Dab/Dab", help="Image dir to sample calibration from")
    ap.add_argument("--limit", type=int, default=200, help="Number of calibration images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op-types", nargs="+", default=["MatMul", "Gemm"],
                    help="Op types to FP8-quantize. MatMul/Gemm are the ViT's heavy "
                         "GEMMs. Conv is excluded: the patch-embed QLinearConv has no "
                         "float8 kernel. Restricting to GEMMs also avoids the -inf "
                         "attention-mask tensors that break Distribution calibration.")
    ap.add_argument("--smoke", action="store_true",
                    help="Write to a temp _smoke path and do not touch the real output")
    ap.add_argument("--out-suffix", default="", help="Extra suffix for the output stem (sweeps)")
    ap.add_argument("--per-channel", action="store_true", help="Per-channel weight quantization")
    ap.add_argument("--calib-method", default="Distribution",
                    choices=["Distribution", "MinMax", "Percentile", "Entropy"])
    ap.add_argument("--calib-cuda", action="store_true",
                    help="Run calibration on CUDA EP (much faster on the fp32 source)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        logger.error(f"Source ONNX not found: {src}")
        sys.exit(1)
    out = Path(args.out)
    stem = out.stem + ("_smoke" if args.smoke else "") + args.out_suffix
    out = out.parent / (stem + ".onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Detect inputs (does the graph take a padding_mask?)
    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    has_mask = "padding_mask" in input_names
    del sess
    logger.info(f"Source inputs: {sorted(input_names)} (padding_mask: {has_mask})")

    logger.info(f"Sampling up to {args.limit} calibration images from {args.data_root} ...")
    cal_images = sample_images([args.data_root], args.limit, args.seed)
    if not cal_images:
        logger.error("No calibration images found.")
        sys.exit(1)
    logger.info(f"Collected {len(cal_images)} calibration image paths")

    reader = ImageCalibrationReader(cal_images, has_mask)

    install_onnx_float8_shims()
    calib_providers = ["CUDAExecutionProvider"] if args.calib_cuda else None
    logger.info(f"Running FP8 (E4M3) static quantization "
                f"(op_types={args.op_types}, per_channel={args.per_channel}, "
                f"calib={args.calib_method}, calib_cuda={args.calib_cuda}) ...")
    quantize_static(
        model_input=str(src),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QFLOAT8E4M3FN,
        weight_type=QuantType.QFLOAT8E4M3FN,
        per_channel=args.per_channel,
        calibrate_method=getattr(CalibrationMethod, args.calib_method),
        op_types_to_quantize=args.op_types,
        calibration_providers=calib_providers,
        use_external_data_format=True,
    )
    logger.info(f"Calibration used {reader._used} images; wrote {out}")

    carry_metadata(src, out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
