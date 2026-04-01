#!/usr/bin/env python3
"""ONNXRuntime inference using embedded vocabulary and preprocessing."""
import argparse
import json
import base64
import gzip
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import time
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFile
import onnxruntime as ort
import yaml
from Configuration_System import ConfigManager, ConfigType

from vocabulary import TagVocabulary, verify_vocabulary_integrity
from schemas import RunMetadata, TagPrediction, ImagePrediction, PredictionOutput


logger = logging.getLogger('onnx_infer')

# _setup_logging and _load_infer_cfg are removed, their logic is now in main()


def _load_metadata(session: ort.InferenceSession):
    meta = session.get_modelmeta().custom_metadata_map

    # Check if vocabulary metadata exists and is non-empty
    vocab_b64_gzip = meta.get('vocab_b64_gzip', '')
    vocab_sha256 = meta.get('vocab_sha256', '')

    # Treat missing or empty vocabulary metadata as no embedded vocab
    if not vocab_b64_gzip or not vocab_sha256:
        import warnings
        warnings.warn(
            "Model is missing or has empty embedded vocabulary metadata. "
            "Attempting to load from external vocabulary.json file. "
            "For reproducible inference, please use models with embedded vocabulary.",
            RuntimeWarning
        )
        # Return None to signal external vocab needed
        return None, None, None, None, None, None, meta

    # Try to decode and decompress the vocabulary with robust error handling
    try:
        # Decode base64
        vocab_b64_decoded = base64.b64decode(vocab_b64_gzip)

        # Decompress gzip
        vocab_bytes = gzip.decompress(vocab_b64_decoded)

        # Verify checksum
        sha = hashlib.sha256(vocab_bytes).hexdigest()
        if vocab_sha256 != sha:
            raise RuntimeError(f'Vocabulary checksum mismatch: expected {vocab_sha256}, got {sha}')

        # Decode UTF-8 and parse JSON
        vocab_json = vocab_bytes.decode('utf-8')
        if not vocab_json.strip():
            raise ValueError("Embedded vocabulary JSON is empty")

        vocab = TagVocabulary.from_json(vocab_json)

        # Verify vocabulary integrity using centralized function
        verify_vocabulary_integrity(vocab, Path("embedded_vocabulary"))

    except Exception as e:
        import warnings
        warnings.warn(
            f"Failed to load embedded vocabulary: {e}. "
            "Will require external vocabulary file (--vocab) and preprocessing parameters "
            "(--mean, --std, --image-size, --patch-size) for inference.",
            RuntimeWarning
        )
        return None, None, None, None, None, None, meta

    mean = json.loads(meta.get('normalize_mean', '[0.5, 0.5, 0.5]'))
    std = json.loads(meta.get('normalize_std', '[0.5, 0.5, 0.5]'))
    image_size = int(meta.get('image_size', 512))
    patch_size = int(meta.get('patch_size', 16))
    pad_color = tuple(json.loads(meta.get('pad_color', '[114, 114, 114]')))
    return vocab, mean, std, image_size, patch_size, pad_color, meta


def _load_image(image_path: str, pad_color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, bool]:
    """Load an image from disk, handling EXIF rotation and transparency.

    Returns:
        (H, W, 3) uint8 RGB numpy array and a flag indicating if transparency
        compositing occurred (used to suppress gray_background tag).
    """
    was_composited = False
    with Image.open(image_path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        # Check for transparency and composite if necessary
        if img.mode in ('RGBA', 'LA') or 'transparency' in img.info:
            was_composited = True
            # Create background using pad_color to match training
            background = Image.new('RGB', img.size, pad_color)
            img_rgba = img.convert('RGBA')
            alpha = img_rgba.getchannel('A')
            rgb = img_rgba.convert('RGB')
            background.paste(rgb, mask=alpha)
            img = background
        else:
            img = img.convert('RGB')

        arr = np.asarray(img, dtype=np.uint8)

    return arr, was_composited


def _preprocess_for_model(
    image: np.ndarray,
    image_size: int,
    mean: list[float],
    std: list[float],
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Letterbox, normalize, and format an image for the ONNX model.

    Matches the training pipeline preprocessing:
    1. Scale to fit inside image_size x image_size (no upscaling)
    2. Center on a pad_color canvas
    3. Normalize with mean/std
    4. Return (1, 3, H, W) float32

    Args:
        image: (H, W, 3) uint8 RGB numpy array.
        image_size: Target square size.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.
        pad_color: RGB fill for letterbox padding.

    Returns:
        (1, 3, image_size, image_size) float32 numpy array, normalized.
    """
    h, w = image.shape[:2]
    target = image_size

    # Scale to fit, never upscale
    scale = min(target / w, target / h, 1.0)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    # Resize with PIL (BILINEAR matches training pipeline)
    if new_w != w or new_h != h:
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        image = np.asarray(pil_img, dtype=np.uint8)

    # Create canvas and center the image
    canvas = np.full((target, target, 3), pad_color, dtype=np.uint8)
    top = (target - new_h) // 2
    left = (target - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = image

    # Convert to float32 [0, 1], normalize, transpose to (C, H, W)
    x = canvas.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean_arr) / std_arr
    x = x.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

    return np.expand_dims(x, axis=0)  # (1, C, H, W)


def _preprocess_legacy(image_path: str, pad_color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, bool]:
    """Legacy preprocessing for old ONNX models that bake preprocessing into the graph.

    Returns (1, H, W, 3) uint8 array and compositing flag.
    """
    arr, was_composited = _load_image(image_path, pad_color)
    return np.expand_dims(arr, axis=0), was_composited


def main():
    from utils.logging_setup import setup_logging
    listener = setup_logging()

    try:
        # Load unified config to get defaults
        try:
            manager = ConfigManager(config_type=ConfigType.FULL)
            unified_config = manager.load_from_file("configs/unified_config.yaml")
            infer_cfg = unified_config.inference
        except Exception as e:
            logger.warning(f"Could not load unified_config.yaml: {e}. Using default settings.")
            # Create dummy configs to avoid crashing
            from dataclasses import dataclass
            @dataclass
            class Dummy:
                def __getattr__(self, name):
                    if name == 'prediction_threshold': return 0.0
                    if name == 'top_k': return 5
                    return None
            infer_cfg = Dummy()

        parser = argparse.ArgumentParser(description='ONNX inference with embedded vocab')
        parser.add_argument('model', type=str, help='Path to ONNX model')
        parser.add_argument('images', nargs='+', help='Image paths')
        parser.add_argument('--top_k', type=int, default=infer_cfg.top_k, help=f"Default: {infer_cfg.top_k}")
        parser.add_argument('--threshold', type=float, default=infer_cfg.prediction_threshold, help=f"Default: {infer_cfg.prediction_threshold}")
        parser.add_argument('--output', type=str, help='Output JSON file')
        parser.add_argument('--vocab', type=str, help='External vocabulary file (if not embedded and model is old)')
        parser.add_argument('--providers', nargs='*', default=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        args = parser.parse_args()

        # Explicitly specify providers for better control
        providers = args.providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(
            args.model,
            providers=providers
        )
        logger.info(f"Using providers: {session.get_providers()}")

        # Load metadata - this contains the vocabulary
        vocab, mean, std, image_size, patch_size, pad_color, meta = _load_metadata(session)
        vocab_embedded = True

        if vocab is None:
            # Fallback for old models without embedded vocab
            if not args.vocab:
                raise RuntimeError(
                    "Model lacks embedded vocabulary and no --vocab file provided. "
                    "Please specify vocabulary with --vocab vocabulary.json"
                )
            vocab = TagVocabulary(Path(args.vocab))
            verify_vocabulary_integrity(vocab, Path(args.vocab))
            vocab_embedded = False
            # Use default pad_color for old models without metadata
            if pad_color is None:
                pad_color = (114, 114, 114)

        # Load calibrated thresholds if available (supports both per-tag and per-bucket)
        calibrated_thresholds = None
        model_dir = Path(args.model).parent
        thresholds_path = model_dir / "thresholds.json"
        if thresholds_path.exists():
            try:
                with open(thresholds_path, 'r') as f:
                    raw_thresholds = json.load(f)
                # Detect per-bucket mode: keys look like "300-499" or "10000+"
                sample_key = next(iter(raw_thresholds)) if raw_thresholds else ""
                is_bucket_mode = bool(sample_key and (sample_key[-1] == '+' or '-' in sample_key) and sample_key[0].isdigit())
                if is_bucket_mode and hasattr(vocab, 'tag_frequencies'):
                    # Expand bucket thresholds to per-tag thresholds
                    calibrated_thresholds = {}
                    for tag_name, freq in vocab.tag_frequencies.items():
                        matched = False
                        for bucket_key, thresh_val in raw_thresholds.items():
                            if bucket_key.endswith('+'):
                                low = int(bucket_key[:-1])
                                if freq >= low:
                                    calibrated_thresholds[tag_name] = thresh_val
                                    matched = True
                                    break
                            elif '-' in bucket_key:
                                low, high = bucket_key.split('-', 1)
                                if int(low) <= freq <= int(high):
                                    calibrated_thresholds[tag_name] = thresh_val
                                    matched = True
                                    break
                        if not matched:
                            calibrated_thresholds[tag_name] = args.threshold
                    logger.info(f"Loaded per-bucket thresholds from {thresholds_path}, expanded to {len(calibrated_thresholds)} tags")
                else:
                    calibrated_thresholds = raw_thresholds
                    logger.info(f"Loaded per-tag thresholds from {thresholds_path} ({len(calibrated_thresholds)} entries)")
            except Exception as e:
                logger.warning(f"Failed to load thresholds from {thresholds_path}: {e}")

        input_name = session.get_inputs()[0].name
        input_info = session.get_inputs()[0]
        logger.info(f"Model expects input '{input_name}' with type: {input_info.type}")

        # Detect model format: new models have preprocessing=external in metadata
        is_new_format = meta.get('preprocessing', '') == 'external'
        has_sigmoid = meta.get('output_activation', '') == 'sigmoid'
        if is_new_format:
            logger.info("New model format: preprocessing is external, output is sigmoid probabilities")
        else:
            logger.info("Legacy model format: preprocessing baked in, applying sigmoid post-hoc")

        results = []
        for path in args.images:
            start = time.time()
            try:
                if is_new_format:
                    # New format: load image, preprocess externally, feed (1, C, H, W) float32
                    raw_image, was_composited = _load_image(path, pad_color)
                    inp = _preprocess_for_model(raw_image, image_size, mean, std, pad_color)
                else:
                    # Legacy format: model expects (1, H, W, 3) uint8
                    inp, was_composited = _preprocess_legacy(path, pad_color)
            except Exception as e:
                logger.error(f"Preprocessing failed for {path}: {e}")
                results.append(ImagePrediction(
                    image=path,
                    tags=[],
                    processing_time=int((time.time() - start) * 1000),
                    error=str(e)
                ))
                continue

            try:
                outputs = session.run(None, {input_name: inp})

                # Single output: probabilities (new) or raw logits (legacy)
                tag_scores = outputs[0][0]

                # Legacy models output raw logits — apply sigmoid
                if not has_sigmoid:
                    tag_scores = 1.0 / (1.0 + np.exp(-tag_scores.astype(np.float64)))
                    tag_scores = tag_scores.astype(np.float32)

                idxs = np.argsort(tag_scores)[::-1][:args.top_k]
                # Resolve special-token indices once
                try:
                    pad_idx = vocab.tag_to_index.get(getattr(vocab, "pad_token", "<PAD>"), 0)
                    unk_idx = getattr(vocab, "unk_index",
                                      vocab.tag_to_index.get(getattr(vocab, "unk_token", "<UNK>"), 1))
                except Exception:
                    pad_idx = 0
                    unk_idx = 1

                tags = []
                predicted_rating = None
                rating_confidence = None
                for idx in idxs:
                    score = float(tag_scores[idx])
                    # Skip special tokens (padding/unknown)
                    if int(idx) in (pad_idx, unk_idx):
                        continue
                    try:
                        tag_name = vocab.get_tag_from_index(int(idx))
                    except ValueError:
                        continue
                    # Use per-tag calibrated threshold if available, else global threshold
                    if calibrated_thresholds is not None:
                        tag_thresh = calibrated_thresholds.get(tag_name, args.threshold)
                    else:
                        tag_thresh = args.threshold
                    if score < tag_thresh:
                        continue
                    # Extract rating from rating:* tags
                    if tag_name.startswith("rating:"):
                        if predicted_rating is None or score > rating_confidence:
                            predicted_rating = tag_name.split(":", 1)[1]
                            rating_confidence = score
                    tags.append(TagPrediction(name=tag_name, score=score))

                # Conditionally filter the gray_background tag if we added it
                if was_composited:
                    tags = [tag for tag in tags if tag.name != 'gray_background']

                result = ImagePrediction(
                    image=path,
                    tags=tags,
                    processing_time=int((time.time() - start) * 1000)
                )
                if predicted_rating is not None:
                    result.rating = predicted_rating
                    result.rating_confidence = rating_confidence

                results.append(result)

            except Exception as e:
                logger.error(f"Inference failed for {path}: {e}")
                # Add failed result
                results.append(ImagePrediction(
                    image=path,
                    tags=[],
                    processing_time=int((time.time() - start) * 1000),
                    error=str(e)
                ))

        metadata = RunMetadata(
            top_k=args.top_k,
            threshold=args.threshold,
            vocab_sha256=meta.get('vocab_sha256', 'unknown'),
            normalize_mean=mean,
            normalize_std=std,
            image_size=image_size,
            patch_size=patch_size,
            model_path=args.model,
            num_tags=len(vocab.tags) if hasattr(vocab, 'tags') else None,
            vocab_embedded=vocab_embedded
        )

        if vocab_embedded:
            print(f"Using embedded vocabulary metadata from model", file=sys.stderr)
        else:
            print(f"Using external vocabulary from {args.vocab}", file=sys.stderr)

        output = PredictionOutput(metadata=metadata, results=results)

        if args.output:
            output.save(Path(args.output))
        else:
            print(output.to_json())
    finally:
        if listener:
            listener.stop()


# Optionally allow truncated images to load (opt-in)
import os as _os
if bool(int(_os.environ.get("OO_ALLOW_TRUNCATED", "0"))):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
if __name__ == '__main__':
    main()
