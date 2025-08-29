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
        return None, None, None, None, None, meta

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
        return None, None, None, None, None, meta

    mean = json.loads(meta.get('normalize_mean', '[0.5, 0.5, 0.5]'))
    std = json.loads(meta.get('normalize_std', '[0.5, 0.5, 0.5]'))
    image_size = int(meta.get('image_size', 448))
    patch_size = int(meta.get('patch_size', 32))
    return vocab, mean, std, image_size, patch_size, meta


def _preprocess(image_path: str) -> tuple[np.ndarray, bool]:
    """
    Loads an image and prepares it for the self-contained ONNX model.
    Handles transparent images by compositing them on a grey background.
    Returns the image array and a flag indicating if compositing occurred.
    """
    was_composited = False
    with Image.open(image_path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        # Check for transparency and composite if necessary
        if img.mode in ('RGBA', 'LA') or 'transparency' in img.info:
            was_composited = True
            # Create a grey background to match training
            background = Image.new('RGB', img.size, (114, 114, 114))
            # Ensure we have explicit alpha and RGB color data, even for 'LA' or 'P' images
            img_rgba = img.convert('RGBA')           # normalizes 'LA'/'P' w/ transparency to RGBA
            alpha = img_rgba.getchannel('A')         # single-band (L) mask
            rgb   = img_rgba.convert('RGB')
            # Paste with a single-channel mask for correct transparency handling
            background.paste(rgb, mask=alpha)
            img = background
        else:
            img = img.convert('RGB')

        # Convert to numpy array
        arr = np.asarray(img, dtype=np.uint8)

    # Add batch dimension -> (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr, was_composited


def _preprocess_simple(image_path: str, image_size: int, mean, std):
    """Simple preprocessing for backward compatibility"""
    return _preprocess(image_path, image_size, mean, std, session=None)


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
        vocab, mean, std, image_size, patch_size, meta = _load_metadata(session)
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

        input_name = session.get_inputs()[0].name
        input_info = session.get_inputs()[0]
        logger.info(f"Model expects input '{input_name}' with type: {input_info.type}")

        results = []
        for path in args.images:
            start = time.time()
            try:
                # Preprocessing now returns a flag for compositing
                inp, was_composited = _preprocess(path)
            except Exception as e:
                logger.error(f"Preprocessing failed for {path}: {e}")
                continue

            outputs = session.run(None, {input_name: inp})

            if len(outputs) == 1:
                scores = outputs[0][0]
            else:
                scores = outputs[-1][0]

            idxs = np.argsort(scores)[::-1][:args.top_k]
            tags = []
            for idx in idxs:
                score = float(scores[idx])
                if score < args.threshold:
                    continue
                tag_name = vocab.get_tag_from_index(int(idx))
                tags.append(TagPrediction(name=tag_name, score=score))

            # Conditionally filter the gray_background tag if we added it
            if was_composited:
                tags = [tag for tag in tags if tag.name != 'gray_background']

            results.append(ImagePrediction(
                image=path,
                tags=tags,
                processing_time=int((time.time() - start) * 1000)
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
