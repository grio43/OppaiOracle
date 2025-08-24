#!/usr/bin/env python3
"""ONNXRuntime inference using embedded vocabulary and preprocessing."""
import argparse
import json
import base64
import gzip
import hashlib
import logging
import time
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

from vocabulary import TagVocabulary, verify_vocabulary_integrity
from schemas import RunMetadata, TagPrediction, ImagePrediction, PredictionOutput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def _preprocess(image_path: str, image_size: int, mean, std):
    """Preprocess image for ONNX inference with explicit float32 handling."""
    img = Image.open(image_path).convert('RGB').resize((image_size, image_size))

    # Ensure all operations stay in float32 to prevent dtype promotion
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Convert mean and std to numpy arrays to prevent float64 promotion
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, :]
    return arr.astype(np.float32)  # Explicit final cast to ensure float32


def main():
    parser = argparse.ArgumentParser(description='ONNX inference with embedded vocab')
    parser.add_argument('model', type=str, help='Path to ONNX model')
    parser.add_argument('images', nargs='+', help='Image paths')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--vocab', type=str, help='External vocabulary file (if not embedded)')
    # Preprocessing parameters (required when metadata is missing)
    parser.add_argument('--mean', type=float, nargs=3, metavar=('R', 'G', 'B'),
                        help='Normalization mean values for RGB channels (required if not in model metadata)')
    parser.add_argument('--std', type=float, nargs=3, metavar=('R', 'G', 'B'),
                        help='Normalization std values for RGB channels (required if not in model metadata)')
    parser.add_argument('--image-size', type=int, help='Input image size (required if not in model metadata)')
    parser.add_argument('--patch-size', type=int, help='Model patch size (required if not in model metadata)')
    args = parser.parse_args()

    # Explicitly specify providers for better control
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(
        args.model,
        providers=providers
    )
    logger.info(f"Using providers: {session.get_providers()}")
    result = _load_metadata(session)
    vocab_embedded = True

    if result[0] is None:
        # Need to load external vocabulary
        if not args.vocab:
            raise RuntimeError(
                "Model lacks embedded vocabulary and no --vocab file provided. "
                "Please specify vocabulary with --vocab vocabulary.json"
            )
        vocab = TagVocabulary(Path(args.vocab))
        # Verify external vocabulary
        verify_vocabulary_integrity(vocab, Path(args.vocab))
        
        # Check for required preprocessing parameters
        missing_params = []
        if args.mean is None:
            missing_params.append("--mean")
        if args.std is None:
            missing_params.append("--std")
        if args.image_size is None:
            missing_params.append("--image-size")
        if args.patch_size is None:
            missing_params.append("--patch-size")

        if missing_params:
            raise RuntimeError(
                f"Model lacks preprocessing metadata and required parameters are missing: {', '.join(missing_params)}\n"
                f"When model metadata is not available, you must explicitly provide:\n"
                f"  --mean R G B        (e.g., --mean 0.5 0.5 0.5)\n"
                f"  --std R G B         (e.g., --std 0.5 0.5 0.5)\n"
                f"  --image-size SIZE   (e.g., --image-size 640)\n"
                f"  --patch-size SIZE   (e.g., --patch-size 16)\n"
                f"These values must match those used during training to ensure correct predictions."
            )

        # Use explicitly provided parameters
        mean = list(args.mean)
        std = list(args.std)
        image_size = args.image_size
        patch_size = args.patch_size
        meta = result[-1]  # Get the meta from the result
        vocab_embedded = False
    else:
        vocab, mean, std, image_size, patch_size, meta = result
    input_name = session.get_inputs()[0].name

    results = []
    for path in args.images:
        start = time.time()
        # Preprocess with explicit float32 handling
        try:
            inp = _preprocess(path, image_size, mean, std)
            # Verify dtype before inference
            if inp.dtype != np.float32:
                logger.warning(f"Input dtype is {inp.dtype}, converting to float32")
                inp = inp.astype(np.float32)
        except Exception as e:
            logger.error(f"Preprocessing failed for {path}: {e}")
            continue

        outputs = session.run(None, {input_name: inp})

        # Handle both old (predictions, scores) and new (scores only) model formats
        if len(outputs) == 1:
            scores = outputs[0][0]  # New format: scores only
        else:
            scores = outputs[-1][0]  # Old format: assume last output are scores

        idxs = np.argsort(scores)[::-1][:args.top_k]
        tags = []
        for idx in idxs:
            score = float(scores[idx])
            if score < args.threshold:
                continue
            # This will raise ValueError if placeholder detected
            tag_name = vocab.get_tag_from_index(int(idx))
            tags.append(TagPrediction(name=tag_name, score=score))

        results.append(ImagePrediction(
            image=path,
            tags=tags,
            processing_time=int((time.time() - start) * 1000)
        ))

    # Create metadata using the schema
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

    # Log metadata source
    if vocab_embedded:
        print(f"Using embedded vocabulary metadata from model", file=sys.stderr)
    else:
        print(f"Using external vocabulary from {args.vocab}", file=sys.stderr)

    # Create output using schema
    output = PredictionOutput(metadata=metadata, results=results)

    if args.output:
        output.save(Path(args.output))
    else:
        print(output.to_json())


if __name__ == '__main__':
    main()
