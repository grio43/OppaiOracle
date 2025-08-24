#!/usr/bin/env python3
"""ONNXRuntime inference using embedded vocabulary and preprocessing."""
import argparse
import json
import base64
import gzip
import hashlib
import time
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

from vocabulary import TagVocabulary, verify_vocabulary_integrity
from schemas import RunMetadata, TagPrediction, ImagePrediction, PredictionOutput


def _load_metadata(session: ort.InferenceSession):
    meta = session.get_modelmeta().custom_metadata_map
    if 'vocab_b64_gzip' not in meta:
        # Try to load from external vocabulary file as fallback
        import warnings
        warnings.warn(
            "Model is missing embedded vocabulary metadata. "
            "Attempting to load from external vocabulary.json file. "
            "For reproducible inference, please use models with embedded vocabulary.",
            RuntimeWarning
        )
        # Return None to signal external vocab needed
        return None, None, None, None, None, meta
    vocab_bytes = gzip.decompress(base64.b64decode(meta['vocab_b64_gzip']))
    sha = hashlib.sha256(vocab_bytes).hexdigest()
    if meta.get('vocab_sha256') and meta['vocab_sha256'] != sha:
        raise RuntimeError('Vocabulary checksum mismatch')
    vocab = TagVocabulary.from_json(vocab_bytes.decode('utf-8'))

    # Verify vocabulary integrity using centralized function
    verify_vocabulary_integrity(vocab, Path("embedded_vocabulary"))
    mean = json.loads(meta.get('normalize_mean', '[0.5, 0.5, 0.5]'))
    std = json.loads(meta.get('normalize_std', '[0.5, 0.5, 0.5]'))
    image_size = int(meta.get('image_size', 448))
    patch_size = int(meta.get('patch_size', 32))
    return vocab, mean, std, image_size, patch_size, meta


def _preprocess(image_path: str, image_size: int, mean, std):
    img = Image.open(image_path).convert('RGB').resize((image_size, image_size))
    arr = np.array(img).astype('float32') / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, :]
    return arr


def main():
    parser = argparse.ArgumentParser(description='ONNX inference with embedded vocab')
    parser.add_argument('model', type=str, help='Path to ONNX model')
    parser.add_argument('images', nargs='+', help='Image paths')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--vocab', type=str, help='External vocabulary file (if not embedded)')
    args = parser.parse_args()

    session = ort.InferenceSession(args.model)
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
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        image_size = 448
        patch_size = 32
        meta = result[-1]  # Get the meta from the result
        vocab_embedded = False
    else:
        vocab, mean, std, image_size, patch_size, meta = result
    input_name = session.get_inputs()[0].name

    results = []
    for path in args.images:
        start = time.time()
        inp = _preprocess(path, image_size, mean, std)
        outputs = session.run(None, {input_name: inp})
        scores = outputs[-1][0]  # assume last output are scores
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
