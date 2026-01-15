"""Batched ONNX inference engine for dataset grading.

Adapted from onnx_infer.py for high-throughput batch processing.
"""

import base64
import gzip
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, ImageFile
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class BatchResult:
    """Result for a single image in a batch."""
    image_path: str
    scores: np.ndarray  # Raw sigmoid scores for all tags
    error: Optional[str] = None


class ONNXBatchInference:
    """High-throughput ONNX inference for grading large datasets.

    Uses batched inference with preprocessing embedded in the ONNX model.
    Vocabulary is extracted from model metadata.
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        providers: Optional[List[str]] = None,
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to ONNX model file
            batch_size: Number of images per batch
            providers: ONNX Runtime execution providers
        """
        self.model_path = Path(model_path)
        self.batch_size = batch_size

        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        logger.info(f"Loading ONNX model from {model_path}")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        logger.info(f"Using providers: {self.session.get_providers()}")

        # Get input info
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Model input: {self.input_name}, shape: {input_shape}")

        # Load vocabulary and metadata from model
        self.vocab, self.tag_frequencies, self.metadata = self._load_metadata()
        self.num_tags = len(self.vocab)
        logger.info(f"Loaded vocabulary with {self.num_tags} tags")

    def _load_metadata(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
        """Load vocabulary and metadata from ONNX model."""
        meta = self.session.get_modelmeta().custom_metadata_map

        vocab_b64_gzip = meta.get('vocab_b64_gzip', '')
        vocab_sha256 = meta.get('vocab_sha256', '')

        if not vocab_b64_gzip or not vocab_sha256:
            raise RuntimeError(
                "Model is missing embedded vocabulary metadata. "
                "Please use a model exported with embedded vocabulary."
            )

        # Decode and decompress vocabulary
        vocab_b64_decoded = base64.b64decode(vocab_b64_gzip)
        vocab_bytes = gzip.decompress(vocab_b64_decoded)

        # Verify checksum
        sha = hashlib.sha256(vocab_bytes).hexdigest()
        if vocab_sha256 != sha:
            raise RuntimeError(f'Vocabulary checksum mismatch: expected {vocab_sha256}, got {sha}')

        # Parse JSON
        import json
        vocab_data = json.loads(vocab_bytes.decode('utf-8'))

        tag_to_index = vocab_data['tag_to_index']
        tag_frequencies = vocab_data.get('tag_frequencies', {})

        # Build index_to_tag for decoding
        self.index_to_tag = {int(idx): tag for tag, idx in tag_to_index.items()}

        return tag_to_index, tag_frequencies, meta

    def _preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess a single image.

        Returns (H, W, 3) uint8 array or None on error.
        The ONNX model handles resize/normalize internally.
        """
        try:
            with Image.open(image_path) as img:
                img.load()
                img = ImageOps.exif_transpose(img)

                # Handle transparency by compositing on grey background
                if img.mode in ('RGBA', 'LA') or 'transparency' in img.info:
                    background = Image.new('RGB', img.size, (114, 114, 114))
                    img_rgba = img.convert('RGBA')
                    alpha = img_rgba.getchannel('A')
                    rgb = img_rgba.convert('RGB')
                    background.paste(rgb, mask=alpha)
                    img = background
                else:
                    img = img.convert('RGB')

                return np.asarray(img, dtype=np.uint8)

        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            return None

    def _run_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Run inference on a batch of preprocessed images.

        Args:
            images: List of (H, W, 3) uint8 arrays (variable sizes OK)

        Returns:
            (batch_size, num_tags) array of sigmoid scores
        """
        # Stack into batch - each image may have different size
        # ONNX model handles dynamic input sizes via letterboxing
        batch = np.stack(images, axis=0)  # (B, H, W, 3)

        outputs = self.session.run(None, {self.input_name: batch})
        scores = outputs[0]  # (B, num_tags) logits

        # Apply sigmoid to get probabilities
        return 1.0 / (1.0 + np.exp(-scores))

    def infer_batch(
        self,
        image_paths: List[str],
    ) -> List[BatchResult]:
        """Run inference on a batch of images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of BatchResult objects
        """
        results = []
        valid_images = []
        valid_paths = []

        # Preprocess all images
        for path in image_paths:
            arr = self._preprocess_image(path)
            if arr is not None:
                valid_images.append(arr)
                valid_paths.append(path)
            else:
                results.append(BatchResult(
                    image_path=path,
                    scores=np.zeros(self.num_tags, dtype=np.float32),
                    error="Failed to load image"
                ))

        if not valid_images:
            return results

        # Check if all images have the same dimensions
        shapes = set(img.shape for img in valid_images)
        if len(shapes) > 1:
            # Different sizes - need to process one at a time
            # (ONNX dynamic axes don't support variable within batch)
            for path, img in zip(valid_paths, valid_images):
                try:
                    batch = np.expand_dims(img, axis=0)
                    outputs = self.session.run(None, {self.input_name: batch})
                    scores = 1.0 / (1.0 + np.exp(-outputs[0][0]))
                    results.append(BatchResult(image_path=path, scores=scores))
                except Exception as e:
                    results.append(BatchResult(
                        image_path=path,
                        scores=np.zeros(self.num_tags, dtype=np.float32),
                        error=str(e)
                    ))
        else:
            # Same size - batch inference
            try:
                scores = self._run_batch(valid_images)
                for i, path in enumerate(valid_paths):
                    results.append(BatchResult(image_path=path, scores=scores[i]))
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                for path in valid_paths:
                    results.append(BatchResult(
                        image_path=path,
                        scores=np.zeros(self.num_tags, dtype=np.float32),
                        error=str(e)
                    ))

        return results

    def infer_stream(
        self,
        image_paths: Iterator[str],
    ) -> Iterator[BatchResult]:
        """Stream inference over many images.

        Yields results as batches complete. More memory efficient
        than loading all images at once.

        Args:
            image_paths: Iterator of image file paths

        Yields:
            BatchResult objects
        """
        batch_paths = []

        for path in image_paths:
            batch_paths.append(path)

            if len(batch_paths) >= self.batch_size:
                for result in self.infer_batch(batch_paths):
                    yield result
                batch_paths = []

        # Process remaining
        if batch_paths:
            for result in self.infer_batch(batch_paths):
                yield result

    def get_tag_name(self, index: int) -> str:
        """Get tag name from index."""
        return self.index_to_tag.get(index, f"<unknown_{index}>")

    def get_tag_index(self, tag: str) -> Optional[int]:
        """Get tag index from name."""
        return self.vocab.get(tag)

    def decode_predictions(
        self,
        scores: np.ndarray,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Decode scores to list of (tag_name, confidence) above threshold."""
        mask = scores >= threshold
        indices = np.where(mask)[0]
        return [
            (self.get_tag_name(int(idx)), float(scores[idx]))
            for idx in indices
        ]
