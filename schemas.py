#!/usr/bin/env python3
"""
Standardized schemas for prediction outputs across all tools.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TagPrediction:
    """Single tag prediction."""
    name: str
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 4)}


@dataclass
class ImagePrediction:
    """Prediction result for a single image."""
    image: str  # Path or identifier
    tags: List[TagPrediction]
    processing_time: Optional[float] = None  # in milliseconds
    error: Optional[str] = None  # Error message if processing failed
    rating: Optional[str] = None  # Predicted rating (safe, sensitive, questionable, explicit, unknown)
    rating_confidence: Optional[float] = None  # Confidence score for rating prediction

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "image": self.image,
            "tags": [t.to_dict() for t in self.tags],
        }
        if self.processing_time is not None:
            result["processing_time"] = self.processing_time
        if self.rating is not None:
            result["rating"] = self.rating
        if self.rating_confidence is not None:
            result["rating_confidence"] = round(self.rating_confidence, 4)
        if self.error is not None:
            result["error"] = self.error
            result["status"] = "failed"
        else:
            result["status"] = "success"
        return result


@dataclass
class RunMetadata:
    """Metadata for a prediction run."""
    top_k: int
    threshold: float
    vocab_sha256: str
    normalize_mean: List[float]
    normalize_std: List[float]
    image_size: int
    patch_size: int
    model_path: Optional[str] = None
    num_tags: Optional[int] = None
    vocab_embedded: bool = True  # Whether vocab came from model metadata
    # Channel order of the image tensor and of ``normalize_mean``/``normalize_std``.
    # Defaults to "RGB" when parsing legacy artifacts that pre-date this field.
    color_order: str = "RGB"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionOutput:
    """Complete output with metadata and results."""
    metadata: RunMetadata
    results: List[ImagePrediction]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "results": [r.to_dict() for r in self.results]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: Path) -> None:
        """Save to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def canonical_vocab_bytes(vocab_data: Dict) -> bytes:
    """Produce the canonical byte representation of vocabulary data for hashing.

    The same vocabulary content must hash identically regardless of how it was
    serialized to disk (compact vs pretty-printed, CRLF vs LF, ensure_ascii vs
    not) or whether dict keys are int or str in memory. This function defines
    the single canonical form: UTF-8 JSON, sorted keys, compact separators,
    string-coerced inner-dict keys.
    """
    normalized: Dict[str, Any] = {}
    for section_name, section in vocab_data.items():
        if isinstance(section, dict):
            normalized[section_name] = {str(k): v for k, v in section.items()}
        else:
            normalized[section_name] = section
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_vocab_sha256(vocab_path: Optional[Path] = None,
                        vocab_data: Optional[Dict] = None) -> str:
    """Compute SHA256 hash of vocabulary in its canonical form.

    Hashing is content-canonical: a path argument is loaded as JSON and then
    canonicalized via :func:`canonical_vocab_bytes`, so reformatting the file
    on disk (indent, line endings) cannot change the hash.

    Args:
        vocab_path: Path to vocabulary file
        vocab_data: Vocabulary data dict (if already loaded)

    Returns:
        SHA256 hash as hex string, or "unknown" if computation fails

    Raises:
        ValueError: If neither vocab_path nor vocab_data provided
        TypeError: If vocab_data is not JSON-serializable
    """
    if vocab_data is None and vocab_path is None:
        raise ValueError("Must provide either vocab_path or vocab_data")

    try:
        if vocab_data is None:
            if not vocab_path.exists():
                logger.warning(f"Vocabulary file does not exist: {vocab_path}")
                return "unknown"
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
            except PermissionError as e:
                logger.warning(f"Permission denied reading vocabulary for hash: {vocab_path}: {e}")
                return "unknown"
            except OSError as e:  # Covers IOError, file system errors
                logger.error(f"OS error reading vocabulary for hash: {vocab_path}: {e}", exc_info=True)
                return "unknown"
            except json.JSONDecodeError as e:
                logger.error(f"Vocabulary file is not valid JSON: {vocab_path}: {e}")
                return "unknown"

        try:
            return hashlib.sha256(canonical_vocab_bytes(vocab_data)).hexdigest()
        except TypeError as e:
            logger.error(f"Vocabulary data not JSON-serializable: {e}")
            raise TypeError(f"Cannot hash non-serializable vocabulary: {e}") from e

    except (TypeError, ValueError):
        # Re-raise expected exceptions
        raise
    except Exception as e:
        # Truly unexpected error - log and re-raise for debugging
        logger.critical(f"Critical error in compute_vocab_sha256: {type(e).__name__}: {e}", exc_info=True)
        raise
