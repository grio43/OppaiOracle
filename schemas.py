#!/usr/bin/env python3
"""
Standardized schemas for prediction outputs across all tools.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "tags": [t.to_dict() for t in self.tags],
            "processing_time": self.processing_time
        }


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


def compute_vocab_sha256(vocab_path: Optional[Path] = None, 
                        vocab_data: Optional[Dict] = None) -> str:
    """Compute SHA256 hash of vocabulary.
    
    Args:
        vocab_path: Path to vocabulary file
        vocab_data: Vocabulary data dict (if already loaded)
    
    Returns:
        SHA256 hash as hex string
    """
    if vocab_data:
        # Hash the vocabulary data directly
        vocab_json = json.dumps(vocab_data, sort_keys=True)
        return hashlib.sha256(vocab_json.encode()).hexdigest()
    elif vocab_path and vocab_path.exists():
        # Hash the file contents
        with open(vocab_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    else:
        return "unknown"

