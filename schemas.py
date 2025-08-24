#!/usr/bin/env python3
"""
Standard schemas for prediction outputs
Ensures consistent output format across all tools
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class TagPrediction:
    """Single tag prediction"""
    name: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 4)}


@dataclass
class ImageResult:
    """Result for a single image"""
    image: str  # Image path or identifier
    tags: List[TagPrediction]
    processing_time: Optional[float] = None  # milliseconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "tags": [t.to_dict() for t in self.tags],
            "processing_time": round(self.processing_time, 2) if self.processing_time else None
        }


@dataclass
class RunMetadata:
    """Metadata for a prediction run"""
    top_k: int
    threshold: float
    vocab_sha256: str
    normalize_mean: List[float]
    normalize_std: List[float]
    image_size: int
    patch_size: int
    model_path: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PredictionFormatter:
    """Formats predictions according to standard schema"""

    @staticmethod
    def format_prediction(
        image_path: str,
        predictions: Dict[str, float],
        processing_time_ms: Optional[float] = None,
        threshold: float = 0.5
    ) -> ImageResult:
        """Format a single prediction"""

        tags = []
        for tag_name, score in predictions.items():
            if score >= threshold:
                tags.append(TagPrediction(name=tag_name, score=score))

        # Sort by score descending
        tags.sort(key=lambda t: t.score, reverse=True)

        return ImageResult(
            image=image_path,
            tags=tags,
            processing_time=processing_time_ms
        )

    @staticmethod
    def format_batch_output(
        results: List[ImageResult],
        metadata: RunMetadata
    ) -> Dict[str, Any]:
        """Format batch prediction output"""

        return {
            "metadata": metadata.to_dict(),
            "results": [r.to_dict() for r in results]
        }

    @staticmethod
    def save_results(
        results: List[ImageResult],
        metadata: RunMetadata,
        output_path: str
    ):
        """Save results to JSON file"""

        output = PredictionFormatter.format_batch_output(results, metadata)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    @staticmethod
    def load_results(input_path: str) -> Tuple[List[ImageResult], RunMetadata]:
        """Load results from JSON file"""

        with open(input_path, 'r') as f:
            data = json.load(f)

        metadata = RunMetadata(**data['metadata'])
        results = []

        for r in data['results']:
            tags = [TagPrediction(**t) for t in r['tags']]
            results.append(ImageResult(
                image=r['image'],
                tags=tags,
                processing_time=r.get('processing_time')
            ))

        return results, metadata

