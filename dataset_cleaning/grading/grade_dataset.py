#!/usr/bin/env python3
"""Grade dataset images using ONNX model and store FP/FN analysis in SQLite.

This script:
1. Loads an ONNX model with embedded vocabulary
2. Iterates through dataset images and their JSON sidecar labels
3. Compares model predictions to ground truth
4. Stores FP/FN counts and priority scores in SQLite database

Usage:
    python -m dataset_cleaning.grading.grade_dataset \\
        --model model.onnx \\
        --dataset /path/to/images \\
        --db grading.db \\
        --threshold 0.5
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterator
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .onnx_batch_inference import ONNXBatchInference, BatchResult
from .priority_scoring import compute_priority, GENDER_TAGS, COUNT_TAGS

logger = logging.getLogger(__name__)


@dataclass
class ImageGrade:
    """Grading result for a single image."""
    image_path: str
    json_path: str
    fp_tags: List[Tuple[str, float]]  # (tag, confidence)
    fn_tags: List[Tuple[str, float]]  # (tag, confidence)
    priority_score: float
    has_gender_error: bool
    has_count_error: bool


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Enable WAL mode for better performance with large datasets
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    conn.execute("PRAGMA temp_store=MEMORY")

    # Load and execute schema
    schema_path = Path(__file__).parent / 'db_schema.sql'
    with open(schema_path, 'r') as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()

    return conn


def find_json_sidecar(image_path: Path) -> Optional[Path]:
    """Find JSON sidecar file for an image."""
    # Try common patterns
    for suffix in ['.json', '_tags.json', '_meta.json']:
        json_path = image_path.with_suffix(suffix)
        if json_path.exists():
            return json_path

    # Try without extension + .json
    json_path = image_path.parent / (image_path.stem + '.json')
    if json_path.exists():
        return json_path

    return None


def load_ground_truth(json_path: Path) -> Set[str]:
    """Load ground truth tags from JSON sidecar."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle various tag formats
        tags = data.get('tags', data.get('tag', []))

        if isinstance(tags, str):
            # Comma-separated string
            return set(t.strip() for t in tags.split(',') if t.strip())
        elif isinstance(tags, (list, tuple)):
            return set(str(t).strip() for t in tags if str(t).strip())
        else:
            return set()

    except Exception as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return set()


def discover_images(
    dataset_path: Path,
    extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.webp', '.gif'},
) -> Iterator[Tuple[Path, Path]]:
    """Discover image files with their JSON sidecars.

    Yields (image_path, json_path) tuples.
    """
    for ext in extensions:
        for image_path in dataset_path.rglob(f'*{ext}'):
            json_path = find_json_sidecar(image_path)
            if json_path is not None:
                yield image_path, json_path


def grade_image(
    result: BatchResult,
    json_path: Path,
    threshold: float,
    tag_frequencies: Dict[str, int],
    index_to_tag: Dict[int, str],
    tag_to_index: Dict[str, int],
) -> Optional[ImageGrade]:
    """Grade a single image by comparing predictions to ground truth.

    Returns None if the image had an error during inference.
    """
    if result.error:
        logger.warning(f"Skipping {result.image_path}: {result.error}")
        return None

    # Load ground truth
    gt_tags = load_ground_truth(json_path)
    if not gt_tags:
        logger.debug(f"No ground truth tags for {result.image_path}")
        return None

    # Filter GT tags to only those in vocabulary
    gt_tags = gt_tags & set(tag_to_index.keys())

    # Get predicted tags above threshold
    pred_indices = np.where(result.scores >= threshold)[0]
    pred_tags = set()
    pred_confidences = {}

    for idx in pred_indices:
        tag = index_to_tag.get(idx)
        if tag and not tag.startswith('<'):  # Skip special tokens
            pred_tags.add(tag)
            pred_confidences[tag] = float(result.scores[idx])

    # Compute FP and FN
    fp_tags = []  # Predicted but not in GT
    for tag in pred_tags - gt_tags:
        fp_tags.append((tag, pred_confidences[tag]))

    fn_tags = []  # In GT but not predicted
    for tag in gt_tags - pred_tags:
        # Get model confidence for this tag
        idx = tag_to_index.get(tag)
        if idx is not None and idx < len(result.scores):
            conf = float(result.scores[idx])
        else:
            conf = 0.0
        fn_tags.append((tag, conf))

    # Compute priority score
    priority, has_gender, has_count = compute_priority(
        fp_tags, fn_tags, tag_frequencies
    )

    return ImageGrade(
        image_path=result.image_path,
        json_path=str(json_path),
        fp_tags=fp_tags,
        fn_tags=fn_tags,
        priority_score=priority,
        has_gender_error=has_gender,
        has_count_error=has_count,
    )


def save_grade(conn: sqlite3.Connection, grade: ImageGrade, tag_frequencies: Dict[str, int]):
    """Save a grading result to the database."""
    cursor = conn.cursor()

    # Insert image grade
    cursor.execute('''
        INSERT OR REPLACE INTO image_grades
        (image_path, json_path, fp_count, fn_count, priority_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        grade.image_path,
        grade.json_path,
        len(grade.fp_tags),
        len(grade.fn_tags),
        grade.priority_score,
    ))

    image_id = cursor.lastrowid

    # Delete existing tag errors for this image (in case of re-grade)
    cursor.execute('DELETE FROM tag_errors WHERE image_id = ?', (image_id,))

    # Insert tag errors
    for tag, conf in grade.fp_tags:
        freq = tag_frequencies.get(tag, 0)
        cursor.execute('''
            INSERT INTO tag_errors
            (image_id, tag_name, error_type, confidence, tag_frequency, is_gender_tag, is_count_tag)
            VALUES (?, ?, 'FP', ?, ?, ?, ?)
        ''', (
            image_id, tag, conf, freq,
            tag in GENDER_TAGS, tag in COUNT_TAGS
        ))

    for tag, conf in grade.fn_tags:
        freq = tag_frequencies.get(tag, 0)
        cursor.execute('''
            INSERT INTO tag_errors
            (image_id, tag_name, error_type, confidence, tag_frequency, is_gender_tag, is_count_tag)
            VALUES (?, ?, 'FN', ?, ?, ?, ?)
        ''', (
            image_id, tag, conf, freq,
            tag in GENDER_TAGS, tag in COUNT_TAGS
        ))


def main():
    parser = argparse.ArgumentParser(
        description='Grade dataset images using ONNX model'
    )
    parser.add_argument(
        '--model', '-m', required=True,
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--dataset', '-d', required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--db', required=True,
        help='Path to SQLite database file'
    )
    parser.add_argument(
        '--threshold', '-t', type=float, default=0.5,
        help='Prediction threshold (default: 0.5)'
    )
    parser.add_argument(
        '--batch-size', '-b', type=int, default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of images to process (for testing)'
    )
    parser.add_argument(
        '--commit-every', type=int, default=1000,
        help='Commit to database every N images (default: 1000)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Initialize
    model_path = Path(args.model)
    dataset_path = Path(args.dataset)
    db_path = Path(args.db)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    # Load model
    logger.info("Loading ONNX model...")
    engine = ONNXBatchInference(
        str(model_path),
        batch_size=args.batch_size,
    )

    # Build mappings
    tag_to_index = engine.vocab
    index_to_tag = {idx: tag for tag, idx in tag_to_index.items()}

    # Initialize database
    logger.info(f"Initializing database: {db_path}")
    conn = init_database(db_path)

    # Record grading run
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO grading_runs (model_path, threshold)
        VALUES (?, ?)
    ''', (str(model_path), args.threshold))
    run_id = cursor.lastrowid
    conn.commit()

    # Discover images
    logger.info(f"Discovering images in {dataset_path}...")
    image_pairs = list(discover_images(dataset_path))
    total_images = len(image_pairs)

    if args.limit:
        image_pairs = image_pairs[:args.limit]
        logger.info(f"Limited to {args.limit} images")

    logger.info(f"Found {total_images} images with JSON sidecars")

    # Grade images
    start_time = time.time()
    processed = 0
    errors = 0
    with_errors = 0

    # Create path to json_path mapping
    path_to_json = {str(img): json for img, json in image_pairs}
    image_paths = [str(img) for img, _ in image_pairs]

    logger.info("Starting grading...")
    pbar = tqdm(total=len(image_paths), desc="Grading", unit="img")

    for result in engine.infer_stream(iter(image_paths)):
        json_path = Path(path_to_json[result.image_path])

        grade = grade_image(
            result,
            json_path,
            args.threshold,
            engine.tag_frequencies,
            index_to_tag,
            tag_to_index,
        )

        if grade is None:
            errors += 1
        else:
            save_grade(conn, grade, engine.tag_frequencies)
            if grade.fp_tags or grade.fn_tags:
                with_errors += 1

        processed += 1
        pbar.update(1)

        # Periodic commit
        if processed % args.commit_every == 0:
            conn.commit()
            pbar.set_postfix({
                'errors': errors,
                'with_fp_fn': with_errors,
            })

    pbar.close()

    # Final commit
    conn.commit()

    # Update run record
    elapsed = time.time() - start_time
    cursor.execute('''
        UPDATE grading_runs
        SET completed_at = CURRENT_TIMESTAMP,
            total_images = ?,
            images_with_errors = ?
        WHERE id = ?
    ''', (processed, with_errors, run_id))
    conn.commit()

    # Summary
    logger.info("=" * 60)
    logger.info("GRADING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total images: {processed}")
    logger.info(f"Load errors: {errors}")
    logger.info(f"Images with FP/FN: {with_errors} ({100*with_errors/max(1,processed):.1f}%)")
    logger.info(f"Time elapsed: {elapsed:.1f}s ({processed/elapsed:.1f} img/s)")
    logger.info(f"Database: {db_path}")
    logger.info("=" * 60)

    # Show distribution
    cursor.execute('''
        SELECT
            CASE
                WHEN total_false = 0 THEN '0 errors'
                WHEN total_false <= 2 THEN '1-2 errors'
                WHEN total_false <= 5 THEN '3-5 errors'
                WHEN total_false <= 10 THEN '6-10 errors'
                ELSE '10+ errors'
            END as bucket,
            COUNT(*) as count
        FROM image_grades
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '0 errors' THEN 1
                WHEN '1-2 errors' THEN 2
                WHEN '3-5 errors' THEN 3
                WHEN '6-10 errors' THEN 4
                ELSE 5
            END
    ''')

    logger.info("\nError Distribution:")
    for row in cursor.fetchall():
        logger.info(f"  {row['bucket']}: {row['count']}")

    conn.close()


if __name__ == '__main__':
    main()
