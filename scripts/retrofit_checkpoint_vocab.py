#!/usr/bin/env python3
"""
Retrofit existing checkpoints with embedded vocabulary and preprocessing params
"""

import argparse
import sys
from pathlib import Path
import torch
import json
import logging
from typing import Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model_metadata import ModelMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrofit_checkpoint(
    checkpoint_path: Path,
    vocab_path: Path,
    output_path: Optional[Path] = None,
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_size: int = 640,
    patch_size: int = 16,
    force: bool = False
) -> bool:
    """Retrofit a checkpoint with embedded vocabulary and preprocessing params"""

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False

    if not vocab_path.exists():
        logger.error(f"Vocabulary not found: {vocab_path}")
        return False

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check if already has embedded vocabulary
    if 'vocab_b64_gzip' in checkpoint and not force:
        logger.warning("Checkpoint already has embedded vocabulary. Use --force to overwrite.")
        return False

    # Embed vocabulary
    checkpoint = ModelMetadata.embed_vocabulary(checkpoint, vocab_path)

    # Embed preprocessing params
    checkpoint = ModelMetadata.embed_preprocessing_params(
        checkpoint,
        normalize_mean,
        normalize_std,
        image_size,
        patch_size
    )

    # Save retrofitted checkpoint
    if output_path is None:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_retrofitted.pt"

    logger.info(f"Saving retrofitted checkpoint: {output_path}")
    torch.save(checkpoint, output_path)

    # Verify the retrofit
    logger.info("Verifying retrofitted checkpoint...")
    test_checkpoint = torch.load(output_path, map_location='cpu')

    vocab_data = ModelMetadata.extract_vocabulary(test_checkpoint)
    if vocab_data is None:
        logger.error("Failed to extract vocabulary from retrofitted checkpoint!")
        return False

    preprocessing = ModelMetadata.extract_preprocessing_params(test_checkpoint)
    if preprocessing is None:
        logger.error("Failed to extract preprocessing params from retrofitted checkpoint!")
        return False

    logger.info(f"✓ Successfully retrofitted checkpoint with:")
    logger.info(f"  - Vocabulary: {len(vocab_data['tag_to_index'])} tags")
    logger.info(f"  - Preprocessing: {preprocessing}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Retrofit checkpoints with embedded vocabulary')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('vocabulary', type=str, help='Path to vocabulary.json file')
    parser.add_argument('-o', '--output', type=str, help='Output path (default: adds _retrofitted suffix)')
    parser.add_argument('--normalize-mean', nargs=3, type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument('--normalize-std', nargs=3, type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--force', action='store_true', help='Overwrite existing embedded vocabulary')

    args = parser.parse_args()

    success = retrofit_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        vocab_path=Path(args.vocabulary),
        output_path=Path(args.output) if args.output else None,
        normalize_mean=tuple(args.normalize_mean),
        normalize_std=tuple(args.normalize_std),
        image_size=args.image_size,
        patch_size=args.patch_size,
        force=args.force
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

