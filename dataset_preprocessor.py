#!/usr/bin/env python3
"""
Data Preparation Script for Direct Training Pipeline

Prepares Danbooru dataset for simplified direct training approach.
Handles single-string tag fields with duplicates.
"""

import json
import h5py
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import pickle
import sys
import os
import yaml
from Configuration_System import ConfigManager, ConfigType


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metadata_ingestion import parse_tags_field, dedupe_preserve_order

# Logging will be set up in main()
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
# DEFAULT_VOCAB_PATH will be determined in main() from the unified config


class DanbooruDataPreprocessor:
    """Preprocessor for Danbooru dataset - direct training approach"""
    
    def __init__(self,
                 vocab_path: Path,
                 output_dir: Optional[Path] = None,
                 num_workers: Optional[int] = None,
                 ignore_tags_file: Optional[Path] = None,
                 ignored_tags: Optional[List[str]] = None):
        """
        Initialize preprocessor
        
        Args:
            vocab_path: Path to vocabulary file
            output_dir: Output directory for processed data
            num_workers: Number of parallel workers
        """
        self.vocab_path = Path(vocab_path)

        # Defaults for output_dir and num_workers are now handled in main() and passed in.
        self.output_dir = Path(output_dir or "./prepared")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_workers = int(num_workers or cpu_count())

        # Tag ignoring: load from file and/or list, union of both is used
        self.ignore_tags: Set[str] = set()
        if ignore_tags_file:
            try:
                with open(ignore_tags_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        self.ignore_tags.add(line)
                if self.ignore_tags:
                    logger.info(f"Loaded {len(self.ignore_tags)} tags to ignore from {ignore_tags_file}")
            except Exception as e:
                logger.warning(f"Failed to load ignore tags from {ignore_tags_file}: {e}")
        if ignored_tags:
            self.ignore_tags.update(ignored_tags)
            if ignored_tags:
                logger.info(f"Ignoring {len(ignored_tags)} additional tags specified in code")
        
        # Load and validate vocabulary
        vocab_file = self._resolve_vocab_file(vocab_path)

        if not vocab_file.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found at {vocab_file}. "
                f"Run vocabulary building script first."
            )

        try:
            with open(vocab_file, 'r', encoding="utf-8") as f:
                vocab_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in vocabulary file {vocab_file}: {e}"
            ) from e

        # Validate structure
        self._validate_vocabulary_structure(vocab_data, vocab_file)

        # Load validated data
        self.tag_to_index = vocab_data['tag_to_index']
        self.index_to_tag = vocab_data['index_to_tag']
        self.tag_counts = vocab_data.get('tag_frequencies', {})

        # Validate consistency
        self._validate_vocabulary_consistency(vocab_file)

        self.vocab_size = len(self.tag_to_index)
        logger.info(f"Loaded vocabulary with {self.vocab_size} tags")

        # Rating mapping
        self.rating_to_index = {
            'general': 0,
            'safe': 0,      # Map safe to general
            'sensitive': 1,
            'questionable': 2,
            'explicit': 3
        }

        # Pre-compute quality indicators (once, not per batch)
        self.quality_indicators = {
            'best_quality', 'high_quality', 'masterpiece',
            'highres', 'absurdres', 'incredibly_absurdres'
        }

        self.negative_indicators = {
            'low_quality', 'worst_quality', 'bad_anatomy',
            'bad_proportions', 'error', 'missing_limbs'
        }

        # Statistics tracking
        self.batch_stats = {
            'total_duplicates_removed': 0,
            'files_with_duplicates': 0,
            'files_with_all_oov': 0,
            'total_oov_tags': 0
        }

    def _resolve_vocab_file(self, vocab_path: Path) -> Path:
        """Resolve vocabulary file path."""
        vocab_path = Path(vocab_path)

        # If path is a directory, look for vocabulary.json inside
        if vocab_path.is_dir():
            return vocab_path / 'vocabulary.json'

        # If path is a JSON file, use it
        if vocab_path.suffix == '.json' and vocab_path.exists():
            return vocab_path

        # Try appending .json if no extension
        if vocab_path.suffix == '':
            candidate = vocab_path.with_suffix('.json')
            if candidate.exists():
                return candidate

        # Try looking for vocabulary.json in the same directory
        candidate = vocab_path / 'vocabulary.json'
        if candidate.exists():
            return candidate

        # Return original path (will fail with clear error in caller)
        return vocab_path

    def _validate_vocabulary_structure(self, vocab_data: dict, vocab_file: Path) -> None:
        """Validate that vocabulary JSON has correct structure.

        Args:
            vocab_data: Loaded JSON data
            vocab_file: Path to vocabulary file (for error messages)

        Raises:
            ValueError: If structure is invalid
        """
        # Check required fields
        required_fields = ['tag_to_index', 'index_to_tag']
        missing_fields = [f for f in required_fields if f not in vocab_data]

        if missing_fields:
            raise ValueError(
                f"Vocabulary file {vocab_file} is missing required fields: {missing_fields}. "
                f"Expected fields: {required_fields}"
            )

        # Validate tag_to_index is a dict with string keys and int values
        tag_to_index = vocab_data['tag_to_index']
        if not isinstance(tag_to_index, dict):
            raise ValueError(
                f"Vocabulary field 'tag_to_index' must be a dict, "
                f"got {type(tag_to_index).__name__}"
            )

        # Check a sample of entries for correct types
        sample_size = min(10, len(tag_to_index))
        for tag, idx in list(tag_to_index.items())[:sample_size]:
            if not isinstance(tag, str):
                raise ValueError(
                    f"Vocabulary 'tag_to_index' keys must be strings, "
                    f"found {type(tag).__name__}: {tag}"
                )
            if not isinstance(idx, int):
                raise ValueError(
                    f"Vocabulary 'tag_to_index' values must be integers, "
                    f"found {type(idx).__name__} for tag '{tag}'"
                )

        # Validate index_to_tag is a dict with string keys (will be converted to int) and string values
        index_to_tag = vocab_data['index_to_tag']
        if not isinstance(index_to_tag, dict):
            raise ValueError(
                f"Vocabulary field 'index_to_tag' must be a dict, "
                f"got {type(index_to_tag).__name__}"
            )

        # Convert string keys to integers (JSON converts int keys to strings)
        try:
            # Validate that keys can be converted to integers
            for key in list(index_to_tag.keys())[:sample_size]:
                int(key)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Vocabulary 'index_to_tag' keys must be integer-convertible strings, "
                f"found non-numeric key: {key}"
            ) from e

        # Check values are strings
        for key, tag in list(index_to_tag.items())[:sample_size]:
            if not isinstance(tag, str):
                raise ValueError(
                    f"Vocabulary 'index_to_tag' values must be strings, "
                    f"found {type(tag).__name__} for index {key}"
                )

        # Validate tag_frequencies if present
        if 'tag_frequencies' in vocab_data:
            tag_frequencies = vocab_data['tag_frequencies']
            if not isinstance(tag_frequencies, dict):
                raise ValueError(
                    f"Vocabulary field 'tag_frequencies' must be a dict, "
                    f"got {type(tag_frequencies).__name__}"
                )

    def _validate_vocabulary_consistency(self, vocab_file: Path) -> None:
        """Validate that vocabulary mappings are consistent.

        Args:
            vocab_file: Path to vocabulary file (for error messages)

        Raises:
            ValueError: If mappings are inconsistent
        """
        # Convert index_to_tag keys to integers
        self.index_to_tag = {int(k): v for k, v in self.index_to_tag.items()}

        # Check that mappings are inverses (sample check for performance)
        sample_size = min(100, len(self.tag_to_index))
        sample_tags = list(self.tag_to_index.keys())[:sample_size]

        inconsistencies = []
        for tag in sample_tags:
            idx = self.tag_to_index[tag]

            # Check reverse mapping
            if idx not in self.index_to_tag:
                inconsistencies.append(
                    f"Tag '{tag}' maps to index {idx}, "
                    f"but index {idx} not in index_to_tag"
                )
            elif self.index_to_tag[idx] != tag:
                inconsistencies.append(
                    f"Tag '{tag}' maps to index {idx}, "
                    f"but index {idx} maps back to '{self.index_to_tag[idx]}'"
                )

        if inconsistencies:
            raise ValueError(
                f"Vocabulary file {vocab_file} has inconsistent mappings:\n" +
                "\n".join(f"  - {inc}" for inc in inconsistencies[:10])
            )

    def process_metadata_batch(self, json_files: List[Path]) -> Dict:
        """Process a batch of JSON metadata files - FIXED VERSION"""
        # Reset batch statistics for this batch
        batch_stats = {
            'total_duplicates_removed': 0,
            'files_with_duplicates': 0,
            'files_with_all_oov': 0,
            'total_oov_tags': 0
        }

        results = {
            'filenames': [],
            'tag_indices': [],
            'ratings': [],
            'tag_counts': [],  # Number of tags per image
            'quality_scores': []  # Based on tag count and presence of quality tags
        }

        # Use instance variables instead of recreating sets
        tag_to_index = self.tag_to_index
        quality_indicators = self.quality_indicators
        negative_indicators = self.negative_indicators
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                entries = [data] if isinstance(data, dict) else data
                
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    
                    # Extract fields from your format
                    filename = entry.get('filename')
                    rating = entry.get('rating', 'general')
                    tags_field = entry.get('tags')  # Single space-delimited string
                    
                    if not filename:
                        logger.debug(f"Entry in {json_file} missing filename")
                        continue

                    tags_list = parse_tags_field(tags_field)

                    if not tags_list:
                        logger.debug(f"No tags found for {filename} in {json_file}")
                        continue

                    original_count = len(tags_list)
                    tags_list = dedupe_preserve_order(tags_list)

                    # Track duplicate statistics (before ignore_tags filtering)
                    dedupe_count = original_count - len(tags_list)
                    if dedupe_count > 0:
                        batch_stats['total_duplicates_removed'] += dedupe_count
                        batch_stats['files_with_duplicates'] += 1
                        logger.debug(f"Removed {dedupe_count} duplicate tags from {filename}")

                    # Remove ignored tags before further processing
                    if self.ignore_tags:
                        tags_list = [t for t in tags_list if t not in self.ignore_tags]
                        if not tags_list:
                            logger.debug(f"Skipping {filename}: all tags are in ignore list")
                            continue
                    # Convert tags to indices (optimized with .get())
                    tag_indices = []
                    oov_count = 0

                    for tag in tags_list:
                        idx = tag_to_index.get(tag)
                        if idx is not None:
                            tag_indices.append(idx)
                        else:
                            oov_count += 1
                            batch_stats['total_oov_tags'] += 1
                    
                    # Log OOV tags for debugging
                    if oov_count > 0:
                        oov_tags = [t for t in tags_list if t not in self.tag_to_index]
                        logger.debug(f"{filename}: {oov_count} OOV tags: {oov_tags[:5]}...")

                    if not tag_indices:
                        batch_stats['files_with_all_oov'] += 1
                        logger.warning(f"Skipping {filename}: all {len(tags_list)} tags are OOV")
                        continue
                    
                    # Calculate quality score (optimized with set intersection)
                    quality_score = len(tag_indices) * 0.1  # Base score from tag count

                    tags_set = set(tags_list)
                    quality_bonus = len(tags_set & quality_indicators) * 2.0
                    quality_penalty = len(tags_set & negative_indicators) * 1.0

                    quality_score += quality_bonus - quality_penalty
                    quality_score = max(0.1, min(10.0, quality_score))  # Clamp between 0.1 and 10
                    
                    # Store results
                    results['filenames'].append(filename)
                    results['tag_indices'].append(tag_indices)
                    results['ratings'].append(self.rating_to_index.get(rating, 0))
                    results['tag_counts'].append(len(tag_indices))
                    results['quality_scores'].append(quality_score)

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {json_file}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue

        # Attach batch stats after processing all files
        results['batch_stats'] = batch_stats

        return results
    
    def prepare_training_data(self, 
                             metadata_dir: Path,
                             max_images: Optional[int] = None,
                             stratified: bool = True):
        """
        Prepare complete training dataset
        
        Args:
            metadata_dir: Directory with JSON metadata
            max_images: Optional limit on number of images
            stratified: Whether to use stratified sampling
        """
        metadata_dir = Path(metadata_dir)
        json_files = sorted(metadata_dir.glob('*.json'))
        
        if max_images:
            json_files = json_files[:max_images]
        
        logger.info(f"Processing {len(json_files)} metadata files")
        
        # Reset statistics
        aggregated_stats = {
            'total_duplicates_removed': 0,
            'files_with_duplicates': 0,
            'files_with_all_oov': 0,
            'total_oov_tags': 0
        }
        
        # Process in parallel batches
        batch_size = 1000
        all_results = {
            'filenames': [],
            'tag_vectors': [],  # Binary vectors for training
            'tag_indices': [],  # Variable-length indices for analysis
            'ratings': [],
            'quality_scores': []
        }
        
        # NOTE: Using instance method with ProcessPoolExecutor may cause pickling issues on Windows.
        # If multiprocessing fails, consider refactoring process_metadata_batch to a module-level function.
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []

            for i in range(0, len(json_files), batch_size):
                batch = json_files[i:i+batch_size]
                futures.append(executor.submit(self.process_metadata_batch, batch))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_results = future.result()

                # Aggregate statistics from batch
                batch_stats = batch_results.pop('batch_stats', {})
                for key in aggregated_stats:
                    if key in batch_stats:
                        aggregated_stats[key] += batch_stats[key]

                # Convert to binary vectors with bounds checking
                for indices in batch_results['tag_indices']:
                    binary_vector = np.zeros(self.vocab_size, dtype=np.float32)
                    if indices:
                        # Validate indices are within vocabulary bounds
                        max_idx = max(indices)
                        min_idx = min(indices)
                        if max_idx >= self.vocab_size or min_idx < 0:
                            logger.error(
                                f"Tag index out of bounds: min={min_idx}, max={max_idx}, "
                                f"vocab_size={self.vocab_size}. Filtering invalid indices."
                            )
                            indices = [i for i in indices if 0 <= i < self.vocab_size]
                        if indices:  # Recheck after filtering
                            binary_vector[indices] = 1.0
                    all_results['tag_vectors'].append(binary_vector)
                
                all_results['filenames'].extend(batch_results['filenames'])
                all_results['tag_indices'].extend(batch_results['tag_indices'])
                all_results['ratings'].extend(batch_results['ratings'])
                all_results['quality_scores'].extend(batch_results['quality_scores'])
        
        # Log statistics
        logger.info(f"Processed {len(all_results['filenames'])} valid images")
        logger.info(f"Removed {aggregated_stats['total_duplicates_removed']} duplicate tags from {aggregated_stats['files_with_duplicates']} files")
        logger.info(f"Skipped {aggregated_stats['files_with_all_oov']} files with only OOV tags")
        logger.info(f"Total OOV tags encountered: {aggregated_stats['total_oov_tags']}")

        
        # Apply stratified sampling if requested
        if stratified:
            indices = self._stratified_sample(all_results)
            all_results = self._subset_results(all_results, indices)
            logger.info(f"After stratified sampling: {len(indices)} images")
        
        # Save to HDF5
        self._save_to_hdf5(all_results)
        
        # Save metadata
        self._save_metadata(all_results, aggregated_stats)
        
        return all_results
    
    def _stratified_sample(self, results: Dict, target_ratio: float = 0.8) -> List[int]:
        """
        Perform stratified sampling to balance dataset
        
        Args:
            results: Processing results
            target_ratio: Ratio of high-quality samples to keep
            
        Returns:
            Indices to keep
        """
        quality_scores = np.array(results['quality_scores'])
        
        # Split into quality tiers
        q75 = np.percentile(quality_scores, 75)
        q50 = np.percentile(quality_scores, 50)
        q25 = np.percentile(quality_scores, 25)
        
        high_quality = np.where(quality_scores >= q75)[0]
        medium_quality = np.where((quality_scores >= q50) & (quality_scores < q75))[0]
        low_quality = np.where((quality_scores >= q25) & (quality_scores < q50))[0]
        very_low_quality = np.where(quality_scores < q25)[0]
        
        # Sample proportionally
        n_total = len(quality_scores)
        indices = []
        
        # Keep most high-quality
        indices.extend(high_quality)
        
        # Sample from other tiers
        indices.extend(np.random.choice(medium_quality, 
                                      size=min(len(medium_quality), int(n_total * 0.3)),
                                      replace=False))
        indices.extend(np.random.choice(low_quality,
                                      size=min(len(low_quality), int(n_total * 0.1)),
                                      replace=False))
        indices.extend(np.random.choice(very_low_quality,
                                      size=min(len(very_low_quality), int(n_total * 0.05)),
                                      replace=False))
        
        return sorted(indices)
    
    def _subset_results(self, results: Dict, indices: List[int]) -> Dict:
        """Subset results by indices.

        Handles both lists and numpy arrays correctly. Lists are subset by
        iterating through indices, numpy arrays use numpy's advanced indexing.
        """
        subset = {}
        for key, values in results.items():
            if isinstance(values, list):
                subset[key] = [values[i] for i in indices]
            elif isinstance(values, np.ndarray):
                # Use numpy array indexing for numpy arrays
                subset[key] = values[np.array(indices)]
            else:
                # Fallback: try direct indexing (may fail for some types)
                try:
                    subset[key] = values[indices]
                except (TypeError, IndexError) as e:
                    raise TypeError(
                        f"Cannot subset key '{key}' of type {type(values).__name__}. "
                        f"Expected list or numpy array. Error: {e}"
                    )
        return subset
    
    def _save_to_hdf5(self, results: Dict):
        """Save processed data to HDF5 format"""
        output_file = self.output_dir / 'training_data.h5'
        
        with h5py.File(output_file, 'w') as f:
            # Save tag vectors as main dataset
            tag_vectors = np.array(results['tag_vectors'], dtype=np.float32)
            f.create_dataset('tag_vectors', data=tag_vectors, compression='gzip')
            
            # Save ratings
            ratings = np.array(results['ratings'], dtype=np.int8)
            f.create_dataset('ratings', data=ratings, compression='gzip')
            
            # Save quality scores
            quality_scores = np.array(results['quality_scores'], dtype=np.float32)
            f.create_dataset('quality_scores', data=quality_scores, compression='gzip')
            
            # Save filenames as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            filenames = f.create_dataset('filenames', (len(results['filenames']),), dtype=dt)
            for i, fname in enumerate(results['filenames']):
                filenames[i] = fname
            
            # Save metadata
            f.attrs['vocab_size'] = self.vocab_size
            f.attrs['num_samples'] = len(results['filenames'])
            f.attrs['num_ratings'] = len(self.rating_to_index)
        
        logger.info(f"Saved HDF5 data to {output_file}")
        
        # Also save indices separately for flexibility (use JSON instead of pickle for security)
        indices_file = self.output_dir / 'tag_indices.json'
        with open(indices_file, 'w', encoding='utf-8') as f:
            # Convert any non-JSON-serializable types to lists
            serializable_indices = [
                list(indices) if isinstance(indices, (set, tuple)) else indices
                for indices in results['tag_indices']
            ]
            json.dump(serializable_indices, f, indent=2)

        logger.info(f"Saved tag indices to {indices_file}")
    
    def _save_metadata(self, results: Dict, aggregated_stats: Dict):
        """Save dataset metadata and statistics"""
        # Calculate tag frequency in dataset
        tag_frequency = Counter()
        for indices in results['tag_indices']:
            tag_frequency.update(indices)
        
        # Calculate co-occurrence matrix for top tags
        top_tags = [idx for idx, _ in tag_frequency.most_common(1000)]
        cooccurrence = np.zeros((len(top_tags), len(top_tags)), dtype=np.int32)
        
        for indices in results['tag_indices']:
            indices_set = set(indices)
            for i, tag1 in enumerate(top_tags):
                if tag1 in indices_set:
                    for j, tag2 in enumerate(top_tags):
                        if i != j and tag2 in indices_set:
                            cooccurrence[i, j] += 1
        
        metadata = {
            'num_samples': len(results['filenames']),
            'vocab_size': self.vocab_size,
            'avg_tags_per_image': np.mean([len(idx) for idx in results['tag_indices']]),
            'median_tags_per_image': np.median([len(idx) for idx in results['tag_indices']]),
            'max_tags_per_image': max(len(idx) for idx in results['tag_indices']),
            'min_tags_per_image': min(len(idx) for idx in results['tag_indices']),
            'rating_distribution': dict(Counter(results['ratings'])),
            'quality_score_stats': {
                'mean': float(np.mean(results['quality_scores'])),
                'std': float(np.std(results['quality_scores'])),
                'min': float(np.min(results['quality_scores'])),
                'max': float(np.max(results['quality_scores'])),
                'median': float(np.median(results['quality_scores']))
            },
            'processing_stats': aggregated_stats  # Include duplicate/OOV statistics
        }
        
        # Save metadata
        with open(self.output_dir / 'dataset_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Save co-occurrence matrix
        np.save(self.output_dir / 'cooccurrence_top1000.npy', cooccurrence)
        
        logger.info(f"Saved metadata to {self.output_dir}")
        logger.info(f"Dataset statistics:")
        logger.info(f"  - Samples: {metadata['num_samples']:,}")
        logger.info(f"  - Avg tags per image: {metadata['avg_tags_per_image']:.1f}")
        logger.info(f"  - Quality score: {metadata['quality_score_stats']['mean']:.2f} ± {metadata['quality_score_stats']['std']:.2f}")
    
    def create_training_splits(self, 
                              val_ratio: float = 0.05,
                              test_ratio: float = 0.05,
                              seed: int = 42):
        """
        Create train/val/test splits
        
        Args:
            val_ratio: Validation set ratio
            test_ratio: Test set ratio  
            seed: Random seed
        """
        np.random.seed(seed)
        
        # Load the HDF5 file
        h5_file = self.output_dir / 'training_data.h5'
        with h5py.File(h5_file, 'r') as f:
            n_samples = f.attrs['num_samples']
        
        # Create indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Calculate split points
        n_test = int(n_samples * test_ratio)
        n_val = int(n_samples * val_ratio)
        n_train = n_samples - n_test - n_val
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Save splits
        splits = {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(),
            'test': test_indices.tolist(),
            'n_train': int(n_train), # need to convert to int for JSON serialization
            'n_val': int(n_val),
            'n_test': int(n_test)
        }
        
        with open(self.output_dir / 'splits.json', 'w', encoding='utf-8') as f:
            json.dump(splits, f)
        
        logger.info(f"Created splits: train={n_train:,}, val={n_val:,}, test={n_test:,}")
        
        return splits


def prepare_phased_training_data(
    metadata_dir: Path,
    vocab_path: Path,
    output_dir: Path,
    num_workers: int,
    ignore_tags_file: Optional[Path],
    phase1_size: int = 4_000_000,
    total_size: int = 8_500_000
):
    """
    Prepare data for phased training strategy
    
    Args:
        metadata_dir: Directory with JSON metadata
        vocab_path: Path to vocabulary
        output_dir: Output directory
        num_workers: Number of parallel workers
        ignore_tags_file: Path to file with tags to ignore
        phase1_size: Size of Phase 1 dataset (high quality)
        total_size: Total dataset size
    """
    output_dir = Path(output_dir)
    
    # Phase 1: High-quality subset
    logger.info("=" * 50)
    logger.info("PHASE 1: Preparing high-quality subset")
    logger.info("=" * 50)
    
    phase1_dir = output_dir / 'phase1_common_tags'
    preprocessor = DanbooruDataPreprocessor(vocab_path, phase1_dir, num_workers=num_workers, ignore_tags_file=ignore_tags_file)
    
    # Process with quality filtering
    results = preprocessor.prepare_training_data(
        metadata_dir,
        max_images=phase1_size,
        stratified=True
    )
    
    preprocessor.create_training_splits()
    
    # Phase 2: Full dataset
    logger.info("=" * 50)
    logger.info("PHASE 2: Preparing full dataset")
    logger.info("=" * 50)
    
    phase2_dir = output_dir / 'phase2_full_vocab'
    preprocessor = DanbooruDataPreprocessor(vocab_path, phase2_dir, num_workers=num_workers, ignore_tags_file=ignore_tags_file)
    
    results = preprocessor.prepare_training_data(
        metadata_dir,
        max_images=total_size,
        stratified=False  # Use all data
    )
    
    preprocessor.create_training_splits()
    
    logger.info("=" * 50)
    logger.info("Data preparation complete!")
    logger.info(f"Phase 1 data: {phase1_dir}")
    logger.info(f"Phase 2 data: {phase2_dir}")
    logger.info("=" * 50)


def main():
    """Main execution"""
    import argparse
    from utils.logging_setup import setup_logging
    listener = setup_logging()

    try:
        # Load unified config to get defaults
        unified_config = None
        try:
            manager = ConfigManager(config_type=ConfigType.FULL)
            unified_config = manager.load_from_file("configs/unified_config.yaml")
            data_cfg = unified_config.data
        except Exception as e:
            logger.warning(f"Could not load unified_config.yaml: {e}. Using default settings.")
            from dataclasses import dataclass
            @dataclass
            class Dummy:
                def __getattr__(self, name): return None
            data_cfg = Dummy()

        # Determine default paths from config (with fallbacks)
        DEFAULT_VOCAB_PATH = (unified_config.vocab_path if unified_config else None) or data_cfg.vocab_dir or "vocabulary"
        DEFAULT_OUTPUT_DIR = data_cfg.output_dir or "./prepared"

        parser = argparse.ArgumentParser(description="Prepare Danbooru data for direct training")
        parser.add_argument('--metadata_dir', type=str, required=True, help='Directory with JSON metadata files')
        parser.add_argument('--vocab_path', type=str, default=str(DEFAULT_VOCAB_PATH), help=f'Path to vocabulary file (default: {DEFAULT_VOCAB_PATH})')
        parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR), help=f'Output directory for processed data (default: {DEFAULT_OUTPUT_DIR})')
        parser.add_argument('--phase1_size', type=int, default=4_000_000, help='Size of Phase 1 dataset')
        parser.add_argument('--total_size', type=int, default=8_500_000, help='Total dataset size')
        parser.add_argument('--num_workers', type=int, default=data_cfg.num_workers, help=f'Number of parallel workers (default: {data_cfg.num_workers})')
        parser.add_argument('--ignore_tags_file', type=str, default=None, help='Path to file with tags to ignore')

        args = parser.parse_args()

        # Determine ignore_tags_file path from args or config
        ignore_tags_file = args.ignore_tags_file or getattr(data_cfg, 'ignore_tags_file', None)
        if ignore_tags_file:
            ignore_tags_file = Path(ignore_tags_file)

        prepare_phased_training_data(
            metadata_dir=Path(args.metadata_dir),
            vocab_path=Path(args.vocab_path),
            output_dir=Path(args.output_dir),
            num_workers=args.num_workers,
            ignore_tags_file=ignore_tags_file,
            phase1_size=args.phase1_size,
            total_size=args.total_size
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if listener:
            listener.stop()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())