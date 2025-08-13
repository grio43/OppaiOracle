#!/usr/bin/env python3
"""
Data Preparation Script for Direct Training Pipeline
Prepares Danbooru dataset for simplified direct training approach
"""

import json
import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DanbooruDataPreprocessor:
    """Preprocessor for Danbooru dataset - direct training approach"""
    
    def __init__(self, 
                 vocab_path: Path,
                 output_dir: Path,
                 num_workers: int = None):
        """
        Initialize preprocessor
        
        Args:
            vocab_path: Path to vocabulary file
            output_dir: Output directory for processed data
            num_workers: Number of parallel workers
        """
        self.vocab_path = Path(vocab_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_workers = num_workers or mp.cpu_count()
        
        # Load vocabulary (using pickle for speed)
        vocab_pkl = vocab_path / 'vocabulary.pkl'
        if vocab_pkl.exists():
            with open(vocab_pkl, 'rb') as f:
                vocab_data = pickle.load(f)
                self.tag_to_index = vocab_data['tag_to_index']
                self.tag_counts = vocab_data['tag_counts']
                self.frequency_buckets = vocab_data['frequency_buckets']
        else:
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
        
        self.vocab_size = len(self.tag_to_index)
        
        # Rating mapping
        self.rating_to_index = {
            'general': 0,
            'safe': 0,      # Map safe to general
            'sensitive': 1,
            'questionable': 2,
            'explicit': 3
        }
        
        logger.info(f"Loaded vocabulary with {self.vocab_size} tags")
    
    def process_metadata_batch(self, json_files: List[Path]) -> Dict:
        """Process a batch of JSON metadata files"""
        results = {
            'filenames': [],
            'tag_indices': [],
            'ratings': [],
            'tag_counts': [],  # Number of tags per image
            'quality_scores': []  # Based on tag count and presence of quality tags
        }
        
        quality_indicators = {
            'best_quality', 'high_quality', 'masterpiece', 
            'highres', 'absurdres', 'incredibly_absurdres'
        }
        
        negative_indicators = {
            'low_quality', 'worst_quality', 'bad_anatomy',
            'bad_proportions', 'error', 'missing_limbs'
        }
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract tags
                if isinstance(data, dict):
                    tags = data.get('tags', [])
                    rating = data.get('rating', 'general')
                    filename = data.get('filename', json_file.stem)
                else:
                    continue
                
                # Convert tags to indices
                tag_indices = []
                for tag in tags:
                    if tag in self.tag_to_index:
                        tag_indices.append(self.tag_to_index[tag])
                
                if not tag_indices:
                    continue
                
                # Calculate quality score
                quality_score = len(tag_indices) * 0.1  # Base score from tag count
                
                for tag in tags:
                    if tag in quality_indicators:
                        quality_score += 2.0
                    elif tag in negative_indicators:
                        quality_score -= 1.0
                
                quality_score = max(0.1, min(10.0, quality_score))  # Clamp between 0.1 and 10
                
                # Store results
                results['filenames'].append(filename)
                results['tag_indices'].append(tag_indices)
                results['ratings'].append(self.rating_to_index.get(rating, 0))
                results['tag_counts'].append(len(tag_indices))
                results['quality_scores'].append(quality_score)
                
            except Exception as e:
                logger.debug(f"Error processing {json_file}: {e}")
                continue
        
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
        
        # Process in parallel batches
        batch_size = 1000
        all_results = {
            'filenames': [],
            'tag_vectors': [],  # Binary vectors for training
            'tag_indices': [],  # Variable-length indices for analysis
            'ratings': [],
            'quality_scores': []
        }
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for i in range(0, len(json_files), batch_size):
                batch = json_files[i:i+batch_size]
                futures.append(executor.submit(self.process_metadata_batch, batch))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_results = future.result()
                
                # Convert to binary vectors
                for indices in batch_results['tag_indices']:
                    binary_vector = np.zeros(self.vocab_size, dtype=np.float32)
                    binary_vector[indices] = 1.0
                    all_results['tag_vectors'].append(binary_vector)
                
                all_results['filenames'].extend(batch_results['filenames'])
                all_results['tag_indices'].extend(batch_results['tag_indices'])
                all_results['ratings'].extend(batch_results['ratings'])
                all_results['quality_scores'].extend(batch_results['quality_scores'])
        
        logger.info(f"Processed {len(all_results['filenames'])} valid images")
        
        # Apply stratified sampling if requested
        if stratified:
            indices = self._stratified_sample(all_results)
            all_results = self._subset_results(all_results, indices)
            logger.info(f"After stratified sampling: {len(indices)} images")
        
        # Save to HDF5
        self._save_to_hdf5(all_results)
        
        # Save metadata
        self._save_metadata(all_results)
        
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
        """Subset results by indices"""
        subset = {}
        for key, values in results.items():
            if isinstance(values, list):
                subset[key] = [values[i] for i in indices]
            else:
                subset[key] = values[indices]
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
        
        # Also save indices separately for flexibility
        indices_file = self.output_dir / 'tag_indices.pkl'
        with open(indices_file, 'wb') as f:
            pickle.dump(results['tag_indices'], f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved tag indices to {indices_file}")
    
    def _save_metadata(self, results: Dict):
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
            }
        }
        
        # Save metadata
        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
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
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test
        }
        
        with open(self.output_dir / 'splits.json', 'w') as f:
            json.dump(splits, f)
        
        logger.info(f"Created splits: train={n_train:,}, val={n_val:,}, test={n_test:,}")
        
        return splits


def prepare_phased_training_data(
    metadata_dir: Path,
    vocab_path: Path,
    output_dir: Path,
    phase1_size: int = 4_000_000,
    total_size: int = 8_500_000
):
    """
    Prepare data for phased training strategy
    
    Args:
        metadata_dir: Directory with JSON metadata
        vocab_path: Path to vocabulary
        output_dir: Output directory
        phase1_size: Size of Phase 1 dataset (high quality)
        total_size: Total dataset size
    """
    output_dir = Path(output_dir)
    
    # Phase 1: High-quality subset
    logger.info("=" * 50)
    logger.info("PHASE 1: Preparing high-quality subset")
    logger.info("=" * 50)
    
    phase1_dir = output_dir / 'phase1_common_tags'
    preprocessor = DanbooruDataPreprocessor(vocab_path, phase1_dir)
    
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
    preprocessor = DanbooruDataPreprocessor(vocab_path, phase2_dir)
    
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
    
    parser = argparse.ArgumentParser(
        description="Prepare Danbooru data for direct training"
    )
    parser.add_argument(
        '--metadata_dir',
        type=str,
        required=True,
        help='Directory with JSON metadata files'
    )
    parser.add_argument(
        '--vocab_path',
        type=str,
        required=True,
        help='Path to vocabulary directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--phase1_size',
        type=int,
        default=4_000_000,
        help='Size of Phase 1 dataset'
    )
    parser.add_argument(
        '--total_size',
        type=int,
        default=8_500_000,
        help='Total dataset size'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    try:
        prepare_phased_training_data(
            metadata_dir=Path(args.metadata_dir),
            vocab_path=Path(args.vocab_path),
            output_dir=Path(args.output_dir),
            phase1_size=args.phase1_size,
            total_size=args.total_size
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())