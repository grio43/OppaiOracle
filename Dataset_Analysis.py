#!/usr/bin/env python3
"""
Dataset Analysis Tools for Anime Image Tagger
Comprehensive analysis of image datasets and tag distributions
"""

import os
import json
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import h5py

# Optional imports
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for dataset analysis"""
    # Paths
    dataset_paths: List[str] = field(default_factory=list)
    output_dir: str = "./dataset_analysis"
    cache_dir: str = "./analysis_cache"
    
    # Analysis options
    analyze_images: bool = True
    analyze_tags: bool = True
    analyze_duplicates: bool = True
    analyze_quality: bool = True
    analyze_cooccurrence: bool = True
    
    # Image analysis
    compute_image_stats: bool = True
    sample_size_for_stats: Optional[int] = 10000
    check_corrupted: bool = True
    extract_color_stats: bool = True
    
    # Tag analysis
    min_tag_frequency: int = 10
    max_tags_to_analyze: int = 10000
    analyze_tag_hierarchy: bool = True
    
    # Duplicate detection
    use_perceptual_hash: bool = True
    perceptual_hash_size: int = 16
    duplicate_threshold: float = 0.95
    
    # Quality metrics
    check_resolution: bool = True
    min_resolution: Tuple[int, int] = (256, 256)
    max_resolution: Tuple[int, int] = (4096, 4096)
    check_aspect_ratio: bool = True
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    
    # Performance
    num_workers: int = 8
    batch_size: int = 100
    
    # Visualization
    create_visualizations: bool = True
    figure_dpi: int = 150
    max_tags_in_plots: int = 50
    
    # Report
    generate_report: bool = True
    report_format: str = "html"  # html, markdown, json


@dataclass
class ImageStats:
    """Statistics for a single image"""
    path: str
    width: int
    height: int
    channels: int
    format: str
    file_size_kb: float
    
    # Optional stats
    mean_color: Optional[Tuple[float, float, float]] = None
    std_color: Optional[Tuple[float, float, float]] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    sharpness: Optional[float] = None
    
    # Hash for deduplication
    file_hash: Optional[str] = None
    perceptual_hash: Optional[str] = None
    
    # Quality flags
    is_corrupted: bool = False
    is_low_quality: bool = False
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class DatasetStats:
    """Overall dataset statistics"""
    total_images: int = 0
    total_tags: int = 0
    unique_tags: int = 0
    
    # Image statistics
    avg_width: float = 0
    avg_height: float = 0
    avg_file_size_kb: float = 0
    
    # Tag statistics
    avg_tags_per_image: float = 0
    tag_frequency_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Format distribution
    format_distribution: Dict[str, int] = field(default_factory=dict)
    resolution_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    corrupted_images: int = 0
    duplicate_images: int = 0
    low_quality_images: int = 0
    
    # Timing
    analysis_duration_seconds: float = 0


class ImageAnalyzer:
    """Analyzes image properties and quality"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def analyze_image(self, image_path: Path) -> Optional[ImageStats]:
        """Analyze a single image"""
        try:
            # Basic stats
            stats = ImageStats(
                path=str(image_path),
                file_size_kb=image_path.stat().st_size / 1024
            )
            
            # Open image
            with Image.open(image_path) as img:
                stats.width = img.width
                stats.height = img.height
                stats.format = img.format or 'unknown'
                stats.channels = len(img.getbands())
                
                # Check for corruption
                if self.config.check_corrupted:
                    try:
                        img.load()
                        img.verify()
                    except:
                        stats.is_corrupted = True
                        stats.quality_issues.append("corrupted")
                        return stats
                
                # Color statistics
                if self.config.extract_color_stats and not stats.is_corrupted:
                    img_array = np.array(img.convert('RGB'))
                    stats.mean_color = tuple(img_array.mean(axis=(0, 1)))
                    stats.std_color = tuple(img_array.std(axis=(0, 1)))
                    
                    # Brightness (average of grayscale)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    stats.brightness = gray.mean()
                    
                    # Contrast (standard deviation of grayscale)
                    stats.contrast = gray.std()
                    
                    # Sharpness (variance of Laplacian)
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    stats.sharpness = laplacian.var()
            
            # File hash
            stats.file_hash = self._compute_file_hash(image_path)
            
            # Perceptual hash
            if self.config.use_perceptual_hash and IMAGEHASH_AVAILABLE:
                stats.perceptual_hash = self._compute_perceptual_hash(image_path)
            
            # Quality checks
            self._check_quality(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return None
    
    def _compute_file_hash(self, path: Path) -> str:
        """Compute file hash"""
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _compute_perceptual_hash(self, path: Path) -> str:
        """Compute perceptual hash"""
        if not IMAGEHASH_AVAILABLE:
            return ""
        
        try:
            img = Image.open(path)
            hash_val = imagehash.dhash(img, hash_size=self.config.perceptual_hash_size)
            return str(hash_val)
        except:
            return ""
    
    def _check_quality(self, stats: ImageStats):
        """Check image quality"""
        # Resolution checks
        if self.config.check_resolution:
            if stats.width < self.config.min_resolution[0] or stats.height < self.config.min_resolution[1]:
                stats.quality_issues.append("low_resolution")
                stats.is_low_quality = True
            elif stats.width > self.config.max_resolution[0] or stats.height > self.config.max_resolution[1]:
                stats.quality_issues.append("excessive_resolution")
        
        # Aspect ratio checks
        if self.config.check_aspect_ratio:
            aspect_ratio = stats.width / stats.height
            if aspect_ratio < self.config.min_aspect_ratio:
                stats.quality_issues.append("too_tall")
            elif aspect_ratio > self.config.max_aspect_ratio:
                stats.quality_issues.append("too_wide")
        
        # Brightness/contrast checks
        if stats.brightness is not None:
            if stats.brightness < 50:  # Too dark
                stats.quality_issues.append("too_dark")
                stats.is_low_quality = True
            elif stats.brightness > 200:  # Too bright
                stats.quality_issues.append("too_bright")
            
            if stats.contrast < 20:  # Low contrast
                stats.quality_issues.append("low_contrast")
                stats.is_low_quality = True
        
        # Sharpness check
        if stats.sharpness is not None and stats.sharpness < 100:
            stats.quality_issues.append("blurry")
            stats.is_low_quality = True


class TagAnalyzer:
    """Analyzes tag distributions and relationships"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.tag_counts = Counter()
        self.tag_cooccurrence = defaultdict(Counter)
        self.images_per_tag = defaultdict(set)
        self.tags_per_image = defaultdict(set)
        
    def add_image_tags(self, image_path: str, tags: List[str]):
        """Add tags for an image"""
        # Update counts
        self.tag_counts.update(tags)
        
        # Update mappings
        for tag in tags:
            self.images_per_tag[tag].add(image_path)
        self.tags_per_image[image_path] = set(tags)
        
        # Update co-occurrence
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i+1:]:
                self.tag_cooccurrence[tag1][tag2] += 1
                self.tag_cooccurrence[tag2][tag1] += 1
    
    def get_tag_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tag statistics"""
        stats = {
            'total_unique_tags': len(self.tag_counts),
            'total_tag_occurrences': sum(self.tag_counts.values()),
            'avg_tags_per_image': np.mean([len(tags) for tags in self.tags_per_image.values()]) if self.tags_per_image else 0,
            'tag_frequency_distribution': self._get_frequency_distribution(),
            'most_common_tags': self.tag_counts.most_common(100),
            'rare_tags': [(tag, count) for tag, count in self.tag_counts.items() if count < self.config.min_tag_frequency],
            'singleton_tags': sum(1 for count in self.tag_counts.values() if count == 1)
        }
        
        return stats
    
    def _get_frequency_distribution(self) -> Dict[str, int]:
        """Get tag frequency distribution"""
        distribution = defaultdict(int)
        for count in self.tag_counts.values():
            if count == 1:
                distribution['1'] += 1
            elif count < 10:
                distribution['2-9'] += 1
            elif count < 100:
                distribution['10-99'] += 1
            elif count < 1000:
                distribution['100-999'] += 1
            elif count < 10000:
                distribution['1000-9999'] += 1
            else:
                distribution['10000+'] += 1
        return dict(distribution)
    
    def analyze_tag_hierarchy(self) -> Dict[str, List[str]]:
        """Analyze tag hierarchies and relationships"""
        hierarchies = defaultdict(list)
        
        # Common patterns
        patterns = {
            'hair_color': r'.*_hair$',
            'eye_color': r'.*_eyes$',
            'clothing': r'.*(shirt|dress|pants|skirt|uniform|clothes|wear).*',
            'emotion': r'.*(smile|laugh|cry|angry|sad|happy).*',
            'pose': r'.*(sitting|standing|lying|running|walking).*'
        }
        
        import re
        for category, pattern in patterns.items():
            regex = re.compile(pattern)
            for tag in self.tag_counts:
                if regex.match(tag):
                    hierarchies[category].append(tag)
        
        return dict(hierarchies)
    
    def find_tag_clusters(self, min_similarity: float = 0.5) -> List[Set[str]]:
        """Find clusters of related tags based on co-occurrence"""
        if not self.tag_cooccurrence:
            return []
        
        # Build similarity matrix
        tags = list(self.tag_counts.keys())[:self.config.max_tags_to_analyze]
        n_tags = len(tags)
        similarity_matrix = np.zeros((n_tags, n_tags))
        
        for i, tag1 in enumerate(tags):
            for j, tag2 in enumerate(tags):
                if i != j:
                    cooc = self.tag_cooccurrence[tag1].get(tag2, 0)
                    total1 = self.tag_counts[tag1]
                    total2 = self.tag_counts[tag2]
                    # Jaccard similarity
                    similarity = cooc / (total1 + total2 - cooc) if (total1 + total2 - cooc) > 0 else 0
                    similarity_matrix[i, j] = similarity
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=1-min_similarity, min_samples=2, metric='precomputed')
        distance_matrix = 1 - similarity_matrix
        clusters = clustering.fit_predict(distance_matrix)
        
        # Group tags by cluster
        tag_clusters = defaultdict(set)
        for tag, cluster_id in zip(tags, clusters):
            if cluster_id >= 0:  # -1 means noise
                tag_clusters[cluster_id].add(tag)
        
        return list(tag_clusters.values())


class DuplicateDetector:
    """Detects duplicate and near-duplicate images"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.file_hashes = defaultdict(list)
        self.perceptual_hashes = defaultdict(list)
        
    def add_image(self, stats: ImageStats):
        """Add image for duplicate detection"""
        if stats.file_hash:
            self.file_hashes[stats.file_hash].append(stats.path)
        
        if stats.perceptual_hash:
            self.perceptual_hashes[stats.perceptual_hash].append(stats.path)
    
    def find_exact_duplicates(self) -> Dict[str, List[str]]:
        """Find exact file duplicates"""
        duplicates = {}
        for hash_val, paths in self.file_hashes.items():
            if len(paths) > 1:
                duplicates[hash_val] = paths
        return duplicates
    
    def find_near_duplicates(self) -> List[Tuple[str, str, float]]:
        """Find near-duplicate images"""
        if not IMAGEHASH_AVAILABLE:
            logger.warning("imagehash not available, skipping near-duplicate detection")
            return []
        
        near_duplicates = []
        hashes = list(self.perceptual_hashes.keys())
        
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                hash1 = imagehash.hex_to_hash(hashes[i])
                hash2 = imagehash.hex_to_hash(hashes[j])
                
                similarity = 1 - (hash1 - hash2) / len(hash1.hash) ** 2
                
                if similarity >= self.config.duplicate_threshold:
                    paths1 = self.perceptual_hashes[hashes[i]]
                    paths2 = self.perceptual_hashes[hashes[j]]
                    
                    for p1 in paths1:
                        for p2 in paths2:
                            near_duplicates.append((p1, p2, similarity))
        
        return near_duplicates


class DatasetAnalyzer:
    """Main dataset analyzer class"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.image_analyzer = ImageAnalyzer(config)
        self.tag_analyzer = TagAnalyzer(config)
        self.duplicate_detector = DuplicateDetector(config)
        
        # Results storage
        self.image_stats = []
        self.dataset_stats = DatasetStats()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup cache
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_dataset(self, dataset_paths: Optional[List[str]] = None) -> DatasetStats:
        """Analyze complete dataset"""
        start_time = datetime.now()
        
        if dataset_paths:
            self.config.dataset_paths = dataset_paths
        
        logger.info(f"Starting dataset analysis for {len(self.config.dataset_paths)} paths")
        
        # Discover images
        image_paths = self._discover_images()
        logger.info(f"Found {len(image_paths)} images")
        
        # Analyze images
        if self.config.analyze_images:
            self._analyze_images(image_paths)
        
        # Analyze tags
        if self.config.analyze_tags:
            self._analyze_tags(image_paths)
        
        # Detect duplicates
        if self.config.analyze_duplicates:
            self._detect_duplicates()
        
        # Compute overall statistics
        self._compute_dataset_stats()
        
        # Generate visualizations
        if self.config.create_visualizations:
            self._create_visualizations()
        
        # Generate report
        if self.config.generate_report:
            self._generate_report()
        
        self.dataset_stats.analysis_duration_seconds = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Analysis completed in {self.dataset_stats.analysis_duration_seconds:.2f} seconds")
        
        return self.dataset_stats
    
    def _discover_images(self) -> List[Path]:
        """Discover all images in dataset paths"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
        image_paths = []
        
        for dataset_path in self.config.dataset_paths:
            path = Path(dataset_path)
            if path.is_file():
                if path.suffix.lower() in image_extensions:
                    image_paths.append(path)
            elif path.is_dir():
                for ext in image_extensions:
                    image_paths.extend(path.rglob(f'*{ext}'))
        
        return sorted(set(image_paths))
    
    def _analyze_images(self, image_paths: List[Path]):
        """Analyze image properties"""
        logger.info("Analyzing image properties...")
        
        # Sample if needed
        if self.config.sample_size_for_stats and len(image_paths) > self.config.sample_size_for_stats:
            import random
            image_paths = random.sample(image_paths, self.config.sample_size_for_stats)
        
        # Parallel processing
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for i in range(0, len(image_paths), self.config.batch_size):
                batch = image_paths[i:i + self.config.batch_size]
                future = executor.submit(self._analyze_image_batch, batch)
                futures.append(future)
            
            # Collect results
            for future in tqdm(futures, desc="Analyzing images"):
                batch_stats = future.result()
                for stats in batch_stats:
                    if stats:
                        self.image_stats.append(stats)
                        self.duplicate_detector.add_image(stats)
    
    def _analyze_image_batch(self, image_paths: List[Path]) -> List[ImageStats]:
        """Analyze a batch of images"""
        results = []
        for path in image_paths:
            stats = self.image_analyzer.analyze_image(path)
            results.append(stats)
        return results
    
    def _analyze_tags(self, image_paths: List[Path]):
        """Analyze tag distributions"""
        logger.info("Analyzing tags...")
        
        # Look for tag files
        for image_path in tqdm(image_paths, desc="Loading tags"):
            tag_file = image_path.with_suffix('.txt')
            if tag_file.exists():
                try:
                    with open(tag_file, 'r') as f:
                        tags = [tag.st
