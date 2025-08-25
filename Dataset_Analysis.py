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
import random

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
# Set backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Cache decorator for compiled regex patterns and other heavy computations
from functools import lru_cache


# Optional heavy dependencies with feature flags
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from tqdm import tqdm as _tqdm
    TQDM_AVAILABLE = True
    def tqdm(iterable=None, **kwargs):
        return _tqdm(iterable, **kwargs)
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable=None, **kwargs):
        # Fallback: no-op progress wrapper
        return iterable

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Note: pandas support planned for future version
# Will be used for advanced statistical analysis and data export
PANDAS_AVAILABLE = False
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    warnings.warn("imagehash not available. Install with: pip install imagehash")

# Optional wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    warnings.warn("wordcloud not available. Install with: pip install wordcloud")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    num_workers: int = min(8, mp.cpu_count())
    batch_size: int = 100
    max_memory_mb: int = 2048  # Maximum memory for image stats
    enable_parallel: bool = True  # Enable parallel processing
    chunk_size: int = 1000  # Process images in chunks
    
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
    width: int = 0
    height: int = 0
    channels: int = 0
    format: str = "unknown"
    file_size_kb: float = 0
    
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
    
    # Additional stats
    tag_statistics: Dict[str, Any] = field(default_factory=dict)
    duplicate_groups: Dict[str, List[str]] = field(default_factory=dict)
    near_duplicates: List[Tuple[str, str, float]] = field(default_factory=list)


class ImageAnalyzer:
    """Analyzes image properties and quality"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def analyze_image(self, image_path: Path) -> Optional[ImageStats]:
        """Analyze a single image.

        Args:
            image_path: Path to image file

        Returns:
            ImageStats object or ``None`` if the image is missing,
            inaccessible or corrupted.

        Raises:
            PermissionError: If file can't be accessed
            RuntimeError: For unexpected errors
        """
        # Check file exists before processing
        if not image_path.exists():
            logger.warning("Image file not found: %s", image_path)
            return None

        if not image_path.is_file():
            logger.warning("Path is not a file: %s", image_path)
            return None
        
        try:
            # Get file stats (may raise PermissionError)
            file_size_kb = image_path.stat().st_size / 1024
        except PermissionError as e:
            logger.error(f"Permission denied accessing {image_path}")
            raise
        except OSError as e:
            logger.error(f"OS error accessing {image_path}: {e}")
            raise
        
        stats = ImageStats(
            path=str(image_path),
            file_size_kb=file_size_kb
        )
        
        # Try to open and analyze image
        try:
            with Image.open(image_path) as img:
                # First verify the image can be loaded
                img.verify()
            
            # Re-open for actual processing (verify closes the file)
            with Image.open(image_path) as img:
                stats.width = img.width
                stats.height = img.height
                stats.format = img.format or 'unknown'
                stats.channels = len(img.getbands())
                
                # Convert to RGB for analysis
                if self.config.extract_color_stats:
                    try:
                        img_rgb = img.convert('RGB')
                        img_array = np.array(img_rgb)
                        
                        # Color statistics
                        stats.mean_color = tuple(float(x) for x in img_array.mean(axis=(0, 1)))
                        stats.std_color = tuple(float(x) for x in img_array.std(axis=(0, 1)))
                        
                        # Brightness/contrast (grayscale)
                        if CV2_AVAILABLE:
                            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                            stats.brightness = float(gray.mean())
                            stats.contrast = float(gray.std())
                            # Sharpness (variance of Laplacian)
                            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                            stats.sharpness = float(laplacian.var())
                        else:
                            # Fallback without OpenCV
                            gray = (0.2989 * img_array[...,0] + 0.5870 * img_array[...,1] + 0.1140 * img_array[...,2]).astype(float)
                            stats.brightness = float(gray.mean())
                            stats.contrast = float(gray.std())
                            # sharpness left as None without cv2
                    except Exception as e:
                        # Color stats are optional, log but don't fail
                        logger.debug(f"Could not extract color stats for {image_path}: {e}")
                        
        except (IOError, OSError) as e:
            # Image is corrupted or unreadable - this is expected for some files
            logger.debug(f"Image corrupted or unreadable: {image_path}: {e}")
            stats.is_corrupted = True
            stats.quality_issues.append("corrupted")
            return stats  # Return partial stats for corrupted images
        except Exception as e:
            # Unexpected error - should not happen
            logger.error(f"Unexpected error analyzing {image_path}: {e}")
            raise RuntimeError(f"Unexpected error analyzing image {image_path}") from e
        
        # File hash
        try:
            stats.file_hash = self._compute_file_hash(image_path)
        except Exception as e:
            logger.debug(f"Could not compute file hash: {e}")
            # Hash is optional, continue
        
        # Perceptual hash
        if self.config.use_perceptual_hash and IMAGEHASH_AVAILABLE:
            try:
                stats.perceptual_hash = self._compute_perceptual_hash(image_path)
            except Exception as e:
                logger.debug(f"Could not compute perceptual hash: {e}")
                # Perceptual hash is optional, continue
        
        # Quality checks
        self._check_quality(stats)
        
        return stats
    
    def _compute_file_hash(self, path: Path) -> str:
        """Compute file hash"""
        try:
            hasher = hashlib.md5()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.debug(f"Could not compute file hash for {path}: {e}")
            return ""
    
    def _compute_perceptual_hash(self, path: Path) -> str:
        """Compute perceptual hash"""
        if not IMAGEHASH_AVAILABLE:
            return ""
        
        try:
            with Image.open(path) as img:
                hash_val = imagehash.dhash(img, hash_size=self.config.perceptual_hash_size)
            return str(hash_val)
        except Exception as e:
            logger.debug(f"Could not compute perceptual hash for {path}: {e}")
            return ""
    
    def _check_quality(self, stats: ImageStats):
        """Check image quality"""
        # Resolution checks
        if self.config.check_resolution and stats.width > 0 and stats.height > 0:
            if stats.width < self.config.min_resolution[0] or stats.height < self.config.min_resolution[1]:
                stats.quality_issues.append("low_resolution")
                stats.is_low_quality = True
            elif stats.width > self.config.max_resolution[0] or stats.height > self.config.max_resolution[1]:
                stats.quality_issues.append("excessive_resolution")
        
        # Aspect ratio checks
        if self.config.check_aspect_ratio:
            if stats.height > 0 and stats.width > 0:
                aspect_ratio = stats.width / stats.height
                if aspect_ratio < self.config.min_aspect_ratio:
                    stats.quality_issues.append("too_tall")
                elif aspect_ratio > self.config.max_aspect_ratio:
                    stats.quality_issues.append("too_wide")
            # Mark invalid dimensions when width or height is zero
            elif stats.height == 0 or stats.width == 0:
                stats.quality_issues.append("invalid_dimensions")
        
        # Brightness/contrast checks
        if stats.brightness is not None:
            if stats.brightness < 50:  # Too dark
                stats.quality_issues.append("too_dark")
                stats.is_low_quality = True
            elif stats.brightness > 200:  # Too bright
                stats.quality_issues.append("too_bright")
            
            if stats.contrast is not None and stats.contrast < 20:  # Low contrast
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

    @lru_cache(maxsize=128)
    def _get_compiled_patterns(self) -> Dict[str, 're.Pattern']:
        """Get compiled regex patterns for tag hierarchy analysis"""
        import re
        patterns = {
            'hair_color': r'.*_(hair|haired)$',
            'eye_color': r'.*_(eyes|eyed)$',
            'clothing': r'.*(shirt|dress|pants|skirt|uniform|clothes|wear|outfit|costume|suit|kimono|bikini|underwear|pajamas|hoodie|jacket|coat).*',
            'emotion': r'.*(smile|laugh|cry|angry|sad|happy|blush|embarrassed|surprised|scared|nervous|confused).*',
            'pose': r'.*(sitting|standing|lying|running|walking|kneeling|squatting|jumping|flying|sleeping).*',
            'character_count': r'^(solo|1girl|2girls|3girls|1boy|2boys|multiple_girls|multiple_boys)$',
            'view_angle': r'.*(from_above|from_below|from_side|from_behind|front_view|portrait|close-up|cowboy_shot|full_body).*',
            'background': r'.*(outdoors|indoors|sky|clouds|city|forest|beach|school|bedroom|kitchen|bathroom).*'
        }
        return {k: re.compile(v, re.IGNORECASE) for k, v in patterns.items()}
        
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
            'median_tags_per_image': np.median([len(tags) for tags in self.tags_per_image.values()]) if self.tags_per_image else 0,
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
        
        # Use compiled patterns (cached) for efficiency
        patterns = self._get_compiled_patterns()
        for category, regex in patterns.items():
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
        
        if n_tags < 2:
            return []
            
        similarity_matrix = np.zeros((n_tags, n_tags))
        
        for i, tag1 in enumerate(tags):
            for j, tag2 in enumerate(tags):
                if i != j:
                    cooc = self.tag_cooccurrence[tag1].get(tag2, 0)
                    total1 = self.tag_counts[tag1]
                    total2 = self.tag_counts[tag2]
                    # Jaccard similarity
                    if total1 > 0 and total2 > 0:
                        denominator = total1 + total2 - cooc
                        similarity = cooc / denominator if denominator > 0 else 0
                    else:
                        similarity = 0
                    similarity_matrix[i, j] = similarity
        
        # Cluster using DBSCAN (if scikit-learn available)
        try:
            from sklearn.cluster import DBSCAN  # deferred import
        except Exception:
            logger.warning("scikit-learn not available, skipping clustering")
            return []
        try:
            clustering = DBSCAN(eps=1-min_similarity, min_samples=2, metric='precomputed')
            distance_matrix = 1 - similarity_matrix
            clusters = clustering.fit_predict(distance_matrix)
            
            # Group tags by cluster
            tag_clusters = defaultdict(set)
            for tag, cluster_id in zip(tags, clusters):
                if cluster_id >= 0:  # -1 means noise
                    tag_clusters[cluster_id].add(tag)
            
            return list(tag_clusters.values())
        except Exception as e:
            logger.warning(f"Could not perform clustering: {e}")
            return []


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
        
        if len(hashes) < 2:
            return []
        
        try:
            import imagehash
            
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    try:
                        # Skip empty or invalid hashes
                        if not hashes[i] or not hashes[j]:
                            continue
                        hash1 = imagehash.hex_to_hash(hashes[i])
                        hash2 = imagehash.hex_to_hash(hashes[j])
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Invalid hash format: {e}")
                        continue
                    
                    # Calculate hamming distance
                    distance = hash1 - hash2
                    max_distance = len(hash1.hash) ** 2
                    similarity = 1 - (distance / max_distance) if max_distance > 0 else 1.0
                    
                    if similarity >= self.config.duplicate_threshold:
                        paths1 = self.perceptual_hashes[hashes[i]]
                        paths2 = self.perceptual_hashes[hashes[j]]
                        
                        for p1 in paths1:
                            for p2 in paths2:
                                if p1 != p2:  # Don't compare with itself
                                    near_duplicates.append((p1, p2, similarity))
        except Exception as e:
            logger.warning(f"Error in near-duplicate detection: {e}")
        
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
        self.image_stats_buffer = []  # Buffer for batch processing
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
        
        if not image_paths:
            logger.warning("No images found to analyze")
            self.dataset_stats.analysis_duration_seconds = (datetime.now() - start_time).total_seconds()
            # Optionally generate an empty report for consistency
            if self.config.generate_report:
                try:
                    self._generate_report()
                except Exception as e:
                    logger.error(f"Error generating empty report: {e}")
            return self.dataset_stats
        
        # Analyze images
        if self.config.analyze_images:
            self._analyze_images(image_paths)
        
        # Analyze tags
        if self.config.analyze_tags:
            tag_stats = self._analyze_tags(image_paths)
            self.dataset_stats.tag_statistics = tag_stats
        
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
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for dataset_path in self.config.dataset_paths:
            path = Path(dataset_path)
            if path.is_file():
                if path.suffix.lower() in image_extensions:
                    image_paths.append(path)
            elif path.is_dir():
                for ext in image_extensions:
                    image_paths.extend(path.rglob(f'*{ext}'))
                    image_paths.extend(path.rglob(f'*{ext.upper()}'))
        
        return sorted(set(image_paths))
    
    def _analyze_images(self, image_paths: List[Path]):
        """Analyze image properties"""
        logger.info("Analyzing image properties...")
        
        # Sample if needed
        if self.config.sample_size_for_stats and len(image_paths) > self.config.sample_size_for_stats:
            image_paths = random.sample(image_paths, self.config.sample_size_for_stats)
            logger.info(f"Sampling {len(image_paths)} images for analysis")
        
        # Process in chunks to manage memory
        chunk_size = self.config.chunk_size
        total_chunks = (len(image_paths) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(image_paths))
            chunk_paths = image_paths[start_idx:end_idx]
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_paths)} images)")
            
            if self.config.enable_parallel and self.config.num_workers > 1:
                # Parallel processing for better performance
                self._analyze_images_parallel(chunk_paths)
            else:
                # Sequential processing (more stable for debugging)
                for path in tqdm(chunk_paths, desc=f"Analyzing images (chunk {chunk_idx + 1})"):
                    try:
                        stats = self.image_analyzer.analyze_image(path)
                        if stats:
                            self.image_stats_buffer.append(stats)
                            self.duplicate_detector.add_image(stats)
                    except Exception as e:
                        logger.error(f"Error analyzing {path}: {e}")
            
            # Flush buffer periodically to manage memory
            self._flush_stats_buffer()

    def _analyze_images_parallel(self, image_paths: List[Path]):
        """Analyze images in parallel"""
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._analyze_single_image_safe, path): path
                for path in image_paths
            }
            # Process results as they complete
            for future in tqdm(future_to_path, desc="Analyzing images (parallel)"):
                try:
                    stats = future.result(timeout=30)
                    if stats:
                        self.image_stats_buffer.append(stats)
                        self.duplicate_detector.add_image(stats)
                except Exception as e:
                    path = future_to_path[future]
                    logger.error(f"Error analyzing {path}: {e}")
    
    def _analyze_single_image_safe(self, path: Path) -> Optional[ImageStats]:
        """Safely analyze a single image (for parallel processing)"""
        try:
            return self.image_analyzer.analyze_image(path)
        except Exception as e:
            logger.error(f"Error analyzing {path}: {e}")
            return None
    
    def _flush_stats_buffer(self):
        """Flush stats buffer to main list"""
        self.image_stats.extend(self.image_stats_buffer)
        self.image_stats_buffer.clear()
    
    def _analyze_tags(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Analyze tag distributions"""
        logger.info("Analyzing tags...")
        
        for image_path in tqdm(image_paths, desc="Loading tags"):
            # Try different tag file formats
            tag_files = [
                image_path.with_suffix('.txt'),
                image_path.with_suffix('.tags'),
                image_path.parent / f"{image_path.stem}_tags.txt"
            ]
            
            for tag_file in tag_files:
                if tag_file.exists():
                    tags = []
                    try:
                        # Use 'replace' to preserve character count and stream line-by-line
                        with open(tag_file, 'r', encoding='utf-8', errors='replace') as f:
                            for line in f:
                                tags.extend(tag.strip() for tag in line.split(',') if tag.strip())
                    except (IOError, OSError) as e:
                        logger.warning(f"Error reading tag file {tag_file}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error loading tags from {tag_file}: {e}")
                        continue

                    if tags:
                        try:
                            # Add tags to analyzer
                            self.tag_analyzer.add_image_tags(str(image_path), tags)
                            break
                        except Exception as e:
                            logger.error(f"Error parsing tags from {tag_file}: {e}")
                            continue
        
        # Get tag statistics
        tag_stats = self.tag_analyzer.get_tag_statistics()
        
        # Find tag hierarchies
        if self.config.analyze_tag_hierarchy:
            hierarchies = self.tag_analyzer.analyze_tag_hierarchy()
            tag_stats['hierarchies'] = hierarchies
        
        # Find tag clusters
        if self.config.analyze_cooccurrence:
            clusters = self.tag_analyzer.find_tag_clusters()
            tag_stats['clusters'] = [list(cluster) for cluster in clusters]
        
        logger.info(f"Analyzed {tag_stats['total_unique_tags']} unique tags")
        return tag_stats
    
    def _detect_duplicates(self):
        """Detect duplicate images"""
        logger.info("Detecting duplicates...")
        
        # Find exact duplicates
        exact_duplicates = self.duplicate_detector.find_exact_duplicates()
        self.dataset_stats.duplicate_groups = exact_duplicates
        self.dataset_stats.duplicate_images = sum(len(paths) - 1 for paths in exact_duplicates.values())
        
        # Find near duplicates
        near_duplicates = self.duplicate_detector.find_near_duplicates()
        self.dataset_stats.near_duplicates = near_duplicates
        
        logger.info(f"Found {self.dataset_stats.duplicate_images} exact duplicates")
        logger.info(f"Found {len(near_duplicates)} near-duplicate pairs")
    
    def _compute_dataset_stats(self):
        """Compute overall dataset statistics"""
        logger.info("Computing dataset statistics...")
        
        if self.image_stats:
            # Basic counts
            self.dataset_stats.total_images = len(self.image_stats)
            
            # Image statistics
            widths = [s.width for s in self.image_stats if s.width > 0]
            heights = [s.height for s in self.image_stats if s.height > 0]
            file_sizes = [s.file_size_kb for s in self.image_stats]
            
            if widths:
                self.dataset_stats.avg_width = np.mean(widths)
            if heights:
                self.dataset_stats.avg_height = np.mean(heights)
            if file_sizes:
                self.dataset_stats.avg_file_size_kb = np.mean(file_sizes)
            
            # Format distribution
            format_counter = Counter(s.format for s in self.image_stats)
            self.dataset_stats.format_distribution = dict(format_counter)
            
            # Resolution distribution
            resolution_buckets = defaultdict(int)
            for s in self.image_stats:
                if s.width > 0 and s.height > 0:
                    if s.width < 512 or s.height < 512:
                        bucket = "low (<512)"
                    elif s.width < 1024 or s.height < 1024:
                        bucket = "medium (512-1024)"
                    elif s.width < 2048 or s.height < 2048:
                        bucket = "high (1024-2048)"
                    else:
                        bucket = "very high (>2048)"
                    resolution_buckets[bucket] += 1
            self.dataset_stats.resolution_distribution = dict(resolution_buckets)
            
            # Quality metrics
            self.dataset_stats.corrupted_images = sum(1 for s in self.image_stats if s.is_corrupted)
            self.dataset_stats.low_quality_images = sum(1 for s in self.image_stats if s.is_low_quality)
        
        # Tag statistics
        if self.dataset_stats.tag_statistics:
            self.dataset_stats.total_tags = self.dataset_stats.tag_statistics.get('total_tag_occurrences', 0)
            self.dataset_stats.unique_tags = self.dataset_stats.tag_statistics.get('total_unique_tags', 0)
            self.dataset_stats.avg_tags_per_image = self.dataset_stats.tag_statistics.get('avg_tags_per_image', 0)
    
    def _create_visualizations(self):
        """Create visualization plots"""
        logger.info("Creating visualizations...")
        
        try:
            # Set style with better fallback handling
            try:
                # Try modern seaborn styles first
                plt.style.use('seaborn-v0_8')
            except Exception:
                try:
                    # Fall back to older seaborn style
                    plt.style.use('seaborn-darkgrid')
                except Exception:
                    # Use default if seaborn not available
                    plt.style.use('default')
            if SEABORN_AVAILABLE:
                sns.set_palette("husl")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            
            # 1. Resolution distribution
            ax1 = plt.subplot(2, 3, 1)
            if self.dataset_stats.resolution_distribution:
                labels = list(self.dataset_stats.resolution_distribution.keys())
                sizes = list(self.dataset_stats.resolution_distribution.values())
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
                ax1.set_title('Resolution Distribution')
            
            # 2. Format distribution
            ax2 = plt.subplot(2, 3, 2)
            if self.dataset_stats.format_distribution:
                formats = list(self.dataset_stats.format_distribution.keys())
                counts = list(self.dataset_stats.format_distribution.values())
                ax2.bar(formats, counts)
                ax2.set_title('Image Format Distribution')
                ax2.set_xlabel('Format')
                ax2.set_ylabel('Count')
                plt.xticks(rotation=45)
            
            # 3. Tag frequency distribution
            ax3 = plt.subplot(2, 3, 3)
            if self.dataset_stats.tag_statistics and 'most_common_tags' in self.dataset_stats.tag_statistics:
                top_tags = self.dataset_stats.tag_statistics['most_common_tags'][:self.config.max_tags_in_plots]
                if top_tags:
                    tags, counts = zip(*top_tags)
                    ax3.barh(range(len(tags)), counts)
                    ax3.set_yticks(range(len(tags)))
                    ax3.set_yticklabels(tags)
                    ax3.set_title(f'Top {len(tags)} Most Common Tags')
                    ax3.set_xlabel('Frequency')
                    ax3.invert_yaxis()
            
            # 4. Image dimensions scatter plot
            ax4 = plt.subplot(2, 3, 4)
            if self.image_stats:
                widths = [s.width for s in self.image_stats if s.width > 0 and s.height > 0][:1000]
                heights = [s.height for s in self.image_stats if s.width > 0 and s.height > 0][:1000]
                if widths and heights:
                    ax4.scatter(widths, heights, alpha=0.5, s=10)
                    ax4.set_title('Image Dimensions Distribution')
                    ax4.set_xlabel('Width (pixels)')
                    ax4.set_ylabel('Height (pixels)')
                    ax4.grid(True, alpha=0.3)
            
            # 5. File size distribution
            ax5 = plt.subplot(2, 3, 5)
            if self.image_stats:
                file_sizes = [s.file_size_kb for s in self.image_stats if s.file_size_kb > 0]
                if file_sizes:
                    ax5.hist(file_sizes, bins=50, edgecolor='black')
                    ax5.set_title('File Size Distribution')
                    ax5.set_xlabel('File Size (KB)')
                    ax5.set_ylabel('Count')
                    ax5.set_xlim(0, np.percentile(file_sizes, 95))  # Limit to 95th percentile
            
            # 6. Quality issues
            ax6 = plt.subplot(2, 3, 6)
            quality_issues = defaultdict(int)
            for s in self.image_stats:
                for issue in s.quality_issues:
                    quality_issues[issue] += 1
            
            if quality_issues:
                issues = list(quality_issues.keys())
                counts = list(quality_issues.values())
                ax6.bar(issues, counts)
                ax6.set_title('Quality Issues Distribution')
                ax6.set_xlabel('Issue Type')
                ax6.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save figure
            output_path = self.output_dir / 'dataset_analysis_visualization.png'
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization to {output_path}")
            
            # Create word cloud if available
            if WORDCLOUD_AVAILABLE and self.tag_analyzer.tag_counts:
                self._create_tag_wordcloud()
                
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _create_tag_wordcloud(self):
        """Create a word cloud of tags"""
        if not WORDCLOUD_AVAILABLE:
            return
        
        plt.figure(figsize=(20, 10))
        try:
            # Prepare tag frequencies
            tag_freq = dict(self.tag_analyzer.tag_counts.most_common(200))
            
            if tag_freq:
                # Create word cloud
                wordcloud = WordCloud(
                    width=1600,
                    height=800,
                    background_color='white',
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate_from_frequencies(tag_freq)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Tag Word Cloud', fontsize=20)
                
                # Save with proper error handling
                output_path = self.output_dir / 'tag_wordcloud.png'
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
                    logger.info(f"Saved tag word cloud to {output_path}")
                except (IOError, OSError) as e:
                    logger.error(f"Error saving word cloud to {output_path}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error saving word cloud: {e}")
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
        finally:
            # Ensure matplotlib resources are always cleaned up
            plt.close('all')
    
    def _generate_report(self):
        """Generate analysis report"""
        logger.info("Generating report...")
        
        try:
            if self.config.report_format == 'json':
                self._generate_json_report()
            elif self.config.report_format == 'markdown':
                self._generate_markdown_report()
            else:  # html
                self._generate_html_report()
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def _generate_json_report(self):
        """Generate JSON report"""
        report_data = {
            'analysis_date': datetime.now().isoformat(),
            'config': asdict(self.config),
            'dataset_stats': asdict(self.dataset_stats),
            'image_quality_summary': self._get_quality_summary(),
            'tag_analysis': self.dataset_stats.tag_statistics
        }
        
        output_path = self.output_dir / 'analysis_report.json'
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Saved JSON report to {output_path}")
    
    def _generate_markdown_report(self):
        """Generate Markdown report"""
        report = []
        report.append("# Dataset Analysis Report\n")
        report.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Analysis Duration:** {self.dataset_stats.analysis_duration_seconds:.2f} seconds\n")
        
        # Dataset Overview
        report.append("\n## Dataset Overview\n")
        report.append(f"- **Total Images:** {self.dataset_stats.total_images:,}\n")
        report.append(f"- **Total Tags:** {self.dataset_stats.total_tags:,}\n")
        report.append(f"- **Unique Tags:** {self.dataset_stats.unique_tags:,}\n")
        report.append(f"- **Average Tags per Image:** {self.dataset_stats.avg_tags_per_image:.2f}\n")
        
        # Image Statistics
        report.append("\n## Image Statistics\n")
        report.append(f"- **Average Width:** {self.dataset_stats.avg_width:.0f} pixels\n")
        report.append(f"- **Average Height:** {self.dataset_stats.avg_height:.0f} pixels\n")
        report.append(f"- **Average File Size:** {self.dataset_stats.avg_file_size_kb:.2f} KB\n")
        
        # Quality Metrics
        report.append("\n## Quality Metrics\n")
        report.append(f"- **Corrupted Images:** {self.dataset_stats.corrupted_images}\n")
        report.append(f"- **Low Quality Images:** {self.dataset_stats.low_quality_images}\n")
        report.append(f"- **Duplicate Images:** {self.dataset_stats.duplicate_images}\n")
        
        # Format Distribution
        if self.dataset_stats.format_distribution:
            report.append("\n## Format Distribution\n")
            for fmt, count in self.dataset_stats.format_distribution.items():
                report.append(f"- **{fmt}:** {count:,}\n")
        
        # Top Tags
        if self.dataset_stats.tag_statistics and 'most_common_tags' in self.dataset_stats.tag_statistics:
            report.append("\n## Top 20 Tags\n")
            for tag, count in self.dataset_stats.tag_statistics['most_common_tags'][:20]:
                report.append(f"1. **{tag}:** {count:,}\n")
        
        # Save report
        output_path = self.output_dir / 'analysis_report.md'
        with open(output_path, 'w') as f:
            f.write(''.join(report))
        
        logger.info(f"Saved Markdown report to {output_path}")
    
    def _generate_html_report(self):
        """Generate HTML report"""
        html = []
        html.append("""<!DOCTYPE html>
<html>
<head>
    <title>Dataset Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .metric { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .metric-label { color: #777; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; background: white; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .warning { color: #ff9800; }
        .error { color: #f44336; }
        .success { color: #4CAF50; }
    </style>
</head>
<body>
""")
        
        html.append(f"<h1>Dataset Analysis Report</h1>")
        html.append(f"<p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<p><strong>Analysis Duration:</strong> {self.dataset_stats.analysis_duration_seconds:.2f} seconds</p>")
        
        # Overview metrics
        html.append("<h2>Dataset Overview</h2>")
        html.append('<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">')
        
        metrics = [
            ("Total Images", f"{self.dataset_stats.total_images:,}"),
            ("Unique Tags", f"{self.dataset_stats.unique_tags:,}"),
            ("Avg Tags/Image", f"{self.dataset_stats.avg_tags_per_image:.2f}"),
            ("Avg Width", f"{self.dataset_stats.avg_width:.0f}px"),
            ("Avg Height", f"{self.dataset_stats.avg_height:.0f}px"),
            ("Avg File Size", f"{self.dataset_stats.avg_file_size_kb:.2f}KB"),
        ]
        
        for label, value in metrics:
            html.append(f'<div class="metric"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>')
        
        html.append("</div>")
        
        # Quality Issues
        html.append("<h2>Quality Metrics</h2>")
        html.append("<table>")
        html.append("<tr><th>Metric</th><th>Count</th><th>Status</th></tr>")
        
        quality_metrics = [
            ("Corrupted Images", self.dataset_stats.corrupted_images, "error" if self.dataset_stats.corrupted_images > 0 else "success"),
            ("Low Quality Images", self.dataset_stats.low_quality_images, "warning" if self.dataset_stats.low_quality_images > 0 else "success"),
            ("Duplicate Images", self.dataset_stats.duplicate_images, "warning" if self.dataset_stats.duplicate_images > 0 else "success"),
        ]
        
        for metric, count, status in quality_metrics:
            # Use HTML entities instead of Unicode characters for better compatibility
            icon = "&#9888;" if status != "success" else "&#10004;"  # Warning sign or checkmark
            html.append(f'<tr><td>{metric}</td><td>{count}</td><td class="{status}">{icon}</td></tr>')
        
        html.append("</table>")
        
        # Top Tags
        if self.dataset_stats.tag_statistics and 'most_common_tags' in self.dataset_stats.tag_statistics:
            html.append("<h2>Top 30 Tags</h2>")
            html.append("<table>")
            html.append("<tr><th>Rank</th><th>Tag</th><th>Count</th><th>Percentage</th></tr>")
            
            total_tags = self.dataset_stats.total_tags or 1
            for i, (tag, count) in enumerate(self.dataset_stats.tag_statistics['most_common_tags'][:30], 1):
                percentage = (count / total_tags) * 100
                html.append(f"<tr><td>{i}</td><td>{tag}</td><td>{count:,}</td><td>{percentage:.2f}%</td></tr>")
            
            html.append("</table>")
        
        # Visualizations
        if (self.output_dir / 'dataset_analysis_visualization.png').exists():
            html.append("<h2>Visualizations</h2>")
            html.append('<img src="dataset_analysis_visualization.png" style="max-width: 100%; height: auto;">')
        
        if (self.output_dir / 'tag_wordcloud.png').exists():
            html.append('<img src="tag_wordcloud.png" style="max-width: 100%; height: auto; margin-top: 20px;">')
        
        html.append("</body></html>")
        
        # Save report
        output_path = self.output_dir / 'analysis_report.html'
        with open(output_path, 'w') as f:
            f.write('\n'.join(html))
        
        logger.info(f"Saved HTML report to {output_path}")
    
    def _get_quality_summary(self) -> Dict[str, Any]:
        """Get quality summary"""
        quality_issues = defaultdict(int)
        for s in self.image_stats:
            for issue in s.quality_issues:
                quality_issues[issue] += 1
        
        return {
            'total_issues': sum(quality_issues.values()),
            'issue_distribution': dict(quality_issues),
            'corrupted_count': self.dataset_stats.corrupted_images,
            'low_quality_count': self.dataset_stats.low_quality_images
        }
    
    def save_cache(self):
        """Save analysis cache for faster re-analysis"""
        cache_file = self.cache_dir / 'analysis_cache.pkl'
        cache_data = {
            'image_stats': self.image_stats,
            'dataset_stats': self.dataset_stats,
            'tag_counts': dict(self.tag_analyzer.tag_counts),
            'timestamp': datetime.now()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved cache to {cache_file}")
    
    def load_cache(self) -> bool:
        """Load analysis cache if available"""
        cache_file = self.cache_dir / 'analysis_cache.pkl'
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.image_stats = cache_data['image_stats']
                self.dataset_stats = cache_data['dataset_stats']
                self.tag_analyzer.tag_counts = Counter(cache_data['tag_counts'])
                
                logger.info(f"Loaded cache from {cache_data['timestamp']}")
                return True
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        
        return False


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze anime image dataset')
    parser.add_argument('dataset_paths', nargs='+', help='Paths to dataset directories or images')
    parser.add_argument('--output-dir', default='./dataset_analysis', help='Output directory for results')
    parser.add_argument('--cache-dir', default='./analysis_cache', help='Cache directory')
    parser.add_argument('--sample-size', type=int, help='Sample size for image analysis')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--no-images', action='store_true', help='Skip image analysis')
    parser.add_argument('--no-tags', action='store_true', help='Skip tag analysis')
    parser.add_argument('--no-duplicates', action='store_true', help='Skip duplicate detection')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip creating visualizations')
    parser.add_argument('--report-format', choices=['html', 'markdown', 'json'], default='html', help='Report format')
    parser.add_argument('--use-cache', action='store_true', help='Use cached results if available')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AnalysisConfig(
        dataset_paths=args.dataset_paths,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        sample_size_for_stats=args.sample_size,
        num_workers=args.num_workers,
        analyze_images=not args.no_images,
        analyze_tags=not args.no_tags,
        analyze_duplicates=not args.no_duplicates,
        create_visualizations=not args.no_visualizations,
        report_format=args.report_format
    )
    
    # Create analyzer
    analyzer = DatasetAnalyzer(config)
    
    # Load cache if requested
    if args.use_cache and analyzer.load_cache():
        logger.info("Using cached results")
        # Still generate visualizations and report
        if config.create_visualizations:
            analyzer._create_visualizations()
        if config.generate_report:
            analyzer._generate_report()
    else:
        # Run analysis
        stats = analyzer.analyze_dataset()
        
        # Save cache
        analyzer.save_cache()
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Total Images: {analyzer.dataset_stats.total_images:,}")
    print(f"Unique Tags: {analyzer.dataset_stats.unique_tags:,}")
    print(f"Corrupted Images: {analyzer.dataset_stats.corrupted_images}")
    print(f"Duplicate Images: {analyzer.dataset_stats.duplicate_images}")
    print(f"Analysis Duration: {analyzer.dataset_stats.analysis_duration_seconds:.2f}s")
    print(f"\nResults saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()