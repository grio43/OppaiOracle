#!/usr/bin/env python3
"""
Inference Engine for Anime Image Tagger
Handles model inference, batch processing, and real-time predictions
Uses Monitor_log.py for monitoring and logging functionality
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import warnings
import traceback
from contextlib import contextmanager
from collections import defaultdict, deque
import threading
import queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2

# Import monitoring system from Monitor_log
try:
    from Monitor_log import MonitorConfig, TrainingMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    warnings.warn("Monitor_log not available. Monitoring will be disabled.")
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine"""
    # Model settings
    model_path: str = "./checkpoints/best_model.pth"
    config_path: str = "./checkpoints/model_config.json"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Batch processing
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Image preprocessing
    image_size: int = 448
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Inference settings
    threshold: float = 0.5
    top_k: int = 10
    use_fp16: bool = True
    use_torch_compile: bool = False
    
    # Performance
    max_queue_size: int = 100
    timeout: float = 30.0
    enable_profiling: bool = False
    
    # Monitoring
    enable_monitoring: bool = True
    monitor_config: Optional[MonitorConfig] = None
    
    # Caching
    enable_cache: bool = True
    cache_size: int = 1000
    
    # Output
    output_format: str = "json"  # json, csv, txt
    save_visualizations: bool = False
    visualization_dir: str = "./visualizations"


class ImagePreprocessor:
    """Handles image preprocessing for inference"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.transform = self._build_transform()
        
    def _build_transform(self):
        """Build image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess a single image"""
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        return self.transform(image)
    
    def preprocess_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> torch.Tensor:
        """Preprocess a batch of images"""
        processed = []
        for img in images:
            try:
                processed.append(self.preprocess_image(img))
            except Exception as e:
                logger.error(f"Failed to preprocess image: {e}")
                # Add black image as placeholder
                processed.append(torch.zeros(3, self.config.image_size, self.config.image_size))
        
        return torch.stack(processed)


class InferenceDataset(Dataset):
    """Dataset for batch inference"""
    
    def __init__(self, image_paths: List[str], preprocessor: ImagePreprocessor):
        self.image_paths = image_paths
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = self.preprocessor.preprocess_image(path)
            return image, path, True
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            # Return black image on error
            return (
                torch.zeros(3, self.preprocessor.config.image_size, self.preprocessor.config.image_size),
                path,
                False
            )


class ModelWrapper:
    """Wrapper for the trained model with optimization"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.tag_names = []
        self.use_fp16 = config.use_fp16 and config.device == "cuda"
        
    def load_model(self):
        """Load the trained model"""
        try:
            # Load model config
            if os.path.exists(self.config.config_path):
                with open(self.config.config_path, 'r') as f:
                    model_config = json.load(f)
                    self.tag_names = model_config.get('tag_names', [])
            
            # Load checkpoint
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # Create model (placeholder - should match your actual model architecture)
            # This is where you'd instantiate your actual model
            # For now, using a dummy model structure
            from torchvision import models
            self.model = models.resnet50(pretrained=False)
            num_classes = len(self.tag_names) if self.tag_names else checkpoint.get('num_classes', 1000)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Optimize model
            if self.config.use_torch_compile and hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
            
            # Setup mixed precision
            if self.use_fp16:
                self.model = self.model.half()
            
            logger.info(f"Model loaded successfully with {num_classes} classes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference on batch of images"""
        images = images.to(self.device)
        
        if self.use_fp16:
            images = images.half()
        
        outputs = self.model(images)
        
        # Apply sigmoid for multi-label classification
        predictions = torch.sigmoid(outputs)
        
        return predictions.cpu().float()


class ResultProcessor:
    """Process and format inference results"""
    
    def __init__(self, config: InferenceConfig, tag_names: List[str]):
        self.config = config
        self.tag_names = tag_names
        
    def process_predictions(
        self,
        predictions: torch.Tensor,
        image_paths: List[str],
        valid_flags: List[bool]
    ) -> List[Dict[str, Any]]:
        """Process raw predictions into formatted results"""
        results = []
        
        for pred, path, valid in zip(predictions, image_paths, valid_flags):
            if not valid:
                results.append({
                    'image': path,
                    'error': 'Failed to process image',
                    'tags': []
                })
                continue
            
            # Get top-k predictions
            scores, indices = torch.topk(pred, min(self.config.top_k, len(pred)))
            
            # Filter by threshold
            mask = scores >= self.config.threshold
            scores = scores[mask]
            indices = indices[mask]
            
            # Format results
            tags = []
            for score, idx in zip(scores.tolist(), indices.tolist()):
                if idx < len(self.tag_names):
                    tags.append({
                        'name': self.tag_names[idx],
                        'score': round(score, 4)
                    })
            
            results.append({
                'image': path,
                'tags': tags,
                'processing_time': None  # Will be filled by engine
            })
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif self.config.output_format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        elif self.config.output_format == 'txt':
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(f"{result['image']}: ")
                    tags = [f"{t['name']}({t['score']:.2f})" for t in result.get('tags', [])]
                    f.write(', '.join(tags) + '\n')


class InferenceCache:
    """Simple LRU cache for inference results"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_order = deque(maxlen=max_size)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # Update access order
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove oldest
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.hits / total if total > 0 else 0,
                'size': len(self.cache)
            }


class InferenceEngine:
    """Main inference engine for anime image tagging"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        self.model_wrapper = ModelWrapper(config)
        self.result_processor = None
        self.monitor = None
        self.cache = None
        
        # Setup components
        self._setup()
        
    def _setup(self):
        """Setup inference engine components"""
        # Load model
        if not self.model_wrapper.load_model():
            raise RuntimeError("Failed to load model")
        
        # Setup result processor
        self.result_processor = ResultProcessor(
            self.config,
            self.model_wrapper.tag_names
        )
        
        # Setup monitoring
        if self.config.enable_monitoring and MONITORING_AVAILABLE:
            monitor_config = self.config.monitor_config or MonitorConfig(
                log_level="INFO",
                use_tensorboard=False,
                use_wandb=False,
                track_gpu_metrics=True,
                enable_alerts=False
            )
            self.monitor = TrainingMonitor(monitor_config)
            logger.info("Monitoring enabled for inference")
        
        # Setup cache
        if self.config.enable_cache:
            self.cache = InferenceCache(self.config.cache_size)
            logger.info(f"Cache enabled with size {self.config.cache_size}")
    
    def predict_single(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Predict tags for a single image"""
        start_time = time.time()
        
        # Check cache if using file path
        if self.cache and isinstance(image, str):
            cached = self.cache.get(image)
            if cached is not None:
                logger.debug(f"Cache hit for {image}")
                return cached
        
        try:
            # Preprocess
            processed = self.preprocessor.preprocess_image(image)
            processed = processed.unsqueeze(0)  # Add batch dimension
            
            # Predict
            predictions = self.model_wrapper.predict(processed)
            
            # Process results
            image_path = image if isinstance(image, str) else "array"
            results = self.result_processor.process_predictions(
                predictions,
                [image_path],
                [True]
            )[0]
            
            # Add timing
            results['processing_time'] = time.time() - start_time
            
            # Cache result
            if self.cache and isinstance(image, str):
                self.cache.put(image, results)
            
            # Log to monitor
            if self.monitor:
                self.monitor.metrics.add_metric('inference_time', results['processing_time'])
                self.monitor.metrics.increment_counter('images_processed')
            
            return results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """Predict tags for a batch of images"""
        start_time = time.time()
        
        try:
            # Check cache for file paths
            results = [None] * len(images)
            uncached_indices = []
            uncached_images = []
            
            if self.cache:
                for i, img in enumerate(images):
                    if isinstance(img, str):
                        cached = self.cache.get(img)
                        if cached is not None:
                            results[i] = cached
                        else:
                            uncached_indices.append(i)
                            uncached_images.append(img)
                    else:
                        uncached_indices.append(i)
                        uncached_images.append(img)
            else:
                uncached_indices = list(range(len(images)))
                uncached_images = images
            
            # Process uncached images
            if uncached_images:
                # Preprocess batch
                processed = self.preprocessor.preprocess_batch(uncached_images)
                
                # Predict
                predictions = self.model_wrapper.predict(processed)
                
                # Process results
                batch_results = self.result_processor.process_predictions(
                    predictions,
                    [img if isinstance(img, str) else f"array_{i}" 
                     for i, img in enumerate(uncached_images)],
                    [True] * len(uncached_images)
                )
                
                # Add timing and cache
                for i, (idx, result) in enumerate(zip(uncached_indices, batch_results)):
                    result['processing_time'] = (time.time() - start_time) / len(images)
                    results[idx] = result
                    
                    # Cache if applicable
                    if self.cache and isinstance(uncached_images[i], str):
                        self.cache.put(uncached_images[i], result)
            
            # Log to monitor
            if self.monitor:
                batch_time = time.time() - start_time
                self.monitor.metrics.add_metric('batch_inference_time', batch_time)
                self.monitor.metrics.add_metric('batch_size', len(images))
                self.monitor.metrics.increment_counter('batches_processed')
                self.monitor.metrics.increment_counter('images_processed', len(images))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [{
                'error': str(e),
                'processing_time': time.time() - start_time
            }] * len(images)
    
    def process_directory(
        self,
        directory: str,
        output_path: str,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp']
    ) -> Dict[str, Any]:
        """Process all images in a directory"""
        directory = Path(directory)
        output_path = Path(output_path)
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.glob(f'**/*{ext}'))
            image_paths.extend(directory.glob(f'**/*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        logger.info(f"Found {len(image_paths)} images to process")
        
        if not image_paths:
            return {'error': 'No images found', 'total': 0}
        
        # Create dataset and dataloader
        dataset = InferenceDataset(image_paths, self.preprocessor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor
        )
        
        # Process batches
        all_results = []
        start_time = time.time()
        
        try:
            for batch_images, batch_paths, batch_valid in dataloader:
                # Run inference
                predictions = self.model_wrapper.predict(batch_images)
                
                # Process results
                batch_results = self.result_processor.process_predictions(
                    predictions,
                    batch_paths,
                    batch_valid.tolist()
                )
                
                all_results.extend(batch_results)
                
                # Log progress
                logger.info(f"Processed {len(all_results)}/{len(image_paths)} images")
                
                # Update monitor
                if self.monitor:
                    self.monitor.metrics.increment_counter('batches_processed')
                    self.monitor.metrics.increment_counter('images_processed', len(batch_images))
            
            # Save results
            self.result_processor.save_results(all_results, output_path)
            
            # Calculate statistics
            total_time = time.time() - start_time
            stats = {
                'total_images': len(image_paths),
                'successful': sum(1 for r in all_results if 'error' not in r),
                'failed': sum(1 for r in all_results if 'error' in r),
                'total_time': total_time,
                'avg_time_per_image': total_time / len(image_paths),
                'output_file': str(output_path)
            }
            
            # Add cache stats
            if self.cache:
                stats['cache'] = self.cache.get_stats()
            
            logger.info(f"Processing complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Directory processing failed: {e}")
            return {
                'error': str(e),
                'total_images': len(image_paths),
                'completed': len(all_results)
            }
    
    @contextmanager
    def profile(self):
        """Context manager for profiling inference"""
        if self.config.enable_profiling:
            import cProfile
            import pstats
            from io import StringIO
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                yield
            finally:
                profiler.disable()
                s = StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(20)
                logger.info(f"Profiling results:\n{s.getvalue()}")
        else:
            yield
    
    def cleanup(self):
        """Cleanup resources"""
        if self.monitor:
            self.monitor.close()
        
        # Clear cache
        if self.cache:
            logger.info(f"Cache stats at cleanup: {self.cache.get_stats()}")
        
        # Clear CUDA cache if using GPU
        if self.config.device == "cuda":
            torch.cuda.empty_cache()


# Example usage
if __name__ == "__main__":
    # Configure inference
    config = InferenceConfig(
        model_path="./checkpoints/best_model.pth",
        batch_size=32,
        threshold=0.5,
        top_k=10,
        enable_monitoring=True,
        enable_cache=True
    )
    
    # Create inference engine
    engine = InferenceEngine(config)
    
    try:
        # Example 1: Single image prediction
        result = engine.predict_single("example.jpg")
        print(f"Single prediction: {result}")
        
        # Example 2: Batch prediction
        images = ["image1.jpg", "image2.jpg", "image3.jpg"]
        results = engine.predict_batch(images)
        print(f"Batch predictions: {results}")
        
        # Example 3: Process directory
        with engine.profile():
            stats = engine.process_directory(
                directory="./test_images",
                output_path="./results/predictions.json"
            )
            print(f"Directory processing stats: {stats}")
            
    finally:
        # Cleanup
        engine.cleanup()