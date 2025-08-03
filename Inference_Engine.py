#!/usr/bin/env python3
"""
Inference Engine for Anime Image Tagger
Production-ready inference pipeline with multiple interfaces
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import io
import base64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as T
from PIL import Image
import cv2

# Optional imports for API
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import our modules
from model_architecture import create_model
from tag_vocabulary import load_vocabulary_for_training
from training_utils import MemoryOptimizer

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    # Model settings
    model_path: str
    vocab_dir: str
    device: str = "cuda"
    use_fp16: bool = True
    
    # Image preprocessing
    image_size: int = 640
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    pad_color: Tuple[int, int, int] = (114, 114, 114)
    
    # Prediction settings
    prediction_threshold: float = 0.5
    adaptive_threshold: bool = True
    min_predictions: int = 5
    max_predictions: int = 50
    top_k: Optional[int] = None
    
    # Tag filtering
    filter_nsfw: bool = False
    nsfw_tags: List[str] = field(default_factory=lambda: ['explicit', 'questionable'])
    blacklist_tags: List[str] = field(default_factory=list)
    whitelist_tags: Optional[List[str]] = None
    
    # Post-processing
    apply_implications: bool = True
    resolve_aliases: bool = True
    group_by_category: bool = False
    
    # Performance
    batch_size: int = 32
    num_workers: int = 4
    use_tensorrt: bool = False
    optimize_for_speed: bool = True
    
    # Output format
    output_format: str = "json"  # json, text, csv
    include_scores: bool = True
    score_decimal_places: int = 3
    
    # API settings
    enable_api: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_max_image_size: int = 10 * 1024 * 1024  # 10MB
    api_rate_limit: int = 100  # requests per minute


class ImagePreprocessor:
    """Optimized image preprocessing for inference"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
        # Create transforms
        self.normalize = T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        
        # Pre-allocate tensors for efficiency
        self.device = torch.device(config.device)
        
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess single image"""
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Handle transparency
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, self.config.pad_color)
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.getchannel('A') if image.mode == 'RGBA' else image.getchannel('L'))
            image = background
        
        # Letterbox resize
        image = self._letterbox_image(image)
        
        # Convert to tensor
        tensor = T.ToTensor()(image)
        tensor = self.normalize(tensor)
        
        return tensor
    
    def _letterbox_image(self, image: Image.Image) -> Image.Image:
        """Letterbox image to target size"""
        w, h = image.size
        target_size = self.config.image_size
        
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create canvas
        canvas = Image.new('RGB', (target_size, target_size), self.config.pad_color)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        
        return canvas
    
    def preprocess_batch(self, images: List[Union[str, Path, Image.Image]]) -> torch.Tensor:
        """Preprocess batch of images"""
        tensors = []
        for image in images:
            tensor = self.preprocess_image(image)
            tensors.append(tensor)
        
        return torch.stack(tensors)


class TagPostProcessor:
    """Post-process predicted tags"""
    
    def __init__(self, vocab, config: InferenceConfig):
        self.vocab = vocab
        self.config = config
        
        # Load implications and aliases if available
        self.implications = {}
        self.aliases = {}
        
        # Build NSFW tag set
        self.nsfw_indices = set()
        if config.filter_nsfw:
            for tag in config.nsfw_tags:
                idx = vocab.get_tag_index(tag)
                if idx != vocab.unk_index:
                    self.nsfw_indices.add(idx)
        
        # Build blacklist indices
        self.blacklist_indices = set()
        for tag in config.blacklist_tags:
            idx = vocab.get_tag_index(tag)
            if idx != vocab.unk_index:
                self.blacklist_indices.add(idx)
        
        # Build whitelist indices
        self.whitelist_indices = None
        if config.whitelist_tags:
            self.whitelist_indices = set()
            for tag in config.whitelist_tags:
                idx = vocab.get_tag_index(tag)
                if idx != vocab.unk_index:
                    self.whitelist_indices.add(idx)
    
    def process_predictions(
        self,
        predictions: torch.Tensor,
        scores: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """Process predictions for a single image"""
        # Get indices above threshold
        if self.config.adaptive_threshold:
            indices = self._adaptive_threshold(scores)
        else:
            indices = torch.where(scores > self.config.prediction_threshold)[0]
        
        # Apply top-k if specified
        if self.config.top_k and len(indices) > self.config.top_k:
            top_scores, top_indices = torch.topk(scores[indices], self.config.top_k)
            indices = indices[top_indices]
        
        # Filter tags
        filtered_tags = []
        for idx in indices:
            idx = idx.item()
            
            # Skip blacklisted
            if idx in self.blacklist_indices:
                continue
            
            # Skip if not in whitelist
            if self.whitelist_indices and idx not in self.whitelist_indices:
                continue
            
            # Skip NSFW if filtering
            if self.config.filter_nsfw and idx in self.nsfw_indices:
                continue
            
            tag = self.vocab.get_tag_from_index(idx)
            score = scores[idx].item()
            
            filtered_tags.append((tag, score))
        
        # Apply implications
        if self.config.apply_implications:
            filtered_tags = self._apply_implications(filtered_tags)
        
        # Sort by score
        filtered_tags.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_tags
    
    def _adaptive_threshold(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply adaptive thresholding"""
        # Start with base threshold
        threshold = self.config.prediction_threshold
        indices = torch.where(scores > threshold)[0]
        
        # Adjust if too few predictions
        if len(indices) < self.config.min_predictions:
            # Get top min_predictions
            _, indices = torch.topk(scores, min(self.config.min_predictions, len(scores)))
        
        # Adjust if too many predictions
        elif len(indices) > self.config.max_predictions:
            # Raise threshold
            threshold_scores = scores[indices]
            _, top_indices = torch.topk(threshold_scores, self.config.max_predictions)
            indices = indices[top_indices]
        
        return indices
    
    def _apply_implications(self, tags: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Apply tag implications"""
        # This is simplified - in practice, load from a implications file
        tag_set = {tag for tag, _ in tags}
        implied_tags = set()
        
        # Example implications
        implications = {
            'cat_ears': ['animal_ears'],
            'dog_ears': ['animal_ears'],
            'thighhighs': ['legwear'],
            'pantyhose': ['legwear'],
        }
        
        for tag, _ in tags:
            if tag in implications:
                for implied in implications[tag]:
                    if implied not in tag_set:
                        implied_tags.add(implied)
        
        # Add implied tags with lower confidence
        for implied in implied_tags:
            tags.append((implied, 0.3))  # Lower confidence for implied
        
        return tags
    
    def format_output(self, tags: List[Tuple[str, float]], image_path: Optional[str] = None) -> Any:
        """Format output based on config"""
        if self.config.output_format == "json":
            output = {
                "tags": [{"tag": tag, "score": round(score, self.config.score_decimal_places)} 
                        for tag, score in tags] if self.config.include_scores
                else [tag for tag, _ in tags],
                "count": len(tags)
            }
            if image_path:
                output["image"] = str(image_path)
            return output
        
        elif self.config.output_format == "text":
            if self.config.include_scores:
                return "\n".join([f"{tag}: {score:.{self.config.score_decimal_places}f}" 
                                for tag, score in tags])
            else:
                return ", ".join([tag for tag, _ in tags])
        
        elif self.config.output_format == "csv":
            if self.config.include_scores:
                return ",".join([f"{tag},{score:.{self.config.score_decimal_places}f}" 
                               for tag, score in tags])
            else:
                return ",".join([tag for tag, _ in tags])
        
        else:
            return tags


class InferenceEngine:
    """Main inference engine"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load vocabulary
        logger.info(f"Loading vocabulary from {config.vocab_dir}")
        self.vocab = load_vocabulary_for_training(Path(config.vocab_dir))
        
        # Load model
        logger.info(f"Loading model from {config.model_path}")
        self.model = self._load_model()
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(config)
        self.postprocessor = TagPostProcessor(self.vocab, config)
        
        # Warmup model
        self._warmup()
        
        # Performance tracking
        self.inference_times = []
        
    def _load_model(self) -> nn.Module:
        """Load and optimize model"""
        # Load checkpoint
        checkpoint = torch.load(self.config.model_path, map_location='cpu')
        
        # Extract config
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            if 'model_config' in model_config:
                model_config = model_config['model_config']
        else:
            from model_architecture import VisionTransformerConfig
            model_config = VisionTransformerConfig()
        
        # Create model
        model = create_model(**model_config if isinstance(model_config, dict) else model_config.__dict__)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DDP weights
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Optimize model
        if self.config.optimize_for_speed:
            model = MemoryOptimizer.optimize_model(model)
        
        # TensorRT optimization (if available)
        if self.config.use_tensorrt:
            model = self._optimize_tensorrt(model)
        
        logger.info(f"Model loaded successfully")
        return model
    
    def _optimize_tensorrt(self, model: nn.Module) -> nn.Module:
        """Optimize model with TensorRT"""
        try:
            import torch_tensorrt
            
            # Compile model
            example_input = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(self.device)
            
            compiled_model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions={torch.float, torch.half} if self.config.use_fp16 else {torch.float}
            )
            
            logger.info("Model optimized with TensorRT")
            return compiled_model
            
        except ImportError:
            logger.warning("TensorRT not available, using standard model")
            return model
    
    def _warmup(self):
        """Warmup model with dummy inference"""
        logger.info("Warming up model...")
        dummy_input = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(self.device)
        
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """Predict tags for single image"""
        # Preprocess
        start_time = time.time()
        tensor = self.preprocessor.preprocess_image(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with autocast(enabled=self.config.use_fp16):
            outputs = self.model(tensor)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # Handle hierarchical output
        if logits.dim() == 3:
            logits = logits.view(1, -1)
        
        # Get probabilities
        scores = torch.sigmoid(logits[0])
        
        # Post-process
        tags = self.postprocessor.process_predictions(logits[0], scores)
        
        # Track timing
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Format output
        result = self.postprocessor.format_output(tags, image if isinstance(image, (str, Path)) else None)
        
        if isinstance(result, dict):
            result['inference_time'] = round(inference_time * 1000, 2)  # ms
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """Predict tags for batch of images"""
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.config.batch_size):
            batch_images = images[i:i + self.config.batch_size]
            
            # Preprocess batch
            start_time = time.time()
            batch_tensor = self.preprocessor.preprocess_batch(batch_images)
            batch_tensor = batch_tensor.to(self.device)
            
            # Inference
            with autocast(enabled=self.config.use_fp16):
                outputs = self.model(batch_tensor)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Handle hierarchical output
            if logits.dim() == 3:
                batch_size = logits.shape[0]
                logits = logits.view(batch_size, -1)
            
            # Get probabilities
            scores = torch.sigmoid(logits)
            
            # Process each image
            for j, (image_logits, image_scores) in enumerate(zip(logits, scores)):
                tags = self.postprocessor.process_predictions(image_logits, image_scores)
                
                image_path = batch_images[j] if isinstance(batch_images[j], (str, Path)) else None
                result = self.postprocessor.format_output(tags, image_path)
                
                results.append(result)
            
            # Track timing
            batch_time = time.time() - start_time
            self.inference_times.extend([batch_time / len(batch_images)] * len(batch_images))
        
        return results
    
    def predict_from_url(self, url: str) -> Dict[str, Any]:
        """Predict tags from image URL"""
        import requests
        from io import BytesIO
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            return self.predict(image)
            
        except Exception as e:
            logger.error(f"Error loading image from URL: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, float]:
        """Get inference statistics"""
        if not self.inference_times:
            return {}
        
        times_ms = [t * 1000 for t in self.inference_times]
        
        return {
            'avg_inference_time_ms': np.mean(times_ms),
            'std_inference_time_ms': np.std(times_ms),
            'min_inference_time_ms': np.min(times_ms),
            'max_inference_time_ms': np.max(times_ms),
            'total_images': len(self.inference_times),
            'images_per_second': 1000 / np.mean(times_ms) if times_ms else 0
        }


class BatchInferenceProcessor:
    """Process large batches of images efficiently"""
    
    def __init__(self, engine: InferenceEngine, num_workers: int = 4):
        self.engine = engine
        self.num_workers = num_workers
    
    async def process_directory_async(
        self,
        directory: Path,
        output_file: Optional[Path] = None,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp']
    ) -> List[Dict[str, Any]]:
        """Process directory asynchronously"""
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.rglob(f'*{ext}'))
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process with thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            tasks = []
            for path in image_paths:
                task = loop.run_in_executor(executor, self.engine.predict, path)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def process_directory(
        self,
        directory: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]: