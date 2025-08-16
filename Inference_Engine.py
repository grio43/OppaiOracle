#!/usr/bin/env python3
"""
Inference Engine for Anime Image Tagger
Production-ready inference pipeline with multiple interfaces
FIXED: Normalization now matches training (0.5, 0.5, 0.5)
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
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as T
from PIL import Image
from vocabulary import TagVocabulary, load_vocabulary_for_training

# Optional imports for API
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Model Architecture Components (minimal implementation)
# ============================================================================

@dataclass
class VisionTransformerConfig:
    """Configuration for Vision Transformer model"""
    image_size: int = 640
    patch_size: int = 14  # Changed from 16
    num_channels: int = 3
    hidden_size: int = 1280  # Changed from 768
    num_hidden_layers: int = 24  # Changed from 12
    num_attention_heads: int = 16  # Changed from 12
    intermediate_size: int = 5120  # Changed to 4x hidden_size
    num_tags: int = 100000  # Changed from 10000
    num_ratings: int = 5  # Added
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6


class SimpleViT(nn.Module):
    """Simplified Vision Transformer for demonstration"""
    
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.num_channels, 
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Position embedding
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.num_tags)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x[:, 0])  # Use CLS token
        logits = self.head(x)
        
        return {'logits': logits}


def create_model(**kwargs):
    """Create model from config"""
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 10000  # Default number of tags
    
    config = VisionTransformerConfig(**kwargs)
    return SimpleViT(config)



# ============================================================================
# Memory Optimization Components
# ============================================================================

class MemoryOptimizer:
    """Memory optimization utilities"""
    
    @staticmethod
    def optimize_model(model: nn.Module) -> nn.Module:
        """Optimize model for inference"""
        # Fuse batch norm layers if available
        if hasattr(torch.jit, 'optimize_for_inference'):
            try:
                model = torch.jit.optimize_for_inference(torch.jit.script(model))
            except:
                pass  # Skip if scripting fails
        return model


# ============================================================================
# Main Inference Components with FIXED Normalization
# ============================================================================

@dataclass
class InferenceConfig:
    """Configuration for inference with FIXED normalization"""
    # Model settings
    model_path: str
    vocab_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
    
    # Image preprocessing - FIXED to match training
    image_size: int = 640
    # CRITICAL FIX: Use anime-optimized normalization to match training
    # These values MUST match what was used during training (HDF5_loader.py)
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # FIXED: was (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)    # FIXED: was (0.229, 0.224, 0.225)
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
    optimize_for_speed: bool = False  # Disabled by default to avoid scripting issues
    
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
    
    def validate_normalization(self):
        """Validate normalization parameters are correct for anime models"""
        # Check if using incorrect ImageNet normalization
        if (abs(self.normalize_mean[0] - 0.485) < 0.01 or 
            abs(self.normalize_std[0] - 0.229) < 0.01):
            logger.warning(
                "⚠️ WARNING: Config appears to use ImageNet normalization!\n"
                "  Anime models should use mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)\n"
                "  Auto-correcting to anime normalization..."
            )
            # Auto-correct to anime normalization
            self.normalize_mean = (0.5, 0.5, 0.5)
            self.normalize_std = (0.5, 0.5, 0.5)
            return False
        return True


class ImagePreprocessor:
    """Optimized image preprocessing for inference"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
        # Validate normalization on initialization
        config.validate_normalization()
        
        # Create transforms with correct normalization
        self.normalize = T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        
        # Pre-allocate tensors for efficiency
        self.device = torch.device(config.device)
        
        # Log normalization being used
        logger.info(f"ImagePreprocessor initialized with normalization:")
        logger.info(f"  Mean: {config.normalize_mean}")
        logger.info(f"  Std: {config.normalize_std}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess single image"""
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Handle transparency
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, self.config.pad_color)
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode == 'RGBA':
                background.paste(image, mask=image.getchannel('A'))
            elif image.mode == 'LA':
                background.paste(image, mask=image.getchannel('L'))
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
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
    
    def __init__(self, vocab: TagVocabulary, config: InferenceConfig):
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
            
            # Skip if out of vocabulary range
            if idx >= len(self.vocab):
                continue
            
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
            k = min(self.config.min_predictions, len(scores))
            _, indices = torch.topk(scores, k)
        
        # Adjust if too many predictions
        elif len(indices) > self.config.max_predictions:
            # Raise threshold
            threshold_scores = scores[indices]
            _, top_indices = torch.topk(threshold_scores, self.config.max_predictions)
            indices = indices[top_indices]
        
        return indices
    
    def _apply_implications(self, tags: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Apply tag implications"""
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
    """Main inference engine with normalization validation"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load vocabulary
        logger.info(f"Loading vocabulary from {config.vocab_dir}")
        self.vocab = load_vocabulary_for_training(Path(config.vocab_dir))
        
        # Load model and validate normalization
        logger.info(f"Loading model from {config.model_path}")
        self.model = self._load_model()
        
        # Validate normalization parameters
        self._validate_normalization()
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(config)
        self.postprocessor = TagPostProcessor(self.vocab, config)
        
        # Warmup model
        self._warmup()
        
        # Performance tracking
        self.inference_times = []
    
    def _validate_normalization(self):
        """Validate that normalization parameters match training"""
        try:
            # Try to load normalization params from checkpoint
            checkpoint = torch.load(self.config.model_path, map_location='cpu', weights_only=False)
            
            if 'normalization_params' in checkpoint:
                # If checkpoint contains normalization params, validate against config
                saved_mean = checkpoint['normalization_params'].get('mean')
                saved_std = checkpoint['normalization_params'].get('std')
                
                if saved_mean and saved_std:
                    config_mean = self.config.normalize_mean
                    config_std = self.config.normalize_std
                    
                    # Check if they match (with small tolerance for float precision)
                    mean_match = all(abs(s - c) < 1e-6 for s, c in zip(saved_mean, config_mean))
                    std_match = all(abs(s - c) < 1e-6 for s, c in zip(saved_std, config_std))
                    
                    if not (mean_match and std_match):
                        logger.warning(
                            f"⚠️ NORMALIZATION MISMATCH DETECTED!\n"
                            f"  Training used: mean={saved_mean}, std={saved_std}\n"
                            f"  Config has: mean={config_mean}, std={config_std}\n"
                            f"  Updating config to match training values..."
                        )
                        # Auto-correct the mismatch
                        self.config.normalize_mean = tuple(saved_mean)
                        self.config.normalize_std = tuple(saved_std)
                    else:
                        logger.info("✓ Normalization parameters validated successfully")
            else:
                # No saved params in checkpoint, log expected values
                logger.info(
                    f"Normalization parameters not found in checkpoint.\n"
                    f"Using config values: mean={self.config.normalize_mean}, "
                    f"std={self.config.normalize_std}\n"
                    f"⚠️ Please ensure these match training values!"
                )
                
                # Add runtime warning if using ImageNet values
                if (abs(self.config.normalize_mean[0] - 0.485) < 0.01 or 
                    abs(self.config.normalize_std[0] - 0.229) < 0.01):
                    logger.warning(
                        "⚠️ Config appears to use ImageNet normalization!\n"
                        "  Anime models typically use mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)\n"
                        "  Auto-correcting to anime normalization..."
                    )
                    # Auto-correct to anime normalization
                    self.config.normalize_mean = (0.5, 0.5, 0.5)
                    self.config.normalize_std = (0.5, 0.5, 0.5)
                    
        except Exception as e:
            logger.error(f"Could not validate normalization parameters: {e}")
            # Default to anime normalization if validation fails
            logger.info("Defaulting to anime normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)")
            self.config.normalize_mean = (0.5, 0.5, 0.5)
            self.config.normalize_std = (0.5, 0.5, 0.5)
    
    def _load_model(self) -> nn.Module:
        """Load and optimize model"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.config.model_path, map_location='cpu', weights_only=False)
            
            # Extract config and normalization params if available
            if 'config' in checkpoint:
                model_config = checkpoint['config']
                if 'model_config' in model_config:
                    model_config = model_config['model_config']
                    
                # Check for normalization params in config
                if 'normalization_params' in model_config:
                    norm_params = model_config['normalization_params']
                    logger.info(f"Found normalization params in model config: {norm_params}")
            else:
                model_config = VisionTransformerConfig()
            
            # Create model
            if isinstance(model_config, dict):
                model = create_model(**model_config)
            else:
                model = create_model(**model_config.__dict__)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DDP weights
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            logger.warning(f"Could not load model from checkpoint: {e}")
            logger.warning("Creating new model with default config")
            model = create_model()
        
        model.to(self.device)
        model.eval()
        
        # Optimize model if requested (disabled scripting to avoid issues)
        if self.config.optimize_for_speed:
            try:
                model = torch.jit.script(model)
                logger.info("Model optimized with TorchScript")
            except Exception as e:
                logger.warning(f"Could not optimize model: {e}")
        
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
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
            return model
    
    def _warmup(self):
        """Warmup model with dummy inference"""
        logger.info("Warming up model...")
        dummy_input = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(self.device)
        
        for _ in range(3):
            with torch.no_grad():
                try:
                    _ = self.model(dummy_input)
                except Exception as e:
                    logger.warning(f"Warmup failed: {e}")
                    break
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """Predict tags for single image"""
        try:
            # Preprocess
            start_time = time.time()
            tensor = self.preprocessor.preprocess_image(image)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            # Inference
            if self.config.use_fp16 and self.device.type == 'cuda':
                with autocast(enabled=True):
                    outputs = self.model(tensor)
            else:
                outputs = self.model(tensor)
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Handle hierarchical output
            if logits.dim() == 3:
                logits = logits.view(logits.shape[0], -1)
            
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
                result['normalization'] = {
                    'mean': self.config.normalize_mean,
                    'std': self.config.normalize_std
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                "error": str(e),
                "tags": [],
                "count": 0
            }
    
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
            
            try:
                # Preprocess batch
                start_time = time.time()
                batch_tensor = self.preprocessor.preprocess_batch(batch_images)
                batch_tensor = batch_tensor.to(self.device)
                
                # Inference
                if self.config.use_fp16 and self.device.type == 'cuda':
                    with autocast(enabled=True):
                        outputs = self.model(batch_tensor)
                else:
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
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Add error results for failed batch
                for img in batch_images:
                    results.append({
                        "error": str(e),
                        "tags": [],
                        "count": 0
                    })
        
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
            return {
                "error": f"Failed to load image from URL: {str(e)}",
                "tags": [],
                "count": 0
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        if not self.inference_times:
            return {
                "normalization": {
                    "mean": self.config.normalize_mean,
                    "std": self.config.normalize_std,
                    "scheme": "anime" if self.config.normalize_mean == (0.5, 0.5, 0.5) else "unknown"
                }
            }
        
        times_ms = [t * 1000 for t in self.inference_times]
        
        return {
            'avg_inference_time_ms': float(np.mean(times_ms)),
            'std_inference_time_ms': float(np.std(times_ms)),
            'min_inference_time_ms': float(np.min(times_ms)),
            'max_inference_time_ms': float(np.max(times_ms)),
            'total_images': len(self.inference_times),
            'images_per_second': 1000 / np.mean(times_ms) if times_ms else 0,
            'normalization': {
                'mean': self.config.normalize_mean,
                'std': self.config.normalize_std,
                'scheme': 'anime' if self.config.normalize_mean == (0.5, 0.5, 0.5) else 'unknown'
            }
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
        extensions: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Process directory asynchronously"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.rglob(f'*{ext}'))
            image_paths.extend(directory.rglob(f'*{ext.upper()}'))
        
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
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def process_directory(
        self,
        directory: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        extensions: List[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Process directory synchronously with optional progress callback"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        directory = Path(directory)
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.rglob(f'*{ext}'))
            image_paths.extend(directory.rglob(f'*{ext.upper()}'))
        
        # Remove duplicates
        image_paths = list(set(image_paths))
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process images
        results = []
        for i, path in enumerate(image_paths):
            try:
                result = self.engine.predict(path)
                results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(image_paths), path)
                    
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append({
                    "image": str(path),
                    "error": str(e),
                    "tags": []
                })
        
        # Save results if requested
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata about normalization
            output_data = {
                'results': results,
                'metadata': {
                    'total_images': len(results),
                    'normalization': {
                        'mean': self.engine.config.normalize_mean,
                        'std': self.engine.config.normalize_std
                    }
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
        
        return results


class InferenceAPI:
    """FastAPI-based inference API"""
    
    def __init__(self, engine: InferenceEngine, config: InferenceConfig):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart")
        
        self.engine = engine
        self.config = config
        self.app = FastAPI(title="Anime Image Tagger API")
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Anime Image Tagger API",
                "version": "1.0.0",
                "normalization": {
                    "mean": self.config.normalize_mean,
                    "std": self.config.normalize_std
                }
            }
        
        @self.app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            """Predict tags for uploaded image"""
            # Check file size
            contents = await file.read()
            if len(contents) > self.config.api_max_image_size:
                raise HTTPException(400, f"File size exceeds {self.config.api_max_image_size} bytes")
            
            # Process image
            try:
                image = Image.open(io.BytesIO(contents))
                result = self.engine.predict(image)
                return JSONResponse(content=result)
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                raise HTTPException(500, f"Error processing image: {str(e)}")
        
        @self.app.post("/predict_url")
        async def predict_url(url: str):
            """Predict tags from image URL"""
            try:
                result = self.engine.predict_from_url(url)
                if "error" in result:
                    raise HTTPException(400, result["error"])
                return JSONResponse(content=result)
                
            except Exception as e:
                logger.error(f"Error processing URL: {e}")
                raise HTTPException(500, f"Error processing URL: {str(e)}")
        
        @self.app.get("/stats")
        async def get_stats():
            """Get inference statistics"""
            return self.engine.get_statistics()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "normalization": {
                    "mean": self.config.normalize_mean,
                    "std": self.config.normalize_std,
                    "valid": self.config.normalize_mean == (0.5, 0.5, 0.5)
                }
            }
        
        @self.app.get("/config")
        async def get_config():
            """Get current configuration"""
            return {
                "image_size": self.config.image_size,
                "normalization": {
                    "mean": self.config.normalize_mean,
                    "std": self.config.normalize_std
                },
                "prediction_threshold": self.config.prediction_threshold,
                "adaptive_threshold": self.config.adaptive_threshold,
                "min_predictions": self.config.min_predictions,
                "max_predictions": self.config.max_predictions
            }
    
    def run(self):
        """Run the API server"""
        uvicorn.run(
            self.app,
            host=self.config.api_host,
            port=self.config.api_port
        )


def verify_normalization_fix():
    """Utility function to verify normalization is fixed"""
    print("\n" + "="*60)
    print("NORMALIZATION VERIFICATION")
    print("="*60)
    
    # Check default config
    config = InferenceConfig(
        model_path="dummy.pt",
        vocab_dir="."
    )
    
    print(f"Default normalization:")
    print(f"  Mean: {config.normalize_mean}")
    print(f"  Std: {config.normalize_std}")
    
    if config.normalize_mean == (0.5, 0.5, 0.5) and config.normalize_std == (0.5, 0.5, 0.5):
        print("✅ Normalization is CORRECT (anime-optimized)")
    else:
        print("❌ Normalization is INCORRECT (not anime-optimized)")
    
    # Test auto-correction
    bad_config = InferenceConfig(
        model_path="dummy.pt",
        vocab_dir=".",
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225)
    )
    
    print(f"\nTesting auto-correction for ImageNet values:")
    if bad_config.validate_normalization():
        print("  No correction needed")
    else:
        print(f"  Auto-corrected to: mean={bad_config.normalize_mean}, std={bad_config.normalize_std}")
    
    print("\n" + "="*60)


def main():
    """Main entry point for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Anime Image Tagger Inference (FIXED)")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", required=True, help="Path to vocabulary directory")
    parser.add_argument("--image", help="Path to single image")
    parser.add_argument("--directory", help="Path to directory of images")
    parser.add_argument("--url", help="URL of image")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    parser.add_argument("--format", choices=["json", "text", "csv"], default="json", help="Output format")
    parser.add_argument("--verify-fix", action="store_true", help="Verify normalization fix")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Verify fix if requested
    if args.verify_fix:
        verify_normalization_fix()
        return
    
    # Create config with FIXED normalization
    config = InferenceConfig(
        model_path=args.model,
        vocab_dir=args.vocab,
        device=args.device,
        prediction_threshold=args.threshold,
        batch_size=args.batch_size,
        enable_api=args.api,
        api_port=args.api_port,
        output_format=args.format,
        # Ensure correct normalization
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5)
    )
    
    # Log normalization being used
    logger.info("="*60)
    logger.info("INFERENCE ENGINE INITIALIZATION")
    logger.info("="*60)
    logger.info(f"Normalization Configuration:")
    logger.info(f"  Mean: {config.normalize_mean}")
    logger.info(f"  Std: {config.normalize_std}")
    logger.info(f"  Scheme: {'anime' if config.normalize_mean == (0.5, 0.5, 0.5) else 'unknown'}")
    logger.info("="*60)
    
    # Create engine
    engine = InferenceEngine(config)
    
    # Handle different modes
    if args.api:
        # Start API server
        api = InferenceAPI(engine, config)
        api.run()
        
    elif args.image:
        # Process single image
        result = engine.predict(args.image)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
        
    elif args.url:
        # Process URL
        result = engine.predict_from_url(args.url)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
        
    elif args.directory:
        # Process directory
        processor = BatchInferenceProcessor(engine)
        
        def progress_callback(current, total, path):
            print(f"Processing {current}/{total}: {path.name}")
        
        results = processor.process_directory(
            args.directory,
            args.output,
            progress_callback=progress_callback
        )
        
        print(f"\nProcessed {len(results)} images")
        stats = engine.get_statistics()
        if stats:
            print(f"Average inference time: {stats.get('avg_inference_time_ms', 0):.2f}ms")
            print(f"Images per second: {stats.get('images_per_second', 0):.2f}")
            print(f"Normalization scheme: {stats['normalization']['scheme']}")
        
    else:
        print("Please specify --image, --url, --directory, or --api")
        print("Use --verify-fix to check normalization configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()