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
import yaml
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import warnings
import traceback
from contextlib import contextmanager
from collections import defaultdict, deque
import threading
import queue

import gzip
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Vocabulary utilities
from model_metadata import ModelMetadata
from vocabulary import TagVocabulary, load_vocabulary_for_training, verify_vocabulary_integrity
from schemas import TagPrediction, ImagePrediction, RunMetadata, PredictionOutput, compute_vocab_sha256

# Make cv2 optional - not needed for basic inference
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CV2_AVAILABLE = False
    warnings.warn("OpenCV (cv2) not available. Some image loading features may be limited.")

# Import monitoring system from Monitor_log
try:
    from Monitor_log import MonitorConfig, TrainingMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    warnings.warn("Monitor_log not available. Monitoring will be disabled.")
    MONITORING_AVAILABLE = False

# Import the actual model architecture
try:
    from model_architecture import SimplifiedTagger, VisionTransformerConfig
except ImportError:
    raise ImportError("model_architecture.py not found. Cannot load SimplifiedTagger model.")   

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
def _load_vocab_path() -> Path:
    """Resolve vocabulary path via unified_config.yaml, with sensible fallbacks."""
    try:
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "unified_config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    data = (cfg.get("data") or {})
    p = cfg.get("vocab_path") or data.get("vocab_path")
    if p:
        return Path(p)
    vd = data.get("vocab_dir")
    return (PROJECT_ROOT / vd / "vocabulary.json") if vd else (PROJECT_ROOT / "vocabulary.json")
DEFAULT_VOCAB_PATH = _load_vocab_path()


@dataclass
class InferenceConfig:
    """
    Thin shim preserving existing attributes.
    Values are overridden from configs/inference_config.yaml if present.
    """
    # Model settings
    vocab_path: Optional[str] = str(DEFAULT_VOCAB_PATH)  # Explicit vocabulary path
    model_path: str = "./checkpoints/best_model.pt"  # Standardize to .pt
    config_path: str = "./checkpoints/model_config.json"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch processing
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Image preprocessing
    image_size: int = 640
    # Use same normalization as training (anime-optimized, not ImageNet)
    normalize_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    normalize_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

    # Inference settings
    threshold: float = 0.5
    top_k: int = 10
    use_fp16: bool = True
    use_torch_compile: bool = False
    thresholds_path: Optional[str] = None
    eye_color_exclusive: bool = False
    tta_flip: bool = False

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
    cache_ttl_seconds: int = 3600  # 1 hour default TTL

    # Output
    output_format: str = "json"  # json, csv, txt
    save_visualizations: bool = False
    visualization_dir: str = "./visualizations"
    input_image_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])

    @classmethod
    def apply_yaml_overrides(cls, path: Path = PROJECT_ROOT / "configs" / "inference_config.yaml"):
        try:
            cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except Exception:
            return
        pp = (cfg.get("preprocessing") or {})
        rt = (cfg.get("runtime") or {})
        post = (cfg.get("postprocessing") or {})
        io = (cfg.get("io") or {})
        inp = (cfg.get("input") or {})
        updates = {
            "image_size": pp.get("image_size"),
            "normalize_mean": pp.get("normalize_mean"),
            "normalize_std": pp.get("normalize_std"),
            "device": rt.get("device"),
            "use_fp16": rt.get("use_fp16"),
            "tta_flip": rt.get("tta_flip"),
            "threshold": post.get("threshold"),
            "top_k": post.get("top_k"),
            "output_format": io.get("output_format"),
            "save_visualizations": io.get("save_visualizations"),
            "visualization_dir": io.get("visualization_dir"),
            "input_image_extensions": inp.get("image_extensions"),
        }
        for key, value in updates.items():
            if value is not None:
                setattr(cls, key, value)
                if key in cls.__dataclass_fields__:
                    field = cls.__dataclass_fields__[key]
                    field.default = value
                    field.default_factory = lambda v=value: v

# Apply YAML overrides on import
InferenceConfig.apply_yaml_overrides()


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
            with Image.open(image) as img:
                image = img.convert('RGB')
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
        self.vocabulary = None  # Loaded vocabulary
        self.normalization_params = None
        self.use_fp16 = config.use_fp16 and config.device == "cuda"
        self.vocab_sha256 = "unknown"  # Will be computed when vocabulary is loaded
        self.patch_size = 16  # Default, will be updated from model config

    def load_model(self):
        """Load the trained model"""
        try:
            # Load checkpoint first
            checkpoint = torch.load(self.config.model_path, map_location=self.device)

            # Priority 1: Check for embedded vocabulary in checkpoint
            if 'vocab_b64_gzip' in checkpoint:
                logger.info("Loading embedded vocabulary from checkpoint")
                vocab_data = ModelMetadata.extract_vocabulary(checkpoint)
                if vocab_data:
                    self.vocabulary = TagVocabulary()
                    self.vocabulary.tag_to_index = vocab_data['tag_to_index']
                    self.vocabulary.index_to_tag = {int(k): v for k, v in vocab_data['index_to_tag'].items()}
                    self.vocabulary.tag_frequencies = vocab_data.get('tag_frequencies', {})
                    self.vocabulary.unk_index = self.vocabulary.tag_to_index.get(self.vocabulary.unk_token, 1)
                    self.tag_names = [
                        self.vocabulary.get_tag_from_index(i)
                        for i in range(len(self.vocabulary.tag_to_index))
                    ]
                    logger.info(f"Successfully loaded {len(self.tag_names)} tags from embedded vocabulary")
                    # Compute vocabulary hash
                    self.vocab_sha256 = compute_vocab_sha256(vocab_data=vocab_data)
                else:
                    logger.error("Failed to extract embedded vocabulary")
                    # Fall back to external vocabulary file
                    self._load_external_vocabulary()
            else:
                # No embedded vocabulary, load from file
                logger.info("No embedded vocabulary found, loading from external file")
                self._load_external_vocabulary()

            # Validate vocabulary contains real tags, not placeholders
            self._verify_vocabulary()

            # Load preprocessing parameters from checkpoint
            if 'preprocessing_params' in checkpoint:
                preprocessing = ModelMetadata.extract_preprocessing_params(checkpoint)
                if preprocessing:
                    self.config.normalize_mean = preprocessing.get('normalize_mean', [0.5, 0.5, 0.5])
                    self.config.normalize_std = preprocessing.get('normalize_std', [0.5, 0.5, 0.5])
                    self.config.image_size = preprocessing.get('image_size', 640)
                    self.normalization_params = preprocessing
                    logger.info(f"Loaded preprocessing params from checkpoint: {preprocessing}")
            elif 'normalization_params' in checkpoint:
                # Legacy format
                self.normalization_params = checkpoint['normalization_params']
                self.config.normalize_mean = self.normalization_params['mean']
                self.config.normalize_std = self.normalization_params['std']
                logger.info("Loaded normalization params from checkpoint (legacy format)")
            else:
                logger.warning("Preprocessing params not found in checkpoint. Using config defaults.")

            # Load model config
            if os.path.exists(self.config.config_path):
                with open(self.config.config_path, 'r') as f:
                    model_config = json.load(f)
            else:
                logger.warning(f"Config file not found at {self.config.config_path}")
                model_config = {}
            
            # Extract model configuration from checkpoint or use defaults
            vit_config_dict = checkpoint.get('model_config', {})
            
            # Merge with config file if available
            if 'vit_config' in model_config:
                vit_config_dict.update(model_config['vit_config'])
            
            # Set number of tags based on checkpoint or tag_names
            if 'num_tags' not in vit_config_dict:
                if self.tag_names:
                    vit_config_dict['num_tags'] = len(self.tag_names)
                    logger.info(f"Setting num_tags={len(self.tag_names)} from tag_names")
                elif 'num_classes' in checkpoint:
                    vit_config_dict['num_tags'] = checkpoint['num_classes']
                    logger.info(f"Setting num_tags={checkpoint['num_classes']} from checkpoint")
                elif 'model_state_dict' in checkpoint:
                    # Try to infer from tag_head dimensions
                    for key, value in checkpoint['model_state_dict'].items():
                        if 'tag_head.weight' in key:
                            vit_config_dict['num_tags'] = value.shape[0]
                            logger.info(f"Inferred num_tags={value.shape[0]} from tag_head.weight")
                            break
                        elif 'tag_head.bias' in key:
                            vit_config_dict['num_tags'] = value.shape[0]
                            logger.info(f"Inferred num_tags={value.shape[0]} from tag_head.bias")
                            break
                else:
                    raise ValueError("Cannot determine num_tags from checkpoint or config")
            
            # Ensure critical parameters are present with sensible defaults
            vit_config_defaults = {
                'image_size': self.config.image_size if self.config.image_size != 448 else 640,
                'patch_size': 16,
                'num_channels': 3,
                'hidden_size': 1280,
                'num_hidden_layers': 24,
                'num_attention_heads': 16,
                'intermediate_size': 5120,
                'num_ratings': 5,
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'layer_norm_eps': 1e-6,
                'use_flash_attention': True,
                'padding_mask_keep_threshold': 0.9,
                'gradient_checkpointing': False  # Disable for inference
            }
            self.patch_size = vit_config_dict.get('patch_size', 16)
            
            # Apply defaults for missing values
            for key, default_value in vit_config_defaults.items():
                if key not in vit_config_dict:
                    vit_config_dict[key] = default_value
                    logger.debug(f"Using default value for {key}: {default_value}")
            
            # Create Vision Transformer model with correct architecture
            vit_config = VisionTransformerConfig(**vit_config_dict)
            self.model = SimplifiedTagger(vit_config)
            
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
            
            logger.info(f"SimplifiedTagger model loaded successfully:")
            logger.info(f"  - Image size: {vit_config.image_size}")
            logger.info(f"  - Patch size: {vit_config.patch_size}")
            logger.info(f"  - Number of tags: {vit_config.num_tags}")
            logger.info(f"  - Number of ratings: {vit_config.num_ratings}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _load_external_vocabulary(self):
        """Load vocabulary from external file (fallback)"""
        if self.config.vocab_path:
            vocab_path = Path(self.config.vocab_path)
        else:
            vocab_path = DEFAULT_VOCAB_PATH

        if not vocab_path.exists():
            search_paths = []
            if self.config.model_path:
                search_paths.append(Path(self.config.model_path).parent / "vocabulary.json")
            if self.config.config_path:
                search_paths.append(Path(self.config.config_path).parent / "vocabulary.json")

            for alt_path in search_paths:
                if alt_path.exists():
                    vocab_path = alt_path
                    logger.info(f"Found vocabulary at: {vocab_path}")
                    break
            else:
                raise FileNotFoundError("Vocabulary file not found! Please use a checkpoint with embedded vocabulary.")

        logger.info(f"Loading external vocabulary from {vocab_path}")
        self.vocabulary = TagVocabulary(vocab_path)
        self.tag_names = [
            self.vocabulary.get_tag_from_index(i)
            for i in range(len(self.vocabulary.tag_to_index))
        ]
        # Compute vocabulary hash for external vocabulary
        self.vocab_sha256 = compute_vocab_sha256(vocab_data={'tag_to_index': self.vocabulary.tag_to_index})

    def _verify_vocabulary(self):
        """Verify vocabulary contains real tags, not placeholders"""
        # Use centralized verification
        verify_vocabulary_integrity(self.vocabulary, Path(self.config.vocab_path))

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference on batch of images"""
        images = images.to(self.device)
        
        if self.use_fp16:
            images = images.half()
        
        outputs = self.model(images)
        if self.config.tta_flip:
            images_flipped = torch.flip(images, dims=[-1])
            outputs_flipped = self.model(images_flipped)
            outputs = 0.5 * (outputs + outputs_flipped)

        # Handle dictionary output from SimplifiedTagger
        if isinstance(outputs, dict):
            tag_outputs = outputs.get('tag_logits', outputs.get('logits'))
            if tag_outputs is None:
                raise ValueError("Model output missing 'tag_logits' or 'logits' key")
            predictions = torch.sigmoid(tag_outputs)
        else:
            predictions = torch.sigmoid(outputs)

        return predictions.cpu().float()


class ResultProcessor:
    """Process and format inference results"""

    def __init__(self, config: InferenceConfig, tag_names: List[str], model_wrapper: ModelWrapper):
        self.config = config
        self.tag_names = tag_names
        self.model_wrapper = model_wrapper
        self._eye_idx = [i for i, t in enumerate(tag_names) if t.endswith("_eyes")]
        self._th_by_idx = None
        if config.thresholds_path:
            try:
                with open(config.thresholds_path, "r", encoding="utf-8") as f:
                    th_map = json.load(f)
                self._th_by_idx = {i: float(th_map.get(t, self.config.threshold)) for i, t in enumerate(tag_names)}
            except Exception as e:
                print(f"Warning: failed to load thresholds: {e}")
        
    def process_predictions(
        self,
        predictions: torch.Tensor,
        image_paths: List[str],
        valid_flags: List[bool]
    ) -> List[ImagePrediction]:
        """Process raw predictions into formatted results"""
        results = []

        for pred, path, valid in zip(predictions, image_paths, valid_flags):
            if not valid:
                result = ImagePrediction(
                    image=path,
                    tags=[],
                    processing_time=None
                )
                results.append(result)
                continue

            if self._th_by_idx:
                above = [i for i in range(pred.numel()) if pred[i].item() > self._th_by_idx.get(i, self.config.threshold)]
            else:
                above = (pred > self.config.threshold).nonzero(as_tuple=False).squeeze(-1).tolist()

            items = [(self.tag_names[i], float(pred[i].item())) for i in above if i < len(self.tag_names)]
            if self.config.eye_color_exclusive and self._eye_idx:
                eye_items = [(t, s, i) for (t, s), i in zip(items, above) if i in self._eye_idx]
                if eye_items:
                    best = max(eye_items, key=lambda x: x[1])
                    items = [(t, s) for (t, s) in items if not t.endswith("_eyes")] + [best[:2]]

            items.sort(key=lambda x: x[1], reverse=True)
            tags = [TagPrediction(name=t, score=s) for t, s in items[: self.config.top_k]]

            result = ImagePrediction(
                image=path,
                tags=tags,
                processing_time=None  # Will be filled by engine
            )
            results.append(result)
        
        return results
    
    def save_results(self, results: List[ImagePrediction], output_path: str,
                     metadata: RunMetadata):
        """Save results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.output_format == 'json':
            output = PredictionOutput(metadata=metadata, results=results)
            output.save(output_path)
        elif self.config.output_format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['# Metadata:', json.dumps(metadata.to_dict())])
                writer.writerow(['image', 'tags', 'scores', 'processing_time'])
                for result in results:
                    tag_names = [t.name for t in result.tags]
                    tag_scores = [str(t.score) for t in result.tags]
                    writer.writerow([
                        result.image,
                        ','.join(tag_names),
                        ','.join(tag_scores),
                        result.processing_time or ''
                    ])
        elif self.config.output_format == 'txt':
            with open(output_path, 'w') as f:
                f.write(f"# Metadata: {json.dumps(metadata.to_dict())}\n")
                for result in results:
                    f.write(f"{result.image}: ")
                    tags = [f"{t.name}({t.score:.2f})" for t in result.tags]
                    f.write(', '.join(tags) + '\n')


class InferenceCache:
    """LRU cache for inference results with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_order = deque()
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None
        self.hits = 0
        self.misses = 0
        self.expired = 0
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, checking TTL if applicable."""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]

                # Check if item has expired
                if self.ttl and (datetime.now() - timestamp > self.ttl):
                    self.expired += 1
                    # Remove expired item
                    del self.cache[key]
                    # This is tricky because key is still in access_order.
                    # We will do a lazy removal.
                    # A better approach might be a background cleanup thread,
                    # but for now, we'll handle it lazily.
                    # To avoid breaking deque.remove(), we will just leave it.
                    # It will be eventually pushed out.
                    self.misses += 1
                    return None

                self.hits += 1
                # Move to end to signify recent use
                self.access_order.remove(key)
                self.access_order.append(key)
                return value

            self.misses += 1
            return None

    def put(self, key: str, value: Any):
        """Put item in cache, evicting if necessary."""
        with self.lock:
            # If key exists, remove it to update its position
            if key in self.cache:
                self.access_order.remove(key)
            # If cache is full, evict the least recently used item
            elif len(self.cache) >= self.max_size and self.max_size > 0:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
            
            # Add the new item
            self.cache[key] = (value, datetime.now())
            self.access_order.append(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            return {
                'hits': self.hits,
                'misses': self.misses,
                'expired': self.expired,
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
            self.model_wrapper.tag_names,
            self.model_wrapper
        )

        # Update preprocessor with loaded normalization params if available
        if self.model_wrapper.normalization_params:
            self.preprocessor = ImagePreprocessor(self.config)        
        
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
            self.cache = InferenceCache(
                max_size=self.config.cache_size,
                ttl_seconds=self.config.cache_ttl_seconds
            )
            logger.info(f"Cache enabled with size {self.config.cache_size} and TTL {self.config.cache_ttl_seconds}s")
    
    def predict_single(self, image: Union[str, np.ndarray, Image.Image]) -> ImagePrediction:
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
            results.processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Cache result
            if self.cache and isinstance(image, str):
                self.cache.put(image, results)
            
            # Log to monitor
            if self.monitor:
                self.monitor.metrics.add_metric('inference_time', results.processing_time / 1000)
                self.monitor.metrics.increment_counter('images_processed')
            
            return results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return ImagePrediction(
                image=image if isinstance(image, str) else "array",
                tags=[],
                processing_time=(time.time() - start_time) * 1000
            )
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[ImagePrediction]:
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
                    result.processing_time = ((time.time() - start_time) / len(images)) * 1000  # ms
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
            return [
                ImagePrediction(
                    image=img if isinstance(img, str) else f"array_{i}",
                    tags=[],
                    processing_time=(time.time() - start_time) * 1000 / len(images)
                )
                for i, img in enumerate(images)
            ]
    
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
            
            # Create metadata
            metadata = RunMetadata(
                top_k=self.config.top_k,
                threshold=self.config.threshold,
                vocab_sha256=self.model_wrapper.vocab_sha256,
                normalize_mean=list(self.config.normalize_mean),
                normalize_std=list(self.config.normalize_std),
                image_size=self.config.image_size,
                patch_size=self.model_wrapper.patch_size,
                model_path=self.config.model_path,
                num_tags=len(self.model_wrapper.tag_names)
            )

            # Save results
            self.result_processor.save_results(all_results, output_path, metadata)
            
            # Calculate statistics
            total_time = time.time() - start_time
            stats = {
                'total_images': len(image_paths),
                'successful': sum(1 for r in all_results if len(r.tags) > 0),
                'failed': sum(1 for r in all_results if len(r.tags) == 0),
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
    # Note: image_size should match the training configuration (640 for ViT, not 448)
    config = InferenceConfig(
        model_path="./checkpoints/best_model.pt",  # Standardize to .pt
        # Explicitly set the vocabulary path
        vocab_path="/media/andrewk/qnap-public/workspace/OppaiOracle/vocabulary.json",
        config_path="./checkpoints/model_config.json",
        batch_size=32,
        threshold=0.5,
        top_k=10,
        enable_monitoring=True,
        enable_cache=True,
        image_size=640  # Match training image size for ViT
    )

    # Example of what should be saved during training (add to training script):
    # When saving checkpoint in training script, include normalization params:
    """
    # In your training script when saving checkpoint:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': vit_config_dict,
        'normalization_params': {
            'mean': [0.5, 0.5, 0.5],  # From SimplifiedDataConfig
            'std': [0.5, 0.5, 0.5]     # From SimplifiedDataConfig
        },
        'tag_names': vocab.tag_names,
        # ... other checkpoint data
    }
    torch.save(checkpoint, 'best_model.pt')  # Standardize to .pt
    """

    # Example of saving model config during training (should be done in training script)
    # This shows what the config file should contain:
    example_model_config = {
        "tag_names": ["tag1", "tag2", "..."],  # List of all tag names
        "normalization_params": {
            "mean": [0.5, 0.5, 0.5],  # Must match training config
            "std": [0.5, 0.5, 0.5]    # Must match training config  
        },
        "vit_config": {
            "image_size": 640,
            "patch_size": 16,
            "num_channels": 3,
            "hidden_size": 1280,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 5120,
            "num_tags": 100000,  # Should match len(tag_names)
            "num_ratings": 5,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "layer_norm_eps": 1e-6,
            "use_flash_attention": True,
            "padding_mask_keep_threshold": 0.9,
            "gradient_checkpointing": False  # Disabled for inference
        }
    }
       
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
