#!/usr/bin/env python3
"""
Inference Engine for Anime Image Tagger
Handles model inference, batch processing, and real-time predictions
Uses Monitor_log.py for monitoring and logging functionality
"""
from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, TYPE_CHECKING, NamedTuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import warnings
import traceback
from contextlib import contextmanager
from collections import defaultdict, deque, OrderedDict
import threading
import queue
import argparse

import gzip
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from training_utils import CheckpointManager
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from PIL import Image, ImageOps, ImageFile

# Vocabulary utilities
from model_metadata import ModelMetadata
from vocabulary import TagVocabulary, load_vocabulary_for_training, verify_vocabulary_integrity
from schemas import TagPrediction, ImagePrediction, RunMetadata, PredictionOutput, compute_vocab_sha256
from Configuration_System import load_config, InferenceConfig as BaseInferenceConfig, MonitorConfig, FullConfig
from orientation_handler import OrientationHandler  # orientation-aware TTA mapping

# Make cv2 optional - not needed for basic inference
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CV2_AVAILABLE = False
    warnings.warn("OpenCV (cv2) not available. Some image loading features may be limited.")

# Import monitoring system from Monitor_log (optional)
try:
    from Monitor_log import TrainingMonitor  # MonitorConfig is provided by Configuration_System
    MONITORING_AVAILABLE = True
except ImportError:
    warnings.warn("Monitor_log not available. Monitoring will be disabled.")
    MONITORING_AVAILABLE = False
    TrainingMonitor = None  # type: ignore[assignment]

# Hint to static type checkers without importing at runtime
if TYPE_CHECKING:
    from Configuration_System import MonitorConfig as _MC  # re-export for checkers
    MonitorConfig = _MC

# Import the actual model architecture
try:
    from model_architecture import SimplifiedTagger, VisionTransformerConfig
except ImportError:
    raise ImportError("model_architecture.py not found. Cannot load SimplifiedTagger model.")   

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "vocabulary.json"
ORIENTATION_MAP_PATH = PROJECT_ROOT / "configs" / "orientation_map.json"

# Legacy image size constant - older models used 448x448 images.
# If config specifies this size, we assume it's a legacy config and default to 512.
_LEGACY_IMAGE_SIZE = 448



@dataclass
class InferenceConfig(BaseInferenceConfig):
    """Inference settings derived from the central configuration system."""
    # Match TrainingConfig for memory layout
    memory_format: str = "contiguous"  # or "channels_last"

    vocab_path: Optional[str] = str(DEFAULT_VOCAB_PATH)
    model_path: str = "./checkpoints/best_model.pt"
    config_path: str = "./checkpoints/model_config.json"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    image_size: int = 512
    normalize_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    normalize_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    use_torch_compile: bool = False
    thresholds_path: Optional[str] = None
    eye_color_exclusive: bool = False
    tta_flip: bool = False
    max_queue_size: int = 100
    timeout: float = 30.0
    enable_monitoring: bool = True
    monitor_config: Optional["MonitorConfig"] = None
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    save_visualizations: bool = False
    visualization_dir: str = "./visualizations"
    input_image_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])
    enable_profiling: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.device not in ("cuda", "cpu") and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be 'cuda', 'cpu', or 'cuda:N', got {self.device}")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"CUDA requested but not available, falling back to CPU")
            object.__setattr__(self, 'device', 'cpu')
        if self.cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got {self.cache_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

    @property
    def threshold(self) -> float:
        return self.prediction_threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.prediction_threshold = value


def load_inference_config(path: Path = PROJECT_ROOT / "configs" / "inference_config.yaml") -> InferenceConfig:
    config = InferenceConfig()
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    pp = cfg.get("preprocessing") or {}
    rt = cfg.get("runtime") or {}
    post = cfg.get("postprocessing") or {}
    io = cfg.get("io") or {}
    inp = cfg.get("input") or {}
    updates = {
        "image_size": pp.get("image_size"),
        "normalize_mean": pp.get("normalize_mean"),
        "normalize_std": pp.get("normalize_std"),
        "device": rt.get("device"),
        "tta_flip": rt.get("tta_flip"),
        "prediction_threshold": post.get("threshold"),
        "top_k": post.get("top_k"),
        "output_format": io.get("output_format"),
        "save_visualizations": io.get("save_visualizations"),
        "visualization_dir": io.get("visualization_dir"),
        "input_image_extensions": inp.get("image_extensions"),
    }
    for key, value in updates.items():
        if value is not None:
            setattr(config, key, value)

    try:
        unified_config = load_config(PROJECT_ROOT / "configs" / "unified_config.yaml")
        if unified_config and unified_config.inference:
            config.precision = unified_config.inference.precision
    except Exception as e:
        logger.warning(f"Could not load settings from unified_config.yaml: {e}")

    return config


class ImagePreprocessor:
    """Handles image preprocessing for inference"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.transform = self._build_transform()
        # Define allowed base directories for path validation
        # Use current working directory if input_dir not specified
        self.allowed_dirs = []
        if hasattr(config, 'input_dir') and config.input_dir:
            self.allowed_dirs.append(Path(config.input_dir).resolve())
        else:
            # Warn when using permissive default - security concern for production
            logger.warning(
                "No input_dir configured. Using current working directory as allowed path. "
                "Set input_dir in config for stricter path validation in production."
            )
            self.allowed_dirs.append(Path.cwd().resolve())

    def _build_transform(self):
        """Build tensor + normalize transform (geometry handled manually to avoid stretching/upscale)."""
        return transforms_v2.Compose([
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])

    def _validate_image_path(self, path: str) -> Path:
        """Validate image path to prevent path traversal.

        Args:
            path: String path to image file

        Returns:
            Validated absolute Path object

        Raises:
            ValueError: If path is unsafe or invalid
        """
        try:
            # Resolve to absolute path
            abs_path = Path(path).resolve()

            # Check if path is in allowed directories
            is_allowed = any(
                abs_path.is_relative_to(allowed)
                for allowed in self.allowed_dirs
            )

            if not is_allowed:
                raise ValueError(
                    f"Image path {path} is outside allowed directories"
                )

            # Check file exists
            if not abs_path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            # Check it's a file, not a directory
            if not abs_path.is_file():
                raise ValueError(f"Path is not a file: {path}")

            # Check extension
            if abs_path.suffix.lower() not in self.config.input_image_extensions:
                raise ValueError(
                    f"Invalid image extension: {abs_path.suffix}. "
                    f"Allowed: {self.config.input_image_extensions}"
                )

            return abs_path

        except Exception as e:
            logger.error(f"Invalid image path {path}: {e}")
            raise

    @staticmethod
    def _letterbox_downscale_only(
        img: Image.Image,
        target: int,
        pad_color=(114, 114, 114),
        return_mask: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, torch.Tensor]]:
        """Letterbox to square target without upscaling.

        - Preserves aspect ratio
        - Downscales if larger than target
        - Never upscales; pads to target with pad_color

        Args:
            img: Input PIL Image
            target: Target square size
            pad_color: RGB tuple for padding color
            return_mask: If True, also returns padding mask (True=PAD)

        Returns:
            If return_mask=False: letterboxed Image
            If return_mask=True: (letterboxed Image, padding mask tensor)
        """
        w, h = img.size
        if w <= 0 or h <= 0:
            canvas = Image.new("RGB", (target, target), tuple(pad_color))
            if return_mask:
                # Entire image is padding
                pmask = torch.ones(target, target, dtype=torch.bool)
                return canvas, pmask
            return canvas

        ratio = min(target / float(w), target / float(h))
        scale = min(1.0, ratio)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((nw, nh), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
        canvas = Image.new("RGB", (target, target), tuple(pad_color))
        left = (target - nw) // 2
        top = (target - nh) // 2
        canvas.paste(resized, (left, top))

        if return_mask:
            # Create padding mask: True where padding, False where actual image
            pmask = torch.ones(target, target, dtype=torch.bool)
            pmask[top:top + nh, left:left + nw] = False
            return canvas, pmask

        return canvas
    
    def preprocess_image(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Preprocess a single image.

        Args:
            image: Input image (path, numpy array, or PIL Image)
            return_mask: If True, also returns padding mask for flex attention

        Returns:
            If return_mask=False: preprocessed tensor (C, H, W)
            If return_mask=True: (preprocessed tensor, padding mask (H, W) with True=PAD)
        """
        # Load image if path
        if isinstance(image, str):
            # Validate path to prevent path traversal
            validated_path = self._validate_image_path(image)
            with Image.open(validated_path) as img:
                img.load()
                img = ImageOps.exif_transpose(img)

                # Handle transparency by compositing onto neutral gray
                if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
                    rgba = img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, (114, 114, 114))
                    alpha = rgba.getchannel("A")
                    bg.paste(rgba, mask=alpha)
                    img = bg
                else:
                    img = img.convert('RGB')

                # Assign after all processing is done
                image = img
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Geometry: letterbox to target without upscaling, no stretching
        if return_mask:
            lb, pmask = self._letterbox_downscale_only(
                image, int(self.config.image_size), return_mask=True
            )
            # Apply tensor + normalize
            return self.transform(lb), pmask
        else:
            lb = self._letterbox_downscale_only(image, int(self.config.image_size))
            # Apply tensor + normalize
            return self.transform(lb)
    
    def preprocess_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        return_masks: bool = False
    ) -> Union[Tuple[torch.Tensor, List[bool]], Tuple[torch.Tensor, torch.Tensor, List[bool]]]:
        """Preprocess a batch of images.

        Args:
            images: List of input images
            return_masks: If True, also returns stacked padding masks

        Returns:
            If return_masks=False: (tensor [N, 3, H, W], valid flags list)
            If return_masks=True: (tensor [N, 3, H, W], masks [N, H, W], valid flags list)
        """
        processed = []
        masks = []
        valid_flags = []

        for img in images:
            try:
                if return_masks:
                    tensor, pmask = self.preprocess_image(img, return_mask=True)
                    processed.append(tensor)
                    masks.append(pmask)
                else:
                    processed.append(self.preprocess_image(img))
                valid_flags.append(True)
            except (IOError, OSError, ValueError, RuntimeError) as e:
                logger.error(f"Failed to preprocess image {img}: {e}")
                # Add black image as placeholder to maintain batch shape
                processed.append(torch.zeros(3, self.config.image_size, self.config.image_size))
                if return_masks:
                    # Full padding mask for invalid images
                    masks.append(torch.ones(self.config.image_size, self.config.image_size, dtype=torch.bool))
                valid_flags.append(False)

        if return_masks:
            return torch.stack(processed), torch.stack(masks), valid_flags
        return torch.stack(processed), valid_flags


class DatasetItem(NamedTuple):
    """Item returned by InferenceDataset"""
    image: torch.Tensor
    path: str
    is_valid: bool


class InferenceDataset(Dataset):
    """Dataset for batch inference"""

    def __init__(self, image_paths: List[str], preprocessor: ImagePreprocessor):
        self.image_paths = image_paths
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> DatasetItem:
        """Load and preprocess an image.

        Returns:
            DatasetItem: Contains image tensor, path, and validity flag.
                        If loading fails, returns zero tensor with is_valid=False.
                        Consumers MUST check is_valid flag!
        """
        path = self.image_paths[idx]
        try:
            image = self.preprocessor.preprocess_image(path)
            return DatasetItem(image, path, True)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return DatasetItem(
                torch.zeros(3, self.preprocessor.config.image_size,
                           self.preprocessor.config.image_size),
                path,
                False
            )


def inference_collate_fn(batch):
    """Custom collate function for inference dataset.

    Properly handles DatasetItem namedtuples to ensure batch_valid
    is a list of booleans instead of a tuple.
    """
    images = torch.stack([item.image for item in batch])
    paths = [item.path for item in batch]
    valid_flags = [item.is_valid for item in batch]
    return images, paths, valid_flags


class ModelWrapper:
    """Wrapper for the trained model with optimization"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.tag_names = []
        self.vocabulary = None  # Loaded vocabulary
        self.normalization_params = None
        self.vocab_sha256 = "unknown"  # Will be computed when vocabulary is loaded
        self.patch_size = 16  # Default, will be updated from model config

    def load_model(self, config: "FullConfig"):
        """Load the trained model"""
        self.config = config.inference
        try:
            # Load checkpoint first
            model_path = Path(self.config.model_path)
            checkpoint_dir = model_path.parent
            manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
            checkpoint = manager.load_checkpoint(checkpoint_path=str(model_path))
            
            if not checkpoint:
                raise FileNotFoundError(f"Could not load checkpoint from {model_path}")

            state_dict = checkpoint.pop('state_dict')
            meta = checkpoint # The rest is meta

            # Priority 1: Check for embedded vocabulary in checkpoint
            if 'vocab_b64_gzip' in meta:
                logger.info("Loading embedded vocabulary from checkpoint")
                vocab_data = ModelMetadata.extract_vocabulary(meta)
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
                    # Track vocabulary source for verification
                    self._vocab_source = "embedded"
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

            # Precompute left↔right index map once for TTA if enabled
            self._tta_index_map = None
            if getattr(self.config, "tta_flip", False) and self.tag_names:
                try:
                    orientation_path = getattr(getattr(config, 'data', None), 'orientation_map_path', None)
                    map_path = Path(orientation_path) if orientation_path else None
                    if map_path and map_path.exists():
                        oh = OrientationHandler(mapping_file=map_path)
                        mapping = oh.precompute_all_mappings(set(self.tag_names))
                        index_map: List[int] = []
                        for tag in self.tag_names:
                            swapped = mapping.get(tag, tag)
                            index_map.append(self.vocabulary.tag_to_index.get(swapped, self.vocabulary.tag_to_index[tag]))
                        self._tta_index_map = torch.as_tensor(index_map, dtype=torch.long, device=self.device)
                    else:
                        logger.warning("TTA flip enabled but orientation_map_path not configured; left/right remap disabled.")
                except Exception as e:
                    logger.warning(f"Failed to build TTA left↔right index map: {e}")

            # Load preprocessing parameters from checkpoint
            # Store original config values for mismatch detection
            config_mean = list(self.config.normalize_mean)
            config_std = list(self.config.normalize_std)
            config_image_size = self.config.image_size

            if 'preprocessing_params' in meta:
                preprocessing = ModelMetadata.extract_preprocessing_params(meta)
                if preprocessing:
                    checkpoint_mean = preprocessing.get('normalize_mean', [0.5, 0.5, 0.5])
                    checkpoint_std = preprocessing.get('normalize_std', [0.5, 0.5, 0.5])
                    checkpoint_image_size = preprocessing.get('image_size', 512)

                    # Warn if user config differs from checkpoint (potential accuracy issue)
                    if config_mean != list(checkpoint_mean):
                        logger.warning(
                            f"Normalization mean mismatch! Config: {config_mean}, Checkpoint: {checkpoint_mean}. "
                            f"Using checkpoint values for correct inference."
                        )
                    if config_std != list(checkpoint_std):
                        logger.warning(
                            f"Normalization std mismatch! Config: {config_std}, Checkpoint: {checkpoint_std}. "
                            f"Using checkpoint values for correct inference."
                        )
                    if config_image_size != checkpoint_image_size:
                        logger.warning(
                            f"Image size mismatch! Config: {config_image_size}, Checkpoint: {checkpoint_image_size}. "
                            f"Using checkpoint values for correct inference."
                        )

                    self.config.normalize_mean = checkpoint_mean
                    self.config.normalize_std = checkpoint_std
                    self.config.image_size = checkpoint_image_size
                    self.normalization_params = preprocessing
                    logger.info(f"Loaded preprocessing params from checkpoint: {preprocessing}")
            elif 'normalization_params' in meta:
                # Legacy format
                self.normalization_params = meta['normalization_params']
                self.config.normalize_mean = self.normalization_params['mean']
                self.config.normalize_std = self.normalization_params['std']
                logger.info("Loaded normalization params from checkpoint (legacy format)")
            else:
                logger.warning("Preprocessing params not found in checkpoint. Using config defaults.")

            # Load model config
            vit_config_dict = config.model.to_dict()

            # Set number of tags based on the loaded vocabulary
            if self.tag_names:
                vit_config_dict['num_tags'] = len(self.tag_names)
                logger.info(f"Setting num_tags={len(self.tag_names)} from tag_names")
            elif 'num_tags' not in vit_config_dict:
                raise ValueError("Cannot determine num_tags from vocabulary or config")

            # Ensure critical parameters are present with sensible defaults
            vit_config_defaults = {
                'image_size': self.config.image_size if self.config.image_size != _LEGACY_IMAGE_SIZE else 512,
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
                'use_flex_attention': True,
                'flex_block_size': 128,
                'attention_bias': True,
                'token_ignore_threshold': 0.9,
                'use_fp32_layernorm': False,  # Inference doesn't need FP32 LayerNorm
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

            # Load weights with explicit strict checking
            # Note: load_state_dict(strict=True) returns None on success, raises RuntimeError on mismatch
            try:
                self.model.load_state_dict(state_dict, strict=True)
                logger.info("Model state dict loaded successfully (strict=True)")

            except RuntimeError as e:
                logger.error(f"Failed to load model state dict: {e}")
                logger.error(f"Model architecture may not match checkpoint")
                logger.error(f"Checkpoint path: {self.config.model_path}")
                logger.error(f"Expected num_tags: {vit_config.num_tags}")
                raise ValueError(
                    f"Model architecture mismatch. Cannot load checkpoint from {self.config.model_path}. "
                    f"This usually means the model architecture has changed since the checkpoint was created."
                ) from e
            
            self.model = self.model.to(self.device)
            if getattr(self.config, "memory_format", "contiguous") == "channels_last":
                self.model = self.model.to(memory_format=torch.channels_last)
            self.model.eval()
            
            # Optimize model
            if self.config.use_torch_compile and hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
            
            # Setup mixed precision with fallback support
            precision = str(getattr(self.config, "precision", "bf16")).lower()
            if precision in {"bf16", "bfloat16"}:
                if self.config.device == 'cuda':
                    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                        self.model = self.model.to(torch.bfloat16)
                        logger.info("Using bf16 for inference.")
                    else:
                        logger.warning("bf16 not supported on this GPU, falling back to fp16")
                        self.model = self.model.to(torch.float16)
                else:
                    # CPU inference with bf16
                    self.model = self.model.to(torch.bfloat16)
                    logger.info("Using bf16 for CPU inference.")
            elif precision in {"fp16", "float16"}:
                self.model = self.model.to(torch.float16)
                logger.info("Using fp16 for inference.")
            elif precision in {"fp32", "float32"}:
                # Model is already fp32 by default, no conversion needed
                logger.info("Using fp32 for inference.")
            else:
                logger.warning(f"Unknown precision '{precision}', defaulting to bf16")
                self.model = self.model.to(torch.bfloat16)
            
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
        vocab_path = Path(self.config.vocab_path)

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
        # Track vocabulary source for verification
        self._vocab_source = str(vocab_path)

    def _verify_vocabulary(self):
        """Verify vocabulary contains real tags, not placeholders"""
        # Use the tracked vocabulary source for accurate error messages
        vocab_source = getattr(self, '_vocab_source', None)
        if vocab_source == "embedded":
            source_desc = "embedded vocabulary from checkpoint"
        elif vocab_source:
            source_desc = vocab_source
        else:
            source_desc = str(Path(self.config.vocab_path))
        # Use centralized verification
        verify_vocabulary_integrity(self.vocabulary, source_desc)

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run inference on batch of images.

        Args:
            images: Batch of images (N, C, H, W)
            padding_mask: Optional padding mask (N, H, W) with True=PAD semantics.
                         Enables proper flex attention masking for letterboxed images.

        Returns:
            Prediction probabilities (N, num_tags)
        """
        # Match model precision and device in a single transfer
        # Using non_blocking=False to ensure transfer completes before model forward
        model_dtype = next(self.model.parameters()).dtype
        memory_format = (torch.channels_last
                        if getattr(self.config, "memory_format", "contiguous") == "channels_last"
                        else torch.contiguous_format)
        images = images.to(
            device=self.device,
            dtype=model_dtype,
            memory_format=memory_format,
            non_blocking=False  # Ensure transfer completes before model.forward()
        )

        # Transfer padding mask to device if provided
        if padding_mask is not None:
            padding_mask = padding_mask.to(device=self.device, dtype=torch.bool)

        outputs = self.model(images, padding_mask=padding_mask)
        if self.config.tta_flip:
            images_flipped = torch.flip(images, dims=[-1])
            # Flip the padding mask horizontally for TTA (if provided)
            padding_mask_flipped = torch.flip(padding_mask, dims=[-1]) if padding_mask is not None else None
            outputs_flipped = self.model(images_flipped, padding_mask=padding_mask_flipped)
            # Orientation-aware averaging if we have a mapping
            if isinstance(outputs, dict):
                if 'tag_logits' in outputs and self._tta_index_map is not None:
                    # Use advanced indexing to reorder flipped outputs
                    # self._tta_index_map[i] = index of tag in flipped image that corresponds to tag i in original
                    batch_size = outputs_flipped['tag_logits'].shape[0]

                    # Create batch indices
                    batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)

                    # Reorder flipped predictions using index map
                    # tags_f[b, i] = outputs_flipped[b, index_map[i]]
                    tags_f = outputs_flipped['tag_logits'][batch_idx, self._tta_index_map.unsqueeze(0)]

                    # Average original and reordered flipped
                    outputs['tag_logits'] = 0.5 * (outputs['tag_logits'] + tags_f)

                    # Ratings are orientation-invariant; average directly if present
                    if 'rating_logits' in outputs and 'rating_logits' in outputs_flipped:
                        outputs['rating_logits'] = 0.5 * (outputs['rating_logits'] + outputs_flipped['rating_logits'])
                else:
                    # Fallback: elementwise average common keys
                    try:
                        for k in outputs:
                            outputs[k] = 0.5 * (outputs[k] + outputs_flipped[k])
                    except Exception as e:
                        logger.warning(f"TTA averaging failed for key '{k}', using original outputs: {e}")
            else:
                # outputs is a tensor – simple average
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

    def _get_special_token_indices(self, vocab) -> tuple:
        """Get PAD and UNK token indices with proper error handling.

        Returns:
            tuple: (pad_idx, unk_idx)
        """
        # Default indices (standard convention)
        default_pad = 0
        default_unk = 1

        if vocab is None:
            logger.warning("No vocabulary available, using default special token indices")
            return default_pad, default_unk

        try:
            # Try to get PAD token
            pad_token = getattr(vocab, "pad_token", "<PAD>")
            if not isinstance(pad_token, str):
                logger.warning(f"pad_token is not a string: {type(pad_token)}, using default")
                pad_idx = default_pad
            else:
                pad_idx = vocab.tag_to_index.get(pad_token, default_pad)
                if pad_idx != default_pad:
                    logger.debug(f"Using non-standard PAD index: {pad_idx}")

            # Try to get UNK token
            # First check if vocabulary has explicit unk_index
            if hasattr(vocab, "unk_index") and isinstance(vocab.unk_index, int):
                unk_idx = vocab.unk_index
            else:
                # Fall back to looking up token
                unk_token = getattr(vocab, "unk_token", "<UNK>")
                if not isinstance(unk_token, str):
                    logger.warning(f"unk_token is not a string: {type(unk_token)}, using default")
                    unk_idx = default_unk
                else:
                    unk_idx = vocab.tag_to_index.get(unk_token, default_unk)
                    if unk_idx != default_unk:
                        logger.debug(f"Using non-standard UNK index: {unk_idx}")

            return pad_idx, unk_idx

        except AttributeError as e:
            logger.error(f"Vocabulary missing required attributes: {e}")
            return default_pad, default_unk
        except KeyError as e:
            logger.error(f"Vocabulary lookup failed: {e}")
            return default_pad, default_unk
        except Exception as e:
            # This should not happen - log and investigate
            logger.error(
                f"Unexpected error getting special token indices: {e}",
                exc_info=True
            )
            return default_pad, default_unk

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
                # Use torch.where for forward-compatibility and simpler 1-D indexing
                above = torch.where(pred > self.config.threshold)[0].tolist()

            # Drop special tokens (<PAD>=0, <UNK>=1) regardless of score
            vocab = getattr(self.model_wrapper, "vocabulary", None)
            pad_idx, unk_idx = self._get_special_token_indices(vocab)
            above = [i for i in above if i not in (pad_idx, unk_idx)]

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
        self.cache = OrderedDict()  # Maintains insertion order
        self.timestamps = {}  # Separate dict for timestamps
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None

        # Separate locks for data and stats to reduce contention
        self.cache_lock = threading.RLock()  # For cache operations
        self.stats_lock = threading.Lock()   # For stats updates

        # Stats with protected access
        self._hits = 0
        self._misses = 0
        self._expired = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, checking TTL if applicable.

        Thread-safe implementation that avoids nested lock acquisition
        by separating cache operations from stats updates.
        """
        # Perform cache lookup with cache_lock only
        result_status = None  # 'hit', 'miss', or 'expired'
        value = None

        with self.cache_lock:
            if key not in self.cache:
                result_status = 'miss'
            elif self.ttl:
                timestamp = self.timestamps[key]
                if datetime.now() - timestamp > self.ttl:
                    del self.cache[key]
                    del self.timestamps[key]
                    result_status = 'expired'
                else:
                    value = self.cache[key]
                    self.cache.move_to_end(key)
                    result_status = 'hit'
            else:
                value = self.cache[key]
                self.cache.move_to_end(key)
                result_status = 'hit'

        # Update stats outside cache_lock to avoid nested lock contention
        with self.stats_lock:
            if result_status == 'miss':
                self._misses += 1
            elif result_status == 'expired':
                self._expired += 1
                self._misses += 1
            else:  # hit
                self._hits += 1

        return value

    def put(self, key: str, value: Any):
        """Put item in cache, evicting if necessary."""
        with self.cache_lock:
            # If key exists, move to end
            if key in self.cache:
                self.cache.move_to_end(key)
            # If cache is full, evict the least recently used item
            elif len(self.cache) >= self.max_size and self.max_size > 0:
                oldest_key = next(iter(self.cache))  # First key
                del self.cache[oldest_key]
                if oldest_key in self.timestamps:
                    del self.timestamps[oldest_key]

            # Add the new item
            self.cache[key] = value
            self.timestamps[key] = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.stats_lock:
            hits = self._hits
            misses = self._misses
            expired = self._expired

        with self.cache_lock:
            size = len(self.cache)

        total = hits + misses
        return {
            'hits': hits,
            'misses': misses,
            'expired': expired,
            'hit_rate': hits / total if total > 0 else 0,
            'size': size
        }

    def clear(self):
        """Thread-safe cache clearing."""
        with self.cache_lock:
            self.cache.clear()
            self.timestamps.clear()


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
        # Track resources created for cleanup on failure
        monitor_created = None

        try:
            # Load full config for model initialization
            try:
                full_config = load_config(PROJECT_ROOT / "configs" / "unified_config.yaml")
                # Override inference config with our config
                full_config.inference = self.config
            except Exception as e:
                logger.error(f"Failed to load unified_config.yaml: {e}")
                raise RuntimeError(f"Cannot load configuration: {e}")

            # Load model
            if not self.model_wrapper.load_model(full_config):
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
                monitor_config = None
                try:
                    # Assuming unified_config.yaml is the single source of truth
                    unified_config_path = PROJECT_ROOT / "configs" / "unified_config.yaml"
                    if unified_config_path.exists():
                        unified_config = load_config(unified_config_path)
                        if hasattr(unified_config, 'monitor'):
                            monitor_config = unified_config.monitor
                            logger.info("Loaded monitor configuration from unified_config.yaml")
                except Exception as e:
                    logger.warning(f"Could not load monitor settings from unified_config.yaml: {e}")

                if monitor_config is None:
                    monitor_config = self.config.monitor_config or MonitorConfig(
                        log_level="INFO",
                        use_tensorboard=False,
                        use_wandb=False,
                        track_gpu_metrics=True,
                        enable_alerts=False,
                        alert_webhook_url=None  # Ensure webhook is not used by default
                    )
                    logger.info("Using default or legacy monitor configuration for inference.")

                monitor_created = TrainingMonitor(monitor_config)
                self.monitor = monitor_created
                logger.info("Monitoring enabled for inference")

            # Setup cache
            if self.config.enable_cache:
                self.cache = InferenceCache(
                    max_size=self.config.cache_size,
                    ttl_seconds=self.config.cache_ttl_seconds
                )
                logger.info(f"Cache enabled with size {self.config.cache_size} and TTL {self.config.cache_ttl_seconds}s")

        except Exception:
            # Clean up monitor if it was created before failure
            if monitor_created is not None:
                try:
                    monitor_created.close()
                    logger.info("Cleaned up monitor after setup failure")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to cleanup monitor: {cleanup_err}")
            raise
    
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
            # Preprocess with padding mask for proper flex attention masking
            processed, pmask = self.preprocessor.preprocess_image(image, return_mask=True)
            processed = processed.unsqueeze(0)  # Add batch dimension
            pmask = pmask.unsqueeze(0)  # Add batch dimension

            # Predict with padding mask
            predictions = self.model_wrapper.predict(processed, padding_mask=pmask)
            
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
                # Preprocess batch with padding masks for proper flex attention masking
                processed_tensor, padding_masks, valid_flags = self.preprocessor.preprocess_batch(
                    uncached_images, return_masks=True
                )

                # Predict with padding masks
                predictions = self.model_wrapper.predict(processed_tensor, padding_mask=padding_masks)

                # Process results - USE ACTUAL VALIDITY FLAGS
                batch_results = self.result_processor.process_predictions(
                    predictions,
                    [img if isinstance(img, str) else f"array_{i}"
                     for i, img in enumerate(uncached_images)],
                    valid_flags  # Use actual flags, not always True!
                )

                # Add timing and cache
                for i, (idx, result) in enumerate(zip(uncached_indices, batch_results)):
                    result.processing_time = ((time.time() - start_time) / len(images)) * 1000  # ms
                    results[idx] = result

                    # Only cache valid results
                    if self.cache and isinstance(uncached_images[i], str) and valid_flags[i]:
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
            logger.error(f"Batch inference failed: {e}", exc_info=True)
            # Preserve partial results and fill in missing with empty predictions
            partial_count = sum(1 for r in results if r is not None)
            if partial_count > 0:
                logger.info(f"Returning {partial_count} partial results out of {len(images)}")
            for i, img in enumerate(images):
                if results[i] is None:
                    results[i] = ImagePrediction(
                        image=img if isinstance(img, str) else f"array_{i}",
                        tags=[],
                        processing_time=(time.time() - start_time) * 1000 / len(images)
                    )
            return results
    
    def process_directory(
        self,
        directory: str,
        output_path: str,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process all images in a directory"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.webp']
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
        # Build worker kwargs safely: only enable multiprocessing knobs when workers > 0
        _worker_kwargs = {
            "num_workers": self.config.num_workers,
            "pin_memory": getattr(self.config, "pin_memory", False),
            # persistent_workers is multiprocessing-only; disable when workers == 0
            "persistent_workers": (
                getattr(self.config, "persistent_workers", False)
                if getattr(self.config, "num_workers", 0) > 0
                else False
            ),
        }
        if getattr(self.config, "num_workers", 0) > 0:
            pf = getattr(self.config, "prefetch_factor", None)
            if pf is not None:
                _worker_kwargs["prefetch_factor"] = pf

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=inference_collate_fn,  # Use custom collate function
            **_worker_kwargs,
        )

        # Process batches
        all_results = []
        start_time = time.time()

        try:
            for batch_images, batch_paths, batch_valid in dataloader:
                # Run inference
                predictions = self.model_wrapper.predict(batch_images)

                # Process results - batch_valid is already a list
                batch_results = self.result_processor.process_predictions(
                    predictions,
                    batch_paths,
                    batch_valid  # Already a list from custom collate function
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
        # Close monitor first
        if self.monitor:
            try:
                self.monitor.close()
            except Exception as e:
                logger.warning(f"Error closing monitor: {e}")
            self.monitor = None

        # Clear cache and log stats
        if self.cache:
            try:
                logger.info(f"Cache stats at cleanup: {self.cache.get_stats()}")
                self.cache.clear()  # Thread-safe clearing
            except Exception as e:
                logger.warning(f"Error clearing cache: {e}")
            self.cache = None

        # Move model to CPU and delete
        if hasattr(self, 'model_wrapper') and self.model_wrapper:
            try:
                if hasattr(self.model_wrapper, 'model') and self.model_wrapper.model:
                    # Move to CPU first - must reassign to release GPU memory
                    self.model_wrapper.model = self.model_wrapper.model.cpu()
                    # Delete model
                    del self.model_wrapper.model
                del self.model_wrapper
            except Exception as e:
                logger.warning(f"Error cleaning up model: {e}")
            self.model_wrapper = None

        # Clear any cached preprocessed images
        if hasattr(self, 'preprocessor') and self.preprocessor:
            self.preprocessor = None

        # Clear result processor
        if hasattr(self, 'result_processor') and self.result_processor:
            self.result_processor = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if using GPU
        if self.config.device == "cuda":
            try:
                torch.cuda.empty_cache()
                # Synchronize to ensure all operations complete
                torch.cuda.synchronize()
                logger.info("GPU memory released")
            except Exception as e:
                logger.warning(f"Error clearing CUDA cache: {e}")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained model")
    parser.add_argument(
        "--model",
        default="./checkpoints/best_model.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        default="./checkpoints/model_config.json",
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--vocab",
        default=str(DEFAULT_VOCAB_PATH),
        help="Path to vocabulary JSON file",
    )
    args = parser.parse_args()

    # Configure inference
    # Load defaults from configuration files and override with CLI args
    config = load_inference_config()
    config.model_path = args.model
    config.vocab_path = args.vocab
    config.config_path = args.config

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
            "image_size": 512,
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
            "use_flex_attention": True,
            "flex_block_size": 128,
            "attention_bias": True,
            "token_ignore_threshold": 0.9,
            "use_fp32_layernorm": False,
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
