#!/usr/bin/env python3
"""
Configuration System for Anime Image Tagger
Centralized configuration management with validation and persistence
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Type, TypeVar, get_type_hints, get_origin, get_args
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from enum import Enum
import argparse
from datetime import datetime
import copy
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

# Type variable for generic config classes
T = TypeVar('T', bound='BaseConfig')


class ConfigError(Exception):
    """Configuration related errors"""
    pass


class ConfigValidationError(ConfigError):
    """Configuration validation errors"""
    pass


class ConfigType(Enum):
    """Types of configuration files"""
    TRAINING = "training"
    INFERENCE = "inference"
    VALIDATION = "validation"
    EXPORT = "export"
    PREPROCESSING = "preprocessing"
    FULL = "full"


@dataclass
class BaseConfig:
    """Base configuration class with common functionality"""
    
    # Config versioning
    _config_version: str = field(default="1.0.0", init=False, repr=False)
    
    def to_dict(self, exclude_private: bool = True) -> Dict[str, Any]:
        """
        Convert config to dictionary
        
        Args:
            exclude_private: Whether to exclude private fields (starting with _)
        """
        result = {}
        for field_obj in fields(self):
            if exclude_private and field_obj.name.startswith('_'):
                continue
            
            value = getattr(self, field_obj.name)
            if is_dataclass(value):
                value = value.to_dict(exclude_private)
            elif isinstance(value, list):
                value = [v.to_dict(exclude_private) if is_dataclass(v) else v for v in value]
            elif isinstance(value, dict):
                value = {k: v.to_dict(exclude_private) if is_dataclass(v) else v 
                        for k, v in value.items()}
            
            result[field_obj.name] = value
        
        return result
    
    def to_yaml(self, path: Union[str, Path], **kwargs):
        """Save config to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(
                self.to_dict(), 
                f, 
                default_flow_style=False, 
                sort_keys=False,
                **kwargs
            )
        logger.info(f"Saved config to {path}")
    
    def to_json(self, path: Union[str, Path], **kwargs):
        """Save config to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, **kwargs)
        logger.info(f"Saved config to {path}")
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create config from dictionary, handling nesting and type conversion."""
        kwargs = {}
        cls_fields = {f.name: f for f in fields(cls)}
        type_hints = get_type_hints(cls)
        
        for key, value in data.items():
            if key.startswith('_'):  # Skip private fields
                continue
                
            if key not in cls_fields:
                logger.warning(f"Unknown config field '{key}' in {cls.__name__}, skipping")
                continue
            
            field_info = cls_fields[key]
            field_type = type_hints.get(key, field_info.type)
            
            # Handle nested dataclasses
            if is_dataclass(field_type) and isinstance(value, dict):
                kwargs[key] = field_type.from_dict(value)
            # Handle Optional types
            elif get_origin(field_type) is Union:
                args = get_args(field_type)
                if len(args) == 2 and type(None) in args:
                    # This is Optional[T]
                    actual_type = args[0] if args[1] is type(None) else args[1]
                    if value is not None and is_dataclass(actual_type) and isinstance(value, dict):
                        kwargs[key] = actual_type.from_dict(value)
                    else:
                        kwargs[key] = value
                else:
                    kwargs[key] = value
            # Handle Lists with dataclass elements
            elif get_origin(field_type) is list:
                args = get_args(field_type)
                if args and is_dataclass(args[0]) and isinstance(value, list):
                    kwargs[key] = [args[0].from_dict(v) if isinstance(v, dict) else v 
                                  for v in value]
                else:
                    kwargs[key] = value
            # Handle Tuples
            elif get_origin(field_type) is tuple and isinstance(value, (list, tuple)):
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
        
        return cls(**kwargs)
    
    def update(self, updates: Dict[str, Any], validate: bool = True):
        """
        Update config values
        
        Args:
            updates: Dictionary of updates
            validate: Whether to validate after updating
        """
        for key, value in updates.items():
            if hasattr(self, key):
                if not key.startswith('_'):  # Don't update private fields
                    setattr(self, key, value)
            else:
                logger.warning(f"Unknown config field: {key}")
        
        if validate:
            self.validate()
    
    def validate(self):
        """Validate configuration values - override in subclasses"""
        pass
    
    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation
        
        Args:
            path: Dot-separated path (e.g., 'model.hidden_size')
            default: Default value if path doesn't exist
        """
        parts = path.split('.')
        current = self
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
        
        return current
    
    def set_nested(self, path: str, value: Any):
        """
        Set nested configuration value using dot notation
        
        Args:
            path: Dot-separated path (e.g., 'model.hidden_size')
            value: Value to set
        """
        parts = path.split('.')
        current = self
        
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise ConfigError(f"Path not found: {path}")
        
        if hasattr(current, parts[-1]):
            setattr(current, parts[-1], value)
        else:
            raise ConfigError(f"Field not found: {parts[-1]}")
    
    def __eq__(self, other):
        """Check equality based on dict representation"""
        if not isinstance(other, self.__class__):
            return False
        return self.to_dict() == other.to_dict()


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration"""
    # Architecture
    architecture_type: str = "vit_large_extended"
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    intermediate_size: int = 6144
    
    # Vision specific
    image_size: int = 640
    patch_size: int = 14
    num_channels: int = 3
    
    # Special tokens
    use_cls_token: bool = True
    use_style_token: bool = True
    use_line_token: bool = True
    use_color_token: bool = True
    num_special_tokens: int = 4
    
    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    drop_path_rate: float = 0.1
    
    # Initialization
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    
    # Attention
    use_flash_attention: bool = True
    attention_bias: bool = True
    
    # Tag prediction
    num_labels: int = 200000
    num_groups: int = 20
    tags_per_group: int = 10000
    
    # Efficiency
    gradient_checkpointing: bool = False
    
    def validate(self):
        """Validate model configuration"""
        errors = []
        
        if self.hidden_size % self.num_attention_heads != 0:
            errors.append(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        if self.num_labels != self.num_groups * self.tags_per_group:
            errors.append(
                f"num_labels ({self.num_labels}) must equal "
                f"num_groups ({self.num_groups}) * tags_per_group ({self.tags_per_group})"
            )
        
        if self.patch_size > self.image_size:
            errors.append(
                f"patch_size ({self.patch_size}) must be <= image_size ({self.image_size})"
            )
        
        if self.image_size % self.patch_size != 0:
            errors.append(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )
        
        # Validate dropout probabilities
        for prob_name in ['hidden_dropout_prob', 'attention_probs_dropout_prob', 'drop_path_rate']:
            prob_value = getattr(self, prob_name)
            if not 0 <= prob_value <= 1:
                errors.append(f"{prob_name} must be in [0, 1], got {prob_value}")
        
        if errors:
            raise ConfigValidationError("Model config validation failed:\n" + "\n".join(errors))


@dataclass
class StorageLocation:
    """Storage location configuration"""
    path: str
    priority: int
    type: str = "local"  # local, das, nas, s3, gcs
    enabled: bool = True
    
    def validate(self):
        """Validate storage location"""
        if not self.path:
            raise ConfigValidationError("Storage location must have a path")
        
        if self.priority < 0:
            raise ConfigValidationError(f"Priority must be non-negative, got {self.priority}")
        
        valid_types = ["local", "das", "nas", "s3", "gcs"]
        if self.type not in valid_types:
            raise ConfigValidationError(f"Storage type must be one of {valid_types}, got {self.type}")


@dataclass
class DataConfig(BaseConfig):
    """Data loading and preprocessing configuration"""
    # Storage locations
    storage_locations: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"path": "/home/user/datasets/anime_curated", "priority": 0, "type": "local", "enabled": True},
        {"path": "/mnt/das/anime_archive", "priority": 1, "type": "das", "enabled": True},
        {"path": "/mnt/nas/anime_dataset/primary", "priority": 2, "type": "nas", "enabled": True},
        {"path": "/mnt/nas/anime_dataset/video_frames", "priority": 3, "type": "nas", "enabled": True}
    ])
    
    # Paths
    hdf5_dir: str = "/home/user/datasets/teacher_features"
    vocab_dir: str = "/home/user/datasets/vocabulary"
    output_dir: str = "./outputs"
    
    # Image processing
    image_size: int = 640
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    pad_color: Tuple[int, int, int] = (114, 114, 114)
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = False
    
    # Caching
    cache_size_gb: float = 4.0
    preload_files: int = 2
    use_memory_cache: bool = True
    
    # Augmentation
    augmentation_enabled: bool = True
    random_flip_prob: float = 0.5
    color_jitter: bool = True
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    random_rotation_degrees: float = 0.0
    
    def validate(self):
        """Validate data configuration"""
        errors = []
        
        # Validate storage locations
        for i, loc in enumerate(self.storage_locations):
            try:
                storage_loc = StorageLocation(**loc)
                storage_loc.validate()
            except Exception as e:
                errors.append(f"Storage location {i}: {str(e)}")
        
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_workers < 0:
            errors.append(f"num_workers must be non-negative, got {self.num_workers}")
        
        if self.cache_size_gb < 0:
            errors.append(f"cache_size_gb must be non-negative, got {self.cache_size_gb}")
        
        # Validate normalization parameters
        for param_name, param_value in [('normalize_mean', self.normalize_mean), 
                                        ('normalize_std', self.normalize_std)]:
            if len(param_value) != 3:
                errors.append(f"{param_name} must have 3 values, got {len(param_value)}")
        
        # Validate augmentation parameters
        if self.random_flip_prob < 0 or self.random_flip_prob > 1:
            errors.append(f"random_flip_prob must be in [0, 1], got {self.random_flip_prob}")
        
        scale_min, scale_max = self.random_crop_scale
        if not (0 < scale_min <= scale_max <= 1):
            errors.append(f"random_crop_scale must satisfy 0 < min <= max <= 1, got {self.random_crop_scale}")
        
        if errors:
            raise ConfigValidationError("Data config validation failed:\n" + "\n".join(errors))


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration"""
    # Basic settings
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 10000
    warmup_ratio: float = 0.0  # Alternative to warmup_steps
    num_cycles: float = 0.5
    lr_end: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    amp_opt_level: str = "O1"
    amp_dtype: str = "float16"  # float16 or bfloat16
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_steps: int = 5000
    save_total_limit: int = 5
    save_best_only: bool = False
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # Loss configuration
    focal_gamma_pos: float = 1.0  # Updated from 0.0
    focal_gamma_neg: float = 3.0  # Updated from 4.0
    focal_alpha: float = 0.75  # Unified weight for focal loss
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    
    # Curriculum learning
    use_curriculum: bool = True
    start_region_training_epoch: int = 20
    region_training_interval: int = 5
    curriculum_difficulty_schedule: str = "linear"  # linear, exponential, step
    
    # Hardware
    device: str = "cuda"
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    ddp_backend: str = "nccl"
    
    # Tracking
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "anime-tagger"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # Training stability
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001
    
    def validate(self):
        """Validate training configuration"""
        errors = []
        
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.num_epochs <= 0:
            errors.append(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.gradient_accumulation_steps <= 0:
            errors.append(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        
        valid_optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
        if self.optimizer not in valid_optimizers:
            errors.append(f"Unknown optimizer: {self.optimizer}. Must be one of {valid_optimizers}")
        
        valid_schedulers = ["linear", "cosine", "constant", "polynomial", "exponential"]
        if self.scheduler not in valid_schedulers:
            errors.append(f"Unknown scheduler: {self.scheduler}. Must be one of {valid_schedulers}")
        
        # Validate beta values for Adam optimizers
        if self.optimizer in ["adam", "adamw"]:
            if not 0 <= self.adam_beta1 < 1:
                errors.append(f"adam_beta1 must be in [0, 1), got {self.adam_beta1}")
            if not 0 <= self.adam_beta2 < 1:
                errors.append(f"adam_beta2 must be in [0, 1), got {self.adam_beta2}")
        
        # Validate focal loss parameters
        if not 0 <= self.focal_alpha <= 1:
            errors.append(f"focal_alpha must be in [0, 1], got {self.focal_alpha}")
        
        # Validate device
        valid_devices = ["cuda", "cpu", "mps"]
        if not any(self.device.startswith(d) for d in valid_devices):
            errors.append(f"Unknown device: {self.device}. Must start with one of {valid_devices}")
        
        if errors:
            raise ConfigValidationError("Training config validation failed:\n" + "\n".join(errors))


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration"""
    # Model
    model_path: Optional[str] = None
    use_fp16: bool = True
    compile_model: bool = False
    
    # Prediction
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
    remove_underscores: bool = True
    
    # Performance
    use_tensorrt: bool = False
    optimize_for_speed: bool = True
    batch_timeout_ms: int = 100
    max_batch_size: int = 32
    
    # Output
    output_format: str = "json"
    include_scores: bool = True
    score_decimal_places: int = 3
    sort_by_score: bool = True
    
    # API Settings
    enable_api: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_max_image_size: int = 10 * 1024 * 1024  # 10MB
    api_rate_limit: int = 100
    api_cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 1000
    
    def validate(self):
        """Validate inference configuration"""
        errors = []
        
        if self.model_path and not Path(self.model_path).exists():
            logger.warning(f"Model path does not exist: {self.model_path}")
        
        if not 0 <= self.prediction_threshold <= 1:
            errors.append(f"prediction_threshold must be in [0, 1], got {self.prediction_threshold}")
        
        if self.min_predictions > self.max_predictions:
            errors.append(
                f"min_predictions ({self.min_predictions}) > max_predictions ({self.max_predictions})"
            )
        
        if self.min_predictions < 0:
            errors.append(f"min_predictions must be non-negative, got {self.min_predictions}")
        
        valid_formats = ["json", "text", "csv", "xml", "yaml"]
        if self.output_format not in valid_formats:
            errors.append(f"Unknown output_format: {self.output_format}. Must be one of {valid_formats}")
        
        if self.score_decimal_places < 0 or self.score_decimal_places > 10:
            errors.append(f"score_decimal_places must be in [0, 10], got {self.score_decimal_places}")
        
        if self.api_port < 1 or self.api_port > 65535:
            errors.append(f"api_port must be in [1, 65535], got {self.api_port}")
        
        if errors:
            raise ConfigValidationError("Inference config validation failed:\n" + "\n".join(errors))


@dataclass
class ExportConfig(BaseConfig):
    """Model export configuration"""
    # Export format
    export_format: str = "onnx"  # onnx, torchscript, tflite, coreml
    
    # ONNX settings
    opset_version: int = 16
    export_params: bool = True
    do_constant_folding: bool = True
    
    # Dynamic axes
    dynamic_batch_size: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 128
    
    # Optimization
    optimize: bool = True
    optimize_for_mobile: bool = False
    quantize: bool = False
    quantization_type: str = "dynamic"  # dynamic, static, qat
    calibration_dataset_size: int = 100
    
    # Validation
    validate_export: bool = True
    tolerance_rtol: float = 1e-3
    tolerance_atol: float = 1e-5
    num_validation_samples: int = 10
    
    # Metadata
    add_metadata: bool = True
    model_description: str = "Anime Image Tagger Model"
    model_author: str = "AnimeTaggers"
    model_version: str = "1.0.0"
    model_license: str = "MIT"
    
    # Output
    output_path: str = "./exported_model"
    
    def validate(self):
        """Validate export configuration"""
        errors = []
        
        valid_formats = ["onnx", "torchscript", "tflite", "coreml", "tensorrt"]
        if self.export_format not in valid_formats:
            errors.append(f"Unknown export_format: {self.export_format}. Must be one of {valid_formats}")
        
        if self.export_format == "onnx":
            if self.opset_version < 9:
                errors.append(f"opset_version must be >= 9 for ONNX, got {self.opset_version}")
        
        valid_quantization = ["dynamic", "static", "qat"]
        if self.quantization_type not in valid_quantization:
            errors.append(f"Unknown quantization_type: {self.quantization_type}. Must be one of {valid_quantization}")
        
        if self.min_batch_size <= 0:
            errors.append(f"min_batch_size must be positive, got {self.min_batch_size}")
        
        if self.max_batch_size < self.min_batch_size:
            errors.append(
                f"max_batch_size ({self.max_batch_size}) must be >= min_batch_size ({self.min_batch_size})"
            )
        
        if errors:
            raise ConfigValidationError("Export config validation failed:\n" + "\n".join(errors))


@dataclass
class FullConfig(BaseConfig):
    """Complete configuration combining all components"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Global settings
    project_name: str = "anime-image-tagger"
    experiment_name: str = field(default_factory=lambda: f"exp_{datetime.now():%Y%m%d_%H%M%S}")
    output_root: str = "./experiments"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Resource limits
    max_memory_gb: Optional[float] = None
    max_gpu_memory_gb: Optional[float] = None
    
    def validate(self):
        """Validate all sub-configurations and cross-config consistency"""
        errors = []
        
        # Validate each sub-config
        for config_name in ['model', 'data', 'training', 'inference', 'export']:
            try:
                getattr(self, config_name).validate()
            except ConfigValidationError as e:
                errors.append(f"{config_name}: {str(e)}")
        
        # Cross-config validation
        if not errors:  # Only do cross-validation if individual configs are valid
            # Check batch size consistency
            effective_batch = self.data.batch_size * self.training.gradient_accumulation_steps
            if effective_batch > 512:
                logger.warning(f"Large effective batch size ({effective_batch}) may cause memory issues")
            
            # Check image size consistency
            if self.model.image_size != self.data.image_size:
                errors.append(
                    f"Model image_size ({self.model.image_size}) != Data image_size ({self.data.image_size})"
                )
            
            # Check device availability
            if self.training.device.startswith("cuda"):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("CUDA device specified but not available")
                except ImportError:
                    logger.warning("PyTorch not installed, cannot check CUDA availability")
            
            # Validate log level
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.log_level not in valid_log_levels:
                errors.append(f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}")
        
        if errors:
            raise ConfigValidationError("Config validation failed:\n" + "\n".join(errors))


class ConfigManager:
    """Manages configuration loading, saving, and merging"""
    
    def __init__(self, config_type: ConfigType = ConfigType.FULL):
        self.config_type = config_type
        self.config = self._create_default_config()
        self.config_history: List[Dict[str, Any]] = []
        self._env_pattern = re.compile(
            r'^([A-Z0-9_]+)__([A-Z0-9_]+)(?:__(.+))?$',
            re.IGNORECASE
        )
    
    def _create_default_config(self) -> BaseConfig:
        """Create default configuration based on type"""
        config_map = {
            ConfigType.TRAINING: TrainingConfig,
            ConfigType.INFERENCE: InferenceConfig,
            ConfigType.EXPORT: ExportConfig,
            ConfigType.MODEL: ModelConfig,
            ConfigType.PREPROCESSING: DataConfig,
            ConfigType.FULL: FullConfig,
        }
        
        config_class = config_map.get(self.config_type, BaseConfig)
        return config_class()
    
    def load_from_file(self, path: Union[str, Path]) -> BaseConfig:
        """Load configuration from file"""
        path = Path(path)
        
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        
        try:
            # Determine file type and load
            if path.suffix in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
            else:
                raise ConfigError(f"Unknown config file format: {path.suffix}")
            
            # Create config from data
            config_class = type(self.config)
            self.config = config_class.from_dict(data)
            
            # Validate
            self.config.validate()
            
            logger.info(f"Successfully loaded config from {path}")
            return self.config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML file: {e}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse JSON file: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load config: {e}")
    
    def save_to_file(self, path: Union[str, Path], backup: bool = True):
        """
        Save configuration to file
        
        Args:
            path: Output path
            backup: Whether to create backup if file exists
        """
        path = Path(path)
        
        # Create backup if requested and file exists
        if backup and path.exists():
            backup_path = path.with_suffix(f'.backup_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}')
            path.rename(backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on extension
        if path.suffix in ['.yaml', '.yml']:
            self.config.to_yaml(path)
        elif path.suffix == '.json':
            self.config.to_json(path)
        else:
            # Default to YAML
            path = path.with_suffix('.yaml')
            self.config.to_yaml(path)
        
        logger.info(f"Saved config to {path}")
    
    def update_from_env(self, prefix: str = "ANIME_TAGGER_"):
        """
        Update configuration from environment variables
        
        Format: ANIME_TAGGER_<SECTION>__<FIELD>[__<NESTED_FIELD>]
        Example: ANIME_TAGGER_TRAINING__LEARNING_RATE=0.001
        """
        updates = defaultdict(dict)
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            # Remove prefix and parse structure
            config_path = key[len(prefix):].lower()
            parts = config_path.split('__')
            
            if len(parts) < 2:
                logger.warning(f"Invalid env var format: {key}")
                continue
            
            # Parse value
            parsed_value = self._parse_env_value(value)
            
            # Build nested update dictionary
            if len(parts) == 2:
                section, field = parts
                updates[section][field] = parsed_value
            elif len(parts) == 3:
                section, field, subfield = parts
                if section not in updates:
                    updates[section] = {}
                if field not in updates[section]:
                    updates[section][field] = {}
                updates[section][field][subfield] = parsed_value
            else:
                logger.warning(f"Too many nesting levels in env var: {key}")
        
        # Apply updates
        if updates:
            self._apply_nested_updates(self.config, dict(updates))
            self.config.validate()
            logger.info(f"Updated config from {len(updates)} environment sections")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try JSON first (handles lists, dicts, etc.)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments"""
        updates = {}
        
        for key, value in vars(args).items():
            if value is not None and key != 'config' and key != 'output_config':
                updates[key] = value
        
        if updates:
            self._apply_nested_updates(self.config, updates)
            self.config.validate()
            logger.info(f"Updated config from {len(updates)} command line arguments")
    
    def _apply_nested_updates(self, config: Any, updates: Dict[str, Any]):
        """Apply nested updates to configuration"""
        for key, value in updates.items():
            if '.' in key:
                # Handle dot notation
                try:
                    config.set_nested(key, value)
                except ConfigError as e:
                    logger.warning(f"Failed to set {key}: {e}")
            elif isinstance(value, dict) and hasattr(config, key):
                # Recursive update for nested configs
                nested_config = getattr(config, key)
                if isinstance(nested_config, BaseConfig):
                    self._apply_nested_updates(nested_config, value)
                else:
                    setattr(config, key, value)
            else:
                # Direct update
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown config field: {key}")
    
    def merge_configs(self, *configs: BaseConfig) -> BaseConfig:
        """
        Merge multiple configurations (later configs override earlier ones)
        """
        merged_dict = {}
        
        for config in configs:
            deep_update(merged_dict, config.to_dict())
        
        config_class = type(self.config)
        merged = config_class.from_dict(merged_dict)
        merged.validate()
        
        return merged
    
    def get_diff(self, other: Union[BaseConfig, 'ConfigManager']) -> Dict[str, Tuple[Any, Any]]:
        """Get differences between configurations"""
        if isinstance(other, ConfigManager):
            other_config = other.config
        else:
            other_config = other
        
        return self._recursive_diff(self.config.to_dict(), other_config.to_dict())
    
    def _recursive_diff(self, dict1: Dict, dict2: Dict, prefix: str = "") -> Dict[str, Tuple[Any, Any]]:
        """Recursively find differences between two dictionaries"""
        diff = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            if isinstance(val1, dict) and isinstance(val2, dict):
                nested_diff = self._recursive_diff(val1, val2, full_key)
                diff.update(nested_diff)
            elif val1 != val2:
                diff[full_key] = (val1, val2)
        
        return diff
    
    def checkpoint(self, name: str, description: str = ""):
        """Save configuration checkpoint"""
        checkpoint = {
            'name': name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'config': copy.deepcopy(self.config)
        }
        self.config_history.append(checkpoint)
        logger.info(f"Created config checkpoint: {name}")
    
    def restore_checkpoint(self, name: str) -> bool:
        """
        Restore configuration from checkpoint
        
        Returns:
            True if checkpoint was found and restored, False otherwise
        """
        for checkpoint in reversed(self.config_history):
            if checkpoint['name'] == name:
                self.config = copy.deepcopy(checkpoint['config'])
                logger.info(f"Restored config from checkpoint: {name}")
                return True
        
        logger.warning(f"Checkpoint not found: {name}")
        return False
    
    def list_checkpoints(self) -> List[Dict[str, str]]:
        """List all available checkpoints"""
        return [
            {
                'name': cp['name'],
                'description': cp['description'],
                'timestamp': cp['timestamp']
            }
            for cp in self.config_history
        ]


def deep_update(target: Dict, source: Dict) -> Dict:
    """Deep update target dictionary with source dictionary"""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def create_config_parser(config_type: ConfigType = ConfigType.FULL) -> argparse.ArgumentParser:
    """Create argument parser for configuration"""
    parser = argparse.ArgumentParser(
        description="Anime Image Tagger Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General arguments
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output-config', type=str, help='Save final config to file')
    parser.add_argument('--validate-only', action='store_true', help='Only validate config and exit')
    
    # Create subparsers for different config sections
    if config_type == ConfigType.FULL:
        # Model arguments
        model_group = parser.add_argument_group('model')
        model_group.add_argument('--model.architecture_type', type=str, help='Model architecture')
        model_group.add_argument('--model.hidden_size', type=int, help='Hidden size')
        model_group.add_argument('--model.num_hidden_layers', type=int, help='Number of layers')
        model_group.add_argument('--model.num_attention_heads', type=int, help='Number of attention heads')
        
        # Data arguments
        data_group = parser.add_argument_group('data')
        data_group.add_argument('--data.batch_size', type=int, help='Batch size')
        data_group.add_argument('--data.num_workers', type=int, help='Number of data workers')
        data_group.add_argument('--data.image_size', type=int, help='Input image size')
        data_group.add_argument('--data.output_dir', type=str, help='Output directory')
        
        # Training arguments
        train_group = parser.add_argument_group('training')
        train_group.add_argument('--training.num_epochs', type=int, help='Number of epochs')
        train_group.add_argument('--training.learning_rate', type=float, help='Learning rate')
        train_group.add_argument('--training.weight_decay', type=float, help='Weight decay')
        train_group.add_argument('--training.device', type=str, help='Device (cuda/cpu)')
        train_group.add_argument('--training.distributed', action='store_true', help='Use distributed training')
        train_group.add_argument('--training.seed', type=int, help='Random seed')
        
        # Inference arguments
        infer_group = parser.add_argument_group('inference')
        infer_group.add_argument('--inference.model_path', type=str, help='Path to model')
        infer_group.add_argument('--inference.prediction_threshold', type=float, help='Prediction threshold')
        infer_group.add_argument('--inference.top_k', type=int, help='Top-k predictions')
        infer_group.add_argument('--inference.filter_nsfw', action='store_true', help='Filter NSFW tags')
        
        # Export arguments
        export_group = parser.add_argument_group('export')
        export_group.add_argument('--export.export_format', type=str, help='Export format')
        export_group.add_argument('--export.optimize', action='store_true', help='Optimize exported model')
        export_group.add_argument('--export.quantize', action='store_true', help='Quantize model')
    
    return parser


def load_config(
    config_file: Optional[str] = None,
    config_type: ConfigType = ConfigType.FULL,
    args: Optional[argparse.Namespace] = None,
    env_prefix: str = "ANIME_TAGGER_",
    validate: bool = True
) -> BaseConfig:
    """
    Load configuration from multiple sources
    
    Priority: args > env > file > defaults
    
    Args:
        config_file: Path to configuration file
        config_type: Type of configuration
        args: Command line arguments
        env_prefix: Environment variable prefix
        validate: Whether to validate the final config
        
    Returns:
        Loaded and validated configuration
    """
    manager = ConfigManager(config_type)
    
    # Load from file if provided
    if config_file:
        try:
            manager.load_from_file(config_file)
        except ConfigError as e:
            logger.error(f"Failed to load config file: {e}")
            raise
    
    # Update from environment
    manager.update_from_env(env_prefix)
    
    # Update from args
    if args:
        manager.update_from_args(args)
    
    # Validate final config
    if validate:
        try:
            manager.config.validate()
        except ConfigValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    return manager.config


def generate_example_configs(output_dir: Path = Path("./config_examples")):
    """Generate example configuration files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = {
        "full_config.yaml": FullConfig(),
        "training_config.yaml": TrainingConfig(),
        "inference_config.yaml": InferenceConfig(),
        "export_config.yaml": ExportConfig(),
        "model_config.yaml": ModelConfig(),
        "data_config.yaml": DataConfig(),
    }
    
    # Save each config
    for filename, config in configs.items():
        config.to_yaml(output_dir / filename)
    
    # Create specialized configs
    # Minimal training config
    minimal_train = {
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "device": "cuda",
        "use_amp": True,
    }
    with open(output_dir / "minimal_training.yaml", 'w') as f:
        yaml.dump(minimal_train, f, default_flow_style=False)
    
    # High performance inference config
    hp_inference = InferenceConfig(
        use_fp16=True,
        use_tensorrt=True,
        optimize_for_speed=True,
        compile_model=True,
        batch_timeout_ms=50,
        max_batch_size=64,
    )
    hp_inference.to_yaml(output_dir / "high_performance_inference.yaml")
    
    # Mobile export config
    mobile_export = ExportConfig(
        export_format="tflite",
        optimize=True,
        optimize_for_mobile=True,
        quantize=True,
        quantization_type="static",
    )
    mobile_export.to_yaml(output_dir / "mobile_export.yaml")
    
    logger.info(f"Generated example configs in {output_dir}")


if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            # Generate example configs
            output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./config_examples")
            generate_example_configs(output_dir)
        elif sys.argv[1] == "validate":
            # Validate a config file
            if len(sys.argv) < 3:
                print("Usage: python Configuration_System.py validate <config_file>")
                sys.exit(1)
            
            try:
                manager = ConfigManager(ConfigType.FULL)
                config = manager.load_from_file(sys.argv[2])
                print(f"✓ Config file '{sys.argv[2]}' is valid")
            except Exception as e:
                print(f"✗ Config validation failed: {e}")
                sys.exit(1)
        else:
            print("Unknown command. Use 'generate' or 'validate'")
            sys.exit(1)
    else:
        # Run tests
        print("Testing Enhanced Configuration System...")
        print("=" * 60)
        
        # Test 1: Create and validate default config
        print("\n1. Testing default configuration creation...")
        config = FullConfig()
        try:
            config.validate()
            print("   ✓ Default config is valid")
        except Exception as e:
            print(f"   ✗ Validation failed: {e}")
        
        # Test 2: Save and load config
        print("\n2. Testing save/load functionality...")
        test_file = Path("test_config.yaml")
        config.to_yaml(test_file)
        
        manager = ConfigManager(ConfigType.FULL)
        loaded = manager.load_from_file(test_file)
        
        if config == loaded:
            print("   ✓ Config saved and loaded correctly")
        else:
            print("   ✗ Loaded config doesn't match original")
        
        # Test 3: Environment variable override
        print("\n3. Testing environment variable override...")
        os.environ["ANIME_TAGGER_TRAINING__LEARNING_RATE"] = "0.0005"
        os.environ["ANIME_TAGGER_MODEL__NUM_GROUPS"] = "25"
        os.environ["ANIME_TAGGER_DATA__BATCH_SIZE"] = "64"
        
        manager.update_from_env()
        
        assert manager.config.training.learning_rate == 0.0005
        assert manager.config.model.num_groups == 25
        assert manager.config.data.batch_size == 64
        print("   ✓ Environment overrides work correctly")
        
        # Test 4: Config diff
        print("\n4. Testing config diff functionality...")
        config2 = FullConfig()
        config2.training.num_epochs = 200
        config2.model.hidden_size = 2048
        
        manager2 = ConfigManager(ConfigType.FULL)
        manager2.config = config2
        
        diff = manager.get_diff(manager2)
        print(f"   Found {len(diff)} differences")
        for key, (val1, val2) in list(diff.items())[:3]:
            print(f"   - {key}: {val1} → {val2}")
        
        # Test 5: Checkpointing
        print("\n5. Testing checkpoint functionality...")
        manager.checkpoint("before_changes", "Testing checkpoint")
        manager.config.training.num_epochs = 500
        
        success = manager.restore_checkpoint("before_changes")
        if success and manager.config.training.num_epochs != 500:
            print("   ✓ Checkpoint restore successful")
        else:
            print("   ✗ Checkpoint restore failed")
        
        # Clean up
        test_file.unlink()
        for key in list(os.environ.keys()):
            if key.startswith("ANIME_TAGGER_"):
                del os.environ[key]
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")