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
from typing import Dict, List, Optional, Tuple, Union, Any, Type
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
import argparse
from datetime import datetime
import copy
import re

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration related errors"""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def to_yaml(self, path: Union[str, Path]):
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: Union[str, Path]):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary"""
        # Filter out unknown fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def update(self, updates: Dict[str, Any]):
        """Update config values"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown config field: {key}")
    
    def validate(self):
        """Validate configuration values"""
        # Override in subclasses
        pass


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
        if self.hidden_size % self.num_attention_heads != 0:
            raise ConfigError(f"hidden_size ({self.hidden_size}) must be divisible by "
                            f"num_attention_heads ({self.num_attention_heads})")
        
        if self.num_labels != self.num_groups * self.tags_per_group:
            raise ConfigError(f"num_labels ({self.num_labels}) must equal "
                            f"num_groups ({self.num_groups}) * tags_per_group ({self.tags_per_group})")
        
        if self.patch_size > self.image_size:
            raise ConfigError(f"patch_size ({self.patch_size}) must be <= image_size ({self.image_size})")


@dataclass
class DataConfig(BaseConfig):
    """Data loading and preprocessing configuration"""
    # Storage locations
    storage_locations: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"path": "/home/user/datasets/anime_curated", "priority": 0, "type": "local"},
        {"path": "/mnt/das/anime_archive", "priority": 1, "type": "das"},
        {"path": "/mnt/nas/anime_dataset/primary", "priority": 2, "type": "nas"},
        {"path": "/mnt/nas/anime_dataset/video_frames", "priority": 3, "type": "nas"}
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
    
    # Caching
    cache_size_gb: float = 4.0
    preload_files: int = 2
    
    # Augmentation
    augmentation_enabled: bool = True
    random_flip_prob: float = 0.5
    color_jitter: bool = True
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    
    def validate(self):
        """Validate data configuration"""
        for loc in self.storage_locations:
            if 'path' not in loc or 'priority' not in loc:
                raise ConfigError("Each storage location must have 'path' and 'priority'")
        
        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_workers < 0:
            raise ConfigError(f"num_workers must be non-negative, got {self.num_workers}")


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
    num_cycles: float = 0.5
    
    # Mixed precision
    use_amp: bool = True
    amp_opt_level: str = "O1"
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_steps: int = 5000
    save_total_limit: int = 5
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # Loss configuration
    focal_gamma_pos: float = 0.0
    focal_gamma_neg: float = 4.0
    focal_alpha_pos: float = 1.0
    focal_alpha_neg: float = 1.0
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    
    # Distillation
    use_distillation: bool = True
    distillation_alpha: float = 0.7
    distillation_temperature: float = 3.0
    anime_teacher_weight: float = 0.7
    clip_teacher_weight: float = 0.3
    
    # Curriculum learning
    use_curriculum: bool = True
    start_region_training_epoch: int = 20
    region_training_interval: int = 5
    
    # Hardware
    device: str = "cuda"
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Tracking
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "anime-tagger"
    wandb_run_name: Optional[str] = None
    
    # Seed
    seed: int = 42
    
    def validate(self):
        """Validate training configuration"""
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.num_epochs <= 0:
            raise ConfigError(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.gradient_accumulation_steps <= 0:
            raise ConfigError(f"gradient_accumulation_steps must be positive")
        
        if self.optimizer not in ["adam", "adamw", "sgd"]:
            raise ConfigError(f"Unknown optimizer: {self.optimizer}")
        
        if self.scheduler not in ["linear", "cosine", "constant", "polynomial"]:
            raise ConfigError(f"Unknown scheduler: {self.scheduler}")


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration"""
    # Model
    model_path: Optional[str] = None
    use_fp16: bool = True
    
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
    
    # Performance
    use_tensorrt: bool = False
    optimize_for_speed: bool = True
    
    # Output
    output_format: str = "json"
    include_scores: bool = True
    score_decimal_places: int = 3
    
    # API
    enable_api: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_max_image_size: int = 10 * 1024 * 1024
    api_rate_limit: int = 100
    
    def validate(self):
        """Validate inference configuration"""
        if self.prediction_threshold < 0 or self.prediction_threshold > 1:
            raise ConfigError(f"prediction_threshold must be in [0, 1], got {self.prediction_threshold}")
        
        if self.min_predictions > self.max_predictions:
            raise ConfigError(f"min_predictions ({self.min_predictions}) > "
                            f"max_predictions ({self.max_predictions})")
        
        if self.output_format not in ["json", "text", "csv"]:
            raise ConfigError(f"Unknown output_format: {self.output_format}")


@dataclass
class ExportConfig(BaseConfig):
    """Model export configuration"""
    # ONNX settings
    opset_version: int = 16
    export_params: bool = True
    do_constant_folding: bool = True
    
    # Dynamic axes
    dynamic_batch_size: bool = True
    
    # Optimization
    optimize: bool = True
    optimize_for_mobile: bool = False
    quantize: bool = False
    quantization_type: str = "dynamic"
    
    # Validation
    validate_export: bool = True
    tolerance_rtol: float = 1e-3
    tolerance_atol: float = 1e-5
    
    # Metadata
    add_metadata: bool = True
    model_description: str = "Anime Image Tagger Model"
    model_author: str = "AnimeTaggers"
    model_version: str = "1.0"
    
    def validate(self):
        """Validate export configuration"""
        if self.opset_version < 9:
            raise ConfigError(f"opset_version must be >= 9, got {self.opset_version}")
        
        if self.quantization_type not in ["dynamic", "static", "qat"]:
            raise ConfigError(f"Unknown quantization_type: {self.quantization_type}")


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
    
    def validate(self):
        """Validate all sub-configurations"""
        self.model.validate()
        self.data.validate()
        self.training.validate()
        self.inference.validate()
        self.export.validate()
        
        # Cross-config validation
        if self.data.batch_size * self.training.gradient_accumulation_steps > 512:
            logger.warning("Very large effective batch size may cause memory issues")
        
        if self.model.image_size != self.data.image_size:
            raise ConfigError(f"Model image_size ({self.model.image_size}) != "
                            f"Data image_size ({self.data.image_size})")


class ConfigManager:
    """Manages configuration loading, saving, and merging"""
    
    def __init__(self, config_type: ConfigType = ConfigType.FULL):
        self.config_type = config_type
        self.config = self._create_default_config()
        self.config_history = []
    
    def _create_default_config(self) -> BaseConfig:
        """Create default configuration based on type"""
        if self.config_type == ConfigType.TRAINING:
            return TrainingConfig()
        elif self.config_type == ConfigType.INFERENCE:
            return InferenceConfig()
        elif self.config_type == ConfigType.EXPORT:
            return ExportConfig()
        elif self.config_type == ConfigType.FULL:
            return FullConfig()
        else:
            return BaseConfig()
    
    def load_from_file(self, path: Union[str, Path]) -> BaseConfig:
        """Load configuration from file"""
        path = Path(path)
        
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        
        # Determine file type
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
        
        logger.info(f"Loaded config from {path}")
        return self.config
    
    def save_to_file(self, path: Union[str, Path]):
        """Save configuration to file"""
        path = Path(path)
        
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
        """Update configuration from environment variables"""
        updates = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert ANIME_TAGGER_MODEL__HIDDEN_SIZE to model.hidden_size
                config_key = key[len(prefix):].lower()
                config_key = config_key.replace('__', '.')
                
                # Parse value
                try:
                    # Try to parse as JSON first (for lists, dicts)
                    parsed_value = json.loads(value)
                except:
                    # Try to parse as number
                    try:
                        if '.' in value:
                            parsed_value = float(value)
                        else:
                            parsed_value = int(value)
                    except:
                        # Keep as string
                        parsed_value = value
                        
                        # Convert string booleans
                        if value.lower() == 'true':
                            parsed_value = True
                        elif value.lower() == 'false':
                            parsed_value = False
                
                updates[config_key] = parsed_value
        
        # Apply updates
        self._apply_nested_updates(self.config, updates)
        
        if updates:
            logger.info(f"Updated config from environment: {list(updates.keys())}")
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments"""
        updates = {}
        
        for key, value in vars(args).items():
            if value is not None:
                updates[key] = value
        
        # Apply updates
        self._apply_nested_updates(self.config, updates)
        
        if updates:
            logger.info(f"Updated config from args: {list(updates.keys())}")
    
    def _apply_nested_updates(self, config: Any, updates: Dict[str, Any]):
        """Apply nested updates to configuration"""
        for key, value in updates.items():
            if '.' in key:
                # Nested update
                parts = key.split('.')
                current = config
                
                # Navigate to nested object
                for part in parts[:-1]:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        logger.warning(f"Unknown config path: {key}")
                        break
                else:
                    # Set final value
                    if hasattr(current, parts[-1]):
                        setattr(current, parts[-1], value)
                    else:
                        logger.warning(f"Unknown config field: {key}")
            else:
                # Direct update
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown config field: {key}")
    
    def merge_configs(self, *configs: BaseConfig) -> BaseConfig:
        """Merge multiple configurations"""
        merged_dict = {}
        
        for config in configs:
            merged_dict.update(config.to_dict())
        
        config_class = type(self.config)
        merged = config_class.from_dict(merged_dict)
        merged.validate()
        
        return merged
    
    def get_diff(self, other: BaseConfig) -> Dict[str, Tuple[Any, Any]]:
        """Get differences between configurations"""
        diff = {}
        
        self_dict = self.config.to_dict()
        other_dict = other.to_dict()
        
        all_keys = set(self_dict.keys()) | set(other_dict.keys())
        
        for key in all_keys:
            self_val = self_dict.get(key)
            other_val = other_dict.get(key)
            
            if self_val != other_val:
                diff[key] = (self_val, other_val)
        
        return diff
    
    def checkpoint(self, name: str):
        """Save configuration checkpoint"""
        checkpoint = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'config': copy.deepcopy(self.config)
        }
        self.config_history.append(checkpoint)
        logger.info(f"Created config checkpoint: {name}")
    
    def restore_checkpoint(self, name: str):
        """Restore configuration from checkpoint"""
        for checkpoint in reversed(self.config_history):
            if checkpoint['name'] == name:
                self.config = copy.deepcopy(checkpoint['config'])
                logger.info(f"Restored config from checkpoint: {name}")
                return
        
        raise ConfigError(f"Checkpoint not found: {name}")


def create_config_parser(config_type: ConfigType = ConfigType.FULL) -> argparse.ArgumentParser:
    """Create argument parser for configuration"""
    parser = argparse.ArgumentParser(description="Anime Image Tagger Configuration")
    
    # General arguments
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output-config', type=str, help='Save final config to file')
    
    if config_type in [ConfigType.FULL, ConfigType.TRAINING]:
        # Training arguments
        parser.add_argument('--num-epochs', type=int, help='Number of training epochs')
        parser.add_argument('--learning-rate', type=float, help='Learning rate')
        parser.add_argument('--batch-size', type=int, help='Batch size')
        parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
        parser.add_argument('--distributed', action='store_true', help='Use distributed training')
        parser.add_argument('--output-dir', type=str, help='Output directory')
    
    if config_type in [ConfigType.FULL, ConfigType.INFERENCE]:
        # Inference arguments
        parser.add_argument('--model-path', type=str, help='Path to model')
        parser.add_argument('--threshold', type=float, help='Prediction threshold')
        parser.add_argument('--top-k', type=int, help='Top-k predictions')
        parser.add_argument('--filter-nsfw', action='store_true', help='Filter NSFW tags')
    
    if config_type in [ConfigType.FULL, ConfigType.EXPORT]:
        # Export arguments
        parser.add_argument('--opset-version', type=int, help='ONNX opset version')
        parser.add_argument('--optimize', action='store_true', help='Optimize exported model')
        parser.add_argument('--quantize', action='store_true', help='Quantize model')
    
    return parser


def load_config(
    config_file: Optional[str] = None,
    config_type: ConfigType = ConfigType.FULL,
    args: Optional[argparse.Namespace] = None,
    env_prefix: str = "ANIME_TAGGER_"
) -> BaseConfig:
    """
    Load configuration from multiple sources
    
    Priority: args > env > file > defaults
    
    Args:
        config_file: Path to configuration file
        config_type: Type of configuration
        args: Command line arguments
        env_prefix: Environment variable prefix
        
    Returns:
        Loaded and validated configuration
    """
    manager = ConfigManager(config_type)
    
    # Load from file if provided
    if config_file:
        manager.load_from_file(config_file)
    
    # Update from environment
    manager.update_from_env(env_prefix)
    
    # Update from args
    if args:
        manager.update_from_args(args)
    
    # Validate final config
    manager.config.validate()
    
    return manager.config


def generate_example_configs(output_dir: Path = Path("./config_examples")):
    """Generate example configuration files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full config
    full_config = FullConfig()
    full_config.to_yaml(output_dir / "full_config.yaml")
    
    # Training config
    train_config = TrainingConfig()
    train_config.to_yaml(output_dir / "training_config.yaml")
    
    # Inference config
    inference_config = InferenceConfig()
    inference_config.to_yaml(output_dir / "inference_config.yaml")
    
    # Minimal training config
    minimal_train = {
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "device": "cuda"
    }
    with open(output_dir / "minimal_training.yaml", 'w') as f:
        yaml.dump(minimal_train, f)
    
    # High performance inference config
    hp_inference = {
        "use_fp16": True,
        "use_tensorrt": True,
        "optimize_for_speed": True,
        "batch_size": 64,
        "adaptive_threshold": True
    }
    with open(output_dir / "high_performance_inference.yaml", 'w') as f:
        yaml.dump(hp_inference, f)
    
    logger.info(f"Generated example configs in {output_dir}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Generate example configs
        generate_example_configs()
    else:
        # Test configuration system
        print("Testing configuration system...")
        
        # Create default config
        config = FullConfig()
        print(f"\nDefault config created with {len(fields(config))} top-level fields")
        
        # Save to YAML
        config.to_yaml("test_config.yaml")
        print("Saved to test_config.yaml")
        
        # Load and validate
        manager = ConfigManager(ConfigType.FULL)
        loaded = manager.load_from_file("test_config.yaml")
        print("Loaded and validated config")
        
        # Test environment override
        os.environ["ANIME_TAGGER_TRAINING__LEARNING_RATE"] = "0.0001"
        os.environ["ANIME_TAGGER_MODEL__NUM_GROUPS"] = "25"
        manager.update_from_env()
        print(f"Updated from env: LR={manager.config.training.learning_rate}, "
              f"Groups={manager.config.model.num_groups}")
        
        # Clean up
        os.unlink("test_config.yaml")
        print("\nConfiguration system test completed!")
