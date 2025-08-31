#!/usr/bin/env python3
"""
Configuration System for Anime Image Tagger
Centralized configuration management with validation and persistence
"""

import os
import json
import yaml
import typing as _typing
from dataclasses import replace  # import for with_updates
try:
    import types as _types
    # Python 3.10 introduced types.UnionType to represent ``X | Y`` unions.
    # It aliases to typing.Union in Python 3.14+, but older versions return
    # a distinct class. Capture it if available to allow cross-version checks.
    _UnionType = getattr(_types, "UnionType", None)
except Exception:
    _UnionType = None

def _is_union_origin(origin: _typing.Any) -> bool:
    """Return True if the given origin comes from a typing.Union or PEP 604 UnionType.

    Python 3.10 introduced the ``|`` operator to build union types. Prior to
    Python 3.14, ``get_origin(int | str)`` returns ``types.UnionType`` instead
    of ``typing.Union``【719174400860031†L79-L90】【165362151616795†L1114-L1147】.  On Python 3.14+
    both syntaxes produce a plain ``typing.Union``.  Checking both allows
    code to detect unions across Python versions.

    Args:
        origin: the result of ``typing.get_origin(some_type)``.

    Returns:
        True if the origin represents a union type, False otherwise.
    """
    return (origin is _typing.Union) or (_UnionType is not None and origin is _UnionType)
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Type, TypeVar, get_type_hints, get_origin, get_args
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from enum import Enum
import argparse
import sys
from datetime import datetime
import copy
import time
from collections import defaultdict
import re
import warnings

# Import sensitive defaults
try:
    from sensitive_config import ALERT_WEBHOOK_URL
except ImportError:  # pragma: no cover - fallback when file missing
    ALERT_WEBHOOK_URL = None

logger = logging.getLogger(__name__)

# Type variable for generic config classes
T = TypeVar('T', bound='BaseConfig')

CONFIG_VERSION = "2.0.0"  # Bumped for unified config format


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
    MODEL = "model"
    EXPORT = "export"
    PREPROCESSING = "preprocessing"
    FULL = "full"


def _object_to_dict(obj: Any, exclude_private: bool = True, preserve_tuples: bool = True) -> Any:
    """Recursively convert an object to a serializable structure.

    Unlike ``dataclasses.asdict`` this helper preserves tuple types by
    converting tuple items individually rather than coercing the tuple to a
    list. It also respects ``exclude_private`` when encountering nested
    dataclass instances.

    Args:
        obj: The object to convert.
        exclude_private: Whether to omit private fields from nested dataclasses.

    Returns:
        A structure composed of ``dict``, ``list``, ``tuple`` and primitive
        types suitable for serialization.
    """
    if is_dataclass(obj):
        # Use obj.to_dict for nested dataclasses; do not pass preserve_tuples because the dataclass
        # methods are responsible for handling their own private fields.
        return obj.to_dict(exclude_private)  # type: ignore[attr-defined]
    if isinstance(obj, dict):
        return {k: _object_to_dict(v, exclude_private, preserve_tuples) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_object_to_dict(v, exclude_private, preserve_tuples) for v in obj]
    if isinstance(obj, tuple):
        # When preserve_tuples=False, emit lists to avoid !!python/tuple tags【752494560202949†L1177-L1181】.
        mapped = [_object_to_dict(v, exclude_private, preserve_tuples) for v in obj]
        return tuple(mapped) if preserve_tuples else mapped
    return obj


@dataclass
class BaseConfig:
    """Base configuration class with common functionality"""
    
    # Config versioning
    _config_version: str = field(default=CONFIG_VERSION, init=False, repr=False)
    
    def to_dict(self, exclude_private: bool = True, *, preserve_tuples: bool = True) -> Dict[str, Any]:
        """
        Convert config to dictionary
        
        Args:
            exclude_private: Whether to exclude private fields (starting with _)
        """
        result: Dict[str, Any] = {}
        for field_obj in fields(self):
            if exclude_private and field_obj.name.startswith('_'):
                continue

            value = getattr(self, field_obj.name)
            result[field_obj.name] = _object_to_dict(value, exclude_private, preserve_tuples)

        return result

    def to_yaml(self, path: Union[str, Path], *, sort_keys: bool = False, preserve_tuples: bool = False, **kwargs) -> None:
        """Save config to a YAML file.

        By default this method emits standard YAML by using ``yaml.safe_dump`` and
        converting tuples to lists when ``preserve_tuples`` is ``False``. Using
        ``safe_dump`` avoids emitting Python-specific tags like ``!!python/tuple``
        which can cause compatibility issues with other tools【246116543212867†L1701-L1712】【752494560202949†L1177-L1181】.

        Args:
            path: destination path where YAML will be written.
            sort_keys: whether to sort dictionary keys in the output.
            preserve_tuples: if True, tuple values are preserved in the emitted YAML;
                if False, tuples are converted to lists for portability.
            **kwargs: additional arguments passed through to ``yaml.safe_dump``.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                self.to_dict(exclude_private=True, preserve_tuples=preserve_tuples),
                f,
                sort_keys=sort_keys,
                default_flow_style=False,
                **kwargs
            )
        logger.info(f"Saved config to {p}")
    
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
            # Handle Optional and Union types including PEP 604 ``X | Y`` unions
            elif _is_union_origin(get_origin(field_type)):
                args = get_args(field_type)
                non_none = [t for t in args if t is not type(None)]
                # If the incoming value is None, treat it as the Optional case and set None
                if value is None:
                    kwargs[key] = None
                else:
                    last_err = None
                    assigned = False
                    # Try each type variant until one succeeds. For dataclasses,
                    # recursively call from_dict; for primitives, cast directly.
                    for t in non_none:
                        try:
                            if is_dataclass(t) and isinstance(value, dict):
                                kwargs[key] = t.from_dict(value)  # type: ignore[attr-defined]
                            else:
                                # If this variant itself is another typing construct (e.g. list[int]),
                                # fall back to assigning the raw value and let later validation handle it.
                                origin_t = get_origin(t)
                                if origin_t is not None:
                                    kwargs[key] = value
                                else:
                                    kwargs[key] = t(value)
                            assigned = True
                            break
                        except Exception as e:
                            last_err = e
                    if not assigned:
                        # None of the variants matched; raise a useful error
                        raise TypeError(f"Cannot coerce {key}={value!r} to {field_type}") from last_err
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
    
    def update(self, updates: Dict[str, Any], validate: bool = True) -> None:
        """
        Update configuration values in-place with validation and type coercion.

        Unknown keys are rejected to avoid silently corrupting the configuration. Values
        for nested dataclasses and Union/Optional types are coerced using the same
        rules as ``from_dict``. When ``validate`` is True the updated instance is
        validated via ``self.validate()``.
        """
        allowed = {f.name: f.type for f in fields(self)}
        # Identify unknown keys to prevent silently adding invalid attributes
        unknown = set(updates) - set(allowed)
        if unknown:
            raise AttributeError(f"Unknown config field(s): {sorted(unknown)}")
        for name, raw in updates.items():
            ftype = allowed[name]
            # Coerce nested dataclasses
            if is_dataclass(ftype) and isinstance(raw, dict):
                coerced = ftype.from_dict(raw)  # type: ignore[attr-defined]
            else:
                origin = get_origin(ftype)
                # Handle Union/Optional types using the helper
                if _is_union_origin(origin):
                    args = get_args(ftype)
                    if raw is None:
                        coerced = None
                    else:
                        last_err = None
                        assigned = False
                        for t in args:
                            if t is type(None):
                                continue
                            try:
                                if is_dataclass(t) and isinstance(raw, dict):
                                    coerced = t.from_dict(raw)  # type: ignore[attr-defined]
                                else:
                                    # If nested typing constructs, leave raw value
                                    if get_origin(t) is not None:
                                        coerced = raw
                                    else:
                                        coerced = t(raw)
                                assigned = True
                                break
                            except Exception as e:
                                last_err = e
                        if not assigned:
                            raise TypeError(f"Cannot coerce {name}={raw!r} to {ftype}") from last_err
                else:
                    coerced = raw
            setattr(self, name, coerced)
        if validate:
            self.validate()

    def with_updates(self, **kwargs):
        """Return a new instance with validated updates applied.

        This method constructs a new dataclass instance by validating and coercing
        the provided keyword arguments against the field types. Unknown keys
        are rejected. Under the hood it uses ``dataclasses.replace`` which
        raises ``TypeError`` if a field name does not exist【752035413495800†L502-L506】.

        Returns:
            A new instance of the same type with updated fields.
        """
        allowed = {f.name: f.type for f in fields(self)}
        unknown = set(kwargs) - set(allowed)
        if unknown:
            raise AttributeError(f"Unknown config field(s): {sorted(unknown)}")
        validated: Dict[str, Any] = {}
        for name, ftype in allowed.items():
            if name in kwargs:
                raw = kwargs[name]
                if is_dataclass(ftype) and isinstance(raw, dict):
                    validated[name] = ftype.from_dict(raw)  # type: ignore[attr-defined]
                else:
                    origin = get_origin(ftype)
                    if _is_union_origin(origin):
                        args = get_args(ftype)
                        if raw is None:
                            validated[name] = None
                        else:
                            last_err = None
                            assigned = False
                            for t in args:
                                if t is type(None):
                                    continue
                                try:
                                    if is_dataclass(t) and isinstance(raw, dict):
                                        validated[name] = t.from_dict(raw)  # type: ignore[attr-defined]
                                    else:
                                        if get_origin(t) is not None:
                                            validated[name] = raw
                                        else:
                                            validated[name] = t(raw)
                                    assigned = True
                                    break
                                except Exception as e:
                                    last_err = e
                            if not assigned:
                                raise TypeError(f"Cannot coerce {name}={raw!r} to {ftype}") from last_err
                    else:
                        validated[name] = raw
        return replace(self, **validated)
    
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
    patch_size: int = 16
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
    
    # Masking
    token_ignore_threshold: float = 0.9  # Fraction of padding pixels to ignore token
    
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
        for prob_name in ['hidden_dropout_prob', 'attention_probs_dropout_prob']:
            prob_value = getattr(self, prob_name)
            if not 0 <= prob_value <= 1:
                errors.append(f"{prob_name} must be in [0, 1], got {prob_value}")

        if not 0 <= self.drop_path_rate < 1:
            errors.append(f"drop_path_rate must be in [0, 1), got {self.drop_path_rate}")
        elif self.drop_path_rate > 0.5:
            warnings.warn(
                f"drop_path_rate={self.drop_path_rate} is unusually high; "
                "values between 0.08 and 0.3 are typical for ViT."
            )
        
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
    storage_locations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Paths
    data_dir: str = "./data"
    hdf5_dir: str = "/home/user/datasets/teacher_features"
    vocab_dir: str = "/home/user/datasets/vocabulary"
    output_dir: str = "./outputs"
    
    # Image processing
    image_size: int = 640
    # Default to inception-style normalization which expects pixel values in
    # [0, 1] range. Using 0.5/0.5/0.5 keeps training, validation and inference
    # in sync unless explicitly overridden in the unified config.
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    pad_color: Tuple[int, int, int] = (114, 114, 114)
    
    # Data loading
    batch_size: int = 60
    num_workers: int = 12
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = False
    
    # Caching
    cache_size_gb: float = 105.0
    l1_per_worker_mb: int = 256
    preload_files: int = 2
    use_memory_cache: bool = True
    
    # Augmentation
    augmentation_enabled: bool = True
    random_flip_prob: float = 0.35
    color_jitter: bool = True
    color_jitter_brightness: float = 0.10
    color_jitter_contrast:   float = 0.10
    color_jitter_hue:        float = 0.03  # cap hue to protect eye-color semantics
    color_jitter_saturation: float = 0.00
    eye_color_weight_boost: float = 1.5  # Boost for eye color tags in sampling
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    random_rotation_degrees: float = 0.0

    # Advanced Augmentation (set alpha/p to 0.0 to disable)
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    random_erasing_p: float = 0.0

    # Orientation mapping (consolidated from augmentation.yaml)
    orientation_map_path: Optional[str] = None
    strict_orientation_validation: bool = True
    skip_unmapped: bool = True
    orientation_safety_mode: str = "conservative"

    # L2 Caching (from dataset_loader.py usage)
    l2_cache_enabled: bool = field(default=False, metadata={"help": "Enable L2 (LMDB) caching"})
    l2_cache_path: str = field(default="./l2_cache", metadata={"help": "Path to L2 cache directory"})
    l2_max_size_gb: float = field(default=48.0, metadata={"help": "Maximum size of L2 cache in GB"})
    l2_max_readers: int = field(default=2048, metadata={"help": "Max readers for LMDB"})
    cache_precision: str = field(default='uint8', metadata={"help": "Precision for cached images ('uint8', 'float16', 'bfloat16', 'float32')"})
    canonical_cache_dtype: str = field(default='uint8', metadata={"help": "Canonical dtype for cache storage"})
    l2_storage_dtype: str = field(default='bfloat16', metadata={"help": "Storage dtype for L2 cache ('float16','bfloat16','float32','uint8')"})
    cpu_bf16_cache_pipeline: bool = field(default=True, metadata={"help": "Process normalization on CPU in bf16 when populating L2"})

    # Dataset behavior (from dataset_loader.py usage)
    patch_size: int = field(default=16, metadata={"help": "Patch size for vision transformer"})
    validate_on_init: bool = field(default=False, metadata={"help": "Validate all images on dataset init"})
    skip_error_samples: bool = field(default=True, metadata={"help": "Skip samples that cause loading errors"})
    collect_augmentation_stats: bool = field(default=False, metadata={"help": "Collect detailed augmentation stats"})

    # Weighted Sampling (from dataset_loader.py usage)
    frequency_weighted_sampling: bool = field(default=False, metadata={"help": "Enable frequency-weighted sampling"})
    sample_weight_power: float = field(default=0.5, metadata={"help": "Power for inverse frequency weighting"})

    # Working Set Sampler (from dataset_loader.py usage)
    use_working_set_sampler: bool = field(default=False, metadata={"help": "Enable working set sampler"})
    working_set_pct: float = field(default=5.0, metadata={"help": "Percentage of dataset in the working set"})
    working_set_max_items: int = field(default=400000, metadata={"help": "Max items in the working set"})
    trickle_in_pct: float = field(default=1.0, metadata={"help": "Percentage of new items to trickle in each epoch"})
    max_new_uniques_per_epoch: int = field(default=80000, metadata={"help": "Max new unique items per epoch"})
    working_set_refresh_epochs: int = field(default=2, metadata={"help": "Epochs before refreshing working set"})

    # Memory Management (from dataset_loader.py usage)
    critical_free_ram_pct: float = field(default=5.0, metadata={"help": "Critical free RAM percentage threshold"})
    low_free_ram_pct: float = field(default=12.0, metadata={"help": "Low free RAM percentage threshold"})
    high_free_ram_pct: float = field(default=25.0, metadata={"help": "High free RAM percentage threshold"})

    def validate(self):
        """Validate data configuration"""
        errors = []

        # Validate storage locations and check unique priorities
        priorities = []
        for i, loc in enumerate(self.storage_locations):
            try:
                # Handle both dict and StorageLocation objects
                if isinstance(loc, StorageLocation):
                    storage_loc = loc
                elif isinstance(loc, dict):
                    storage_loc = StorageLocation(**loc)
                else:
                    errors.append(f"Storage location {i}: Invalid type {type(loc)}")
                    continue

                storage_loc.validate()
                # Only check priority uniqueness for enabled locations
                if storage_loc.enabled:
                    priorities.append(storage_loc.priority)
            except Exception as e:
                errors.append(f"Storage location {i}: {str(e)}")
        # Unique priority validation for enabled locations only
        if priorities and len(priorities) != len(set(priorities)):
            dupes = [p for p in set(priorities) if priorities.count(p) > 1]
            errors.append(f"Duplicate storage location priorities detected: {sorted(dupes)}")

        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")

        if self.num_workers < 0:
            errors.append(f"num_workers must be non-negative, got {self.num_workers}")

        if self.cache_size_gb < 0:
            errors.append(f"cache_size_gb must be non-negative, got {self.cache_size_gb}")

        # New bounds checks
        if self.prefetch_factor < 1:
            errors.append(f"prefetch_factor must be >= 1, got {self.prefetch_factor}")
        if self.preload_files < 0:
            errors.append(f"preload_files must be >= 0, got {self.preload_files}")
        if self.random_rotation_degrees < 0:
            errors.append(f"random_rotation_degrees must be >= 0, got {self.random_rotation_degrees}")

        # Validate normalization parameters
        for param_name, param_value in [('normalize_mean', self.normalize_mean), 
                                        ('normalize_std', self.normalize_std)]:
            if len(param_value) != 3:
                errors.append(f"{param_name} must have 3 values, got {len(param_value)}")

        # Validate pad_color
        if len(self.pad_color) != 3:
            errors.append(f"pad_color must have 3 values, got {len(self.pad_color)}")
        else:
            for i, c in enumerate(self.pad_color):
                if not isinstance(c, int) or not (0 <= c <= 255):
                    errors.append(f"pad_color[{i}] must be int in [0,255], got {c}")

        # Validate augmentation parameters
        if self.random_flip_prob < 0 or self.random_flip_prob > 1:
            errors.append(f"random_flip_prob must be in [0, 1], got {self.random_flip_prob}")

        if self.orientation_safety_mode not in {"conservative", "balanced", "permissive"}:
            errors.append(
                f"orientation_safety_mode must be one of 'conservative', 'balanced', 'permissive', got {self.orientation_safety_mode}"
            )

        # Orientation mapping checks (strict mode only, and only if flips enabled)
        try:
            rfp = float(getattr(self, "random_flip_prob", 0.0) or 0.0)
        except Exception:
            rfp = 0.0

        try:
            if bool(getattr(self, "strict_orientation_validation", False)) and rfp > 0:
                from pathlib import Path
                from orientation_handler import OrientationHandler
                handler = OrientationHandler(
                    mapping_file=Path(self.orientation_map_path) if self.orientation_map_path else None,
                    random_flip_prob=rfp,
                    strict_mode=True,
                    safety_mode=self.orientation_safety_mode,
                    skip_unmapped=bool(getattr(self, "skip_unmapped", False)),
                )
                mapping_issues = handler.validate_mappings()
                if mapping_issues:
                    errors.append(f"orientation_mapping issues: {mapping_issues}")
        except Exception as e:
            errors.append(f"orientation_mapping_error: {e}")

        scale_min, scale_max = self.random_crop_scale
        if not (0 < scale_min <= scale_max <= 1):
            errors.append(f"random_crop_scale must satisfy 0 < min <= max <= 1, got {self.random_crop_scale}")

        # Validate advanced augmentations
        if self.randaugment_num_ops < 1:
            errors.append(f"randaugment_num_ops must be positive, got {self.randaugment_num_ops}")
        if not 0 <= self.randaugment_magnitude <= 30:
            errors.append(f"randaugment_magnitude must be in [0, 30], got {self.randaugment_magnitude}")
        if self.mixup_alpha < 0:
            errors.append(f"mixup_alpha must be non-negative, got {self.mixup_alpha}")
        if self.cutmix_alpha < 0:
            errors.append(f"cutmix_alpha must be non-negative, got {self.cutmix_alpha}")
        if not 0 <= self.random_erasing_p <= 1:
            errors.append(f"random_erasing_p must be in [0, 1], got {self.random_erasing_p}")

        valid_precisions = ["uint8", "float16", "bfloat16", "float32"]
        if self.cache_precision not in valid_precisions:
            errors.append(f"Invalid cache_precision: {self.cache_precision}. Must be one of {valid_precisions}")
        if self.l2_storage_dtype not in valid_precisions:
            errors.append(f"Invalid l2_storage_dtype: {self.l2_storage_dtype}. Must be one of {valid_precisions}")

        if errors:
            raise ConfigValidationError("Data config validation failed:\n" + "\n".join(errors))


@dataclass
class GradientClippingConfig(BaseConfig):
    enabled: bool = True
    max_norm: float = 1.0


@dataclass
class OverflowBackoffConfig(BaseConfig):
    enabled: bool = False
    factor: float = 0.1


@dataclass
class LossConfig(BaseConfig):
    """Hyperparameters for loss functions."""
    alpha: float = 0.5
    gamma_neg: float = 1.0
    gamma_pos: float = 1.0
    label_smoothing: float = 0.0
    clip: float = 0.05

    def validate(self):
        errors = []
        if not 0.0 <= self.alpha <= 1.0:
            errors.append(f"alpha must be in [0, 1], got {self.alpha}")
        if self.gamma_neg < 0 or self.gamma_pos < 0:
            errors.append("gamma_neg and gamma_pos must be >= 0")
        if not 0.0 <= self.label_smoothing <= 1.0:
            errors.append(
                f"label_smoothing must be in [0, 1], got {self.label_smoothing}"
            )
        if not 0.0 <= self.clip < 0.5:
            errors.append(f"clip must be in [0, 0.5), got {self.clip}")
        if errors:
            raise ConfigValidationError("Loss config validation failed:\n" + "\n".join(errors))


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration"""
    # Memory layout to unlock Tensor Core perf on Ampere+ (and Blackwell)
    memory_format: str = "contiguous"  # or "channels_last"
    # Basic settings
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adan_beta3: float = 0.99  # Beta3 for Adan optimizer
    adam_epsilon: float = 1e-8
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 10000
    warmup_ratio: float = 0.0  # Deprecated - use warmup_steps directly
    num_cycles: float = 0.5
    lr_end: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    amp_opt_level: str = "O1"
    amp_dtype: str = "bfloat16"  # float16 or bfloat16
    enable_anomaly_detection: bool = False

    # Gradient clipping
    max_grad_norm: float = 1.0  # deprecated, use gradient_clipping.max_norm
    gradient_clipping: GradientClippingConfig = field(default_factory=GradientClippingConfig)

    # Optional LR backoff when loss becomes non-finite
    overflow_backoff_on_nan: OverflowBackoffConfig = field(default_factory=OverflowBackoffConfig)
    
    # Checkpointing
    save_steps: int = 5000
    save_total_limit: int = 5
    save_best_only: bool = False
    # Resume behavior:
    #   "none"   -> start fresh
    #   "latest" -> resume most recent checkpoint under <output_root>/<experiment>/checkpoints
    #   "best"   -> resume checkpoints/best_model.pt if present
    #   any other non-empty string is treated as an absolute/relative path to a .pt file
    resume_from: str = "latest"
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # Loss configuration
    tag_loss: LossConfig = field(default_factory=LossConfig)
    rating_loss: LossConfig = field(default_factory=LossConfig)
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
    seed: Optional[int] = None   # None => fresh, logged seed per run
    deterministic: bool = False  # turn on only when seed is set
    benchmark: bool = True
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001
    # Ignore the first N epochs for early-stopping decisions.
    # After burn-in, reset the early-stopping baseline to a robust summary
    # of the burn-in window to avoid outlier first-epoch spikes.
    early_stopping_burn_in_epochs: int = 0
    # One of: 'median', 'mean', 'last', 'max'
    early_stopping_burn_in_strategy: str = "median"

    # Knowledge distillation (from training_config.yaml comments)
    use_distillation: bool = False
    distillation_alpha: float = 0.7
    distillation_temperature: float = 3.0
    
    def validate(self):
        """Validate training configuration"""
        errors = []
        
        # cuDNN benchmark can conflict with deterministic execution
        if self.deterministic and self.benchmark:
            logger.warning("deterministic=True forces benchmark=False to ensure repeatability")
            self.benchmark = False
        
        # AMP backend note
        if self.use_amp and self.amp_opt_level:
            logger.warning("amp_opt_level appears to target NVIDIA Apex; if using torch.amp, prefer configuring amp_dtype and ignore amp_opt_level")
        
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.num_epochs <= 0:
            errors.append(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.gradient_accumulation_steps <= 0:
            errors.append(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        
        valid_optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad", "adan"]
        if self.optimizer not in valid_optimizers:
            errors.append(f"Unknown optimizer: {self.optimizer}. Must be one of {valid_optimizers}")
        
        valid_schedulers = ["cosine", "cosine_restarts", "step", "multistep", "plateau", "exponential"]
        if self.scheduler not in valid_schedulers:
            errors.append(f"Unknown scheduler: {self.scheduler}. Must be one of {valid_schedulers}")
        
        # Validate beta values for Adam optimizers
        if self.optimizer in ["adam", "adamw"]:
            if not 0 <= self.adam_beta1 < 1:
                errors.append(f"adam_beta1 must be in [0, 1), got {self.adam_beta1}")
            if not 0 <= self.adam_beta2 < 1:
                errors.append(f"adam_beta2 must be in [0, 1), got {self.adam_beta2}")
        
        # Validate loss configurations
        try:
            self.tag_loss.validate()
        except ConfigValidationError as e:
            errors.append(f"tag_loss: {e}")
        try:
            self.rating_loss.validate()
        except ConfigValidationError as e:
            errors.append(f"rating_loss: {e}")
        
        # Validate device
        valid_devices = ["cuda", "cpu", "mps"]
        if not any(self.device.startswith(d) for d in valid_devices):
            errors.append(f"Unknown device: {self.device}. Must start with one of {valid_devices}")

        # Early-stopping burn-in
        if self.early_stopping_burn_in_epochs is None or int(self.early_stopping_burn_in_epochs) < 0:
            errors.append("early_stopping_burn_in_epochs must be >= 0")
        allowed_es_strategies = {"median", "mean", "last", "max"}
        if str(self.early_stopping_burn_in_strategy).lower() not in allowed_es_strategies:
            errors.append(
                f"early_stopping_burn_in_strategy must be one of {sorted(allowed_es_strategies)}"
            )
        
        if errors:
            raise ConfigValidationError("Training config validation failed:\n" + "\n".join(errors))


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration"""
    # Model
    model_path: Optional[str] = None
    precision: str = "bf16"  # Options: "fp32", "fp16", "bf16"
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceConfig":
        """Create config from dictionary with backward compatibility."""
        data = dict(data)
        if 'use_fp16' in data:
            warnings.warn(
                "The 'use_fp16' field is deprecated. Use 'precision' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if data.pop('use_fp16'):
                data['precision'] = 'fp16'
        return super().from_dict(data)

    def validate(self):
        """Validate inference configuration"""
        errors = []

        # Security posture warnings if API is enabled
        if self.enable_api:
            if self.api_host == '0.0.0.0':
                logger.warning("API is bound to 0.0.0.0; consider 127.0.0.1 or a firewall in production")
            if self.api_cors_origins and ('*' in self.api_cors_origins):
                logger.warning("API CORS origins allow '*'; require explicit origins for production")

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

        valid_precisions = ["fp32", "fp16", "bf16"]
        if self.precision not in valid_precisions:
            errors.append(f"Invalid precision: {self.precision}. Must be one of {valid_precisions}")

        valid_formats = ["json", "text", "csv", "xml", "yaml"]
        if self.output_format not in valid_formats:
            errors.append(f"Unknown output_format: {self.output_format}. Must be one of {valid_formats}")

        if self.score_decimal_places < 0 or self.score_decimal_places > 10:
            errors.append(f"score_decimal_places must be in [0, 10], got {self.score_decimal_places}")

        if self.api_port < 1 or self.api_port > 65535:
            errors.append(f"api_port must be in [1, 65535], got {self.api_port}")

        # Additional field validations
        if self.top_k is not None and self.top_k <= 0:
            errors.append(f"top_k must be > 0 when set, got {self.top_k}")
        if self.api_workers < 1:
            errors.append(f"api_workers must be >= 1, got {self.api_workers}")
        if self.api_rate_limit < 0:
            errors.append(f"api_rate_limit must be >= 0, got {self.api_rate_limit}")
        if self.cache_ttl_seconds < 0:
            errors.append(f"cache_ttl_seconds must be >= 0, got {self.cache_ttl_seconds}")
        if self.max_batch_size <= 0:
            errors.append(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.batch_timeout_ms < 0:
            errors.append(f"batch_timeout_ms must be >= 0, got {self.batch_timeout_ms}")

        if errors:
            raise ConfigValidationError("Inference config validation failed:\n" + "\n".join(errors))


@dataclass
class ExportConfig(BaseConfig):
    """Model export configuration"""
    # Export format
    export_format: str = "onnx"  # onnx, torchscript, tflite, coreml
    
    # Use the latest available opset for transformer models with LayerNormalization
    # Opset 19 aligns with ONNX Runtime 1.22 and CUDA 12.8
    opset_version: int = 19  # default, will be clamped at export time
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


# --- ONNX opset resolution helpers ---
def resolve_opset(requested: int | None = None) -> int:
    """Resolve the ONNX opset version based on installed PyTorch/ONNX support.

    PyTorch’s ONNX exporter supports a limited range of opset versions.  As of
    recent releases, ``torch.onnx._constants.ONNX_MIN_OPSET`` and
    ``torch.onnx._constants.ONNX_MAX_OPSET`` describe the supported interval.
    ``ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET`` may cap the TorchScript exporter at
    a lower version (e.g., 20) than the global maximum【464953657985222†L4-L9】.  ONNX
    Runtime, on the other hand, can run models up to opset 21 with version
    1.20【598437280542461†L224-L248】.  To choose a safe default, clamp the requested
    version into the supported range.  When the requested value is ``None``,
    ``ExportConfig.opset_version`` is used as the target.

    Args:
        requested: desired opset version, or ``None`` to use the default on
            ``ExportConfig``.

    Returns:
        A version within the exporter’s supported range.
    """
    try:
        import torch.onnx._constants as _c  # type: ignore
        min_o = getattr(_c, "ONNX_MIN_OPSET", 7)
        # Prefer the TorchScript exporter max if available; fall back to ONNX_MAX_OPSET
        max_o = getattr(_c, "ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET", None)
        if max_o is None:
            max_o = getattr(_c, "ONNX_MAX_OPSET", 17)
    except Exception:
        # Conservative defaults when constants are unavailable
        min_o, max_o = 7, 21
    target = requested if requested is not None else ExportConfig.opset_version
    if target < min_o:
        return min_o
    if target > max_o:
        return max_o
    return target


@dataclass
class ValidationDataloaderConfig(BaseConfig):
    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int = 2
    persistent_workers: bool = True

@dataclass
class ValidationPreprocessingConfig(BaseConfig):
    # Match training defaults; these should be kept in sync with DataConfig
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_size: int = 640
    patch_size: int = 16

@dataclass
class ValidationConfig(BaseConfig):
    dataloader: ValidationDataloaderConfig = field(default_factory=ValidationDataloaderConfig)
    preprocessing: ValidationPreprocessingConfig = field(default_factory=ValidationPreprocessingConfig)


@dataclass
class TBImageLoggingConfig(BaseConfig):
    """Configuration for TensorBoard image and text sample logging."""
    max_samples: int = 4
    topk: int = 15
    log_native_resolution: bool = True
    dpi_for_figures: int = 220


@dataclass
class MonitorConfig(BaseConfig):
    """Configuration for monitoring system"""
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_to_file: bool = True
    log_to_console: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Metrics tracking
    track_system_metrics: bool = True
    system_metrics_interval: float = 30.0  # seconds
    track_gpu_metrics: bool = True
    track_disk_io: bool = True
    track_network_io: bool = False

    # Visualization
    use_tensorboard: bool = True
    tensorboard_dir: str = "./tensorboard"
    use_wandb: bool = False
    wandb_project: str = "anime-tagger"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Alerts
    enable_alerts: bool = True
    alert_on_gpu_memory_threshold: float = 0.9  # 90% usage
    alert_on_cpu_memory_threshold: float = 0.9
    alert_on_disk_space_threshold: float = 0.95
    alert_on_training_stuck_minutes: int = 30
    alert_on_loss_explosion: float = 10.0
    alert_on_nan_loss: bool = True
    # Webhook URL is loaded from sensitive_config to avoid hardcoding secrets
    alert_webhook_url: Optional[str] = ALERT_WEBHOOK_URL

    # Performance profiling
    enable_profiling: bool = False
    profile_interval_steps: int = 1000
    profile_duration_steps: int = 10
    profile_memory: bool = True
    profile_shapes: bool = True
    profile_output_dir: str = "./profiles"

    # Data pipeline monitoring
    monitor_data_pipeline: bool = True
    data_pipeline_stats_interval: int = 100  # batches
    augmentation_stats_interval: int = 100  # batches
    log_augmentation_histograms: bool = True
    log_augmentation_images: bool = False
    augmentation_image_interval: int = 500  # batches
    # Parameter / gradient histogram logging
    log_param_histograms: bool = True
    log_grad_histograms: bool = True
    # Log every N steps (set high if training is slow or memory-limited)
    param_hist_interval_steps: int = 200
    grad_hist_interval_steps: int = 200

    # Unified histogram logging
    log_histograms: bool = False
    histogram_log_interval: int = 500
    histogram_params: Optional[List[str]] = None

    # Remote monitoring
    enable_prometheus: bool = False
    prometheus_port: int = 8080

    # TensorBoard image logging helpers
    tb_image_logging: TBImageLoggingConfig = field(default_factory=TBImageLoggingConfig)

    # History
    max_history_size: int = 10000
    history_save_interval: int = 100
    checkpoint_metrics: bool = True

    # Distributed training
    distributed: bool = False
    rank: int = 0
    world_size: int = 1

    # Safety
    auto_recovery: bool = True
    max_retries: int = 3
    safe_mode: bool = True  # Disable features that might crash


@dataclass
class DebugConfig(BaseConfig):
    """Configuration for debugging and diagnosing training issues."""
    # When enabled, additional checks and logging are activated.
    # This may impact performance and should be disabled for regular training.
    enabled: bool = False

    # If true, dump the input tensors and model outputs to a .pt file
    # when a non-finite value is detected in the model's output logits.
    dump_tensors_on_error: bool = False

    # If true, log detailed information about the batch that caused a
    # non-finite error. This includes file paths or other identifiers.
    log_batch_info_on_error: bool = False

    # Enable PyTorch's anomaly detection for debugging gradients.
    detect_anomaly: bool = False

    # If true, log the gradient norm of the model's parameters to TensorBoard.
    log_gradient_norm: bool = False

    # If true, perform a pre-training validation step to check the integrity of the input data.
    validate_input_data: bool = False

    # If true, save intermediate images during the data augmentation process for visual inspection.
    visualize_augmentations: bool = False

    # If true, log statistics (min/mean/max) of input batches.
    log_input_stats: bool = False

    # If true, log statistics (min/mean/max) of model activations such as logits.
    log_activation_stats: bool = False

    # Directory to save the visualized augmentation images.
    augmentation_visualization_path: str = "./aug_visualizations"

    def validate(self):
        """Validate debug configuration."""
        if self.enabled:
            logger.warning("Debug mode is enabled. This may slow down training.")
        if self.dump_tensors_on_error and not self.enabled:
            logger.warning("`dump_tensors_on_error` is true but debug mode is disabled.")
        if self.log_batch_info_on_error and not self.enabled:
            logger.warning("`log_batch_info_on_error` is true but debug mode is disabled.")
        if self.log_input_stats and not self.enabled:
            logger.warning("`log_input_stats` is true but debug mode is disabled.")
        if self.log_activation_stats and not self.enabled:
            logger.warning("`log_activation_stats` is true but debug mode is disabled.")


@dataclass
class FullConfig(BaseConfig):
    """Complete configuration combining all components"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    
    # Global settings
    project_name: str = "anime-image-tagger"
    experiment_name: str = field(default_factory=lambda: f"exp_{datetime.now():%Y%m%d_%H%M%S}")
    output_root: str = "./experiments"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Paths (consolidated from paths.yaml)
    vocab_path: str = "./vocabulary.json"
    log_dir: str = "./logs"
    default_output_dir: str = "./outputs"

    # Logging settings (from logging.yaml)
    file_logging_enabled: bool = True
    log_rotation_max_bytes: int = 10485760
    log_rotation_backups: int = 5
    
    # Resource limits
    max_memory_gb: Optional[float] = None
    max_gpu_memory_gb: Optional[float] = None
    
    def validate(self):
        """Validate all sub-configurations and cross-config consistency"""
        errors = []
        
        # Validate each sub-config
        for config_name in ['model', 'data', 'training', 'inference', 'export', 'monitor', 'debug']:
            try:
                config_obj = getattr(self, config_name)
                if hasattr(config_obj, 'validate') and callable(getattr(config_obj, 'validate')):
                    config_obj.validate()
                else:
                    logger.debug(f"Config section '{config_name}' has no validate() method, skipping.")
            except ConfigValidationError as e:
                errors.append(f"{config_name}: {str(e)}")
        
        # Cross-config validation
        if not errors:  # Only do cross-validation if individual configs are valid
            # Check batch size consistency
            effective_batch = self.data.batch_size * self.training.gradient_accumulation_steps
            # Make threshold configurable based on available memory
            memory_threshold = 512
            if self.max_memory_gb:
                memory_threshold = int(self.max_memory_gb * 1024 / 16)  # Rough heuristic
            if effective_batch > memory_threshold:
                logger.warning(f"Large effective batch size ({effective_batch}) may cause memory issues (threshold: {memory_threshold})")
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

        legacy_configs = {
            "training_config.yaml",
            "inference_config.yaml",
            "dataset_prep.yaml",
            "export_config.yaml",
            "runtime.yaml",
            "logging.yaml",
            "vocabulary.yaml",
            "orientation_map.json"
        }

        if path.name in legacy_configs:
            logger.warning(
                f"Loading legacy config file '{path.name}'. "
                f"Please migrate to 'configs/unified_config.yaml'."
            )
        
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

        # Normalize extension first so backups follow the final filename
        if path.suffix not in ['.yaml', '.yml', '.json']:
            path = path.with_suffix('.yaml')

        # Create backup if requested and file exists
        if backup and path.exists():
            timestamp = f"{datetime.now():%Y%m%d_%H%M%S}_{int(time.time() * 1000000) % 1000000:06d}"
            backup_path = path.with_suffix(f'.backup_{timestamp}{path.suffix}')
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
            # Fallback to YAML
            self.config.to_yaml(path.with_suffix('.yaml'))

        logger.info(f"Saved config to {path}")


    def update_from_env(self, prefix: str = "ANIME_TAGGER_"):
        """
        Update configuration from environment variables
        
        Format: ANIME_TAGGER_<SECTION>__<FIELD>[__<SUBFIELD>[__<...>]]
        Examples:
            ANIME_TAGGER_TRAINING__LEARNING_RATE=0.001
            ANIME_TAGGER_MODEL__HEADS__ATTN=16
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
            
            # Build nested update dictionary for arbitrary depth
            section, *rest = parts
            if section not in updates:
                updates[section] = {}
            cursor = updates[section]
            if not rest:
                logger.warning(f"Missing field name in env var: {key}")
                continue
            for subkey in rest[:-1]:
                if subkey not in cursor or not isinstance(cursor[subkey], dict):
                    cursor[subkey] = {}
                cursor = cursor[subkey]
            cursor[rest[-1]] = parsed_value

        # Apply updates
        if updates:
            self._apply_nested_updates(self.config, dict(updates))
            self.config.validate()
            logger.info(f"Updated config from {len(updates)} environment sections")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False

        # Try JSON for complex types (only if it looks like JSON)
        if value.strip().startswith(('[', '{')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Try JSON for null values
        if value.lower() == 'null':
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
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
            if value is not None and key not in {'config', 'output_config', 'validate_only'}:
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
    parser.add_argument('--validate-only', action='store_true', default=None, help='Only validate config and exit')
    
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
        train_group.add_argument('--training.distributed', action='store_true', default=None, help='Use distributed training')
        train_group.add_argument('--training.seed', type=int, help='Random seed')
        
        # Inference arguments
        infer_group = parser.add_argument_group('inference')
        infer_group.add_argument('--inference.model_path', type=str, help='Path to model')
        infer_group.add_argument('--inference.prediction_threshold', type=float, help='Prediction threshold')
        infer_group.add_argument('--inference.top_k', type=int, help='Top-k predictions')
        infer_group.add_argument('--inference.filter_nsfw', action='store_true', default=None, help='Filter NSFW tags')
        
        # Export arguments
        export_group = parser.add_argument_group('export')
        export_group.add_argument('--export.export_format', type=str, help='Export format')
        export_group.add_argument('--export.optimize', action='store_true', default=None, help='Optimize exported model')
        export_group.add_argument('--export.quantize', action='store_true', default=None, help='Quantize model')
    
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
        precision="fp16",
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
    # Defer heavy logging for 'validate' to avoid importing any training deps.
    is_validate = any(arg == "validate" for arg in sys.argv[1:])

    if len(sys.argv) > 1 and is_validate:
        # Lightweight console logging only; avoid importing logging_setup (which may import heavy deps)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # Validate a config file
        if len(sys.argv) < 3:
            print("Usage: python Configuration_System.py validate <config_file>")
            sys.exit(1)

        try:
            manager = ConfigManager(ConfigType.FULL)
            _ = manager.load_from_file(sys.argv[2])
            print(f"✓ Config file '{sys.argv[2]}' is valid")
        except Exception as e:
            print(f"✗ Config validation failed: {e}")
            sys.exit(1)
    else:
        # Full logging for other commands and interactive tests
        from utils.logging_setup import setup_logging
        listener = setup_logging()
        try:
            if len(sys.argv) > 1:
                if sys.argv[1] == "generate":
                    # Generate example configs
                    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./config_examples")
                    generate_example_configs(output_dir)
                elif sys.argv[1] == "validate":
                    # Should have been handled above; keep here for safety
                    if len(sys.argv) < 3:
                        print("Usage: python Configuration_System.py validate <config_file>")
                        sys.exit(1)
                    try:
                        manager = ConfigManager(ConfigType.FULL)
                        _ = manager.load_from_file(sys.argv[2])
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
                os.environ["ANIME_TAGGER_MODEL__NUM_GROUPS"] = "20"  # Keep consistent with default tags_per_group
                os.environ["ANIME_TAGGER_DATA__BATCH_SIZE"] = "64"

                manager.update_from_env()

                assert manager.config.training.learning_rate == 0.0005
                assert manager.config.model.num_groups == 20
                assert manager.config.data.batch_size == 64
                print("   ✓ Environment overrides work correctly")

                # Test 4: Config diff
                print("\n4. Testing config diff functionality...")
                config2 = FullConfig()
                # Create a fresh manager to avoid contamination from previous test
                manager2 = ConfigManager(ConfigType.FULL)
                manager2.config = config2
                config2.training.num_epochs = 200
                config2.model.hidden_size = 2048

                # Compare with original unmodified config
                original_manager = ConfigManager(ConfigType.FULL)

                diff = original_manager.get_diff(manager2)
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
        finally:
            if listener:
                listener.stop()
