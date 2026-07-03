#!/usr/bin/env python3
"""
Configuration System for Anime Image Tagger
Centralized configuration management with validation and persistence
"""

import copy
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
except (ImportError, AttributeError):
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
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum

# Type alias for JSON-serializable values
JsonSerializable = Union[None, bool, int, float, str, List['JsonSerializable'], Dict[str, 'JsonSerializable'], Tuple['JsonSerializable', ...]]
import argparse
import sys
from datetime import datetime
import time
from collections import defaultdict
import warnings
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Allowlist of trusted webhook domains for alert URLs
ALLOWED_WEBHOOK_DOMAINS = [
    'hooks.slack.com',
    'discord.com',
    'hooks.microsoft.com',  # Teams
    'api.telegram.org',     # Telegram
    # Add other trusted webhook providers here
]

def _is_allowed_domain(netloc: str, allowed_domains: list) -> bool:
    """Check if netloc matches an allowed domain securely.

    Prevents subdomain spoofing attacks like 'evil.com.hooks.slack.com'.
    Only allows exact domain matches or legitimate subdomains (prefixed with '.').

    Args:
        netloc: The network location from URL (may include port)
        allowed_domains: List of allowed domain patterns

    Returns:
        True if domain is allowed, False otherwise
    """
    # Strip port if present and normalize to lowercase
    host = netloc.split(':')[0].lower()

    for domain in allowed_domains:
        domain = domain.lower()
        # Exact match
        if host == domain:
            return True
        # Legitimate subdomain: host must end with ".{domain}"
        # e.g., "workspace.hooks.slack.com" matches ".hooks.slack.com"
        # but "evil.com.hooks.slack.com" would also match, so we need additional validation
        if host.endswith('.' + domain):
            # Verify this is a real subdomain, not a spoofed domain
            # The part before the domain must be a valid subdomain (no extra dots in suspicious positions)
            prefix = host[:-len(domain) - 1]  # Get the subdomain part
            # A valid subdomain prefix should not contain the domain itself
            # e.g., for "evil.com.hooks.slack.com", prefix is "evil.com" which contains "."
            # For legitimate "workspace.hooks.slack.com", prefix is "workspace" (no dots typically)
            # However, multi-level subdomains like "a.b.hooks.slack.com" are valid
            # The key check: the prefix should not look like another TLD (e.g., "evil.com")
            # Simplest secure approach: only allow subdomains that don't contain common TLD patterns
            if '.' in prefix and any(prefix.endswith(tld) for tld in ['.com', '.org', '.net', '.io', '.co']):
                # This looks like a spoofed domain (e.g., evil.com.hooks.slack.com)
                continue
            return True
    return False


def validate_webhook_url(url: str | None) -> str | None:
    """Validate webhook URL for security.

    Args:
        url: The webhook URL to validate

    Returns:
        The validated URL or None if no URL provided

    Raises:
        ValueError: If URL is invalid or uses untrusted domain
    """
    if url is None:
        return None

    # Ensure HTTPS only (never plain HTTP)
    parsed = urlparse(url)
    if parsed.scheme != 'https':
        raise ValueError(f"Webhook URL must use HTTPS, got: {parsed.scheme}")

    # Validate against URL credential injection (user:pass@host)
    if parsed.username or parsed.password:
        raise ValueError("Webhook URL must not contain credentials")

    # Check domain allowlist with secure matching
    if not _is_allowed_domain(parsed.netloc, ALLOWED_WEBHOOK_DOMAINS):
        raise ValueError(
            f"Webhook domain '{parsed.netloc}' not in allowlist. "
            f"Allowed domains: {ALLOWED_WEBHOOK_DOMAINS}"
        )

    logger.info(f"Validated webhook URL for domain: {parsed.netloc}")
    return url

try:
    from sensitive_config import ALERT_WEBHOOK_URL as _ALERT_WEBHOOK_URL
    ALERT_WEBHOOK_URL = validate_webhook_url(_ALERT_WEBHOOK_URL)
except ImportError:  # pragma: no cover - fallback when file missing
    ALERT_WEBHOOK_URL = None
except ValueError as e:
    logger.error(f"Invalid webhook URL in sensitive_config.py: {e}")
    ALERT_WEBHOOK_URL = None

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


def _object_to_dict(obj: Any, exclude_private: bool = True, preserve_tuples: bool = True) -> JsonSerializable:
    """Recursively convert an object to a serializable structure.

    Unlike ``dataclasses.asdict`` this helper preserves tuple types by
    converting tuple items individually rather than coercing the tuple to a
    list. It also respects ``exclude_private`` when encountering nested
    dataclass instances.

    Args:
        obj: The object to convert (dataclass, dict, list, tuple, or primitive).
        exclude_private: Whether to omit private fields from nested dataclasses.
        preserve_tuples: If True, preserve tuple types; if False, convert to lists.

    Returns:
        A structure composed of ``dict``, ``list``, ``tuple`` and primitive
        types suitable for serialization.
    """
    # Handle Enum types first - convert to their value for JSON/pickle compatibility
    if isinstance(obj, Enum):
        return obj.value
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

    # Class-level static caches (shared across all instances)
    # Using class attributes with setdefault for thread-safe initialization
    _fields_cache_static: _typing.ClassVar[Dict[type, Dict[str, Any]]] = {}
    _type_hints_cache_static: _typing.ClassVar[Dict[type, Dict[str, type]]] = {}

    @classmethod
    def _get_cached_fields(cls) -> Dict[str, Any]:
        """Get fields dict with caching to avoid repeated introspection."""
        # Cache fields() result per class to avoid repeated dataclass introspection.
        cache = BaseConfig._fields_cache_static
        if cls not in cache:
            cache[cls] = {f.name: f for f in fields(cls)}
        return cache[cls]

    @classmethod
    def _get_cached_type_hints(cls) -> Dict[str, type]:
        """Get type hints dict with caching to avoid repeated introspection."""
        # Cache get_type_hints() result per class to avoid repeated introspection.
        cache = BaseConfig._type_hints_cache_static
        if cls not in cache:
            cache[cls] = get_type_hints(cls)
        return cache[cls]

    def to_dict(self, exclude_private: bool = True, *, preserve_tuples: bool = True) -> Dict[str, Any]:
        """
        Convert config to dictionary

        Args:
            exclude_private: Whether to exclude private fields (starting with _)
        """
        result: Dict[str, Any] = {}
        # Use cached fields to avoid repeated introspection
        for name, field_obj in self.__class__._get_cached_fields().items():
            if exclude_private and name.startswith('_'):
                continue

            value = getattr(self, name)
            result[name] = _object_to_dict(value, exclude_private, preserve_tuples)

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

        Raises:
            OSError: If file cannot be written (disk full, permissions, etc.)
            ConfigError: If configuration cannot be serialized
        """
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ConfigError(f"Cannot create directory {p.parent}: {e}") from e

        # Use atomic write pattern: write to temp file, then rename
        temp_path = p.with_suffix(p.suffix + '.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    self.to_dict(exclude_private=True, preserve_tuples=preserve_tuples),
                    f,
                    sort_keys=sort_keys,
                    default_flow_style=False,
                    **kwargs
                )
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk

            # Verify temp file was written
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise ConfigError(f"Failed to write config to {temp_path}: file is empty")

            # Atomic rename
            temp_path.replace(p)
            logger.info(f"Successfully saved config to {p} ({p.stat().st_size} bytes)")

        except OSError as e:
            # Clean up temp file if it exists (CR-007 fix: improve logging)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_path}")
                except OSError as cleanup_error:
                    # Log failure instead of silent pass (CR-007)
                    logger.warning(
                        f"Failed to remove temp file {temp_path}: {cleanup_error}. "
                        f"Manual cleanup may be required."
                    )
            raise ConfigError(f"Failed to save config to {p}: {e}") from e
        except Exception as e:
            # Clean up temp file on any error (CR-007 fix: improve logging)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_path}")
                except OSError as cleanup_error:
                    # Log failure instead of silent pass (CR-007)
                    logger.warning(
                        f"Failed to remove temp file {temp_path}: {cleanup_error}. "
                        f"Manual cleanup may be required."
                    )
            raise ConfigError(f"Error serializing config to {p}: {e}") from e
    
    def to_json(self, path: Union[str, Path], **kwargs):
        """Save config to JSON file.

        Raises:
            OSError: If file cannot be written
            ConfigError: If configuration cannot be serialized
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ConfigError(f"Cannot create directory {path.parent}: {e}") from e

        temp_path = path.with_suffix(path.suffix + '.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, **kwargs)
                f.flush()
                os.fsync(f.fileno())

            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise ConfigError(f"Failed to write config to {temp_path}: file is empty")

            temp_path.replace(path)
            logger.info(f"Successfully saved config to {path} ({path.stat().st_size} bytes)")

        except OSError as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_path}")
                except OSError as cleanup_error:
                    logger.warning(
                        f"Failed to remove temp file {temp_path}: {cleanup_error}. "
                        f"Manual cleanup may be required."
                    )
            raise ConfigError(f"Failed to save config to {path}: {e}") from e
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_path}")
                except OSError as cleanup_error:
                    logger.warning(
                        f"Failed to remove temp file {temp_path}: {cleanup_error}. "
                        f"Manual cleanup may be required."
                    )
            raise ConfigError(f"Error serializing config to {path}: {e}") from e
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create config from dictionary, handling nesting and type conversion."""
        kwargs = {}
        # Use cached introspection results (200-800ms savings on repeated calls)
        cls_fields = cls._get_cached_fields()
        type_hints = cls._get_cached_type_hints()
        
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
                                # Handle nested typing constructs (e.g. List[int], Dict[str, str])
                                origin_t = get_origin(t)
                                if origin_t is list and isinstance(value, list):
                                    # Handle List[T] within Union
                                    args_t = get_args(t)
                                    if args_t and is_dataclass(args_t[0]):
                                        kwargs[key] = [args_t[0].from_dict(v) if isinstance(v, dict) else v
                                                      for v in value]
                                    else:
                                        kwargs[key] = value
                                elif origin_t is tuple and isinstance(value, (list, tuple)):
                                    # Handle Tuple within Union
                                    kwargs[key] = tuple(value)
                                elif origin_t is dict and isinstance(value, dict):
                                    # Handle Dict within Union
                                    kwargs[key] = value
                                elif _is_union_origin(origin_t):
                                    # Handle nested Union types recursively
                                    temp_args = get_args(t)
                                    temp_non_none = [tt for tt in temp_args if tt is not type(None)]
                                    if value is None and type(None) in temp_args:
                                        kwargs[key] = None
                                    else:
                                        # Recursively try nested union types
                                        nested_assigned = False
                                        for tt in temp_non_none:
                                            try:
                                                if is_dataclass(tt) and isinstance(value, dict):
                                                    kwargs[key] = tt.from_dict(value)
                                                else:
                                                    kwargs[key] = tt(value)
                                                nested_assigned = True
                                                break
                                            except (TypeError, ValueError, AttributeError):
                                                continue
                                        if not nested_assigned:
                                            raise TypeError(f"Cannot coerce to nested union {t}")
                                elif origin_t is not None:
                                    # Other generic types - assign raw value
                                    kwargs[key] = value
                                else:
                                    # Primitive type - try direct coercion
                                    if t is int and isinstance(value, float):
                                        if value != int(value):
                                            raise ConfigValidationError(
                                                f"Lossy coercion not allowed: {key}={value} (float) would be truncated to {int(value)}. "
                                                f"Please use an integer value in your config file."
                                            )
                                    kwargs[key] = t(value)
                            assigned = True
                            break
                        except (TypeError, ValueError, AttributeError) as e:
                            # These are expected during type coercion attempts
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

        # First pass: coerce all values without modifying state
        coerced_updates = {}
        for name, raw in updates.items():
            ftype = allowed[name]
            # Initialize coerced to avoid UnboundLocalError
            coerced = None
            coerced_assigned = False

            # Coerce nested dataclasses
            if is_dataclass(ftype) and isinstance(raw, dict):
                coerced = ftype.from_dict(raw)  # type: ignore[attr-defined]
                coerced_assigned = True
            else:
                origin = get_origin(ftype)
                # Handle Union/Optional types using the helper
                if _is_union_origin(origin):
                    args = get_args(ftype)
                    if raw is None:
                        coerced = None
                        coerced_assigned = True
                    else:
                        last_err = None
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
                                coerced_assigned = True
                                break
                            except (TypeError, ValueError, AttributeError) as e:
                                # Expected during type coercion attempts
                                last_err = e
                        if not coerced_assigned:
                            raise TypeError(f"Cannot coerce {name}={raw!r} to {ftype}") from last_err
                else:
                    coerced = raw
                    coerced_assigned = True

            # Sanity check before adding to updates dict
            if not coerced_assigned:
                raise RuntimeError(f"Internal error: failed to coerce value for {name}")

            coerced_updates[name] = coerced

        # Validate with temporary updates if requested
        if validate:
            # Create temporary snapshot for validation
            original_values = {name: getattr(self, name) for name in coerced_updates}
            try:
                # Apply updates temporarily
                for name, value in coerced_updates.items():
                    setattr(self, name, value)
                # Validate
                self.validate()
            except Exception:
                # Rollback on validation failure
                for name, value in original_values.items():
                    setattr(self, name, value)
                raise
        else:
            # No validation, apply directly
            for name, value in coerced_updates.items():
                setattr(self, name, value)

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
                                except (TypeError, ValueError, AttributeError) as e:
                                    # Expected during type coercion attempts
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
            path: Dot-separated path (e.g., 'model.hidden_size').
                  Empty string returns self.
            default: Default value if path doesn't exist

        Returns:
            The value at the specified path, or default if not found

        Raises:
            ValueError: If path contains empty parts (e.g., 'model..size')
            TypeError: If path is not a string
        """
        if not path:
            # Empty path - return self as special case
            return self

        if not isinstance(path, str):
            raise TypeError(f"Path must be a string, got {type(path)}")

        parts = path.split('.')

        # Check for empty parts (e.g., "model..hidden_size" or ".model")
        if any(not part for part in parts):
            raise ValueError(
                f"Invalid path '{path}': contains empty parts. "
                f"Use 'model.hidden_size', not 'model..hidden_size'"
            )

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

        Raises:
            ValueError: If path is empty or contains empty parts
            TypeError: If path is not a string
            ConfigError: If path doesn't exist
        """
        if not path:
            raise ValueError("Cannot set value with empty path")

        if not isinstance(path, str):
            raise TypeError(f"Path must be a string, got {type(path)}")

        parts = path.split('.')

        # Check for empty parts
        if any(not part for part in parts):
            raise ValueError(
                f"Invalid path '{path}': contains empty parts. "
                f"Use 'model.hidden_size', not 'model..hidden_size'"
            )

        if len(parts) == 1:
            # Direct attribute set
            if hasattr(self, parts[0]):
                setattr(self, parts[0], value)
            else:
                raise ConfigError(f"Field not found: {parts[0]}")
            return

        # Navigate to parent
        current = self
        for i, part in enumerate(parts[:-1]):
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                traversed = '.'.join(parts[:i+1])
                raise ConfigError(
                    f"Path not found: '{path}'. "
                    f"Failed at '{traversed}' - attribute '{part}' does not exist"
                )

        # Set final attribute
        final_attr = parts[-1]
        if hasattr(current, final_attr):
            setattr(current, final_attr, value)
        else:
            parent_path = '.'.join(parts[:-1])
            raise ConfigError(
                f"Field not found: '{path}'. "
                f"Parent '{parent_path}' exists but has no attribute '{final_attr}'"
            )
    
    def __eq__(self, other):
        """Check equality based on dict representation"""
        if not isinstance(other, self.__class__):
            return False
        return self.to_dict() == other.to_dict()


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration.

    Recommended ViT learning rate (base value, scales with batch size): 2.5e-4.
    Use get_recommended_learning_rate() to get the batch-scaled LR.
    """
    # Architecture
    architecture_type: str = "vit"  # Must be "vit" — the only supported architecture (hard-asserted in validate())
    hidden_size: int = 1024
    num_hidden_layers: int = 18
    num_attention_heads: int = 16
    intermediate_size: int = 4096

    # Vision specific
    image_size: int = 448
    patch_size: int = 16
    num_channels: int = 3

    # Regularization
    hidden_dropout_prob: float = 0.1
    pos_dropout: float = 0.0  # Position embedding dropout (0.0 = modern best practice)
    attention_dropout: float = 0.1
    drop_path_rate: float = 0.1
    
    # Initialization
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6

    # Precision and numerical stability
    use_fp32_layernorm: bool = field(default=False, metadata={"help": "Use FP32 for LayerNorm (better stability, slight speed cost). Set to False for full bfloat16."})

    # Attention (Flex Attention)
    use_flex_attention: bool = True  # Enables Flex Attention (requires PyTorch 2.5+)
    attention_bias: bool = True
    flex_block_size: int = 128  # Block size for Flex Attention sparse computation

    # Masking
    token_ignore_threshold: float = 0.9  # Fraction of padding pixels to ignore token
    
    # Tag prediction
    num_labels: int = 0
    num_groups: int = 20
    tags_per_group: int = 10000

    # Efficiency
    gradient_checkpointing: bool = False
    checkpoint_every_n_layers: int = 1  # 1=all layers, 2=every 2nd, 4=every 4th, etc.

    def validate(self):
        """Validate model configuration"""
        errors = []
        
        if self.hidden_size % self.num_attention_heads != 0:
            errors.append(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Note: num_labels validation is deferred until after vocabulary is loaded
        # because the actual vocabulary size is only known at runtime.
        # A value of 0 indicates "use vocabulary size" (computed dynamically).
        # At runtime train_direct sets num_labels = vocabulary size (~19K), so this
        # is a CAPACITY check, not strict equality: the grouped-head prototype was
        # never shipped (model_architecture ignores num_groups/tags_per_group), and
        # requiring exact equality would make any post-vocab-load validate() raise.
        if self.num_labels != 0 and self.num_labels > self.num_groups * self.tags_per_group:
            errors.append(
                f"num_labels ({self.num_labels}) exceeds grouped-head capacity: "
                f"num_groups ({self.num_groups}) * tags_per_group ({self.tags_per_group}) = "
                f"{self.num_groups * self.tags_per_group}. Increase num_groups/tags_per_group, "
                f"or set num_labels to 0 (auto-detect from vocabulary)."
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
        for prob_name in ['hidden_dropout_prob', 'attention_dropout']:
            prob_value = getattr(self, prob_name)
            if not 0 <= prob_value <= 1:
                errors.append(f"{prob_name} must be in [0, 1], got {prob_value}")

        if not 0 <= self.drop_path_rate <= 0.8:
            errors.append(
                f"drop_path_rate must be in [0, 0.8], got {self.drop_path_rate}. "
                "Values above 0.8 would drop too many gradient paths and prevent training."
            )
        elif self.drop_path_rate > 0.5:
            warnings.warn(
                f"drop_path_rate={self.drop_path_rate} is unusually high; "
                "values between 0.08 and 0.3 are typical for ViT."
            )

        # Only the ViT architecture is supported.
        if self.architecture_type != "vit":
            errors.append(
                f"architecture_type must be 'vit' (the only supported architecture), "
                f"got '{self.architecture_type}'"
            )

        if errors:
            raise ConfigValidationError("Model config validation failed:\n" + "\n".join(errors))

    def get_recommended_learning_rate(self, base_lr: float = 2.5e-4) -> float:
        """Get architecture-appropriate learning rate recommendation.

        Args:
            base_lr: Base learning rate tuned for ViT (default: 2.5e-4)

        Returns:
            Recommended learning rate for the current architecture
        """
        # Architecture-specific learning rate multipliers
        lr_multipliers = {
            "vit": 1.0,
            "vit_wide_shallow": 1.0,
        }

        multiplier = lr_multipliers.get(self.architecture_type, 1.0)
        recommended_lr = base_lr * multiplier

        return recommended_lr

    def get_learning_rate_guidance(self) -> str:
        """Get detailed learning rate guidance for the current architecture.

        Returns:
            Human-readable guidance string for learning rate selection.
        """
        return (
            f"{self.architecture_type.upper()} Learning Rate Guidance:\n"
            "  - Recommended base LR: 2.5e-4\n"
            "  - Scale with sqrt(effective_batch_size / 256)\n"
            "  - Use warmup for stable training start\n"
        )


@dataclass
class StorageLocation:
    """Storage location configuration"""
    path: str
    priority: int
    type: str = "local"  # local, das, s3, gcs
    enabled: bool = True

    def validate(self):
        """Validate storage location"""
        if not self.path:
            raise ConfigValidationError("Storage location must have a path")

        if self.priority < 0:
            raise ConfigValidationError(f"Priority must be non-negative, got {self.priority}")

        valid_types = ["local", "das", "s3", "gcs"]
        if self.type not in valid_types:
            raise ConfigValidationError(f"Storage type must be one of {valid_types}, got {self.type}")


@dataclass
class DataConfig(BaseConfig):
    """Data loading and preprocessing configuration"""
    # Storage locations
    storage_locations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Paths
    data_dir: str = "./data"
    vocab_dir: str = "/home/user/datasets/vocabulary"
    output_dir: str = "./outputs"
    
    # Image processing
    image_size: int = 512
    # Channel ordering of image tensors emitted by the dataloader. RGB is the
    # project default (and how legacy checkpoints/configs are interpreted). All
    # other per-channel parameters in this block (normalize_mean, normalize_std,
    # pad_color) are interpreted in this same channel order with no implicit
    # reordering anywhere in the pipeline.
    color_order: str = "RGB"
    # Default normalization for ViT: inception-style (0.5/0.5/0.5) expects pixel values in
    # [0, 1] range. Using 0.5/0.5/0.5 keeps training, validation and inference
    # in sync unless explicitly overridden in the unified config. Values are
    # interpreted in the channel order specified by ``color_order`` (channel-
    # symmetric here, so identical for RGB and BGR).
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
    
    # Vocabulary
    vocab_min_frequency: int = 125

    # Worker logging (WARNING allows debugging, CRITICAL minimizes queue overhead)
    worker_log_level: str = field(default="WARNING", metadata={"help": "Log level for DataLoader workers (DEBUG, INFO, WARNING, ERROR, CRITICAL)"})

    # Validation split limiting
    max_val_samples: Optional[int] = field(default=None, metadata={"help": "Limit validation set size at split time (before Arrow cache loading)"})
    
    # Caching
    preload_files: int = 2

    # Metadata cache configuration
    metadata_cache_enabled: bool = True
    metadata_cache_workers: int = 16
    force_rebuild_metadata_cache: bool = False
    metadata_cache_staleness_check_samples: int = 100
    split_cache_version: str = "2.0"
    metadata_cache_version: str = "2.0"
    metadata_cache_use_dynamic_sampling: bool = True
    metadata_cache_use_stratified_sampling: bool = True
    cache_count_tolerance_percent: float = 0.1
    cache_count_tolerance_min: int = 100

    # Augmentation
    # Only random_flip_prob is implemented. With large datasets (5-6M images),
    # additional augmentation is generally unnecessary.
    random_flip_prob: float = 0.0  # Horizontal flip probability (disabled by default)

    # Color jitter augmentation
    color_jitter_enabled: bool = field(default=False, metadata={"help": "Enable color jitter augmentation"})
    color_jitter_brightness: float = field(default=0.1, metadata={"help": "Brightness jitter range"})
    color_jitter_brightness_p: float = field(default=0.15, metadata={"help": "Probability of brightness jitter"})
    color_jitter_contrast: float = field(default=0.1, metadata={"help": "Contrast jitter range"})
    color_jitter_contrast_p: float = field(default=0.15, metadata={"help": "Probability of contrast jitter"})
    color_jitter_saturation: float = field(default=0.1, metadata={"help": "Saturation jitter range"})
    color_jitter_saturation_p: float = field(default=0.15, metadata={"help": "Probability of saturation jitter"})

    # Random erasing augmentation
    random_erasing_enabled: bool = field(default=False, metadata={"help": "Enable random erasing"})
    random_erasing_p: float = field(default=0.25, metadata={"help": "Probability of random erasing"})
    random_erasing_scale_min: float = field(default=0.02, metadata={"help": "Min erased area fraction"})
    random_erasing_scale_max: float = field(default=0.20, metadata={"help": "Max erased area fraction"})
    random_erasing_ratio_min: float = field(default=0.3, metadata={"help": "Min aspect ratio of erased region"})
    random_erasing_ratio_max: float = field(default=3.3, metadata={"help": "Max aspect ratio of erased region"})

    # Random rotation augmentation
    random_rotation_enabled: bool = field(default=False, metadata={"help": "Enable random rotation augmentation"})
    random_rotation_p: float = field(default=0.3, metadata={"help": "Probability of applying rotation per sample"})
    random_rotation_min_degrees: float = field(default=5.0, metadata={"help": "Minimum rotation angle in degrees"})
    random_rotation_max_degrees: float = field(default=10.0, metadata={"help": "Maximum rotation angle in degrees"})

    # Gaussian blur augmentation (DeiT III 3-Augment; Touvron et al. ECCV 2022)
    gaussian_blur_enabled: bool = field(default=False, metadata={"help": "Enable Gaussian blur augmentation (DeiT III 3-Augment)"})
    gaussian_blur_p: float = field(default=0.15, metadata={"help": "Probability of applying blur per sample"})
    gaussian_blur_kernel_size: int = field(default=3, metadata={"help": "Gaussian blur kernel size (must be odd)"})
    gaussian_blur_sigma_min: float = field(default=0.1, metadata={"help": "Minimum sigma for Gaussian blur"})
    gaussian_blur_sigma_max: float = field(default=1.5, metadata={"help": "Maximum sigma for Gaussian blur"})

    # Dtype Configuration for various components
    tag_vector_dtype: str = field(default='bfloat16', metadata={"help": "Dtype for tag/label vectors ('float16', 'bfloat16', 'float32')"})
    cache_dequant_dtype: str = field(default='bfloat16', metadata={"help": "Target dtype for uint8 cache dequantization ('float16', 'bfloat16', 'float32')"})
    metric_compute_dtype: str = field(default='float32', metadata={"help": "Dtype for metric computation ('float16', 'bfloat16', 'float32')"})

    # Dataset behavior (from dataset_loader.py usage)
    patch_size: int = field(default=16, metadata={"help": "Patch size for vision transformer"})
    validate_on_init: bool = field(default=False, metadata={"help": "Validate all images on dataset init"})
    skip_error_samples: bool = field(default=True, metadata={"help": "Skip samples that cause loading errors"})
    collect_augmentation_stats: bool = field(default=False, metadata={"help": "Collect detailed augmentation stats"})

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
                    # Validate enabled storage paths exist (early detection of config errors)
                    storage_path = Path(storage_loc.path)
                    if not storage_path.exists():
                        errors.append(
                            f"Storage location {i}: path does not exist: {storage_loc.path}"
                        )
                    elif not storage_path.is_dir():
                        errors.append(
                            f"Storage location {i}: path is not a directory: {storage_loc.path}"
                        )
            except (TypeError, ValueError, ConfigValidationError) as e:
                # Expected validation errors
                errors.append(f"Storage location {i}: {str(e)}")
        # Unique priority validation for enabled locations only
        if priorities and len(priorities) != len(set(priorities)):
            dupes = [p for p in set(priorities) if priorities.count(p) > 1]
            errors.append(f"Duplicate storage location priorities detected: {sorted(dupes)}")

        # Ensure at least one storage location is enabled
        if not priorities:
            errors.append("No enabled storage locations found. At least one must be enabled.")

        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")

        if self.num_workers < 0:
            errors.append(f"num_workers must be non-negative, got {self.num_workers}")
            
        if self.vocab_min_frequency < 1:
            errors.append(f"vocab_min_frequency must be >= 1, got {self.vocab_min_frequency}")

        # New bounds checks
        if self.prefetch_factor < 1:
            errors.append(f"prefetch_factor must be >= 1, got {self.prefetch_factor}")
        if self.preload_files < 0:
            errors.append(f"preload_files must be >= 0, got {self.preload_files}")

        # Validate normalization parameters
        for param_name, param_value in [('normalize_mean', self.normalize_mean),
                                        ('normalize_std', self.normalize_std)]:
            if len(param_value) != 3:
                errors.append(f"{param_name} must have 3 values, got {len(param_value)}")

        # Validate color_order
        valid_color_orders = {"RGB", "BGR"}
        if self.color_order not in valid_color_orders:
            errors.append(
                f"color_order must be one of {sorted(valid_color_orders)}, got {self.color_order!r}"
            )

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

        if self.random_rotation_p < 0 or self.random_rotation_p > 1:
            errors.append(f"random_rotation_p must be in [0, 1], got {self.random_rotation_p}")
        if self.random_rotation_min_degrees < 0:
            errors.append(f"random_rotation_min_degrees must be >= 0, got {self.random_rotation_min_degrees}")
        if self.random_rotation_max_degrees < self.random_rotation_min_degrees:
            errors.append(f"random_rotation_max_degrees must be >= min_degrees, got {self.random_rotation_max_degrees}")
        if self.random_rotation_max_degrees > 45:
            errors.append(f"random_rotation_max_degrees must be <= 45, got {self.random_rotation_max_degrees}")

        # Validate additional dtype configurations
        valid_float_dtypes = ["float16", "bfloat16", "float32"]
        if self.tag_vector_dtype not in valid_float_dtypes:
            errors.append(f"Invalid tag_vector_dtype: {self.tag_vector_dtype}. Must be one of {valid_float_dtypes}")
        if self.cache_dequant_dtype not in valid_float_dtypes:
            errors.append(f"Invalid cache_dequant_dtype: {self.cache_dequant_dtype}. Must be one of {valid_float_dtypes}")
        if self.metric_compute_dtype not in valid_float_dtypes:
            errors.append(f"Invalid metric_compute_dtype: {self.metric_compute_dtype}. Must be one of {valid_float_dtypes}")

        if errors:
            raise ConfigValidationError("Data config validation failed:\n" + "\n".join(errors))


@dataclass
class GradientClippingConfig(BaseConfig):
    enabled: bool = True
    max_norm: float = 1.0



@dataclass
class LossConfig(BaseConfig):
    """Hyperparameters for loss functions."""
    alpha: float = 0.5
    gamma_neg: float = 3.0
    # Manual guarded gamma_neg step (todos/ASL_plan.md SS3): set to the target
    # value and restart; the ASL drive manager validates the step against the
    # phase window / hold / dwell guards and applies it, then this key should
    # be cleared back to null. When null, the checkpoint's persisted gamma_neg
    # wins over the gamma_neg value above on resume.
    gamma_neg_override: Optional[float] = None
    gamma_pos: float = 1.0
    label_smoothing: float = 0.0
    clip: float = 0.05
    class_weights: Optional[List[float]] = None  # Manual per-class weight override
    class_weight_strategy: Optional[str] = None  # None | "inverse_sqrt" | "effective_number"
    class_weight_clip_min: float = 0.05  # Floor for computed weights
    class_weight_clip_max: float = 5.0  # Cap for computed weights
    class_weight_beta: float = 0.9999  # Beta for effective_number strategy (Cui et al. 2019)

    def validate(self):
        errors = []
        if not 0.0 <= self.alpha <= 1.0:
            errors.append(f"alpha must be in [0, 1], got {self.alpha}")
        if self.gamma_neg < 0 or self.gamma_pos < 0:
            errors.append("gamma_neg and gamma_pos must be >= 0")
        if self.gamma_neg_override is not None and self.gamma_neg_override < 0:
            errors.append(
                f"gamma_neg_override must be >= 0 or null, got {self.gamma_neg_override}"
            )
        if not 0.0 <= self.label_smoothing <= 1.0:
            errors.append(
                f"label_smoothing must be in [0, 1], got {self.label_smoothing}"
            )
        if not 0.0 <= self.clip < 1.0:
            errors.append(f"clip must be in [0, 1), got {self.clip}")
        if self.class_weight_strategy is not None and self.class_weight_strategy not in ("inverse_sqrt", "effective_number"):
            errors.append(f"class_weight_strategy must be null, 'inverse_sqrt', or 'effective_number', got '{self.class_weight_strategy}'")
        if not 0.0 < self.class_weight_beta < 1.0:
            errors.append(f"class_weight_beta must be in (0, 1), got {self.class_weight_beta}")
        if self.class_weight_clip_min <= 0:
            errors.append(f"class_weight_clip_min must be > 0, got {self.class_weight_clip_min}")
        if self.class_weight_clip_max <= self.class_weight_clip_min:
            errors.append(f"class_weight_clip_max must be > class_weight_clip_min")
        if errors:
            raise ConfigValidationError("Loss config validation failed:\n" + "\n".join(errors))


@dataclass
class ASLPhaseWindowConfig(BaseConfig):
    """Per-phase gamma_neg window for the ASL drive plan (todos/ASL_plan.md SS3).

    hold_epochs: gamma_neg is frozen for the first N (phase-local, 1-based)
    epochs of the phase — Phase 1's warmup/early-learning hard hold, Phase 2's
    re-warmup freeze at the resolution switch.
    """
    gamma_neg_min: float = 5.0
    gamma_neg_max: float = 7.0
    hold_epochs: int = 8

    def validate(self):
        errors = []
        if self.gamma_neg_min < 0 or self.gamma_neg_max < 0:
            errors.append("gamma_neg_min/max must be >= 0")
        if self.gamma_neg_max < self.gamma_neg_min:
            errors.append(
                f"gamma_neg_max ({self.gamma_neg_max}) must be >= gamma_neg_min ({self.gamma_neg_min})"
            )
        if int(self.hold_epochs) < 0:
            errors.append("hold_epochs must be >= 0")
        if errors:
            raise ConfigValidationError("ASL phase window validation failed:\n" + "\n".join(errors))


@dataclass
class ASLScheduleConfig(BaseConfig):
    """Guards for MANUAL gamma_neg steps (todos/ASL_plan.md SS3).

    The descent is driven by hand (stop / edit gamma_neg_override / resume);
    this config only enforces the plan's safety rails: per-phase clamp windows,
    hold/freeze windows, and the minimum dwell between unit steps. The SS4
    adaptive controller has zero authority (shadow/logging-only, see
    ASLTelemetryConfig).
    """
    enabled: bool = True
    min_dwell_epochs: int = 3
    phase1: ASLPhaseWindowConfig = field(default_factory=lambda: ASLPhaseWindowConfig(
        gamma_neg_min=5.0, gamma_neg_max=7.0, hold_epochs=8))
    phase2: ASLPhaseWindowConfig = field(default_factory=lambda: ASLPhaseWindowConfig(
        gamma_neg_min=5.0, gamma_neg_max=6.0, hold_epochs=2))
    phase3: ASLPhaseWindowConfig = field(default_factory=lambda: ASLPhaseWindowConfig(
        gamma_neg_min=5.0, gamma_neg_max=6.0, hold_epochs=0))

    def validate(self):
        errors = []
        if int(self.min_dwell_epochs) < 0:
            errors.append("min_dwell_epochs must be >= 0")
        for name in ("phase1", "phase2", "phase3"):
            try:
                getattr(self, name).validate()
            except ConfigValidationError as e:
                errors.append(f"{name}: {e}")
        if errors:
            raise ConfigValidationError("ASL schedule validation failed:\n" + "\n".join(errors))


@dataclass
class ASLTelemetryConfig(BaseConfig):
    """Always-on ASL telemetry set (todos/ASL_plan.md SS5).

    Train-side metrics ride the already-computed detached logits at
    optimizer-update boundaries; val-side variants consume the accumulated
    probability/target matrices. The shadow controller logs the gamma the
    paper's adaptive-asymmetry law WOULD set — it never sets gamma (SS4).
    """
    enabled: bool = True
    # Compute train-side telemetry every N optimizer updates (EMA cadence)
    interval_updates: int = 100
    # Write TB scalars / persist EMA floats every N optimizer updates
    # (must be a multiple of interval_updates to fire on a compute step)
    log_every_updates: int = 500
    ema_beta: float = 0.98
    # top-K for the dp_hard non-GT capture (SS5: excludes PAD/UNK + rating tags)
    topk_hard: int = 10
    num_deciles: int = 10
    # Non-GT score histogram range/bins; watch band = the SS2 clip-cost band
    hist_min: float = 0.05
    hist_max: float = 0.95
    hist_bins: int = 18
    watch_band_low: float = 0.2
    watch_band_high: float = 0.5
    # EPR trend alarm: sustained relative drop in any decile within N epochs
    # of a gamma step -> step back up (SS5)
    epr_alarm_rel_drop: float = 0.05
    epr_alarm_window_epochs: int = 2
    # SS5 hygiene: rating tags inflate mean(p_pos); exclude them from dp metrics
    exclude_rating_tags: bool = True
    # JSON {group_name: [tags]} for the sibling-gap metric; null disables
    sibling_groups_path: Optional[str] = "./configs/confusable_groups.json"
    # SS4 shadow controller (logging only, zero authority)
    shadow_controller_enabled: bool = True
    shadow_lambda: float = 0.05
    shadow_delta_p_target: float = 0.2

    def validate(self):
        errors = []
        if int(self.interval_updates) < 1:
            errors.append("interval_updates must be >= 1")
        if int(self.log_every_updates) < 1:
            errors.append("log_every_updates must be >= 1")
        elif int(self.log_every_updates) % max(1, int(self.interval_updates)) != 0:
            errors.append(
                f"log_every_updates ({self.log_every_updates}) must be a multiple of "
                f"interval_updates ({self.interval_updates}) so logging lands on a compute step"
            )
        if not 0.0 < self.ema_beta < 1.0:
            errors.append(f"ema_beta must be in (0, 1), got {self.ema_beta}")
        if int(self.topk_hard) < 1:
            errors.append("topk_hard must be >= 1")
        if int(self.num_deciles) < 1:
            errors.append("num_deciles must be >= 1")
        if not (0.0 <= self.hist_min < self.hist_max <= 1.0):
            errors.append(f"hist range invalid: [{self.hist_min}, {self.hist_max}]")
        if int(self.hist_bins) < 1:
            errors.append("hist_bins must be >= 1")
        if not (0.0 <= self.watch_band_low < self.watch_band_high <= 1.0):
            errors.append(f"watch band invalid: [{self.watch_band_low}, {self.watch_band_high}]")
        if not 0.0 < self.epr_alarm_rel_drop < 1.0:
            errors.append(f"epr_alarm_rel_drop must be in (0, 1), got {self.epr_alarm_rel_drop}")
        if int(self.epr_alarm_window_epochs) < 0:
            errors.append("epr_alarm_window_epochs must be >= 0")
        if errors:
            raise ConfigValidationError("ASL telemetry validation failed:\n" + "\n".join(errors))


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration"""
    # Memory layout to unlock Tensor Core perf on Ampere+ (and Blackwell)
    memory_format: str = "contiguous"  # or "channels_last"
    # Basic settings
    num_epochs: int = 100
    learning_rate: float = 1e-4
    # Learning rate scaling: automatically adjusts learning_rate based on effective batch size.
    # The learning_rate field should be the BASE rate (tuned for lr_base_batch_size).
    # Modes: "sqrt" (recommended for AdamW), "linear", or "none" (use learning_rate as-is).
    lr_scaling_mode: str = "sqrt"
    lr_base_batch_size: int = 256
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    # Number of distributed processes (1 = single GPU/CPU). Referenced by the
    # FullConfig effective-batch scaling helpers; default 1 keeps them from
    # raising AttributeError if they are ever called.
    world_size: int = 1

    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adan_beta3: float = 0.99  # Beta3 for Adan optimizer
    adam_epsilon: float = 1e-8
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5  # Linear LR warmup over this many epochs
    # Number of cosine cycles (integer COUNT, as consumed by train_direct's
    # int(num_cycles)): 1 = single cosine decay (no restarts), >1 = SGDR restarts.
    # NOT the HuggingFace "fraction of a cosine wave" float semantics.
    num_cycles: int = 1
    # SGDR per-restart max_lr decay (gamma in CosineAnnealingWarmupRestarts).
    # Inert at num_cycles=1 (no restarts); read at train_direct.py via getattr,
    # so a missing field silently defaulted to 0.9 — define it to make it
    # configurable and remove that footgun before any multi-cycle run.
    cycle_decay: float = 0.9
    lr_end: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # only "bfloat16" is supported (bf16-only invariant)
    enable_anomaly_detection: bool = False

    # Gradient clipping
    max_grad_norm: float = 1.0  # deprecated, use gradient_clipping.max_norm
    gradient_clipping: GradientClippingConfig = field(default_factory=GradientClippingConfig)

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
    # Resume-aware reseeding: when true, change RNG streams per (checkpoint, epoch)
    # to avoid replaying the same early-epoch examples after a resume.
    # Deterministic per checkpoint; set false to preserve strict reproducibility across resumes.
    resume_reseed: bool = True
    # Vocabulary-resume safety: by default a resume is REFUSED unless the current
    # vocabulary's SHA256 can be verified against the SHA embedded in the checkpoint.
    # Loading against a mismatched vocabulary silently scrambles every label index.
    # Set True only to load a legacy checkpoint that predates embedded vocab SHAs
    # (you are then responsible for confirming the vocabulary matches).
    allow_unverified_vocab_resume: bool = False
    
    # Loss configuration
    tag_loss: LossConfig = field(default_factory=LossConfig)

    # Unified training-phase selector (todos/ASL_plan.md; progressive plan):
    #   1 = 320px from-scratch, 2 = 448px fine-tune, 3 = optional 512px.
    # Resuming a checkpoint whose recorded phase differs from this value
    # triggers a PHASE TRANSITION: weights load, optimizer/scheduler/scaler
    # start fresh (re-warmup), epoch counters reset to 0, and gamma_neg is
    # carried over frozen from the checkpoint's loss state.
    phase: int = 1
    # ASL gamma_neg drive guards + always-on telemetry (todos/ASL_plan.md)
    asl_schedule: ASLScheduleConfig = field(default_factory=ASLScheduleConfig)
    asl_telemetry: ASLTelemetryConfig = field(default_factory=ASLTelemetryConfig)
    # Hardware
    device: str = "cuda"
    # torch.compile() Optimization (PyTorch 2.0+)
    use_compile: bool = True
    compile_mode: str = "max-autotune"  # Options: default, reduce-overhead, max-autotune
    compile_fullgraph: bool = False  # Allow graph breaks for dynamic shapes
    compile_dynamic: bool = True  # Support varying padding mask shapes
    # Tracking
    use_tensorboard: bool = True
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
            errors.append(
                "Cannot have both deterministic=True and benchmark=True. "
                "cuDNN benchmark mode uses non-deterministic algorithms. "
                "Set benchmark=False for deterministic training, or "
                "set deterministic=False to use benchmark mode for speed."
            )
        
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")

        if self.num_epochs <= 0:
            errors.append(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.gradient_accumulation_steps <= 0:
            errors.append(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        
        valid_optimizers = ["adam", "adamw", "adamw8bit", "sgd", "rmsprop", "adagrad", "adan"]
        if self.optimizer not in valid_optimizers:
            errors.append(f"Unknown optimizer: {self.optimizer}. Must be one of {valid_optimizers}")

        valid_schedulers = ["cosine", "cosine_restarts", "step", "multistep", "plateau", "exponential"]
        if self.scheduler not in valid_schedulers:
            errors.append(f"Unknown scheduler: {self.scheduler}. Must be one of {valid_schedulers}")

        # num_cycles is an integer cycle COUNT (train_direct does int(num_cycles));
        # fractional values would silently truncate (int(0.5) == 0).
        if int(self.num_cycles) < 1:
            errors.append(f"num_cycles must be an integer >= 1 (cycle count), got {self.num_cycles}")

        # Validate beta values for Adam optimizers
        if self.optimizer in ["adam", "adamw", "adamw8bit"]:
            if not 0 <= self.adam_beta1 < 1:
                errors.append(f"adam_beta1 must be in [0, 1), got {self.adam_beta1}")
            if not 0 <= self.adam_beta2 < 1:
                errors.append(f"adam_beta2 must be in [0, 1), got {self.adam_beta2}")
        
        # Validate loss configurations
        try:
            self.tag_loss.validate()
        except ConfigValidationError as e:
            errors.append(f"tag_loss: {e}")

        # Unified phase selector + ASL drive configs
        if int(self.phase) not in (1, 2, 3):
            errors.append(f"phase must be 1, 2, or 3, got {self.phase}")
        try:
            self.asl_schedule.validate()
        except ConfigValidationError as e:
            errors.append(f"asl_schedule: {e}")
        try:
            self.asl_telemetry.validate()
        except ConfigValidationError as e:
            errors.append(f"asl_telemetry: {e}")
        
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

        # bf16-only invariant: enforce in the config layer (fail-fast) instead of
        # only deep inside train_direct/MixedPrecisionTrainer AMP setup.
        if str(self.amp_dtype).lower() not in {"bfloat16", "bf16"}:
            errors.append(
                f"amp_dtype must be 'bfloat16' (only bf16 AMP is supported), got '{self.amp_dtype}'"
            )

        if errors:
            raise ConfigValidationError("Training config validation failed:\n" + "\n".join(errors))


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration"""
    # Model
    model_path: Optional[str] = None
    precision: str = "bf16"  # Options: "fp32", "fp16", "bf16"
    # Prediction
    prediction_threshold: float = 0.2653
    top_k: Optional[int] = None
    eye_color_exclusive: bool = False  # Enforce mutual exclusivity for eye color tags

    # Performance
    max_batch_size: int = 32
    
    # Output
    output_format: str = "json"
    
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

        if self.model_path and not Path(self.model_path).exists():
            logger.debug(f"Model path does not exist: {self.model_path} (expected during training)")

        if not 0 <= self.prediction_threshold <= 1:
            errors.append(f"prediction_threshold must be in [0, 1], got {self.prediction_threshold}")

        valid_precisions = ["fp32", "fp16", "bf16"]
        if self.precision not in valid_precisions:
            errors.append(f"Invalid precision: {self.precision}. Must be one of {valid_precisions}")

        valid_formats = ["json", "text", "csv", "xml", "yaml"]
        if self.output_format not in valid_formats:
            errors.append(f"Unknown output_format: {self.output_format}. Must be one of {valid_formats}")

        if self.top_k is not None and self.top_k <= 0:
            errors.append(f"top_k must be > 0 when set, got {self.top_k}")
        if self.cache_ttl_seconds < 0:
            errors.append(f"cache_ttl_seconds must be >= 0, got {self.cache_ttl_seconds}")
        if self.max_batch_size <= 0:
            errors.append(f"max_batch_size must be >= 1, got {self.max_batch_size}")

        if errors:
            raise ConfigValidationError("Inference config validation failed:\n" + "\n".join(errors))


@dataclass
class ExportConfig(BaseConfig):
    """Model export configuration"""
    # Export format
    export_format: str = "onnx"  # onnx, torchscript, tflite, coreml
    
    # Opset 21 for broader ORT fusion support (LayerNorm, attention, GELU).
    # Opset 23+ has native Attention op but PyTorch doesn't map SDPA to it yet.
    opset_version: int = 21  # default, will be clamped at export time
    export_params: bool = True
    do_constant_folding: bool = True
    
    # Dynamic axes
    dynamic_batch_size: bool = True
    max_batch_size: int = 128

    # Optimization
    optimize: bool = True
    quantize: bool = False
    quantization_type: str = "dynamic"  # dynamic, static, qat
    
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
    # Output
    output_path: str = "./exported_model"

    # Which variant(s) to produce. Default is float16 since the model is trained
    # in bf16 — keeping inference at 16-bit roughly halves the file size and is
    # natively supported by ORT's CUDA EP. Supported: 'full' (fp32), 'fp16', 'quantized'.
    export_variants: List[str] = field(default_factory=lambda: ['fp16'])

    # Vocabulary embedding
    require_embedded_vocabulary: bool = True  # Require vocabulary to be embedded in ONNX model

    # Dynamo-based ONNX export (PyTorch 2.5+)
    # Uses torch.onnx.export(dynamo=True) instead of the legacy TorchScript tracer.
    # Produces cleaner graphs with fewer redundant nodes and lower memory usage.
    use_dynamo_export: bool = True

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
        
        if self.max_batch_size <= 0:
            errors.append(f"max_batch_size must be positive, got {self.max_batch_size}")
        
        if errors:
            raise ConfigValidationError("Export config validation failed:\n" + "\n".join(errors))


@dataclass
class ValidationDataloaderConfig(BaseConfig):
    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True

@dataclass
class ValidationPreprocessingConfig(BaseConfig):
    # Match training defaults; these should be kept in sync with DataConfig
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_size: int = 512
    patch_size: int = 16

@dataclass
class ThresholdCalibrationConfig(BaseConfig):
    """Configuration for post-training per-tag/per-bucket threshold calibration."""
    enabled: bool = False
    mode: str = "per_bucket"  # "per_tag" | "per_bucket"
    # Must equal inference.prediction_threshold (single source of truth; enforced
    # in FullConfig.validate()).
    default_threshold: float = 0.2653
    search_min: float = 0.1
    search_max: float = 0.9
    search_step: float = 0.02
    save_path: str = "./thresholds.json"

    def validate(self):
        """Validate threshold calibration configuration"""
        errors = []

        valid_modes = ["per_tag", "per_bucket"]
        if self.mode not in valid_modes:
            errors.append(f"mode must be one of {valid_modes}, got {self.mode!r}")

        if not 0 <= self.default_threshold <= 1:
            errors.append(f"default_threshold must be in [0, 1], got {self.default_threshold}")

        if not 0 <= self.search_min < self.search_max <= 1:
            errors.append(
                f"Require 0 <= search_min < search_max <= 1, "
                f"got search_min={self.search_min}, search_max={self.search_max}"
            )

        if self.search_step <= 0:
            errors.append(f"search_step must be > 0, got {self.search_step}")

        if errors:
            raise ConfigValidationError("Threshold calibration config validation failed:\n" + "\n".join(errors))


@dataclass
class ValidationConfig(BaseConfig):
    dataloader: ValidationDataloaderConfig = field(default_factory=ValidationDataloaderConfig)
    preprocessing: ValidationPreprocessingConfig = field(default_factory=ValidationPreprocessingConfig)
    # Applies only to the standalone validation_loop.py runner; the in-training
    # validation split is capped separately by data.max_val_samples.
    max_samples: Optional[int] = None  # Maximum samples to use for validation (None = use all)
    # Frequency-bin edges for the bucketed validation metrics (val_bucketed/* in
    # train_direct). None = use the built-in default [300, 500, 1000, 5000, 10000, inf].
    frequency_bins: Optional[List[float]] = None

    def validate(self):
        """Validate validation configuration"""
        errors = []

        if self.dataloader.batch_size <= 0:
            errors.append(f"dataloader.batch_size must be positive, got {self.dataloader.batch_size}")
        if self.dataloader.num_workers < 0:
            errors.append(f"dataloader.num_workers must be non-negative, got {self.dataloader.num_workers}")
        if self.dataloader.prefetch_factor < 1:
            errors.append(f"dataloader.prefetch_factor must be >= 1, got {self.dataloader.prefetch_factor}")

        if self.preprocessing.image_size <= 0:
            errors.append(f"preprocessing.image_size must be positive, got {self.preprocessing.image_size}")
        if self.preprocessing.patch_size <= 0:
            errors.append(f"preprocessing.patch_size must be positive, got {self.preprocessing.patch_size}")
        for param_name, param_value in [('preprocessing.normalize_mean', self.preprocessing.normalize_mean),
                                        ('preprocessing.normalize_std', self.preprocessing.normalize_std)]:
            if len(param_value) != 3:
                errors.append(f"{param_name} must have 3 values, got {len(param_value)}")

        if self.max_samples is not None and self.max_samples <= 0:
            errors.append(f"max_samples must be positive or null, got {self.max_samples}")

        if self.frequency_bins is not None:
            if not self.frequency_bins:
                errors.append("frequency_bins must be a non-empty list or null")
            elif any(b <= 0 for b in self.frequency_bins):
                errors.append(f"frequency_bins values must be positive, got {self.frequency_bins}")
            elif list(self.frequency_bins) != sorted(self.frequency_bins):
                errors.append(f"frequency_bins must be ascending, got {self.frequency_bins}")

        if errors:
            raise ConfigValidationError("Validation config validation failed:\n" + "\n".join(errors))


@dataclass
class TBImageLoggingConfig(BaseConfig):
    """Configuration for TensorBoard image and text sample logging."""
    max_samples: int = 32
    topk: int = 15
    log_native_resolution: bool = True
    dpi_for_figures: int = 220
    image_log_steps: int = 0  # Log images every N steps (0 = disabled)


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
    system_metrics_log_interval_steps: int = 3000  # Log system metrics to TensorBoard every N steps
    track_gpu_metrics: bool = True
    track_disk_io: bool = True
    # Visualization
    use_tensorboard: bool = True
    tensorboard_dir: str = "./tensorboard"

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
    # Remote monitoring
    enable_prometheus: bool = False
    prometheus_port: int = 8080

    # TensorBoard image logging helpers
    tb_image_logging: TBImageLoggingConfig = field(default_factory=TBImageLoggingConfig)

    # Normalization params for image denormalization before TensorBoard display
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    # History
    max_history_size: int = 1000
    history_save_interval: int = 100
    checkpoint_metrics: bool = True

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


# NOTE: AdamW8bitConfig / SchedulerType / SchedulerConfig sub-configs were removed
# from FullConfig — they were never consumed by training (a "silent second source of
# truth" with base_lr=1e-4/weight_decay=0.01 that looked authoritative but did
# nothing). The active optimizer/scheduler settings live under config.training.* and
# the standalone helpers in training_config.py.


@dataclass
class FullConfig(BaseConfig):
    """Complete configuration combining all components"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    threshold_calibration: ThresholdCalibrationConfig = field(default_factory=ThresholdCalibrationConfig)
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
        for config_name in ['model', 'data', 'training', 'inference', 'export',
                            'validation', 'threshold_calibration', 'monitor', 'debug']:
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
            # Sync image_size: data.image_size is the single source of truth
            if self.model.image_size != self.data.image_size:
                logger.info(
                    f"Syncing model.image_size ({self.model.image_size}) "
                    f"from data.image_size ({self.data.image_size})"
                )
                self.model.image_size = self.data.image_size
            if self.validation.preprocessing.image_size != self.data.image_size:
                logger.info(
                    f"Syncing validation.preprocessing.image_size ({self.validation.preprocessing.image_size}) "
                    f"from data.image_size ({self.data.image_size})"
                )
                self.validation.preprocessing.image_size = self.data.image_size
            if self.validation.preprocessing.patch_size != self.model.patch_size:
                logger.info(
                    f"Syncing validation.preprocessing.patch_size ({self.validation.preprocessing.patch_size}) "
                    f"from model.patch_size ({self.model.patch_size})"
                )
                self.validation.preprocessing.patch_size = self.model.patch_size
            # data.patch_size drives the token-level padding-mask grid and is part
            # of the resume-compatibility check; keep it locked to model.patch_size
            # (the source of truth) so the mask grid can't silently diverge from
            # the model's patch grid.
            if self.data.patch_size != self.model.patch_size:
                logger.info(
                    f"Syncing data.patch_size ({self.data.patch_size}) "
                    f"from model.patch_size ({self.model.patch_size})"
                )
                self.data.patch_size = self.model.patch_size

            # Re-validate patch divisibility on the SYNCED sizes. Each sub-config's
            # validate() ran BEFORE the image_size sync above, so it checked the
            # stale model.image_size; without this a non-divisible data.image_size
            # passes config validation and only crashes later in model construction.
            if self.model.patch_size <= 0 or self.model.image_size % self.model.patch_size != 0:
                errors.append(
                    f"model.image_size ({self.model.image_size}) must be divisible by "
                    f"model.patch_size ({self.model.patch_size}) after image-size sync"
                )
            _vp = self.validation.preprocessing
            if _vp.patch_size <= 0 or _vp.image_size % _vp.patch_size != 0:
                errors.append(
                    f"validation.preprocessing.image_size ({_vp.image_size}) must be divisible by "
                    f"validation.preprocessing.patch_size ({_vp.patch_size}) after sync"
                )

            # Prediction-threshold single source of truth: inference.prediction_threshold.
            # threshold_calibration.default_threshold must match it exactly; the two
            # knobs feed the same decision boundary (in-train F1 vs bucketed/inference
            # metrics) and silently desynchronize if edited independently.
            if self.threshold_calibration.default_threshold != self.inference.prediction_threshold:
                errors.append(
                    f"threshold_calibration.default_threshold ({self.threshold_calibration.default_threshold}) "
                    f"must equal inference.prediction_threshold ({self.inference.prediction_threshold}). "
                    f"inference.prediction_threshold is the single source of truth — "
                    f"update both keys together in configs/unified_config.yaml."
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

    # NOTE: The dataset-aware optimizer/scheduler "scaling API" that used to live
    # here (compute_effective_batch_size / scale_learning_rate / scale_weight_decay /
    # get_optimizer_kwargs / get_scheduler_kwargs, plus the AdamW8bitConfig +
    # SchedulerConfig sub-configs) was removed. It was never wired into training —
    # train_direct.py reads config.training.* directly and scales the LR via
    # training_config.scale_learning_rate — and the dead `scale_weight_decay` here
    # implemented a research-refuted 1/sqrt(N) formula. Keeping it as a second,
    # authoritative-looking source of truth was a footgun. Weight decay is fixed at
    # config.training.weight_decay by design (see memory project-weight-decay-fixed).


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
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ConfigError(f"Unknown config file format: {path.suffix}")
            
            # Create config from data
            config_class = type(self.config)
            self.config = config_class.from_dict(data)

            # Note: Validation is deferred to load_config() after env overrides are merged
            logger.info(f"Successfully loaded config from {path}")
            return self.config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML file: {e}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse JSON file: {e}")
        except (OSError, TypeError, ValueError, AttributeError, ConfigValidationError) as e:
            raise ConfigError(f"Failed to load config: {e}")
        except Exception as e:
            # Unexpected error - log details and re-raise as ConfigError
            logger.error(f"Unexpected error loading config from {path}", exc_info=True)
            raise ConfigError(f"Unexpected error loading config: {e}")
    
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
        # Use try-catch to handle race condition where file could be deleted between check and rename
        if backup:
            try:
                timestamp = f"{datetime.now():%Y%m%d_%H%M%S}_{int(time.time() * 1000000) % 1000000:06d}"
                backup_path = path.with_suffix(f'.backup_{timestamp}{path.suffix}')
                path.rename(backup_path)
                logger.info(f"Created backup at {backup_path}")
            except FileNotFoundError:
                # File doesn't exist, no backup needed
                pass
            except OSError as e:
                # Other OS errors (permissions, etc.) - log warning but continue
                logger.warning(f"Could not create backup of {path}: {e}")

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
    
    def _parse_env_value(self, value: str) -> Union[bool, int, float, str, List[Any], Dict[str, Any], None]:
        """Parse environment variable value to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Parsed value as bool, int, float, str, list, dict, or None
        """
        # Number FIRST: parse '1'/'0' as int, not bool. Otherwise an integer env
        # override of 1 or 0 (e.g. MODEL__PATCH_SIZE=1) is silently coerced to a
        # Python bool and written into an int field with no per-field validation.
        # int then float unconditionally: gating float on '.' missed scientific
        # notation ('1e-5' stayed a string and reached float fields uncoerced).
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass

        # Boolean (textual only; numeric 1/0 are handled as ints above)
        low = value.lower()
        if low in ('true', 'yes'):
            return True
        if low in ('false', 'no'):
            return False

        # Try JSON for complex types (only if it looks like JSON)
        # SECURITY NOTE JSON parsing from environment variables without size/depth limits
        # would normally be a security concern. However, this application uses only locally created
        # configuration files and environment variables under direct user control. There is no
        # external/untrusted input to environment variables in the deployment context.
        # This is acceptable for local-only applications. If deployment model changes to accept
        # external env vars (containers, cloud, CI/CD), add validation per CR-002 recommendations.
        if value.strip().startswith(('[', '{')):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                # Log warning when JSON-like value fails to parse
                # This helps catch typos in environment variable configurations
                logger.warning(
                    f"Environment variable value looks like JSON but failed to parse: "
                    f"'{value[:50]}{'...' if len(value) > 50 else ''}' - Error: {e}. "
                    f"Treating as string."
                )

        # Try JSON for null values
        if value.lower() == 'null':
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String (default)
        return value

    # NOTE: update_from_args was removed. This project is configured exclusively
    # via the unified YAML (configs/unified_config.yaml) plus ANIME_TAGGER_* env
    # overrides — there are no per-field CLI overrides (see create_config_parser).

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
                # Direct update — route through BaseConfig.update() so the value
                # gets the same dataclass type coercion as from_dict (env-var
                # strings/ints land with correct field types) instead of a raw
                # setattr that bypasses Optional/Union/nested handling.
                if hasattr(config, key):
                    if isinstance(config, BaseConfig):
                        try:
                            config.update({key: value}, validate=False)
                        except (TypeError, ValueError, AttributeError) as e:
                            logger.warning(f"Failed to set {key}={value!r}: {e}")
                    else:
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
    """Deep update target dictionary with source dictionary.

    Mutable values (lists, nested dicts/objects) from ``source`` are deep-copied
    into ``target`` so later mutation of the merged config cannot alias back into
    the source — merges must be non-destructive.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def create_config_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the training entrypoint.

    Configuration is YAML-only: configs/unified_config.yaml (plus ANIME_TAGGER_*
    env overrides) is the single source of truth. There are deliberately no
    per-field CLI overrides (--training.learning_rate etc.) — the parser exposes
    only the arguments train_direct.py actually consumes.
    """
    parser = argparse.ArgumentParser(
        description="Anime Image Tagger Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--validate-only', action='store_true', default=None, help='Only validate config and exit')

    return parser


def load_config(
    config_file: Optional[str] = None,
    config_type: ConfigType = ConfigType.FULL,
    env_prefix: str = "ANIME_TAGGER_",
    validate: bool = True
) -> BaseConfig:
    """
    Load configuration from multiple sources

    Priority: env > file > defaults
    (There are no CLI overrides — the unified YAML is the configuration source.)

    Args:
        config_file: Path to configuration file
        config_type: Type of configuration
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
        precision="bf16",
        max_batch_size=64,
    )
    hp_inference.to_yaml(output_dir / "high_performance_inference.yaml")
    
    # Mobile export config
    mobile_export = ExportConfig(
        export_format="tflite",
        optimize=True,
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
            print(f"[OK] Config file '{sys.argv[2]}' is valid")
        except Exception as e:
            print(f"[FAIL] Config validation failed: {e}")
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
                        print(f"[OK] Config file '{sys.argv[2]}' is valid")
                    except Exception as e:
                        print(f"[FAIL] Config validation failed: {e}")
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
                    print("   [OK] Default config is valid")
                except Exception as e:
                    print(f"   [FAIL] Validation failed: {e}")

                # Test 2: Save and load config
                print("\n2. Testing save/load functionality...")
                test_file = Path("test_config.yaml")
                config.to_yaml(test_file)

                manager = ConfigManager(ConfigType.FULL)
                loaded = manager.load_from_file(test_file)

                if config == loaded:
                    print("   [OK] Config saved and loaded correctly")
                else:
                    print("   [FAIL] Loaded config doesn't match original")

                # Test 3: Environment variable override
                print("\n3. Testing environment variable override...")
                os.environ["ANIME_TAGGER_TRAINING__LEARNING_RATE"] = "0.0005"
                os.environ["ANIME_TAGGER_MODEL__NUM_GROUPS"] = "20"  # Keep consistent with default tags_per_group
                os.environ["ANIME_TAGGER_DATA__BATCH_SIZE"] = "64"

                manager.update_from_env()

                assert manager.config.training.learning_rate == 0.0005
                assert manager.config.model.num_groups == 20
                assert manager.config.data.batch_size == 64
                print("   [OK] Environment overrides work correctly")

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
                    print("   [OK] Checkpoint restore successful")
                else:
                    print("   [FAIL] Checkpoint restore failed")

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
