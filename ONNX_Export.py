#!/usr/bin/env python3
"""
ONNX Export for Anime Image Tagger
Export trained model to ONNX format for deployment
"""

import os
import json
import pathlib
from packaging.version import Version
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import yaml
import base64
import gzip
import hashlib
import time
import warnings
import sys
# Base libraries

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from training_utils import CheckpointManager

try:
    from model_metadata import ModelMetadata
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnx import shape_inference
    import onnx.checker
    import onnx.helper
    import onnx.numpy_helper
    from Configuration_System import ConfigManager, ConfigType, FullConfig
except ImportError as e:
    print(f"Error importing ONNX libraries: {e}")
    print("Please install: pip install onnx  (CPU: pip install onnxruntime  |  GPU: pip install onnxruntime-gpu)")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None, total=None):
        return iterable

# Import our modules
from model_architecture import create_model, VisionTransformerConfig, SwinV2Tagger
from vocabulary import load_vocabulary_for_training, TagVocabulary


# Configure logging
logger = logging.getLogger(__name__)


MIN_ONNX = Version("1.16.0")
MIN_ORT  = Version("1.20.0")


def _fail(msg: str):
    logger.error(msg)
    raise RuntimeError(msg)


def _check_versions_and_env(opset: int) -> None:
    onnx_v = Version(onnx.__version__)
    ort_v = Version(ort.__version__)
    if onnx_v < MIN_ONNX:
        _fail(f"ONNX >= {MIN_ONNX} required (found {onnx_v}). Try: pip install -U 'onnx>={MIN_ONNX},<2'")
    if ort_v < MIN_ORT:
        _fail(f"onnxruntime >= {MIN_ORT} required (found {ort_v}). For GPU: pip install -U 'onnxruntime-gpu>={MIN_ORT}'")
    if opset < 18:
        _fail(f"opset >= 18 required (requested {opset}). Set export.opset_version to 18 or 19.")

class InferenceWrapper(nn.Module):
    """Thin wrapper: preprocessed input -> model -> sigmoid probabilities.

    Preprocessing (letterbox, pad, normalize) is NOT included in the ONNX graph.
    Consumers must preprocess externally using params from the model metadata.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

        # Enable ONNX-compatible attention mode (uses SDPA instead of flex_attention)
        # The actual tagger (SimplifiedTagger/SwinV2Tagger) may be directly in self.model
        # or nested inside a wrapper at self.model.model
        inner_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(inner_model, 'set_onnx_mode'):
            inner_model.set_onnx_mode(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on preprocessed input.

        Args:
            x: Preprocessed tensor (B, C, H, W) float32, already normalized.

        Returns:
            probabilities: (B, num_tags) sigmoid probabilities in [0, 1].
        """
        outputs = self.model(x)
        tag_logits = outputs['tag_logits'] if isinstance(outputs, dict) else outputs
        return torch.sigmoid(tag_logits)

class ONNXExporter:
    """Main ONNX export class"""

    def __init__(self, config: FullConfig):
        self.config = config
        self.export_config = config.export
        if self.export_config.opset_version < 18:
            logger.warning(f"Raising opset_version from {self.export_config.opset_version} to 18 (minimum).")
            self.export_config.opset_version = 18
        _check_versions_and_env(self.export_config.opset_version)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(self.export_config.output_path).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load vocabulary
        try:
            # Check for embedded vocabulary in checkpoint first
            checkpoint_path = Path(self.config.training.resume_from)
            if checkpoint_path.exists():
                checkpoint_dir = checkpoint_path.parent
                manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
                checkpoint = manager.load_checkpoint(checkpoint_path=str(checkpoint_path))
                if checkpoint:
                    meta = checkpoint
                    if 'vocab_b64_gzip' in meta:
                        logger.info("Found embedded vocabulary in checkpoint, extracting...")
                        vocab_data = ModelMetadata.extract_vocabulary(meta)
                        if vocab_data:
                            self.vocab = TagVocabulary()
                            self.vocab.tag_to_index = vocab_data['tag_to_index']
                            self.vocab.index_to_tag = {int(k): v for k, v in vocab_data['index_to_tag'].items()}
                            self.vocab.tag_frequencies = vocab_data.get('tag_frequencies', {})
                            self.num_tags = len(self.vocab.tag_to_index)
                            logger.info(
                                f"Successfully extracted embedded vocabulary with {self.num_tags} tags"
                            )
                        else:
                            logger.error("Failed to extract embedded vocabulary, falling back to external file")

            # If vocabulary not loaded from checkpoint, load from file
            if not hasattr(self, 'vocab'):
                vocab_path = Path(self.config.vocab_path)
                
                if vocab_path.is_dir():
                    vocab_path = vocab_path / "vocabulary.json"
                
                # Validate vocabulary path before attempting to use it
                if not vocab_path.exists():
                    # Try canonical fallback path (script directory)
                    canonical_path = Path(os.path.dirname(__file__)) / "vocabulary.json"
                    if canonical_path.exists():
                        logger.info(f"Using canonical vocabulary path: {canonical_path}")
                        vocab_path = canonical_path

                if not vocab_path.exists():
                    raise FileNotFoundError(
                        f"Vocabulary file not found at {vocab_path}.\n"
                        f"Cannot export model without valid vocabulary.\n"
                        f"Please provide vocabulary using:\n"
                        f"  --vocab_path /path/to/vocabulary.json"
                    )

                logger.info(f"Loading vocabulary from {vocab_path}")
                self.vocab = TagVocabulary(vocab_path)
                logger.info(f"Loaded vocabulary with {len(self.vocab.tag_to_index)} tags")
                self.num_tags = len(self.vocab.tag_to_index)

            # CRITICAL: Verify vocabulary before export
            self._verify_vocabulary()

        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            raise

    def _verify_vocabulary(self):
        """Verify that vocabulary contains real tags, not placeholders"""
        logger.info("Verifying vocabulary integrity before export...")
        
        placeholder_tags = []
        real_tags_sample = []
        
        for tag, idx in self.vocab.tag_to_index.items():
            if tag.startswith("tag_") and len(tag) > 4 and tag[4:].isdigit():
                placeholder_tags.append(tag)
            elif tag not in ["<PAD>", "<UNK>"]:
                real_tags_sample.append(tag)
                if len(real_tags_sample) >= 20:  # Sample more tags for verification
                    break
        
        # Check for placeholder tags
        if len(placeholder_tags) > 100:  # More than 100 placeholders is definitely wrong
            raise ValueError(
                f"CRITICAL: Vocabulary contains {len(placeholder_tags)} placeholder tags!\n"
                f"Examples: {placeholder_tags[:10]}\n"
                f"This vocabulary is corrupted with 'tag_XXX' placeholders instead of real tags.\n"
                f"The exported ONNX model would be unusable.\n"
                f"Please use the correct vocabulary.json from training."
            )
        
        logger.info(f"✓ Vocabulary verification passed")
        logger.info(f"  Sample real tags: {real_tags_sample[:5]}")
        

        # Load model
        self.model = self._load_model()
        self.model_config = self._extract_model_config()
        
        # Extract and update preprocessing params from checkpoint if available
        self._update_preprocessing_params()

    def _update_preprocessing_params(self):
        """Update preprocessing parameters from checkpoint metadata if available.

        Mirrors the logic in Inference_Engine.py:537-577 to extract and use
        preprocessing params from the checkpoint, with fallback to config defaults.
        """
        checkpoint_path = Path(self.config.training.resume_from)
        if not checkpoint_path.exists():
            logger.warning("Checkpoint not found, using config defaults for preprocessing params")
            return

        # Load checkpoint metadata
        checkpoint_dir = checkpoint_path.parent
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        checkpoint = manager.load_checkpoint(checkpoint_path=str(checkpoint_path))

        if not checkpoint:
            logger.warning("Could not load checkpoint, using config defaults for preprocessing params")
            return

        meta = checkpoint

        # Store original config values for mismatch detection
        config_mean = list(self.config.data.normalize_mean)
        config_std = list(self.config.data.normalize_std)
        config_image_size = self.config.model.image_size

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

                self.config.data.normalize_mean = checkpoint_mean
                self.config.data.normalize_std = checkpoint_std
                self.config.data.image_size = checkpoint_image_size
                logger.info(f"Loaded preprocessing params from checkpoint: mean={checkpoint_mean}, std={checkpoint_std}, image_size={checkpoint_image_size}")
        elif 'normalization_params' in meta:
            # Legacy format
            normalization_params = meta['normalization_params']
            self.config.data.normalize_mean = normalization_params.get('mean', [0.5, 0.5, 0.5])
            self.config.data.normalize_std = normalization_params.get('std', [0.5, 0.5, 0.5])
            logger.info(f"Loaded normalization params from checkpoint (legacy format): mean={self.config.data.normalize_mean}, std={self.config.data.normalize_std}")
        else:
            logger.warning("Preprocessing params not found in checkpoint. Using config defaults.")
            logger.info(f"Using config preprocessing params: mean={self.config.data.normalize_mean}, std={self.config.data.normalize_std}")

    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract configuration from the model"""
        config = {
            'num_heads': 12,  # Default values
            'hidden_size': 768,
            'num_layers': 12,
            'patch_size': 16,  # Default to 16 (standard for ViT). Note: SwinV2 always uses 4.
        }

        # SwinV2 models have different architecture - extract actual config values
        if hasattr(self.model, 'model') and isinstance(self.model.model, SwinV2Tagger):
            logger.info("SwinV2Tagger detected, extracting config for ONNX export")
            swin_model = self.model.model

            # Extract from _swin_config if available
            if hasattr(swin_model, '_swin_config') and swin_model._swin_config is not None:
                swin_cfg = swin_model._swin_config

                # num_heads is typically a tuple (e.g., (8, 16, 32, 64)) - use first stage for optimization
                if hasattr(swin_cfg, 'num_heads'):
                    num_heads = swin_cfg.num_heads
                    config['num_heads'] = num_heads[0] if isinstance(num_heads, (list, tuple)) else num_heads

                # embed_dim is the base embedding dimension
                if hasattr(swin_cfg, 'embed_dim'):
                    config['embed_dim'] = swin_cfg.embed_dim
                    # hidden_size for SwinV2 scales with stages, use embed_dim as base
                    # NOTE: This will be overridden by feature_dim below if available
                    config['hidden_size'] = swin_cfg.embed_dim

                # depths gives number of transformer blocks per stage
                if hasattr(swin_cfg, 'depths'):
                    depths = swin_cfg.depths
                    config['num_layers'] = sum(depths) if isinstance(depths, (list, tuple)) else depths
                    config['depths'] = depths
                
                # Extract patch_size if available in swin_config
                if hasattr(swin_cfg, 'patch_size'):
                    config['patch_size'] = swin_cfg.patch_size

                # Additional SwinV2-specific parameters that may be useful
                if hasattr(swin_cfg, 'window_size'):
                    config['window_size'] = swin_cfg.window_size
                if hasattr(swin_cfg, 'mlp_ratio'):
                    config['mlp_ratio'] = swin_cfg.mlp_ratio

                logger.info(f"  Extracted SwinV2 config: num_heads={config.get('num_heads')}, "
                           f"embed_dim={config.get('embed_dim')}, num_layers={config.get('num_layers')}, "
                           f"window_size={config.get('window_size')}, patch_size={config.get('patch_size')}")

            # Extract feature_dim from the model itself (this is the correct final dimension)
            # For SwinV2, feature_dim = embed_dim * 2^(num_stages-1), NOT embed_dim
            if hasattr(swin_model, 'feature_dim'):
                config['feature_dim'] = swin_model.feature_dim
                # Use feature_dim as hidden_size for SwinV2 (it's the actual output dimension)
                config['hidden_size'] = swin_model.feature_dim
                logger.info(f"  Feature dimension: {config['feature_dim']}")
            elif 'embed_dim' in config and 'depths' in config:
                # Fallback: calculate feature_dim from embed_dim and num_stages
                num_stages = len(config['depths']) if isinstance(config['depths'], (list, tuple)) else 4
                config['feature_dim'] = config['embed_dim'] * (2 ** (num_stages - 1))
                config['hidden_size'] = config['feature_dim']
                logger.info(f"  Calculated feature dimension: {config['feature_dim']} (embed_dim * 2^{num_stages-1})")

            # Extract from backbone if available (fallback)
            if hasattr(swin_model, 'backbone'):
                backbone = swin_model.backbone
                if hasattr(backbone, 'num_features'):
                    config['hidden_size'] = backbone.num_features
                if hasattr(backbone, 'embed_dim'):
                    config['embed_dim'] = backbone.embed_dim
                # Try to get patch_size from backbone
                if hasattr(backbone, 'patch_embed') and hasattr(backbone.patch_embed, 'patch_size'):
                    # patch_size might be a tuple
                    ps = backbone.patch_embed.patch_size
                    config['patch_size'] = ps[0] if isinstance(ps, (tuple, list)) else ps

            return config

        # Try to extract from model
        if hasattr(self.model, 'model'):
            base_model = self.model.model
            
            # Try to get config from various possible attributes
            if hasattr(base_model, 'config'):
                model_cfg = base_model.config
                if hasattr(model_cfg, 'num_heads'):
                    config['num_heads'] = model_cfg.num_heads
                if hasattr(model_cfg, 'hidden_size'):
                    config['hidden_size'] = model_cfg.hidden_size
                if hasattr(model_cfg, 'num_layers'):
                    config['num_layers'] = model_cfg.num_layers
                if hasattr(model_cfg, 'patch_size'):
                    config['patch_size'] = model_cfg.patch_size
            
            # Try to infer from model architecture
            elif hasattr(base_model, 'encoder'):
                encoder = base_model.encoder
                if hasattr(encoder, 'layers'):
                    config['num_layers'] = len(encoder.layers)
                    if len(encoder.layers) > 0:
                        layer = encoder.layers[0]
                        if hasattr(layer, 'self_attn'):
                            if hasattr(layer.self_attn, 'num_heads'):
                                config['num_heads'] = layer.self_attn.num_heads
                            if hasattr(layer.self_attn, 'embed_dim'):
                                config['hidden_size'] = layer.self_attn.embed_dim
        
        # Calculate sequence length based on architecture
        # Check if this is a SwinV2 model first to set correct patch_size
        is_swin = hasattr(self.model, 'model') and isinstance(self.model.model, SwinV2Tagger)

        # Extract patch_size from config if available, otherwise use architecture defaults
        if is_swin:
            # Try to use patch_size extracted from model config (set earlier in this method)
            if 'patch_size' in config and config['patch_size'] is not None:
                patch_size = config['patch_size']
                logger.debug(f"Using SwinV2 patch_size from model config: {patch_size}")
            else:
                # Fallback: SwinV2 typically uses patch_size=4 in timm implementations
                patch_size = 4
                config['patch_size'] = patch_size
                logger.warning(
                    f"Could not extract patch_size from SwinV2 model config, "
                    f"using fallback value: {patch_size}. This may cause incorrect "
                    f"sequence length calculations if the model uses a different patch size."
                )
        else:
            patch_size = config.get('patch_size', 16)
        num_patches = (self.config.model.image_size // patch_size) ** 2

        if is_swin:
            # SwinV2 doesn't use CLS tokens - calculate final resolution after all stages
            # Each stage (except first) reduces spatial dimensions by 2x via patch merging
            # Final resolution = image_size / (patch_size * 2^(num_stages-1))
            depths = config.get('depths', [2, 2, 6, 2])  # Default SwinV2 depths
            num_stages = len(depths) if isinstance(depths, (list, tuple)) else 4
            
            # Use initial_patch_size=4 for SwinV2 (standard from timm implementation)
            # If we extracted patch_size=4 from config, use it; otherwise assume standard SwinV2 patch size
            swin_patch_size = 4
            if patch_size == 4:
                swin_patch_size = 4
            
            # Total downsampling factor = initial_patch_size * 2^(num_stages-1)
            total_downsample = swin_patch_size * (2 ** (num_stages - 1))
            final_resolution = self.config.model.image_size // total_downsample
            
            config['sequence_length'] = final_resolution ** 2
            logger.info(f"  SwinV2 sequence length: {config['sequence_length']} "
                       f"(resolution: {final_resolution}x{final_resolution}, "
                       f"downsample: {total_downsample}x)")
        else:
            # ViT uses CLS token(s), add +2 for special tokens
            config['sequence_length'] = num_patches + 2  # +2 for special tokens (CLS, etc.)

        return config
        
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        checkpoint_path = self.config.training.resume_from
        logger.info(f"Loading model from {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_dir = Path(checkpoint_path).parent
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        checkpoint = manager.load_checkpoint(checkpoint_path=str(checkpoint_path))
        if not checkpoint:
            raise FileNotFoundError(f"Could not load checkpoint from {checkpoint_path}")

        state_dict = checkpoint.pop('state_dict')
        meta = checkpoint
        
        num_tags = self.num_tags

        model_config = asdict(self.config.model)
        model_config['num_labels'] = num_tags
        
        logger.info(f"Creating model with {num_tags} tags")
        model = create_model(**model_config)

        # Additional check: Verify the model's tag_head matches vocabulary
        if hasattr(model, 'tag_head'):
            tag_head_out_features = model.tag_head.out_features
            if tag_head_out_features != num_tags:
                raise RuntimeError(
                    f"Model tag_head output size ({tag_head_out_features}) doesn't match "
                    f"vocabulary size ({num_tags}). This checkpoint is incompatible with "
                    f"the vocabulary file. Please use:\n"
                    f"  1. The correct vocabulary that was used during training, OR\n"
                    f"  2. A checkpoint that was trained with this vocabulary.\n"
                    f"Cannot export model with mismatched vocabulary."
                )

        
        # Handle DDP weights
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Filter out removed rating_head keys from old checkpoints
        state_dict = {k: v for k, v in state_dict.items() if 'rating_head' not in k}

        # Load state dict with strict=False to handle minor mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys[:5]}...")
        
        model.eval()
        
        # Wrap model for inference (preprocessing is external; sigmoid applied on output)
        wrapped_model = InferenceWrapper(model)
        wrapped_model.to(self.device)
        
        return wrapped_model
    
    def export(self):
        """Export model to ONNX format"""
        logger.info("Starting ONNX export...")

        # Final vocabulary check before export
        if not hasattr(self, 'vocab') or self.vocab is None:
            raise RuntimeError("Vocabulary not loaded, cannot export")

        if len(self.vocab.tag_to_index) < 100:
            raise ValueError(
                f"Vocabulary too small ({len(self.vocab.tag_to_index)} tags). "
                f"This appears to be an invalid vocabulary."
            )

        # SwinV2 ONNX export - BLOCKED BY DEFAULT due to torch.roll tracing limitations
        architecture_type = getattr(self.config.model, 'architecture_type', 'vit')
        if architecture_type == 'swinv2':
            # Check if user explicitly wants to force SwinV2 ONNX export (not recommended)
            force_swinv2_export = getattr(self.export_config, 'force_swinv2_export', False)

            error_message = (
                "\n" + "=" * 70 + "\n"
                "ERROR: SwinV2 ONNX EXPORT IS NOT SUPPORTED\n"
                "=" * 70 + "\n"
                "SwinV2's shifted window attention mechanism uses torch.roll() for\n"
                "cyclic shifts, which cannot be properly exported to ONNX.\n\n"
                "TECHNICAL REASON:\n"
                "  torch.roll() performs cyclic tensor shifts that ONNX cannot represent.\n"
                "  The ONNX exporter traces a single execution path, but torch.roll()\n"
                "  behavior depends on runtime tensor values. This causes:\n"
                "  - Incorrect attention patterns in exported model\n"
                "  - Shifted window positions become static/wrong\n"
                "  - Model outputs differ significantly from PyTorch\n\n"
                "RECOMMENDED SOLUTIONS:\n"
                "  1. Use ViT architecture for ONNX export (model.architecture_type='vit')\n"
                "     ViT uses standard attention that exports cleanly to ONNX\n"
                "  2. Use TorchScript export instead of ONNX for SwinV2 models\n"
                "  3. Deploy SwinV2 directly with PyTorch for production\n\n"
                "TO FORCE EXPORT (NOT RECOMMENDED - may produce broken models):\n"
                "  Set export.force_swinv2_export=True in config or via CLI\n"
                "  WARNING: Exported model will likely produce incorrect predictions!\n"
                + "=" * 70
            )

            if not force_swinv2_export:
                logger.error(error_message)
                raise RuntimeError(
                    "SwinV2 ONNX export is blocked by default.\n\n"
                    "REASON: SwinV2's shifted window attention uses torch.roll() which cannot\n"
                    "be properly traced to ONNX. The resulting model will have incorrect\n"
                    "attention patterns and produce wrong predictions.\n\n"
                    "OPTIONS:\n"
                    "  1. Use ViT architecture instead (recommended for ONNX)\n"
                    "  2. Use TorchScript export for SwinV2\n"
                    "  3. Set export.force_swinv2_export=True to override (NOT RECOMMENDED)\n\n"
                    "See https://github.com/microsoft/Swin-Transformer/issues/104 for details."
                )
            else:
                logger.warning(error_message)
                logger.warning(
                    "PROCEEDING WITH SwinV2 ONNX EXPORT (force_swinv2_export=True)\n"
                    "The exported model may produce INCORRECT PREDICTIONS!\n"
                    "Thoroughly validate against PyTorch inference before any use."
                )

        logger.info(f"Export variants: {self.config.export_variants}")
        
        results = {}

        # Export variants
        for variant in self.config.export_variants:
            logger.info(f"\nExporting variant: {variant}")            
            try:
                if variant == "full":
                    results[variant] = self._export_full_model()
                elif variant == "quantized":
                    results[variant] = self._export_quantized_model()
                else:
                    logger.warning(f"Unknown variant: {variant}")
            except Exception as e:
                logger.error(f"Failed to export {variant} variant: {e}")
                results[variant] = None
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("EXPORT SUMMARY")
        logger.info("="*60)
        for variant, path in results.items():
            if path:
                logger.info(f"✓ {variant}: {path}")
            else:
                logger.info(f"✗ {variant}: Failed")
        logger.info("="*60)
        
        return results
    
    def _export_full_model(self) -> Optional[Path]:
        """Export full precision model"""
        output_path = Path(self.export_config.output_path)
        
        try:
            # Dummy input: preprocessed tensor (B, C, H, W) float32
            # Input is always image_size x image_size after external preprocessing
            image_size = self.config.data.image_size
            dummy_input = torch.randn(
                1, 3, image_size, image_size,
                dtype=torch.float32,
                device=self.device,
            )
            logger.debug(f"Using dummy input shape for export: (1, 3, {image_size}, {image_size})")

            logger.info("Ensuring model is in float32 for export")
            self.model.float()

            # Export
            logger.info(f"Exporting to {output_path}")

            use_dynamo = getattr(self.export_config, 'use_dynamo_export', False)

            if use_dynamo:
                # Dynamo-based export (PyTorch 2.5+) — better graph capture
                logger.info("Using Dynamo-based ONNX exporter")
                dynamic_shapes = None
                if self.export_config.dynamic_batch_size:
                    batch_dim = torch.export.Dim("batch_size", min=1, max=self.export_config.max_batch_size)
                    dynamic_shapes = {"x": {0: batch_dim}}

                export_output = torch.onnx.export(
                    self.model,
                    (dummy_input,),
                    dynamo=True,
                    dynamic_shapes=dynamic_shapes,
                    input_names=["pixel_values"],
                    output_names=["probabilities"],
                )
                export_output.save(str(output_path))
            else:
                # Legacy TorchScript-based export
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
                    warnings.filterwarnings("ignore", message=".*TracerWarning.*")

                    torch.onnx.export(
                        self.model,
                        dummy_input,
                        str(output_path),
                        export_params=self.export_config.export_params,
                        opset_version=self.export_config.opset_version,
                        do_constant_folding=self.export_config.do_constant_folding,
                        input_names=["pixel_values"],
                        output_names=["probabilities"],
                        dynamic_axes={
                            "pixel_values": {0: "batch_size"},
                            "probabilities": {0: "batch_size"},
                        } if self.export_config.dynamic_batch_size else None,
                        verbose=False
                    )
            
            # Validate structure before optimization
            if self.export_config.validate_export:
                self._validate_model(output_path)

            # Optimize (must run before adding metadata — optimizer rebuilds
            # the graph and strips metadata_props)
            if self.export_config.optimize:
                self._optimize_model(output_path)

            # Add metadata AFTER optimization so it persists in the final model
            if self.export_config.add_metadata:
                self._add_metadata(output_path)

            # Validate ORT inference on the final model
            if self.export_config.validate_export:
                if not self._validate_ort_inference(output_path):
                    logger.error("Post-optimization inference validation failed! "
                                 "The exported model may produce incorrect outputs.")
            
            logger.info(f"✓ Full model exported to {output_path}")
            
            # Print model info
            self._print_model_info(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export full model: {e}")
            return None
    
    
    def _export_quantized_model(self) -> Optional[Path]:
        """Export quantized model"""
        base_path = Path(self.export_config.output_path)
        
        try:
            # Ensure full model exists
            if not base_path.exists():
                logger.info("Full model not found, exporting first...")
                if not self._export_full_model():
                    raise RuntimeError("Failed to export full model")
            
            if self.export_config.quantization_type == "dynamic":
                quantized_path = base_path.parent / f"{base_path.stem}_quantized_dynamic.onnx"
                self._quantize_dynamic(base_path, quantized_path)
                
            elif self.export_config.quantization_type == "static":
                quantized_path = base_path.parent / f"{base_path.stem}_quantized_static.onnx"
                self._quantize_static(base_path, quantized_path)
                
            else:
                logger.warning(f"Unknown quantization type: {self.export_config.quantization_type}")
                return None
            
            logger.info(f"✓ Quantized model exported to {quantized_path}")
            return quantized_path
            
        except Exception as e:
            logger.error(f"Failed to export quantized model: {e}")
            return None
    
    def _optimize_model(self, model_path: Path):
        """Optimize ONNX model"""
        logger.info("Optimizing ONNX model...")

        # SwinV2 models are not compatible with BERT/ViT transformer optimizer
        # Use basic optimization with onnx-simplifier instead
        if hasattr(self.model, 'model') and isinstance(self.model.model, SwinV2Tagger):
            logger.info("SwinV2Tagger detected, using basic optimization (BERT/ViT optimizer not compatible)")
            self._basic_optimize(model_path)
            return

        # Check opset version to determine optimization strategy
        try:
            model = onnx.load(str(model_path))
            opset_version = model.opset_import[0].version if model.opset_import else 16

            # Warn if using older opset that doesn't support LayerNormalization
            if opset_version < 17:
                logger.warning(
                    f"Model uses opset {opset_version} which doesn't support LayerNormalization. "
                    f"Some transformer optimizations will be skipped. "
                    f"Consider upgrading to opset 17+ for better performance."
                )
        except Exception as e:
            logger.warning(f"Could not determine opset version: {e}")
            opset_version = self.export_config.opset_version

        try:
            # Try to use ONNX Runtime transformer optimizer (correct API)
            from onnxruntime.transformers.optimizer import optimize_model
            from onnxruntime.transformers.fusion_options import FusionOptions

            optimized_path = model_path.parent / f"{model_path.stem}_temp_opt.onnx"

            # Get model config
            cfg = self.model_config

            # Create fusion options — prefer 'vit' (ORT 1.17+), fall back to 'bert'
            try:
                fusion_options = FusionOptions('vit')
                model_type = 'vit'
            except Exception:
                fusion_options = FusionOptions('bert')
                model_type = 'bert'
                logger.info("ORT does not support model_type='vit', using 'bert' fallback")
            fusion_options.enable_gelu = True
            fusion_options.enable_bias_gelu = True
            fusion_options.enable_attention = True

            # Only enable LayerNorm fusions if opset supports it
            if opset_version >= 17:
                fusion_options.enable_skip_layer_norm = True
                fusion_options.enable_layer_norm = True
                logger.info(f"Enabling LayerNormalization fusions (opset {opset_version})")
            else:
                fusion_options.enable_skip_layer_norm = False
                fusion_options.enable_layer_norm = False
                logger.info(
                    f"Disabling LayerNormalization fusions for opset {opset_version} compatibility"
                )

            # Optimize using the correct API
            optimized_model = optimize_model(
                input=str(model_path),
                model_type=model_type,
                num_heads=cfg.get('num_heads', 12),
                hidden_size=cfg.get('hidden_size', 768),
                optimization_options=fusion_options,
                opt_level=2,
                use_gpu=torch.cuda.is_available(),
                only_onnxruntime=False,  # Apply both ONNX and ORT optimizations
                verbose=0,  # Set to 1 for debugging optimization issues
                # Note: float16 and input_int32 parameters removed - not supported in current API
            )

            # Save the optimized model
            optimized_model.save_model_to_file(str(optimized_path))

            # Replace original with optimized
            shutil.move(str(optimized_path), str(model_path))

            logger.info("✓ Model optimization complete")

        except (ImportError, AttributeError) as e:
            logger.warning(f"ONNX Runtime transformer optimizer not available ({e}), trying basic optimization")
            self._basic_optimize(model_path)
        except Exception as e:
            logger.warning(f"Optimization failed: {e}, keeping original model")
    
    def _basic_optimize(self, model_path: Path):
        """Basic ONNX optimization using onnx-simplifier"""
        try:
            from onnxsim import simplify

            model = onnx.load(str(model_path))

            batch_size = self.config.data.batch_size
            image_size = self.config.data.image_size

            # Simplify with onnx-simplifier
            model_simp, check = simplify(
                model,
                check_n=3,
                perform_optimization=True,
                skip_fuse_bn=False,
                input_shapes={'pixel_values': [1, 3, image_size, image_size]}
            )

            if check:
                onnx.save(model_simp, str(model_path))
                logger.info("✓ Basic optimization complete with onnx-simplifier")
            else:
                logger.warning("Simplification check failed, keeping original model")

        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping basic optimization")
            logger.info("Install with: pip install onnx-simplifier")
        except Exception as e:
            logger.warning(f"Basic optimization failed: {e}")

    def _validate_ort_inference(self, model_path: Path):
        """Validate model through ORT inference and compare with PyTorch outputs.

        Validates probabilities output matches between PyTorch and ONNX Runtime
        within tolerance. Uses preprocessed float32 input matching the new export format.
        """
        logger.info("Validating model inference (comparing PyTorch vs ONNX)...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(str(model_path), providers=providers)

            # Test with preprocessed float32 input: (B, C, H, W)
            image_size = self.config.data.image_size
            test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

            # Run ONNX inference — use actual input name from session
            input_name = session.get_inputs()[0].name
            onnx_outputs = session.run(None, {input_name: test_input})

            if len(onnx_outputs) < 1:
                logger.warning(f"Expected 1 output, got {len(onnx_outputs)}. Skipping comparison.")
                logger.info("✓ Model inference validation passed (basic)")
                return True

            onnx_probs = onnx_outputs[0]

            # Run PyTorch inference for comparison
            torch_input = torch.from_numpy(test_input).to(self.device)
            self.model.eval()
            with torch.no_grad():
                torch_probs = self.model(torch_input)
                torch_probs = torch_probs.cpu().numpy()

            # Compare outputs
            rtol = self.export_config.tolerance_rtol
            atol = self.export_config.tolerance_atol

            max_diff = np.max(np.abs(torch_probs - onnx_probs))
            mean_diff = np.mean(np.abs(torch_probs - onnx_probs))

            logger.info(f"  Probabilities - max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")

            # Sanity check: output should be in [0, 1] (sigmoid)
            if onnx_probs.min() < -0.01 or onnx_probs.max() > 1.01:
                logger.warning(
                    f"Output range [{onnx_probs.min():.4f}, {onnx_probs.max():.4f}] "
                    f"outside expected [0, 1] — sigmoid may not be applied correctly"
                )

            ok = np.allclose(torch_probs, onnx_probs, rtol=rtol, atol=atol)

            if ok:
                logger.info("✓ Model inference validation passed (outputs match)")
                return True
            else:
                logger.warning(f"Outputs exceed tolerance (rtol={rtol}, atol={atol})")
                logger.warning(f"  Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")
                logger.error("Model inference validation FAILED — outputs differ beyond acceptable tolerance")
                return False

        except Exception as e:
            logger.error(f"Inference validation failed: {e}")
            return False

    def _quantize_dynamic(self, input_path: Path, output_path: Path):
        """Apply dynamic quantization"""
        logger.info("Applying dynamic quantization...")
        
        try:
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8,
                optimize_model=False,  # Already optimized upstream
                per_channel=True,
                reduce_range=True
            )
            
            # Validate quantized model
            if self.export_config.validate_export:
                self._validate_model(output_path)
            
            logger.info("✓ Dynamic quantization complete")
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            raise
    
    def _quantize_static(self, input_path: Path, output_path: Path):
        """Apply static quantization (requires calibration data)"""
        logger.info("Static quantization requires calibration data")
        logger.warning("Static quantization not fully implemented, using dynamic quantization instead")
        
        # For now, fall back to dynamic quantization
        # A full implementation would:
        # 1. Load or generate calibration dataset
        # 2. Create calibration data reader
        # 3. Run static quantization with calibration
        
        self._quantize_dynamic(input_path, output_path)
    
    def _add_metadata(self, model_path: Path):
        """Add metadata to ONNX model"""
        try:
            model = onnx.load(str(model_path))

            # Clear existing metadata
            del model.metadata_props[:]

            # Prepare vocabulary for embedding
            vocab_b64 = ''
            vocab_sha = ''
            vocab_embedded_successfully = False  # Initialize to prevent NameError

            # First, try to use the vocabulary we already loaded
            if hasattr(self, 'vocab') and self.vocab is not None:
                try:
                    # Create vocabulary data structure
                    vocab_data = {
                        'tag_to_index': self.vocab.tag_to_index,
                        'index_to_tag': {str(k): v for k, v in self.vocab.index_to_tag.items()},
                        'tag_frequencies': getattr(self.vocab, 'tag_frequencies', {})
                    }

                    # Compress vocabulary
                    vocab_json = json.dumps(vocab_data, ensure_ascii=False)
                    vocab_bytes = vocab_json.encode('utf-8')
                    vocab_compressed = gzip.compress(vocab_bytes)
                    vocab_b64 = base64.b64encode(vocab_compressed).decode('utf-8')
                    vocab_sha = hashlib.sha256(vocab_bytes).hexdigest()

                    vocab_embedded_successfully = True
                    logger.info(f"Embedded vocabulary from loaded vocab with {len(self.vocab.tag_to_index)} tags")

                except Exception as e:
                    logger.error(f"Failed to embed loaded vocabulary: {e}")
                    if self.export_config.require_embedded_vocabulary:
                        raise RuntimeError(
                            f"Failed to embed vocabulary in ONNX model: {e}\n"
                            f"Cannot export model without embedded vocabulary.\n"
                            f"To export anyway (not recommended), set require_embedded_vocabulary=False"
                        ) from e

            # Fallback: try to load from file if not already embedded
            if not vocab_embedded_successfully:
                vp = Path(self.config.vocab_path)
                vocab_path = vp / "vocabulary.json" if vp.is_dir() else vp
                if vocab_path.exists():
                    temp_checkpoint: Dict[str, Any] = {}
                    temp_checkpoint = ModelMetadata.embed_vocabulary(temp_checkpoint, vocab_path)
                    # Check if embedding was successful by verifying non-empty values
                    vocab_b64 = temp_checkpoint.get('vocab_b64_gzip', '')
                    vocab_sha = temp_checkpoint.get('vocab_sha256', '')
                    if vocab_b64 and vocab_sha:
                        # Validate the embedded vocabulary data
                        try:
                            # Verify the embedded data is valid and checksum matches
                            vocab_bytes = gzip.decompress(base64.b64decode(vocab_b64))
                            computed_sha = hashlib.sha256(vocab_bytes).hexdigest()
                            if computed_sha == vocab_sha:
                                vocab_embedded_successfully = True
                                logger.info(f"\u2713 Vocabulary successfully embedded (SHA256: {vocab_sha[:8]}...)")
                            else:
                                logger.warning(f"Vocabulary checksum mismatch: expected {vocab_sha}, got {computed_sha}")
                        except Exception as e:
                            logger.warning(f"Failed to validate embedded vocabulary: {e}")
                else:
                    logger.warning(f"Vocabulary file not found at {vocab_path}, model will require external vocabulary")

            # Check if embedding succeeded
            if not vocab_embedded_successfully:
                if self.export_config.require_embedded_vocabulary:
                    raise RuntimeError(
                        "No vocabulary available for embedding.\n"
                        "ONNX export requires embedded vocabulary for reproducible inference.\n"
                        "Please provide vocabulary via --vocab_path or ensure checkpoint has embedded vocab.\n"
                        "To export anyway (not recommended), set require_embedded_vocabulary=False"
                    )
                else:
                    logger.warning(
                        "Exporting model without embedded vocabulary. "
                        "Inference will require external vocabulary file!"
                    )

            # Add metadata
            metadata = {
                'model_description': self.export_config.model_description,
                'model_author': self.export_config.model_author,
                'model_version': self.export_config.model_version,
                'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_tags': str(len(self.vocab.tag_to_index)),
                'image_size': str(self.config.data.image_size),
                'patch_size': str(self.config.model.patch_size),
                'normalize_mean': json.dumps(self.config.data.normalize_mean),
                'normalize_std': json.dumps(self.config.data.normalize_std),
                'pad_color': json.dumps(list(self.config.data.pad_color)),
                'output_activation': 'sigmoid',
                'input_format': 'BCHW_float32_normalized',
                'preprocessing': 'external',
                'framework': 'PyTorch',
                'framework_version': torch.__version__,
                'onnx_version': onnx.__version__,
                'opset_version': str(self.export_config.opset_version),
                'device': str(self.device),
            }

            # Only add vocabulary metadata if embedding was successful
            # This prevents empty strings from being added to metadata
            if vocab_embedded_successfully:
                metadata['vocab_format_version'] = '1'
                metadata['vocab_sha256'] = vocab_sha
                metadata['vocab_b64_gzip'] = vocab_b64
            
            for key, value in metadata.items():
                meta = model.metadata_props.add()
                meta.key = key
                meta.value = value
            
            # Save model
            onnx.save(model, str(model_path))

            if vocab_embedded_successfully:
                logger.info("✓ Metadata added to model (including embedded vocabulary)")
            else:
                logger.info("✓ Metadata added to model (external vocabulary required for inference)")

        except Exception as e:
            logger.warning(f"Failed to add metadata: {e}")
    
    def _validate_model(self, model_path: Path):
        '''Validate ONNX model structure (pre-optimization only)'''
        logger.info("Validating ONNX model...")

        try:
            # Check model structure
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            logger.info("✓ ONNX model structure is valid")
            # Do NOT run inference validation here - save for after optimization
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

    def _print_model_info(self, model_path: Path):
        """Print information about exported model"""
        try:
            model = onnx.load(str(model_path))
            
            # Get model size
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            
            # Count operations
            op_types = defaultdict(int)
            for node in model.graph.node:
                op_types[node.op_type] += 1
            
            # Get input/output info
            inputs = []
            for i in model.graph.input:
                shape = []
                if i.type.HasField('tensor_type'):
                    for d in i.type.tensor_type.shape.dim:
                        if d.HasField('dim_value'):
                            shape.append(d.dim_value)
                        elif d.HasField('dim_param'):
                            shape.append(d.dim_param)
                        else:
                            shape.append('?')
                inputs.append((i.name, shape))
            
            outputs = []
            for o in model.graph.output:
                shape = []
                if o.type.HasField('tensor_type'):
                    for d in o.type.tensor_type.shape.dim:
                        if d.HasField('dim_value'):
                            shape.append(d.dim_value)
                        elif d.HasField('dim_param'):
                            shape.append(d.dim_param)
                        else:
                            shape.append('?')
                outputs.append((o.name, shape))
            
            # Count parameters
            total_params = 0
            for init in model.graph.initializer:
                dims = init.dims
                if dims:
                    params = np.prod(dims)
                    total_params += params
            
            logger.info("\n" + "="*60)
            logger.info("ONNX MODEL INFORMATION")
            logger.info("="*60)
            logger.info(f"Model path: {model_path}")
            logger.info(f"Model size: {model_size:.2f} MB")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Opset version: {model.opset_import[0].version if model.opset_import else 'Unknown'}")
            logger.info(f"\nInputs:")
            for name, shape in inputs:
                logger.info(f"  {name}: {shape}")
            logger.info(f"\nOutputs:")
            for name, shape in outputs:
                logger.info(f"  {name}: {shape}")
            logger.info(f"\nOperation types ({len(op_types)} unique):")
            for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {op_type}: {count}")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to print model info: {e}")
    
    def benchmark(self, model_path: Path, num_runs: int = 100):
        """Benchmark ONNX model performance"""
        logger.info(f"\nBenchmarking model: {model_path}")

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None

        batch_size = self.config.data.batch_size
        image_size = self.config.data.image_size

        try:
            # Create session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(model_path), providers=providers)

            # Log which provider is being used
            logger.info(f"Using providers: {session.get_providers()}")

            # Prepare input matching export format: (B, C, H, W) float32
            input_data = np.random.randn(
                batch_size, 3, image_size, image_size
            ).astype(np.float32)

            input_name = session.get_inputs()[0].name

            # Warmup runs
            logger.info("Warming up...")
            for _ in range(5):
                _ = session.run(None, {input_name: input_data})

            # Benchmark runs
            logger.info(f"Running {num_runs} inference iterations...")
            times = []
            for _ in tqdm(range(num_runs), desc="Benchmarking"):
                start = time.perf_counter()
                _ = session.run(None, {input_name: input_data})
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            # Compute statistics
            times = np.array(times)
            results = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'median_ms': np.median(times),
                'p95_ms': np.percentile(times, 95),
                'p99_ms': np.percentile(times, 99),
                'throughput_fps': 1000 / np.mean(times) * batch_size
            }

            logger.info("\n" + "="*60)
            logger.info("BENCHMARK RESULTS")
            logger.info("="*60)
            logger.info(f"Model: {model_path.name}")
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"Image size: {image_size}x{image_size}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Iterations: {num_runs}")
            logger.info("-"*60)
            logger.info(f"Mean latency: {results['mean_ms']:.2f} ms")
            logger.info(f"Std deviation: {results['std_ms']:.2f} ms")
            logger.info(f"Min latency: {results['min_ms']:.2f} ms")
            logger.info(f"Max latency: {results['max_ms']:.2f} ms")
            logger.info(f"Median latency: {results['median_ms']:.2f} ms")
            logger.info(f"P95 latency: {results['p95_ms']:.2f} ms")
            logger.info(f"P99 latency: {results['p99_ms']:.2f} ms")
            logger.info(f"Throughput: {results['throughput_fps']:.1f} FPS")
            logger.info("="*60 + "\n")
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return None



def main():
    """Main entry point for ONNX export"""
    import argparse
    from utils.logging_setup import setup_logging

    listener = setup_logging()

    try:
        # Load unified config first to get defaults
        try:
            manager = ConfigManager(config_type=ConfigType.FULL)
            unified_config = manager.load_from_file("configs/unified_config.yaml")
        except Exception as e:
            logger.error(f"Could not load unified_config.yaml: {e}. Cannot proceed without configuration.")
            sys.exit(1)

        parser = argparse.ArgumentParser(description='Export Anime Tagger model to ONNX')
        parser.add_argument('checkpoint', nargs='?', default=None, help='Path to model checkpoint')
        parser.add_argument('--vocab_path', type=str, default=None, help='Path to vocabulary file or directory')
        parser.add_argument('-o', '--output', type=str, default=None, help=f'Output ONNX model path')
        parser.add_argument('-b', '--batch-size', type=int, default=None, help='Batch size for export')
        parser.add_argument('-s', '--image-size', type=int, default=None, help=f'Input image size')
        parser.add_argument('--opset', type=int, default=None, help=f'ONNX opset version')
        parser.add_argument('--variants', nargs='+', default=None, choices=['full', 'quantized'], help='Export variants to generate')
        parser.add_argument('--optimize', action='store_true', default=None, help='Optimize exported model')
        parser.add_argument('--no-optimize', action='store_true', default=None, help='Do not optimize exported model')
        parser.add_argument('--quantize', action='store_true', default=None, help='Enable quantization')
        parser.add_argument('--quantization-type', type=str, default=None, choices=['dynamic', 'static'], help=f'Quantization type')
        parser.add_argument('--no-validate', action='store_true', default=None, help='Skip validation')
        parser.add_argument('--benchmark', action='store_true', help='Run benchmark after export')
        parser.add_argument('--force-rebuild-head', action='store_true', help='Recreate tag head if its out_features does not match the vocabulary size')
        parser.add_argument('--force-swinv2', action='store_true', help='Force SwinV2 export despite torch.roll limitations')
        parser.add_argument('--benchmark-runs', type=int, default=100, help='Number of benchmark iterations')

        args = parser.parse_args()

        # Override config with CLI args
        if args.checkpoint:
            unified_config.training.resume_from = args.checkpoint
        if args.vocab_path:
            unified_config.vocab_path = args.vocab_path
        if args.output:
            unified_config.export.output_path = args.output
        if args.batch_size:
            unified_config.data.batch_size = args.batch_size
        if args.image_size:
            unified_config.model.image_size = args.image_size
        if args.opset:
            unified_config.export.opset_version = args.opset
        if args.variants:
            unified_config.export.export_variants = args.variants
        if args.optimize is not None:
            unified_config.export.optimize = not args.no_optimize
        if args.quantize is not None:
            unified_config.export.quantize = args.quantize
        if args.quantization_type:
            unified_config.export.quantization_type = args.quantization_type
        if args.no_validate is not None:
            unified_config.export.validate_export = not args.no_validate
        if args.force_swinv2:
            unified_config.export.force_swinv2_export = True
        
        # Create exporter
        exporter = ONNXExporter(unified_config)
        
        # Export
        results = exporter.export()

        # Benchmark if requested
        if args.benchmark:
            logger.info("\n" + "="*60)
            logger.info("RUNNING BENCHMARKS")
            logger.info("="*60)

            for variant, path in results.items():
                if path and path.exists():
                    exporter.benchmark(path, args.benchmark_runs)

        logger.info("\n✓ Export complete!")
    finally:
        if listener:
            listener.stop()


if __name__ == '__main__':
    main()
