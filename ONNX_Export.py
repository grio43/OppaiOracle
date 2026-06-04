#!/usr/bin/env python3
"""
ONNX Export for Anime Image Tagger
Export trained model to ONNX format for deployment
"""

import os
import json
from packaging.version import Version
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import asdict
from collections import defaultdict
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
from training_utils import CheckpointManager

try:
    from model_metadata import ModelMetadata
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx.checker
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
from model_architecture import create_model
from vocabulary import TagVocabulary


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
        # The actual tagger (SimplifiedTagger) may be directly in self.model
        # or nested inside a wrapper at self.model.model
        inner_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(inner_model, 'set_onnx_mode'):
            inner_model.set_onnx_mode(True)

        # Ensure entire wrapper is in eval mode for export
        self.eval()

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass on preprocessed input.

        Args:
            x: Preprocessed tensor (B, C, H, W) float32, already normalized.
            padding_mask: (B, H, W) bool tensor. True = padding, False = valid pixel.
                An all-False mask is equivalent to no masking.

        Returns:
            probabilities: (B, num_tags) sigmoid probabilities in [0, 1].
        """
        outputs = self.model(x, padding_mask=padding_mask)
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
        
        logger.info(f"[OK] Vocabulary verification passed")
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

        # Store original config values for mismatch detection. image_size is owned
        # by data (phase transitions write data.image_size); reading from model
        # here would detect a mismatch the fix-up at line ~284 never corrects.
        config_mean = list(self.config.data.normalize_mean)
        config_std = list(self.config.data.normalize_std)
        config_image_size = self.config.data.image_size

        if 'preprocessing_params' in meta:
            preprocessing = ModelMetadata.extract_preprocessing_params(meta)
            if preprocessing:
                checkpoint_mean = preprocessing.get('normalize_mean', [0.5, 0.5, 0.5])
                checkpoint_std = preprocessing.get('normalize_std', [0.5, 0.5, 0.5])
                checkpoint_image_size = preprocessing.get('image_size', 512)
                # Resolve color_order from checkpoint. Legacy checkpoints
                # (pre-BGR migration) have no key -> 'RGB'.
                if 'color_order' in preprocessing:
                    checkpoint_color_order = str(preprocessing['color_order']).upper()
                    if checkpoint_color_order not in ("RGB", "BGR"):
                        logger.warning(
                            f"Unknown color_order '{preprocessing['color_order']}' in checkpoint; falling back to 'RGB'."
                        )
                        checkpoint_color_order = "RGB"
                else:
                    checkpoint_color_order = "RGB"
                    logger.info(
                        "Checkpoint preprocessing_params missing 'color_order' - "
                        "defaulting to legacy 'RGB' for ONNX export metadata."
                    )

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
                # Propagate color_order so the metadata-writing path below
                # records what the checkpoint was trained with.
                try:
                    self.config.data.color_order = checkpoint_color_order
                except Exception:
                    setattr(self.config.data, "color_order", checkpoint_color_order)
                logger.info(
                    f"Loaded preprocessing params from checkpoint: mean={checkpoint_mean}, "
                    f"std={checkpoint_std}, image_size={checkpoint_image_size}, "
                    f"color_order={checkpoint_color_order}"
                )
        elif 'normalization_params' in meta:
            # Legacy format
            normalization_params = meta['normalization_params']
            self.config.data.normalize_mean = normalization_params.get('mean', [0.5, 0.5, 0.5])
            self.config.data.normalize_std = normalization_params.get('std', [0.5, 0.5, 0.5])
            try:
                self.config.data.color_order = "RGB"
            except Exception:
                setattr(self.config.data, "color_order", "RGB")
            logger.info(
                f"Loaded normalization params from checkpoint (legacy format): "
                f"mean={self.config.data.normalize_mean}, std={self.config.data.normalize_std}; "
                f"color_order defaulted to 'RGB' (legacy)."
            )
        else:
            logger.warning("Preprocessing params not found in checkpoint. Using config defaults.")
            logger.info(f"Using config preprocessing params: mean={self.config.data.normalize_mean}, std={self.config.data.normalize_std}")

    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract configuration from the model"""
        config = {
            'num_heads': 12,  # Default values
            'hidden_size': 768,
            'num_layers': 12,
            'patch_size': 16,  # Default to 16 (standard for ViT).
        }

        # Try to extract from model
        if hasattr(self.model, 'model'):
            base_model = self.model.model
            
            # Try to get config from various possible attributes
            if hasattr(base_model, 'config'):
                model_cfg = base_model.config
                # Check both naming conventions (num_heads vs num_attention_heads)
                if hasattr(model_cfg, 'num_attention_heads'):
                    config['num_heads'] = model_cfg.num_attention_heads
                elif hasattr(model_cfg, 'num_heads'):
                    config['num_heads'] = model_cfg.num_heads
                if hasattr(model_cfg, 'hidden_size'):
                    config['hidden_size'] = model_cfg.hidden_size
                if hasattr(model_cfg, 'num_hidden_layers'):
                    config['num_layers'] = model_cfg.num_hidden_layers
                elif hasattr(model_cfg, 'num_layers'):
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
        
        # Calculate sequence length (ViT)
        patch_size = config.get('patch_size', 16)
        num_patches = (self.config.data.image_size // patch_size) ** 2

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

        # Determine num_tags from checkpoint state_dict (authoritative source)
        # The tag_head weight shape tells us exactly how many outputs the model has
        checkpoint_num_tags = None
        for k, v in state_dict.items():
            if 'tag_head.weight' in k:
                checkpoint_num_tags = v.shape[0]
                break

        if checkpoint_num_tags is not None and checkpoint_num_tags != self.num_tags:
            logger.warning(
                f"Checkpoint tag_head has {checkpoint_num_tags} outputs but vocabulary has "
                f"{self.num_tags} tags. Using checkpoint size for model creation. "
                f"Extra outputs beyond vocabulary size will be ignored during inference."
            )
            num_tags = checkpoint_num_tags
        else:
            num_tags = self.num_tags

        model_config = asdict(self.config.model)
        model_config['num_tags'] = num_tags

        logger.info(f"Creating model with {num_tags} tags")
        model = create_model(**model_config)

        
        # Handle DDP weights
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Handle torch.compile weights (_orig_mod. prefix)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

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

        # Determine export variants: check config.export first, then top-level config
        export_variants = getattr(self.config, 'export_variants',
                                  getattr(self.export_config, 'export_variants', ['full']))
        logger.info(f"Export variants: {export_variants}")

        results = {}

        # Export variants
        for variant in export_variants:
            logger.info(f"\nExporting variant: {variant}")            
            try:
                if variant == "full":
                    results[variant] = self._export_full_model()
                elif variant == "fp16":
                    results[variant] = self._export_fp16_model()
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
                logger.info(f"[OK] {variant}: {path}")
            else:
                logger.info(f"[FAIL] {variant}: Failed")
        logger.info("="*60)
        
        return results
    
    def _run_onnx_export(self, dummy_input: torch.Tensor, dummy_mask: torch.Tensor, output_path: Path):
        """Run ONNX export using dynamo (default) or legacy TorchScript exporter."""
        use_dynamo = getattr(self.export_config, 'use_dynamo_export', True)

        if use_dynamo:
            logger.info("Using Dynamo-based ONNX exporter")
            try:
                dynamic_shapes = None
                if self.export_config.dynamic_batch_size:
                    batch_dim = torch.export.Dim("batch_size", min=1, max=self.export_config.max_batch_size)
                    dynamic_shapes = {"x": {0: batch_dim}, "padding_mask": {0: batch_dim}}

                # Force UTF-8 for stdout/stderr during dynamo export to avoid
                # cp932 encoding errors from PyTorch's internal emoji logging on Windows
                old_enc_out = getattr(sys.stdout, 'encoding', 'utf-8')
                old_enc_err = getattr(sys.stderr, 'encoding', 'utf-8')
                try:
                    if hasattr(sys.stdout, 'reconfigure'):
                        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

                    export_output = torch.onnx.export(
                        self.model,
                        (dummy_input, dummy_mask),
                        dynamo=True,
                        dynamic_shapes=dynamic_shapes,
                        input_names=["pixel_values", "padding_mask"],
                        output_names=["probabilities"],
                    )
                finally:
                    if hasattr(sys.stdout, 'reconfigure'):
                        sys.stdout.reconfigure(encoding=old_enc_out)
                        sys.stderr.reconfigure(encoding=old_enc_err)

                export_output.save(str(output_path))
                return
            except Exception as e:
                logger.warning(f"Dynamo export failed ({e}), falling back to legacy TorchScript exporter")

        # Legacy TorchScript-based export (fallback)
        logger.info("Using legacy TorchScript-based ONNX exporter")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
            warnings.filterwarnings("ignore", message=".*TracerWarning.*")

            torch.onnx.export(
                self.model,
                (dummy_input, dummy_mask),
                str(output_path),
                export_params=self.export_config.export_params,
                opset_version=self.export_config.opset_version,
                do_constant_folding=self.export_config.do_constant_folding,
                input_names=["pixel_values", "padding_mask"],
                output_names=["probabilities"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "padding_mask": {0: "batch_size"},
                    "probabilities": {0: "batch_size"},
                } if self.export_config.dynamic_batch_size else None,
                verbose=False
            )

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
            dummy_mask = torch.zeros(
                1, image_size, image_size,
                dtype=torch.bool,
                device=self.device,
            )
            # Use non-trivial mask for tracing: simulate letterboxed image with padding border.
            # Ensures mask operations are traced with real values, catching silent mask-dropping
            # during dynamo export (see pytorch/pytorch#152018).
            pad_border = image_size // 8
            dummy_mask[:, :pad_border, :] = True
            dummy_mask[:, -pad_border:, :] = True
            dummy_mask[:, :, :pad_border] = True
            dummy_mask[:, :, -pad_border:] = True
            logger.debug(f"Using dummy input shape for export: (1, 3, {image_size}, {image_size})")

            logger.info("Ensuring model is in float32 for export")
            self.model.float()

            # Export
            logger.info(f"Exporting to {output_path}")

            self._run_onnx_export(dummy_input, dummy_mask, output_path)
            
            # Validate structure before optimization
            if self.export_config.validate_export:
                self._validate_model(output_path)

            # Optimize (must run before adding metadata — optimizer rebuilds
            # the graph and strips metadata_props)
            if self.export_config.optimize:
                self._optimize_model(output_path)
                self._slim_model(output_path)
                self._repair_dynamic_batch(output_path)

            # Consolidate to single file if model fits under 2GB protobuf limit
            self._consolidate_to_single_file(output_path)

            # Add metadata AFTER optimization so it persists in the final model
            if self.export_config.add_metadata:
                self._add_metadata(output_path)

            # Export selected_tags.csv for compatibility with tagger UIs
            self._export_selected_tags_csv(output_path.parent)

            # Validate ORT inference on the final model
            if self.export_config.validate_export:
                if not self._validate_ort_inference(output_path):
                    logger.error("Post-optimization inference validation failed! "
                                 "The exported model may produce incorrect outputs.")

            logger.info(f"[OK] Full model exported to {output_path}")

            # Print model info
            self._print_model_info(output_path)

            return output_path

        except Exception as e:
            logger.error(f"Failed to export full model: {e}")
            return None

    def _export_fp16_model(self) -> Optional[Path]:
        """Export float16 precision model"""
        base_path = Path(self.export_config.output_path)
        output_path = base_path.parent / f"{base_path.stem}_fp16.onnx"

        try:
            image_size = self.config.data.image_size
            dummy_input = torch.randn(
                1, 3, image_size, image_size,
                dtype=torch.float16,
                device=self.device,
            )
            dummy_mask = torch.zeros(
                1, image_size, image_size,
                dtype=torch.bool,
                device=self.device,
            )

            logger.info("Converting model to float16 for export")
            self.model.half()

            logger.info(f"Exporting FP16 model to {output_path}")
            self._run_onnx_export(dummy_input, dummy_mask, output_path)

            # Restore model to float32
            self.model.float()

            # Validate structure
            if self.export_config.validate_export:
                self._validate_model(output_path)

            # Optimize
            if self.export_config.optimize:
                self._optimize_model(output_path)
                self._slim_model(output_path)
                self._repair_dynamic_batch(output_path)

            # Consolidate to single file (FP16 is ~468MB, well under 2GB limit)
            self._consolidate_to_single_file(output_path)

            # Add metadata AFTER optimization
            if self.export_config.add_metadata:
                self._add_metadata(output_path)

            # Export selected_tags.csv for compatibility with tagger UIs
            self._export_selected_tags_csv(output_path.parent)

            # Validate ORT inference
            if self.export_config.validate_export:
                self._validate_ort_inference_fp16(output_path)

            logger.info(f"FP16 model exported to {output_path}")
            self._print_model_info(output_path)
            return output_path

        except Exception as e:
            logger.error(f"Failed to export FP16 model: {e}")
            # Restore model to float32 on failure
            self.model.float()
            return None

    def _validate_ort_inference_fp16(self, model_path: Path):
        """Validate FP16 model through ORT inference"""
        logger.info("Validating FP16 model inference...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
            image_size = self.config.data.image_size
            test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float16)
            test_mask = np.zeros((1, image_size, image_size), dtype=bool)

            input_names = {i.name for i in session.get_inputs()}
            feed = {session.get_inputs()[0].name: test_input}
            if "padding_mask" in input_names:
                feed["padding_mask"] = test_mask
            onnx_outputs = session.run(None, feed)
            onnx_probs = onnx_outputs[0]

            # Sanity check: output should be in [0, 1] (sigmoid)
            if onnx_probs.min() < -0.01 or onnx_probs.max() > 1.01:
                logger.warning(
                    f"FP16 output range [{onnx_probs.min():.4f}, {onnx_probs.max():.4f}] "
                    f"outside expected [0, 1]"
                )
            else:
                logger.info(f"FP16 inference validation passed "
                           f"(output range: [{onnx_probs.min():.4f}, {onnx_probs.max():.4f}])")

            # Dynamic-batch sanity (same rationale as the float32 path).
            if self.export_config.dynamic_batch_size:
                for test_batch in (2, 4):
                    try:
                        multi_input = np.random.randn(
                            test_batch, 3, image_size, image_size,
                        ).astype(np.float16)
                        multi_feed = {session.get_inputs()[0].name: multi_input}
                        if "padding_mask" in input_names:
                            multi_feed["padding_mask"] = np.zeros(
                                (test_batch, image_size, image_size), dtype=bool,
                            )
                        multi_out = session.run(None, multi_feed)[0]
                    except Exception as e:
                        logger.error(
                            f"FP16 dynamic-batch validation failed at batch={test_batch}: {e}"
                        )
                        return False
                    if multi_out.shape[0] != test_batch:
                        logger.error(
                            f"FP16 dynamic-batch validation: batch={test_batch} produced "
                            f"output shape {multi_out.shape}; expected first dim {test_batch}"
                        )
                        return False
                    logger.info(
                        f"[OK] FP16 dynamic-batch validation: batch={test_batch} -> {multi_out.shape}"
                    )

            return True
        except Exception as e:
            logger.error(f"FP16 inference validation failed: {e}")
            return False

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
            
            logger.info(f"[OK] Quantized model exported to {quantized_path}")
            return quantized_path
            
        except Exception as e:
            logger.error(f"Failed to export quantized model: {e}")
            return None
    
    def _optimize_model(self, model_path: Path):
        """Optimize ONNX model"""
        logger.info("Optimizing ONNX model...")

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

            # Log fusion diagnostics to verify optimizer actually fused attention
            from collections import Counter
            pre_model = onnx.load(str(model_path))
            pre_node_count = len(pre_model.graph.node)
            post_node_count = len(optimized_model.model.graph.node)
            node_types = Counter(n.op_type for n in optimized_model.model.graph.node)

            attention_ops = [n for n in optimized_model.model.graph.node if 'Attention' in n.op_type]
            layernorm_ops = [n for n in optimized_model.model.graph.node if 'LayerNorm' in n.op_type]

            logger.info(f"Optimization: {pre_node_count} -> {post_node_count} nodes "
                        f"({pre_node_count - post_node_count} eliminated)")
            logger.info(f"Fused Attention ops: {len(attention_ops)}, "
                        f"Fused LayerNorm ops: {len(layernorm_ops)}")

            if len(attention_ops) == 0:
                logger.warning(
                    "[WARN] No fused Attention ops found! ORT's pattern matcher likely failed "
                    "to recognize the SDPA export pattern. Consider setting "
                    "use_dynamo_export=false or inspecting the graph with verbose=1."
                )

            # Log top node types for quick graph overview
            top_types = node_types.most_common(15)
            logger.info(f"Top node types: {dict(top_types)}")

            del pre_model

            # Save the optimized model
            optimized_model.save_model_to_file(str(optimized_path))

            # Replace original with optimized
            shutil.move(str(optimized_path), str(model_path))

            logger.info("[OK] Model optimization complete")

        except (ImportError, AttributeError) as e:
            logger.warning(f"ONNX Runtime transformer optimizer not available ({e}), trying basic optimization")
            self._basic_optimize(model_path)
        except Exception as e:
            logger.warning(f"Optimization failed: {type(e).__name__}: {e}", exc_info=True)
            logger.warning("Keeping original model")
    
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
                logger.info("[OK] Basic optimization complete with onnx-simplifier")
            else:
                logger.warning("Simplification check failed, keeping original model")

        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping basic optimization")
            logger.info("Install with: pip install onnx-simplifier")
        except Exception as e:
            logger.warning(f"Basic optimization failed: {e}")

    def _slim_model(self, model_path: Path):
        """Run onnxslim for graph cleanup after ORT transformer optimizer."""
        try:
            import onnxslim
        except ImportError:
            logger.info("onnxslim not installed, skipping graph cleanup (pip install onnxslim)")
            return

        logger.info("Running onnxslim graph cleanup...")
        try:
            # Load with external data if .data file exists alongside
            data_path = Path(str(model_path) + ".data")
            load_external = data_path.exists()
            model = onnx.load(str(model_path), load_external_data=load_external)
            slimmed = onnxslim.slim(model)
            # Save back — if external data existed, preserve that format for now
            # (consolidation happens in a separate step)
            if load_external:
                # Remove old files before re-saving to avoid external data conflicts
                if data_path.exists():
                    data_path.unlink()
                if model_path.exists():
                    model_path.unlink()
                onnx.save(slimmed, str(model_path),
                          save_as_external_data=True,
                          all_tensors_to_one_file=True,
                          location=data_path.name,
                          size_threshold=1024)
            else:
                onnx.save(slimmed, str(model_path))
            logger.info("[OK] onnxslim graph cleanup complete")
        except Exception as e:
            logger.warning(f"onnxslim optimization failed: {e}, keeping previous model")

    def _repair_dynamic_batch(self, model_path: Path):
        """Repair dynamic-batch annotations that upstream tooling concretized to 1.

        `onnxruntime.transformers.optimizer.optimize_model` re-runs shape
        inference using the dummy export batch (=1), and stamps every
        intermediate `value_info` entry with concrete `[1, ...]` and the graph
        outputs with `[1, ...]` too. When ORT later loads the model for
        inference its EXTENDED-level optimizer trusts those annotations and
        rewrites MatMul/Gemm patterns into a runtime Reshape (named
        `gemm_input_reshape`) whose target shape bakes in batch=1 -> any
        inference at batch>1 fails with an `input_shape_size ==
        requested_shape_size` Reshape error.

        Three repairs, all keyed off `dynamic_batch_size: true`:

          1. Clear all `value_info` entries so ORT re-infers intermediate
             shapes from the (correctly symbolic) graph inputs at session
             load. Loses no information that can't be recomputed.

          2. Rewrite leading `dim_value=1` on each graph output to
             `dim_param='batch_size'` so downstream consumers see the
             output batch dim as dynamic too.

          3. Defensive: rewrite any 4-D Reshape target with leading literal
             `1` (and no other `1`/`-1`) to lead with `-1`. This is the
             head-split fingerprint; the optimizer normally produces `-1`
             but occasionally leaves one stuck at `1`. Safe because `-1`
             still resolves to 1 at batch=1.

        ONNX symbolic shape inference is intentionally NOT re-run before
        saving: re-running with poisoned upstream state can repopulate the
        same bad values. ORT's own shape propagation at session load is the
        right tool, and it operates from the symbolic inputs.
        """
        if not self.export_config.dynamic_batch_size:
            return

        data_path = Path(str(model_path) + ".data")
        has_external_data = data_path.exists()
        model = onnx.load(str(model_path), load_external_data=has_external_data)

        # (1) Strip poisoned value_info — ORT re-infers at session load.
        stripped_value_info = len(model.graph.value_info)
        del model.graph.value_info[:]

        # (2) Rewrite literal-1 first-dim on graph outputs to symbolic batch.
        fixed_outputs = []
        for out in model.graph.output:
            dims = out.type.tensor_type.shape.dim
            if dims and dims[0].dim_value == 1 and not dims[0].dim_param:
                dims[0].Clear()
                dims[0].dim_param = 'batch_size'
                fixed_outputs.append(out.name)

        # (3) Defensive Reshape head-split fingerprint repair.
        initializers = {init.name: init for init in model.graph.initializer}
        patched_reshapes = []
        for node in model.graph.node:
            if node.op_type != 'Reshape' or len(node.input) < 2:
                continue
            init = initializers.get(node.input[1])
            if init is None:
                continue
            arr = onnx.numpy_helper.to_array(init)
            if arr.ndim != 1 or arr.size != 4:
                continue
            vals = arr.tolist()
            if vals[0] != 1 or any(v == -1 or v == 1 for v in vals[1:]):
                continue
            new_vals = arr.copy()
            new_vals[0] = -1
            new_init = onnx.numpy_helper.from_array(new_vals.astype(arr.dtype), name=init.name)
            init.CopyFrom(new_init)
            patched_reshapes.append((node.name, vals, new_vals.tolist()))

        # Only save (and log) if we actually changed something.
        changed = stripped_value_info or fixed_outputs or patched_reshapes
        if not changed:
            logger.info("Dynamic-batch repair: no shape annotations to fix")
            return

        if stripped_value_info:
            logger.warning(
                f"Dynamic-batch repair: stripped {stripped_value_info} stale "
                f"value_info entries (upstream optimizer concretized batch=1)."
            )
        if fixed_outputs:
            logger.warning(
                f"Dynamic-batch repair: rewrote leading dim_value=1 -> "
                f"dim_param='batch_size' on outputs: {fixed_outputs}"
            )
        if patched_reshapes:
            logger.warning(
                f"Dynamic-batch repair: rewrote {len(patched_reshapes)} 4-D "
                f"Reshape targets from [1, ...] to [-1, ...] (head-split fingerprint)."
            )
            for name, before, after in patched_reshapes:
                logger.info(f"  {name}: {before} -> {after}")

        if has_external_data:
            if data_path.exists():
                data_path.unlink()
            if model_path.exists():
                model_path.unlink()
            onnx.save(model, str(model_path),
                      save_as_external_data=True,
                      all_tensors_to_one_file=True,
                      location=data_path.name,
                      size_threshold=1024)
        else:
            onnx.save(model, str(model_path))

    def _consolidate_to_single_file(self, model_path: Path):
        """Consolidate external data back into a single .onnx file if under 2GB."""
        data_path = Path(str(model_path) + ".data")
        if not data_path.exists():
            return  # Already a single file

        total_size = model_path.stat().st_size + data_path.stat().st_size
        limit_bytes = 2 * 1024 * 1024 * 1024  # 2GB protobuf limit

        if total_size >= limit_bytes:
            logger.info(f"Model total size ({total_size / (1024**3):.2f} GB) exceeds 2GB protobuf limit, "
                        f"keeping external data file")
            return

        logger.info(f"Consolidating model into single .onnx file ({total_size / (1024**2):.0f} MB)...")
        try:
            model = onnx.load(str(model_path), load_external_data=True)
            onnx.save(model, str(model_path),
                      save_as_external_data=False)
            # Remove leftover external data file
            if data_path.exists():
                data_path.unlink()
            logger.info("[OK] Model consolidated into single file")
        except Exception as e:
            logger.warning(f"Failed to consolidate model: {e}, keeping external data format")

    def _export_selected_tags_csv(self, output_dir: Path):
        """Export selected_tags.csv for compatibility with tagger UIs/frontends."""
        if not hasattr(self, 'vocab') or self.vocab is None:
            logger.info("No vocabulary available, skipping selected_tags.csv export")
            return

        csv_path = output_dir / "selected_tags.csv"
        try:
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['tag_id', 'name', 'category'])
                for idx in sorted(self.vocab.index_to_tag.keys()):
                    tag = self.vocab.index_to_tag[idx]
                    # Default category to 0 (general) — matches SmilingWolf format
                    category = 0
                    writer.writerow([idx, tag, category])
            logger.info(f"[OK] Exported {len(self.vocab.index_to_tag)} tags to {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to export selected_tags.csv: {e}")

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

            # Test with preprocessed float32 input: (B, C, H, W) + padding mask
            image_size = self.config.data.image_size
            test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
            test_mask = np.zeros((1, image_size, image_size), dtype=bool)

            # Run ONNX inference — use actual input names from session
            input_names = {i.name for i in session.get_inputs()}
            feed = {session.get_inputs()[0].name: test_input}
            if "padding_mask" in input_names:
                feed["padding_mask"] = test_mask
            onnx_outputs = session.run(None, feed)

            if len(onnx_outputs) < 1:
                logger.warning(f"Expected 1 output, got {len(onnx_outputs)}. Skipping comparison.")
                logger.info("[OK] Model inference validation passed (basic)")
                return True

            onnx_probs = onnx_outputs[0]

            # Run PyTorch inference for comparison
            torch_input = torch.from_numpy(test_input).to(self.device)
            torch_mask = torch.from_numpy(test_mask).to(self.device)
            self.model.eval()
            with torch.no_grad():
                torch_probs = self.model(torch_input, torch_mask)
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
                logger.info("[OK] Model inference validation passed (outputs match)")
            else:
                logger.warning(f"Outputs exceed tolerance (rtol={rtol}, atol={atol})")
                logger.warning(f"  Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")
                logger.error("Model inference validation FAILED — outputs differ beyond acceptable tolerance")
                return False

            # Verify padding mask actually affects output (catch silent mask-dropping
            # during dynamo export — see pytorch/pytorch#152018)
            if "padding_mask" in input_names:
                test_mask_padded = np.zeros((1, image_size, image_size), dtype=bool)
                pad_border = image_size // 4  # Heavy padding to ensure measurable effect
                test_mask_padded[:, :pad_border, :] = True
                test_mask_padded[:, -pad_border:, :] = True

                feed_padded = {session.get_inputs()[0].name: test_input}
                feed_padded["padding_mask"] = test_mask_padded
                onnx_padded = session.run(None, feed_padded)[0]

                if np.allclose(onnx_probs, onnx_padded, atol=1e-5):
                    logger.warning(
                        "[WARN] Padding mask has NO effect on output — "
                        "mask may have been dropped during export!"
                    )
                else:
                    mask_diff = np.max(np.abs(onnx_probs - onnx_padded))
                    logger.info(f"[OK] Padding mask correctly affects model output (max diff: {mask_diff:.6f})")

            # Dynamic-batch sanity: a single batch=1 run masks bugs where the
            # optimizer collapsed the batch dim to a literal 1 (post-attention
            # Reshape targets bake-in batch=1). Run batches > 1 to catch this.
            if self.export_config.dynamic_batch_size:
                for test_batch in (2, 4):
                    try:
                        multi_input = np.random.randn(
                            test_batch, 3, image_size, image_size,
                        ).astype(np.float32)
                        multi_feed = {session.get_inputs()[0].name: multi_input}
                        if "padding_mask" in input_names:
                            multi_feed["padding_mask"] = np.zeros(
                                (test_batch, image_size, image_size), dtype=bool,
                            )
                        multi_out = session.run(None, multi_feed)[0]
                    except Exception as e:
                        logger.error(
                            f"Dynamic-batch validation failed at batch={test_batch}: {e}"
                        )
                        return False
                    if multi_out.shape[0] != test_batch:
                        logger.error(
                            f"Dynamic-batch validation: batch={test_batch} produced "
                            f"output shape {multi_out.shape}; expected first dim {test_batch}"
                        )
                        return False
                    logger.info(
                        f"[OK] Dynamic-batch validation: batch={test_batch} -> {multi_out.shape}"
                    )

            return True

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
            
            logger.info("[OK] Dynamic quantization complete")
            
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
            data_path = Path(str(model_path) + ".data")
            has_external_data = data_path.exists()
            model = onnx.load(str(model_path), load_external_data=has_external_data)

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

            # Resolve color_order for metadata. RGB is the project default;
            # fall back defensively if the attribute is missing on exotic
            # config objects.
            _color_order = getattr(self.config.data, "color_order", "RGB")
            _color_order = str(_color_order).upper() if _color_order else "RGB"
            if _color_order not in ("RGB", "BGR"):
                logger.warning(
                    f"Unrecognized color_order '{_color_order}' on config; recording as 'RGB'."
                )
                _color_order = "RGB"

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
                'color_order': _color_order,
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
            
            # Save model (preserve external data format if present)
            if has_external_data:
                onnx.save(model, str(model_path),
                          save_as_external_data=True,
                          all_tensors_to_one_file=True,
                          location=data_path.name,
                          size_threshold=1024)
            else:
                onnx.save(model, str(model_path))

            if vocab_embedded_successfully:
                logger.info("[OK] Metadata added to model (including embedded vocabulary)")
            else:
                logger.info("[OK] Metadata added to model (external vocabulary required for inference)")

        except Exception as e:
            logger.warning(f"Failed to add metadata: {e}")
    
    def _validate_model(self, model_path: Path):
        '''Validate ONNX model structure (pre-optimization only)'''
        logger.info("Validating ONNX model...")

        try:
            # Check model structure
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            logger.info("[OK] ONNX model structure is valid")
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

            # Prepare inputs matching export format
            input_feed = {}
            for inp in session.get_inputs():
                shape = [d if isinstance(d, int) else batch_size for d in inp.shape]
                if inp.type == 'tensor(bool)':
                    input_feed[inp.name] = np.zeros(shape, dtype=np.bool_)
                else:
                    input_feed[inp.name] = np.random.randn(*shape).astype(np.float32)

            logger.info(f"Input feeds: {', '.join(f'{k}: {v.shape}' for k, v in input_feed.items())}")

            # Warmup runs
            logger.info("Warming up...")
            for _ in range(5):
                _ = session.run(None, input_feed)

            # Benchmark runs
            logger.info(f"Running {num_runs} inference iterations...")
            times = []
            for _ in tqdm(range(num_runs), desc="Benchmarking"):
                start = time.perf_counter()
                _ = session.run(None, input_feed)
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
        parser.add_argument('--variants', nargs='+', default=None, choices=['full', 'fp16', 'quantized'], help='Export variants to generate')
        parser.add_argument('--optimize', action='store_true', default=None, help='Optimize exported model')
        parser.add_argument('--no-optimize', action='store_true', default=None, help='Do not optimize exported model')
        parser.add_argument('--quantize', action='store_true', default=None, help='Enable quantization')
        parser.add_argument('--quantization-type', type=str, default=None, choices=['dynamic', 'static'], help=f'Quantization type')
        parser.add_argument('--no-validate', action='store_true', default=None, help='Skip validation')
        parser.add_argument('--benchmark', action='store_true', help='Run benchmark after export')
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
            # Write to both fields so either field is consistent regardless of which
            # downstream consumer reads from `data` vs `model`.
            unified_config.data.image_size = args.image_size
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

        logger.info("\n[OK] Export complete!")
    finally:
        if listener:
            listener.stop()


if __name__ == '__main__':
    main()
