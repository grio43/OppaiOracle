#!/usr/bin/env python3
"""
ONNX Export for Anime Image Tagger
Export trained model to ONNX format for deployment
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import time
import warnings
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnx import shape_inference
    import onnx.checker
    import onnx.helper
    import onnx.numpy_helper
except ImportError as e:
    print(f"Error importing ONNX libraries: {e}")
    print("Please install: pip install onnx onnxruntime")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None, total=None):
        return iterable

# Import our modules
from model_architecture import create_model, VisionTransformerConfig
from tag_vocabulary import load_vocabulary_for_training


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export"""
    # Model paths
    checkpoint_path: str
    vocab_dir: str
    output_path: str = "model.onnx"
    
    # Export settings
    opset_version: int = 16
    input_names: List[str] = field(default_factory=lambda: ["input_image"])
    output_names: List[str] = field(default_factory=lambda: ["predictions", "scores"])
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    
    # Model configuration
    batch_size: int = 1
    image_size: int = 640
    export_params: bool = True
    do_constant_folding: bool = True
    
    # Optimization settings
    optimize: bool = True
    optimize_for_mobile: bool = False
    quantize: bool = False
    quantization_type: str = "dynamic"  # dynamic, static, qat
    
    # Validation
    validate_export: bool = True
    tolerance_rtol: float = 1e-3
    tolerance_atol: float = 1e-5
    
    # Export variants
    export_variants: List[str] = field(default_factory=lambda: ["full"])
    
    # Metadata
    add_metadata: bool = True
    model_description: str = "Anime Image Tagger Model"
    model_author: str = "AnimeTaggers"
    model_version: str = "1.0"
    
    def __post_init__(self):
        if self.dynamic_axes is None:
            self.dynamic_axes = {
                "input_image": {0: "batch_size"},
                "predictions": {0: "batch_size"},
                "scores": {0: "batch_size"}
            }


class ModelWrapper(nn.Module):
    """Wrapper to simplify model output for ONNX export"""
    
    def __init__(self, model: nn.Module, config: ONNXExportConfig, threshold: float = 0.5):
        super().__init__()
        self.model = model
        self.config = config
        self.threshold = threshold
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with simplified output
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            
        Returns:
            predictions: Binary predictions (batch_size, num_tags)
            scores: Confidence scores (batch_size, num_tags)
        """
        # Get model output
        outputs = self.model(x)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('output', outputs))
            if isinstance(logits, dict):
                # Handle nested dict structure
                logits = logits.get('logits', next(iter(logits.values())))
        else:
            logits = outputs
        
        # Handle hierarchical output - flatten it
        if logits.dim() == 3:
            batch_size = logits.shape[0]
            logits = logits.view(batch_size, -1)
        
        # Convert to probabilities
        scores = torch.sigmoid(logits)
        
        # Get binary predictions
        predictions = (scores > self.threshold).float()
        
        return predictions, scores


class ONNXExporter:
    """Main ONNX export class"""
    
    def __init__(self, config: ONNXExportConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config.output_path).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load vocabulary
        try:
            self.vocab = load_vocabulary_for_training(Path(config.vocab_dir))
            logger.info(f"Loaded vocabulary with {len(self.vocab.tag_to_index)} tags")
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            raise
        
        # Load model
        self.model = self._load_model()
        self.model_config = self._extract_model_config()
        
    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract configuration from the model"""
        config = {
            'num_heads': 12,  # Default values
            'hidden_size': 768,
            'num_layers': 12,
            'patch_size': 32,
        }
        
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
        
        # Calculate sequence length
        num_patches = (self.config.image_size // config['patch_size']) ** 2
        config['sequence_length'] = num_patches + 2  # +2 for special tokens (CLS, etc.)
        
        return config
        
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        logger.info(f"Loading model from {self.config.checkpoint_path}")
        
        if not Path(self.config.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint_path}")
        
        try:
            checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Extract model config
        model_config = None
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            if isinstance(model_config, dict) and 'model_config' in model_config:
                model_config = model_config['model_config']
        
        if model_config is None:
            logger.warning("No config found in checkpoint, using default VisionTransformerConfig")
            model_config = VisionTransformerConfig()
        
        # Create model
        try:
            if isinstance(model_config, dict):
                model = create_model(**model_config)
            else:
                model = create_model(**asdict(model_config))
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DDP weights
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load state dict with strict=False to handle minor mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys[:5]}...")
        
        model.eval()
        
        # Wrap model
        wrapped_model = ModelWrapper(model, self.config)
        wrapped_model.to(self.device)
        
        return wrapped_model
    
    def export(self):
        """Export model to ONNX format"""
        logger.info("Starting ONNX export...")
        logger.info(f"Export variants: {self.config.export_variants}")
        
        results = {}
        
        # Export variants
        for variant in self.config.export_variants:
            logger.info(f"\nExporting variant: {variant}")
            
            try:
                if variant == "full":
                    results[variant] = self._export_full_model()
                elif variant == "mobile":
                    results[variant] = self._export_mobile_model()
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
        output_path = Path(self.config.output_path)
        
        try:
            # Create dummy input
            dummy_input = torch.randn(
                self.config.batch_size,
                3,
                self.config.image_size,
                self.config.image_size,
                device=self.device
            )
            
            # Export
            logger.info(f"Exporting to {output_path}")
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    str(output_path),
                    export_params=self.config.export_params,
                    opset_version=self.config.opset_version,
                    do_constant_folding=self.config.do_constant_folding,
                    input_names=self.config.input_names,
                    output_names=self.config.output_names,
                    dynamic_axes=self.config.dynamic_axes,
                    verbose=False
                )
            
            # Add metadata
            if self.config.add_metadata:
                self._add_metadata(output_path)
            
            # Optimize
            if self.config.optimize:
                self._optimize_model(output_path)
            
            # Validate
            if self.config.validate_export:
                self._validate_model(output_path)
            
            logger.info(f"✓ Full model exported to {output_path}")
            
            # Print model info
            self._print_model_info(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export full model: {e}")
            return None
    
    def _export_mobile_model(self) -> Optional[Path]:
        """Export optimized model for mobile deployment"""
        base_path = Path(self.config.output_path)
        mobile_path = base_path.parent / f"{base_path.stem}_mobile.onnx"
        
        try:
            # Use smaller batch size for mobile
            original_batch_size = self.config.batch_size
            self.config.batch_size = 1
            
            # Create temp path
            temp_path = base_path.parent / "temp_mobile.onnx"
            
            # Export
            dummy_input = torch.randn(
                1, 3, self.config.image_size, self.config.image_size,
                device=self.device
            )
            
            torch.onnx.export(
                self.model,
                dummy_input,
                str(temp_path),
                export_params=True,
                opset_version=self.config.opset_version,
                do_constant_folding=True,
                input_names=self.config.input_names,
                output_names=self.config.output_names,
                dynamic_axes=None,  # Fixed batch size for mobile
                verbose=False
            )
            
            # Optimize for mobile
            self._optimize_for_mobile(temp_path, mobile_path)
            
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
            
            self.config.batch_size = original_batch_size
            
            logger.info(f"✓ Mobile model exported to {mobile_path}")
            return mobile_path
            
        except Exception as e:
            logger.error(f"Failed to export mobile model: {e}")
            self.config.batch_size = original_batch_size
            return None
    
    def _export_quantized_model(self) -> Optional[Path]:
        """Export quantized model"""
        base_path = Path(self.config.output_path)
        
        try:
            # Ensure full model exists
            if not base_path.exists():
                logger.info("Full model not found, exporting first...")
                if not self._export_full_model():
                    raise RuntimeError("Failed to export full model")
            
            if self.config.quantization_type == "dynamic":
                quantized_path = base_path.parent / f"{base_path.stem}_quantized_dynamic.onnx"
                self._quantize_dynamic(base_path, quantized_path)
                
            elif self.config.quantization_type == "static":
                quantized_path = base_path.parent / f"{base_path.stem}_quantized_static.onnx"
                self._quantize_static(base_path, quantized_path)
                
            else:
                logger.warning(f"Unknown quantization type: {self.config.quantization_type}")
                return None
            
            logger.info(f"✓ Quantized model exported to {quantized_path}")
            return quantized_path
            
        except Exception as e:
            logger.error(f"Failed to export quantized model: {e}")
            return None
    
    def _optimize_model(self, model_path: Path):
        """Optimize ONNX model"""
        logger.info("Optimizing ONNX model...")
        
        try:
            # Try to use ONNX Runtime transformer optimizer
            from onnxruntime.transformers import optimizer
            
            optimized_path = model_path.parent / f"{model_path.stem}_temp_opt.onnx"
            
            # Get model config
            cfg = self.model_config
            
            # Create optimizer with dynamic config
            model_optimizer = optimizer.create_optimizer(
                model=str(model_path),
                num_heads=cfg.get('num_heads', 12),
                hidden_size=cfg.get('hidden_size', 768),
                sequence_length=cfg.get('sequence_length', 1024),
                input_int32=False,
                float16=False,  # Keep FP32 for compatibility
                use_gpu=torch.cuda.is_available(),
                opt_level=2,  # Use moderate optimization level
                optimization_options=None,
                provider='CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'
            )
            
            # Optimize
            model_optimizer.optimize()
            model_optimizer.save_model_to_file(str(optimized_path))
            
            # Replace original with optimized
            shutil.move(str(optimized_path), str(model_path))
            
            logger.info("✓ Model optimization complete")
            
        except ImportError:
            logger.warning("ONNX Runtime transformer optimizer not available, trying basic optimization")
            self._basic_optimize(model_path)
        except Exception as e:
            logger.warning(f"Optimization failed: {e}, keeping original model")
    
    def _basic_optimize(self, model_path: Path):
        """Basic ONNX optimization without transformer-specific optimizations"""
        try:
            from onnx import optimizer as onnx_optimizer
            
            model = onnx.load(str(model_path))
            
            # Basic optimization passes
            passes = [
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_pad',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'fuse_consecutive_concats',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
            ]
            
            optimized_model = onnx_optimizer.optimize(model, passes)
            onnx.save(optimized_model, str(model_path))
            
            logger.info("✓ Basic optimization complete")
            
        except Exception as e:
            logger.warning(f"Basic optimization failed: {e}")
    
    def _optimize_for_mobile(self, input_path: Path, output_path: Path):
        """Optimize model specifically for mobile deployment"""
        try:
            from onnx import optimizer as onnx_optimizer
            
            # Load model
            model = onnx.load(str(input_path))
            
            # Run ONNX optimizer passes suitable for mobile
            passes = [
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_monotone_argmax',
                'eliminate_nop_pad',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'eliminate_deadend',
                'fuse_consecutive_concats',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'simplify_lexsort',
                'nop'
            ]
            
            optimized_model = onnx_optimizer.optimize(model, passes)
            
            # Update model with shape inference
            optimized_model = shape_inference.infer_shapes(optimized_model)
            
            # Save
            onnx.save(optimized_model, str(output_path))
            
            # Try mobile-specific optimizer if available
            try:
                from onnxruntime.tools.mobile_optimizer import optimize_model
                optimize_model(str(output_path), str(output_path))
                logger.info("✓ Applied mobile-specific optimizations")
            except ImportError:
                logger.info("✓ Applied general optimizations (mobile optimizer not available)")
                
        except Exception as e:
            logger.warning(f"Mobile optimization failed: {e}, using original model")
            shutil.copy(str(input_path), str(output_path))
    
    def _quantize_dynamic(self, input_path: Path, output_path: Path):
        """Apply dynamic quantization"""
        logger.info("Applying dynamic quantization...")
        
        try:
            quantize_dynamic(
                model_input=str(input_path),
                model_output=str(output_path),
                weight_type=QuantType.QUInt8,
                optimize_model=True,
                per_channel=True,
                reduce_range=True
            )
            
            # Validate quantized model
            if self.config.validate_export:
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
            
            # Add metadata
            metadata = {
                'model_description': self.config.model_description,
                'model_author': self.config.model_author,
                'model_version': self.config.model_version,
                'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_tags': str(len(self.vocab.tag_to_index)),
                'image_size': str(self.config.image_size),
                'framework': 'PyTorch',
                'framework_version': torch.__version__,
                'onnx_version': onnx.__version__,
                'opset_version': str(self.config.opset_version),
                'device': str(self.device),
            }
            
            for key, value in metadata.items():
                meta = model.metadata_props.add()
                meta.key = key
                meta.value = value
            
            # Save model
            onnx.save(model, str(model_path))
            logger.info("✓ Metadata added to model")
            
        except Exception as e:
            logger.warning(f"Failed to add metadata: {e}")
    
    def _validate_model(self, model_path: Path):
        """Validate exported ONNX model"""
        logger.info("Validating ONNX model...")
        
        try:
            # Check model structure
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            logger.info("✓ ONNX model structure is valid")
            
            # Create test input
            test_input = np.random.randn(
                self.config.batch_size,
                3,
                self.config.image_size,
                self.config.image_size
            ).astype(np.float32)
            
            # Run inference with PyTorch
            torch_input = torch.from_numpy(test_input).to(self.device)
            self.model.eval()
            
            with torch.no_grad():
                torch_predictions, torch_scores = self.model(torch_input)
            
            torch_predictions = torch_predictions.cpu().numpy()
            torch_scores = torch_scores.cpu().numpy()
            
            # Run inference with ONNX Runtime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(model_path), providers=providers)
            
            onnx_outputs = session.run(
                None,
                {self.config.input_names[0]: test_input}
            )
            
            onnx_predictions = onnx_outputs[0]
            onnx_scores = onnx_outputs[1]
            
            # Compare outputs
            predictions_close = np.allclose(
                torch_predictions,
                onnx_predictions,
                rtol=self.config.tolerance_rtol,
                atol=self.config.tolerance_atol
            )
            
            scores_close = np.allclose(
                torch_scores,
                onnx_scores,
                rtol=self.config.tolerance_rtol,
                atol=self.config.tolerance_atol
            )
            
            if predictions_close and scores_close:
                logger.info("✓ Model validation passed!")
                logger.info(f"  Max prediction difference: {np.max(np.abs(torch_predictions - onnx_predictions)):.6f}")
                logger.info(f"  Max score difference: {np.max(np.abs(torch_scores - onnx_scores)):.6f}")
            else:
                logger.error("✗ Model validation failed!")
                logger.error(f"  Predictions match: {predictions_close}")
                logger.error(f"  Scores match: {scores_close}")
                if not predictions_close:
                    logger.error(f"  Max prediction diff: {np.max(np.abs(torch_predictions - onnx_predictions))}")
                if not scores_close:
                    logger.error(f"  Max score diff: {np.max(np.abs(torch_scores - onnx_scores))}")
                raise ValueError("ONNX model output does not match PyTorch model")
                
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
        
        try:
            # Create session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(model_path), providers=providers)
            
            # Log which provider is being used
            logger.info(f"Using providers: {session.get_providers()}")
            
            # Prepare input
            input_data = np.random.randn(
                self.config.batch_size,
                3,
                self.config.image_size,
                self.config.image_size
            ).astype(np.float32)
            
            # Warmup runs
            logger.info("Warming up...")
            for _ in range(5):
                _ = session.run(None, {self.config.input_names[0]: input_data})
            
            # Benchmark runs
            logger.info(f"Running {num_runs} inference iterations...")
            times = []
            for _ in tqdm(range(num_runs), desc="Benchmarking"):
                start = time.perf_counter()
                _ = session.run(None, {self.config.input_names[0]: input_data})
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
                'throughput_fps': 1000 / np.mean(times) * self.config.batch_size
            }
            
            logger.info("\n" + "="*60)
            logger.info("BENCHMARK RESULTS")
            logger.info("="*60)
            logger.info(f"Model: {model_path.name}")
            logger.info(f"Batch size: {self.config.batch_size}")
            logger.info(f"Image size: {self.config.image_size}x{self.config.image_size}")
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
    
    parser = argparse.ArgumentParser(description='Export Anime Tagger model to ONNX')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('vocab_dir', type=str, help='Path to vocabulary directory')
    parser.add_argument('-o', '--output', type=str, default='model.onnx',
                        help='Output ONNX model path')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size for export')
    parser.add_argument('-s', '--image-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--opset', type=int, default=16,
                        help='ONNX opset version')
    parser.add_argument('--variants', nargs='+', default=['full'],
                        choices=['full', 'mobile', 'quantized'],
                        help='Export variants to generate')
    parser.add_argument('--optimize', action='store_true', default=True,
                        help='Optimize exported model')
    parser.add_argument('--quantize', action='store_true',
                        help='Enable quantization')
    parser.add_argument('--quantization-type', type=str, default='dynamic',
                        choices=['dynamic', 'static'],
                        help='Quantization type')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark after export')
    parser.add_argument('--benchmark-runs', type=int, default=100,
                        help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    # Create config
    config = ONNXExportConfig(
        checkpoint_path=args.checkpoint,
        vocab_dir=args.vocab_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        image_size=args.image_size,
        opset_version=args.opset,
        export_variants=args.variants,
        optimize=args.optimize,
        quantize=args.quantize,
        quantization_type=args.quantization_type,
        validate_export=not args.no_validate
    )
    
    # Create exporter
    exporter = ONNXExporter(config)
    
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


if __name__ == '__main__':
    main()