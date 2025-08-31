#!/usr/bin/env python3

"""
Validation Loop for Anime Image Tagger
Comprehensive validation pipeline with multiple evaluation modes
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from collections import defaultdict
import gc
import multiprocessing as mp
import yaml
import csv
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from torch.amp import autocast
from tqdm import tqdm

# Scientific computing imports
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix
)
# Headless/optional plotting setup
import matplotlib as mpl
# Use a non-interactive backend so validation can run headless (e.g., on CI)
mpl.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # optional
    _HAVE_SEABORN = True
except Exception:  # pragma: no cover
    sns = None     # type: ignore
    _HAVE_SEABORN = False

# Import our modules
from evaluation_metrics import MetricComputer
from vocabulary import TagVocabulary, load_vocabulary_for_training, verify_vocabulary_integrity
from dataset_loader import create_dataloaders
from Configuration_System import (
    DataConfig as CSDataConfig,
    ValidationConfig as CSValConfig,
    ValidationDataloaderConfig as CSDataloaderConfig,
)
from training_utils import DistributedTrainingHelper
from model_architecture import create_model, VisionTransformerConfig
from model_metadata import ModelMetadata
from schemas import TagPrediction, ImagePrediction, RunMetadata, PredictionOutput, compute_vocab_sha256
from safe_checkpoint import safe_load_checkpoint, InvalidCheckpointError


logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "vocabulary.json"
UNIFIED_CONFIG_PATH = PROJECT_ROOT / "configs" / "unified_config.yaml"

# Default validation preprocessing (image size/patch may be overridden from
# unified_config.yaml if present)
_DEFAULT_VAL_IMAGE_SIZE = 640


@dataclass
class ValidationConfig:
    """Configuration for validation"""
    # Model and data paths
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    data_dir: str = "data/images"
    json_dir: str = "data/annotations"
    vocab_path: str = str(DEFAULT_VOCAB_PATH)
    output_dir: str = "./validation_results"
    
    # Validation modes
    mode: str = "full"  # "full", "fast", "tags", "hierarchical"
    specific_tags: Optional[List[str]] = None  # For "tags" mode
    
    # Batch settings
    batch_size: int = 64
    num_workers: int = 8
    
    # Evaluation settings
    max_samples: Optional[int] = None  # Limit samples for fast validation
    prediction_threshold: float = 0.5
    adaptive_threshold: bool = True
    save_predictions: bool = False
    save_per_image_results: bool = False
    
    # Metric computation
    metric_config: Optional[Dict] = None
    compute_expensive_metrics: bool = True
    
    # Visualization
    create_visualizations: bool = True
    plot_dir: str = "./validation_plots"
    
    # Tag analysis
    analyze_tag_groups: bool = True
    analyze_by_frequency: bool = True
    frequency_bins: List[int] = None
    
    # Performance analysis
    measure_inference_time: bool = True
    profile_memory: bool = False
    
    # Distributed
    distributed: bool = False
    local_rank: int = -1
    
    # Device
    device: str = "cuda"
    use_amp: bool = True


class ValidationRunner:
    """Main validation runner"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.amp_enabled = self.config.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
        if self.amp_enabled:
            logger.info(f"Validation AMP dtype set to {self.amp_dtype}.")
        # Logging queue used by workers
        self._log_queue: Optional[mp.Queue] = mp.Queue()
        self._listener = None  # Initialize listener attribute

        # Load validation overrides from unified_config.yaml
        self._val_mean: Optional[Tuple[float, float, float]] = None
        self._val_std: Optional[Tuple[float, float, float]] = None
        self._val_image_size = _DEFAULT_VAL_IMAGE_SIZE
        self._val_patch_size = 16
        if not UNIFIED_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Unified config not found at {UNIFIED_CONFIG_PATH}")

        try:
            unified = yaml.safe_load(UNIFIED_CONFIG_PATH.read_text(encoding="utf-8")) or {}
            v = (unified.get("validation") or {})
            dl = (v.get("dataloader") or {})
            pp = (v.get("preprocessing") or {})
            data_cfg = unified.get("data") or {}

            # Map dataloader settings → ValidationConfig
            self.config.batch_size = int(dl.get("batch_size", self.config.batch_size))
            self.config.num_workers = int(dl.get("num_workers", self.config.num_workers))

            # Preprocessing overrides (normalization, sizes)
            mean = pp.get("normalize_mean") or data_cfg.get("normalize_mean")
            std = pp.get("normalize_std") or data_cfg.get("normalize_std")
            if not (mean and std):
                raise ValueError("normalize_mean and normalize_std must be provided in unified_config.yaml")

            if isinstance(mean, (list, tuple)) and len(mean) == 3:
                self._val_mean = tuple(float(x) for x in mean)
            else:
                raise ValueError("normalize_mean must be a sequence of three floats")

            if isinstance(std, (list, tuple)) and len(std) == 3:
                self._val_std = tuple(float(x) for x in std)
            else:
                raise ValueError("normalize_std must be a sequence of three floats")

            self._val_image_size = int(pp.get("image_size", data_cfg.get("image_size", self._val_image_size)))
            self._val_patch_size = int(pp.get("patch_size", data_cfg.get("patch_size", self._val_patch_size)))
        except Exception as e:
            raise RuntimeError(f"Failed to load unified validation config: {e}")

        # Warn if legacy validation_config.yaml still exists (migration)
        legacy_val = PROJECT_ROOT / "configs" / "validation_config.yaml"
        if legacy_val.exists():
            logger.warning("Detected legacy configs/validation_config.yaml; it is no longer used now that "
                           "validation settings live under 'validation' in unified_config.yaml.")

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_sha256 = "unknown"
        self.patch_size = self._val_patch_size  # Default
        
        # Setup plotting directory
        self.plot_dir = Path(config.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load model first to check for embedded vocabulary
        self.model, checkpoint = self._load_model()

        # Priority 1: Try to load vocabulary from checkpoint first
        vocab_loaded = False
        if checkpoint and 'vocab_b64_gzip' in checkpoint:
            logger.info("Attempting to load embedded vocabulary from checkpoint")
            vocab_data = ModelMetadata.extract_vocabulary(checkpoint)
            if vocab_data:
                try:
                    self.vocab = TagVocabulary()
                    self.vocab.tag_to_index = vocab_data['tag_to_index']
                    self.vocab.index_to_tag = {int(k): v for k, v in vocab_data['index_to_tag'].items()}
                    self.vocab.tag_frequencies = vocab_data.get('tag_frequencies', {})
                    self.vocab.unk_index = self.vocab.tag_to_index.get(self.vocab.unk_token, 1)

                    # Verify vocabulary integrity
                    verify_vocabulary_integrity(self.vocab, Path("embedded"))
                    vocab_loaded = True
                    logger.info(
                        f"Successfully loaded embedded vocabulary with {len(self.vocab.tag_to_index)} tags"
                    )
                    # Compute vocabulary hash
                    self.vocab_sha256 = compute_vocab_sha256(vocab_data=vocab_data)
                except Exception as e:
                    logger.error(f"Failed to use embedded vocabulary: {e}")
                    vocab_loaded = False

        # Priority 2: Fall back to external vocabulary file
        if not vocab_loaded:
            logger.info("Loading vocabulary from external file")
            vocab_path = Path(config.vocab_path)
            if vocab_path.exists():
                if vocab_path.suffix == '.json':
                    self.vocab = TagVocabulary(vocab_path)
                else:
                    self.vocab = load_vocabulary_for_training(vocab_path)
            else:
                for path in [DEFAULT_VOCAB_PATH, PROJECT_ROOT / "vocabulary/vocabulary.json"]:
                    if path.exists():
                        self.vocab = TagVocabulary(path)
                        vocab_path = path
                        break
                else:
                    raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
            # Verify external vocabulary
            verify_vocabulary_integrity(self.vocab, vocab_path)
            # Compute vocabulary hash for external file
            self.vocab_sha256 = compute_vocab_sha256(vocab_path=vocab_path)

        logger.info(f"Loaded vocabulary with {len(self.vocab.tag_to_index)} tags")

        self.num_tags = len(self.vocab.tag_to_index)

        # Initialize metric computer for F1 and mAP tracking
        self.metric_computer = MetricComputer(num_labels=self.num_tags)

        # Track validation history
        self.validation_history = []
        
        # Default frequency bins if not provided
        if config.frequency_bins is None:
            self.config.frequency_bins = [0, 10, 100, 1000, 10000, float('inf')]
    
    def _infer_num_tags(self, checkpoint: Optional[Dict]) -> int:
        """
        Decide the correct number of tags before model creation.
        Priority:
          1) embedded vocab in checkpoint
          2) external vocab file (config.vocab_path, then defaults)
        """
        try:
            vocab_data = ModelMetadata.extract_vocabulary(checkpoint) if checkpoint else None
            if vocab_data and 'tag_to_index' in vocab_data:
                return len(vocab_data['tag_to_index'])
        except Exception:
            pass
        # External fallback(s)
        vocab_path = Path(self.config.vocab_path)
        if not vocab_path.exists():
            for p in (DEFAULT_VOCAB_PATH, PROJECT_ROOT / "vocabulary" / "vocabulary.json"):
                if p.exists():
                    vocab_path = p
                    break
        vocab = TagVocabulary(vocab_path)
        return len(vocab.tag_to_index)
    
    def _metadata_dict_to_list(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert a batch-level metadata dict-of-lists (as produced by the
        custom collate_fn) into a per-sample list of dicts so downstream
        code can serialize and analyze per-image results easily.
        """
        size = len(meta.get('paths', []))
        items: List[Dict[str, Any]] = []
        for i in range(size):
            item: Dict[str, Any] = {}
            for k, v in meta.items():
                if isinstance(v, (list, tuple)):
                    if i < len(v):
                        item[k] = v[i]
                else:
                    item[k] = v
            items.append(item)
        return items


    def _setup_logging(self):
        """Setup validation-specific logging"""
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Always have a console handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

        # Only primary process writes files
        is_primary = True
        try:
            is_primary = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:
            is_primary = True

        # Queue-based logging: main process listens & writes to file
        if is_primary:
            log_file = self.output_dir / f"validation_{datetime.now():%Y%m%d_%H%M%S}.log"
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            try:
                from logging.handlers import QueueListener
                self._listener = QueueListener(self._log_queue, fh, respect_handler_level=True)
                self._listener.start()
            except Exception:
                # Fallback to direct file handler if QueueListener unavailable
                logger.addHandler(fh)
        # else: workers only attach QueueHandler in worker_init_fn (see dataset_loader)
    
    def _load_model(self) -> Tuple[nn.Module, Optional[Dict]]:
        """Load model from checkpoint"""
        checkpoint = None
        if self.config.checkpoint_path:
            logger.info(f"Loading model from checkpoint: {self.config.checkpoint_path}")
            state_dict, meta = safe_load_checkpoint(self.config.checkpoint_path)
            checkpoint = {"state_dict": state_dict, **meta}
            
            # Extract model config
            if 'config' in checkpoint:
                model_config = checkpoint['config']
                # Handle nested configs
                if 'model_config' in model_config:
                    model_config = model_config['model_config']
            else:
               # Default config
                model_config = VisionTransformerConfig()

            # Convert to dict if needed
            if not isinstance(model_config, dict):
                model_config = asdict(model_config)

            # Ensure num_tags is defined before creating the model
            # (self.num_tags may not be set yet the first time we get here)
            if not hasattr(self, "num_tags") or not isinstance(getattr(self, "num_tags", None), int):
                self.num_tags = self._infer_num_tags(checkpoint)
            model_config['num_tags'] = int(self.num_tags)
            logger.info(f"Creating model with {self.num_tags} tags (derived from vocab)")

            # Create model
            model = create_model(**model_config)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DDP wrapped models
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            
            model.load_state_dict(state_dict)

            # Always try to extract preprocessing; handle legacy inside the extractor.
            preprocessing = ModelMetadata.extract_preprocessing_params(checkpoint) if checkpoint else None
            if not preprocessing:
                # Fall back to unified_config.yaml values (already loaded), or sensible defaults.
                preprocessing = {
                    "normalize_mean": list(self._val_mean) if self._val_mean else [0.5, 0.5, 0.5],
                    "normalize_std": list(self._val_std) if self._val_std else [0.5, 0.5, 0.5],
                    "image_size": int(self._val_image_size),
                    "patch_size": int(self._val_patch_size),
                }
                logger.info("No preprocessing in checkpoint; using unified_config.yaml / defaults.")
            self.preprocessing_params = preprocessing
            self.patch_size = int(self.preprocessing_params.get("patch_size", self._val_patch_size))

        elif self.config.model_path:
            raise InvalidCheckpointError(
                "Loading pickled model objects is disabled. Save state_dict checkpoints instead."
            )
        else:
            raise ValueError("Either checkpoint_path or model_path must be provided")
        
        model.to(self.device)
        model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")

        return model, checkpoint
    
    def create_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        # Build configs from preprocessing params (already extracted from checkpoint)
        data_cfg = CSDataConfig(
            data_dir=self.config.data_dir,
            vocab_dir=str(Path(self.config.vocab_path).parent),
            image_size=self.preprocessing_params.get('image_size', 640),
            normalize_mean=tuple(self.preprocessing_params.get('normalize_mean', [0.5, 0.5, 0.5])),
            normalize_std=tuple(self.preprocessing_params.get('normalize_std', [0.5, 0.5, 0.5])),
            random_flip_prob=0.0,
            pin_memory=True,
            batch_size=8,  # ignored by val loader below
        )
        val_cfg = CSValConfig(
            dataloader=CSDataloaderConfig(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
            )
        )
        _, val_loader, _ = create_dataloaders(
            data_config=data_cfg,
            validation_config=val_cfg,
            vocab_path=Path(self.config.vocab_path),
            active_data_path=Path(self.config.json_dir),
            distributed=self.config.distributed,
            log_queue=self._log_queue,
            frequency_sampling=False,
        )
        
        # Limit samples if requested
        if self.config.max_samples and self.config.max_samples < len(val_loader.dataset):
            indices = np.random.choice(len(val_loader.dataset), self.config.max_samples, replace=False)
            subset = Subset(val_loader.dataset, indices)
            val_loader = DataLoader(
                subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            logger.info(f"Limited validation to {self.config.max_samples} samples")
        
        return val_loader
    
    def validate(self) -> Dict[str, Any]:
        """Run validation based on configured mode"""
        logger.info(f"Starting validation in '{self.config.mode}' mode")
        
        # Create dataloader
        dataloader = self.create_dataloader()
        # Store dataloader reference for cleanup
        self._last_dataloader = dataloader
        
        # Run appropriate validation mode
        if self.config.mode == "full":
            results = self.validate_full(dataloader)
        elif self.config.mode == "fast":
            results = self.validate_fast(dataloader)
        elif self.config.mode == "tags":
            results = self.validate_specific_tags(dataloader)
        elif self.config.mode == "hierarchical":
            results = self.validate_hierarchical(dataloader)
        else:
            raise ValueError(f"Unknown validation mode: {self.config.mode}")
        
        # Save results
        self._save_results(results)

        # Update history
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'mode': self.config.mode,
            'results': results
        })

        # Cleanup logging resources
        self._cleanup_logging()

        # Best-effort worker shutdown for cleanliness
        try:
            dl = None
            try:
                dl = self._last_dataloader  # if stored
            except AttributeError:
                pass
            for obj in filter(None, [dl]):
                if hasattr(obj, "_iterator") and obj._iterator is not None:
                    obj._iterator._shutdown_workers()  # noqa: SLF001
        except Exception:
            pass
        gc.collect()

        return results

    def _cleanup_logging(self):
        """Clean up logging resources including QueueListener"""
        if self._listener is not None:
            try:
                self._listener.stop()
                logger.info("QueueListener stopped successfully")
                self._listener = None
            except Exception as e:
                logger.warning(f"Error stopping QueueListener: {e}")
        # Clear the queue reference
        self._log_queue = None
     

    def validate_full(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Complete validation with all metrics"""
        logger.info("Running full validation...")
        
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        all_metadata = []
        all_processing_times = []  # Track per-image times
        
        # Measure inference time
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
                images = batch['images'].to(self.device)
                tag_labels = batch['tag_labels']
                
                # Time inference
                if self.config.measure_inference_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_time = time.time()
                
                # Forward pass (propagate padding masks when available)
                pmask = batch.get('padding_mask', None)
                if pmask is not None:
                    pmask = pmask.to(self.device)
                with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.amp_enabled):
                    outputs = self.model(images, padding_mask=pmask)
                    logits = outputs['tag_logits'] if isinstance(outputs, dict) else outputs
                
                if self.config.measure_inference_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    per_image_time = (inference_time / images.shape[0]) * 1000  # Convert to ms
                    inference_times.append(per_image_time / 1000)  # Keep in seconds for backward compat
                    all_processing_times.extend([per_image_time] * images.shape[0])
                
                # Handle hierarchical output
                if logits.dim() == 3:
                    B, G, T = logits.shape
                    logits = logits.reshape(B, -1)
                # Make sure labels width matches model head (helpful when vocab/head disagree)
                if tag_labels is not None and tag_labels.ndim == 2 and tag_labels.shape[1] != logits.shape[1]:
                    raise RuntimeError(
                        f"Prediction dim {logits.shape[1]} != target dim {tag_labels.shape[1]}. "
                        "The checkpoint head and vocabulary/targets disagree. "
                        "Rebuild the head or align the vocabulary."
                    )
                
                # Convert to probabilities
                predictions = torch.sigmoid(logits)
                
                # Collect results
                all_predictions.append(predictions.cpu())
                all_targets.append(tag_labels.cpu())
                # Normalize metadata to per-sample dict list
                meta = batch.get('metadata', None)
                if isinstance(meta, dict):
                    all_metadata.extend(self._metadata_dict_to_list(meta))
                elif isinstance(meta, list):
                    all_metadata.extend(meta)
                else:
                    # Fallback: preserve count
                    all_metadata.extend([{'index': j} for j in range(images.shape[0])])
                
                # Memory profiling
                if self.config.profile_memory and batch_idx % 10 == 0 and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"GPU memory allocated: {allocated:.2f} GB")
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        logger.info(f"Collected predictions for {len(all_predictions)} samples")
        
        # Compute metrics
        logger.info("Computing metrics...")

        metrics = self.metric_computer.compute_all_metrics(
            all_predictions,
            all_targets
        )

        # Get tag names and frequencies for optional analysis
        tag_names = []
        for i in range(len(self.vocab.tag_to_index)):
            try:
                tag_name = self.vocab.get_tag_from_index(i)
            except ValueError as e:
                logger.error(f"Vocabulary corruption detected: {e}")
                raise
            tag_names.append(tag_name)
        tag_frequencies = self._compute_tag_frequencies(all_targets) if self.config.analyze_by_frequency else None
        
        # Add timing information
        if self.config.measure_inference_time and inference_times:
            metrics['timing'] = {
                'avg_inference_time_ms': np.mean(inference_times) * 1000,
                'std_inference_time_ms': np.std(inference_times) * 1000,
                'total_inference_time_s': sum(inference_times)
            }
        
        # Create visualizations
        if self.config.create_visualizations:
            self._create_visualizations(all_predictions, all_targets, metrics, tag_names)
        
        # Save predictions if requested
        if self.config.save_predictions:
            self._save_predictions_standardized(
                all_predictions, all_targets, all_metadata, all_processing_times
            )
        
        # Per-image results
        if self.config.save_per_image_results:
            self._save_per_image_results(all_predictions, all_targets, all_metadata)
        
        return metrics
    
    def validate_fast(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Fast validation with basic metrics only"""
        logger.info("Running fast validation...")

        # Limit samples for speed
        max_batches = 50
        limited_data = []
        
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            limited_data.append(batch)
        
        # Create temporary dataloader
        class ListDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        # Forward batches AS-IS (avoid stacking an extra leading dimension)
        fast_loader = DataLoader(
            ListDataset(limited_data),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )
        
        # Run validation
        results = self.validate_full(fast_loader)

        return results
    
    def validate_specific_tags(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Validate performance on specific tags"""
        if not self.config.specific_tags:
            raise ValueError("specific_tags must be provided for 'tags' mode")
        
        logger.info(f"Validating specific tags: {self.config.specific_tags}")
        
        # Get tag indices
        tag_indices = []
        for tag in self.config.specific_tags:
            idx = self.vocab.get_tag_index(tag)
            if idx != self.vocab.unk_index:
                tag_indices.append(idx)
            else:
                logger.warning(f"Tag '{tag}' not found in vocabulary")
        
        if not tag_indices:
            raise ValueError("No valid tags found in vocabulary")
        
        tag_indices = torch.tensor(tag_indices)
        
        # Collect predictions for specific tags
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating specific tags"):
                images = batch['images'].to(self.device)
                tag_labels = batch['tag_labels']
                
                # Forward pass (propagate padding masks when available)
                pmask = batch.get('padding_mask', None)
                if pmask is not None:
                    pmask = pmask.to(self.device)
                with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.amp_enabled):
                    outputs = self.model(images, padding_mask=pmask)
                    logits = outputs['tag_logits'] if isinstance(outputs, dict) else outputs
                
                # Handle hierarchical output
                if logits.dim() == 3:
                    B, G, T = logits.shape
                    logits = logits.reshape(B, -1)
                if tag_labels is not None and tag_labels.ndim == 2 and logits.shape[1] != tag_labels.shape[1]:
                    raise RuntimeError(
                        f"Prediction dim {logits.shape[1]} != target dim {tag_labels.shape[1]}. "
                        "The checkpoint head and vocabulary/targets disagree. "
                        "Rebuild the head or align the vocabulary."
                    )
                
                # Get predictions for specific tags only
                predictions = torch.sigmoid(logits[:, tag_indices])
                targets = tag_labels[:, tag_indices]
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute per-tag metrics
        results = {'specific_tags': {}}
        
        for i, (tag, tag_idx) in enumerate(zip(self.config.specific_tags, tag_indices)):
            tag_preds = all_predictions[:, i]
            tag_targets = all_targets[:, i]
            
            # Skip if tag never appears
            if tag_targets.sum() == 0:
                results['specific_tags'][tag] = {
                    'error': 'Tag never appears in validation set'
                }
                continue
            
            # Compute metrics
            tag_binary = tag_preds > self.config.prediction_threshold
            
            tp = ((tag_binary == 1) & (tag_targets == 1)).sum().item()
            fp = ((tag_binary == 1) & (tag_targets == 0)).sum().item()
            fn = ((tag_binary == 0) & (tag_targets == 1)).sum().item()
            tn = ((tag_binary == 0) & (tag_targets == 0)).sum().item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # Average precision
            ap = average_precision_score(tag_targets.numpy(), tag_preds.numpy())
            
            results['specific_tags'][tag] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'average_precision': ap,
                'support': int(tag_targets.sum()),
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            }
            
            # Create PR curve for this tag
            if self.config.create_visualizations:
                self._create_tag_pr_curve(tag, tag_preds.numpy(), tag_targets.numpy())
        
        # Summary statistics
        f1_scores = [v['f1'] for v in results['specific_tags'].values() if 'f1' in v]
        if f1_scores:
            results['summary'] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'min_f1': np.min(f1_scores),
                'max_f1': np.max(f1_scores)
            }
        
        return results
    
    def validate_hierarchical(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Validate hierarchical group structure"""
        logger.info("Validating hierarchical structure...")
        
        # Collect predictions by group
        group_predictions = defaultdict(list)
        group_targets = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating hierarchical"):
                images = batch['images'].to(self.device)
                # Collate does not provide hierarchical labels; use flat tag labels
                flat_labels = batch.get('tag_labels')
                if flat_labels is None:
                    logger.error("Batch is missing 'tag_labels'; cannot run hierarchical validation.")
                    return {'error': "Batch missing 'tag_labels' for hierarchical validation"}
                
                # Forward pass
                pmask = batch.get('padding_mask', None)
                if pmask is not None:
                    pmask = pmask.to(self.device)
                with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.amp_enabled):
                    outputs = self.model(images, padding_mask=pmask)
                    tag_logits = outputs['tag_logits'] if isinstance(outputs, dict) else outputs
                
                # Expect (batch, num_groups, tags_per_group)
                if tag_logits.dim() != 3:
                    logger.warning(
                        "Model output is not hierarchical (got shape %s). "
                        "Returning a helpful error instead of crashing. "
                        "Use mode='full' or 'tags' with this checkpoint.",
                        tuple(tag_logits.shape)
                    )
                    return {
                        'error': "Model output is not hierarchical (expected 3D logits).",
                        'hint': "Run with mode='full' or 'tags', or load a hierarchical checkpoint."
                    }
                
                predictions = torch.sigmoid(tag_logits)
                
                # Collect by group
                num_groups = predictions.size(1)
                # Best-effort reshape of flat labels into (B, G, T) if sizes line up
                B, G, T = predictions.size(0), predictions.size(1), predictions.size(2)
                if flat_labels.size(1) == G * T:
                    labels_h = flat_labels.view(B, G, T)
                else:
                    logger.warning(
                        "Cannot reshape flat labels of shape %s into (B=%d, G=%d, T=%d); "
                        "filling targets with zeros for metrics.", tuple(flat_labels.shape), B, G, T
                    )
                    labels_h = torch.zeros((B, G, T), dtype=flat_labels.dtype)
                for g in range(num_groups):
                    group_predictions[g].append(predictions[:, g, :].cpu())
                    group_targets[g].append(labels_h[:, g, :].cpu())
        
        # Compute metrics per group
        results = {'groups': {}}
        group_f1_scores = []
        
        for g in range(len(group_predictions)):
            if not group_predictions[g]:
                continue
            
            # Concatenate group data
            g_preds = torch.cat(group_predictions[g], dim=0)
            g_targets = torch.cat(group_targets[g], dim=0)
            
            # Flatten for metrics
            g_preds_flat = g_preds.view(-1)
            g_targets_flat = g_targets.view(-1)
            
            # Compute group metrics
            g_binary = g_preds_flat > self.config.prediction_threshold
            
            tp = ((g_binary == 1) & (g_targets_flat == 1)).sum().item()
            fp = ((g_binary == 1) & (g_targets_flat == 0)).sum().item()
            fn = ((g_binary == 0) & (g_targets_flat == 1)).sum().item()
            tn = ((g_binary == 0) & (g_targets_flat == 0)).sum().item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # Get group name
            group_name = (
                self.vocab.get_group_name(g) if hasattr(self.vocab, 'get_group_name') else f"Group_{g}"
            )
            
            results['groups'][group_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(g_targets_flat.sum()),
                'num_samples': len(g_preds)
            }
            
            group_f1_scores.append(f1)
        
        # Summary statistics
        if group_f1_scores:
            results['summary'] = {
                'mean_group_f1': np.mean(group_f1_scores),
                'std_group_f1': np.std(group_f1_scores),
                'min_group_f1': np.min(group_f1_scores),
                'max_group_f1': np.max(group_f1_scores),
                'num_groups': len(group_f1_scores)
            }
        
        return results
    
    def _compute_tag_frequencies(self, targets: torch.Tensor) -> np.ndarray:
        """Compute tag frequencies from targets"""
        # Sum across samples to get frequency counts
        tag_counts = targets.sum(dim=0).numpy()
        return tag_counts
    
    def _create_visualizations(self, predictions: torch.Tensor, targets: torch.Tensor, 
                              metrics: Dict, tag_names: List[str]):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # 1. Overall metrics bar plot
        plt.figure(figsize=(10, 6))
        overall_metrics = [m for m in ('f1_macro', 'f1_micro', 'mAP') if m in metrics]
        values = [metrics.get(m, 0) for m in overall_metrics]
        plt.bar(overall_metrics, values)
        plt.title('Overall Validation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'overall_metrics.png')
        plt.close()
        
        # 2. Per-tag F1 distribution
        if 'per_tag_metrics' in metrics:
            per_tag = metrics['per_tag_metrics']
            f1_scores = [m['f1'] for m in per_tag.values() if 'f1' in m]
            
            if f1_scores:
                plt.figure(figsize=(10, 6))
                plt.hist(f1_scores, bins=50, edgecolor='black')
                plt.axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
                plt.xlabel('F1 Score')
                plt.ylabel('Number of Tags')
                plt.title('Distribution of Per-Tag F1 Scores')
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.plot_dir / 'f1_distribution.png')
                plt.close()
        
        # 3. Precision-Recall scatter plot
        if 'per_tag_metrics' in metrics:
            precisions = []
            recalls = []
            supports = []
            
            for tag_metrics in metrics['per_tag_metrics'].values():
                if 'precision' in tag_metrics and 'recall' in tag_metrics:
                    precisions.append(tag_metrics['precision'])
                    recalls.append(tag_metrics['recall'])
                    supports.append(tag_metrics.get('support', 1))
            
            if precisions:
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(recalls, precisions, c=supports, 
                                    s=50, alpha=0.6, cmap='viridis')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision vs Recall for All Tags')
                plt.colorbar(scatter, label='Support (count)')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.plot_dir / 'precision_recall_scatter.png')
                plt.close()
        
        # 4. Top/Bottom performing tags
        if 'per_tag_metrics' in metrics:
            tag_f1s = [(tag, m['f1']) for tag, m in metrics['per_tag_metrics'].items() if 'f1' in m]
            tag_f1s.sort(key=lambda x: x[1], reverse=True)
            
            # Top 20 and bottom 20
            top_20 = tag_f1s[:20]
            bottom_20 = tag_f1s[-20:] if len(tag_f1s) > 20 else []
            
            if top_20:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Top performers
                tags, scores = zip(*top_20)
                ax1.barh(range(len(tags)), scores, color='green')
                ax1.set_yticks(range(len(tags)))
                ax1.set_yticklabels(tags)
                ax1.set_xlabel('F1 Score')
                ax1.set_title('Top 20 Performing Tags')
                ax1.set_xlim(0, 1)
                
                # Bottom performers
                if bottom_20:
                    tags, scores = zip(*bottom_20)
                    ax2.barh(range(len(tags)), scores, color='red')
                    ax2.set_yticks(range(len(tags)))
                    ax2.set_yticklabels(tags)
                    ax2.set_xlabel('F1 Score')
                    ax2.set_title('Bottom 20 Performing Tags')
                    ax2.set_xlim(0, 1)
                
                plt.tight_layout()
                plt.savefig(self.plot_dir / 'top_bottom_tags.png')
                plt.close()
    
    def _create_tag_pr_curve(self, tag_name: str, predictions: np.ndarray, targets: np.ndarray):
        """Create precision-recall curve for a specific tag"""
        precision, recall, thresholds = precision_recall_curve(targets, predictions)
        ap = average_precision_score(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve: {tag_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create tag-specific directory
        tag_dir = self.plot_dir / 'tag_curves'
        tag_dir.mkdir(exist_ok=True)
        
        # Safe filename
        safe_name = tag_name.replace('/', '_').replace(' ', '_')
        plt.savefig(tag_dir / f'pr_curve_{safe_name}.png')
        plt.close()

    def _save_predictions_standardized(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                      metadata: List, processing_times: List):
        """Save predictions in standardized format."""
        logger.info("Saving predictions in standardized format...")

        # Create metadata
        run_metadata = RunMetadata(
            top_k=None,  # thresholded mode; not using fixed top_k
            threshold=float(self.config.prediction_threshold),
            vocab_sha256=self.vocab_sha256,
            normalize_mean=list(self.preprocessing_params.get('normalize_mean', [0.5, 0.5, 0.5])),
            normalize_std=list(self.preprocessing_params.get('normalize_std', [0.5, 0.5, 0.5])),
            image_size=int(self.preprocessing_params.get('image_size', 640)),
            patch_size=int(self.patch_size),
            model_path=str(self.config.checkpoint_path or self.config.model_path),
            num_tags=int(self.num_tags)
        )

        # Create image predictions
        results = []
        for i in range(len(predictions)):
            # Get predictions for this image
            pred = predictions[i]

            # Get top predictions above threshold
            scores, indices = torch.topk(pred, min(20, len(pred)))  # Top 20 by default

            # Filter by threshold
            mask = scores >= self.config.prediction_threshold
            scores = scores[mask]
            indices = indices[mask]

            # Create tag predictions
            tags = []
            for score, idx in zip(scores.tolist(), indices.tolist()):
                try:
                    tag_name = self.vocab.get_tag_from_index(idx)
                except ValueError as e:
                    logger.error(f"Vocabulary corruption detected during prediction save: {e}")
                    raise
                tags.append(TagPrediction(name=tag_name, score=score))

            # Get image identifier
            image_id = metadata[i].get('path', metadata[i].get('image_id', f'image_{i}')) if i < len(metadata) else f'image_{i}'

            # Get processing time if available
            proc_time = processing_times[i] if i < len(processing_times) else None

            result = ImagePrediction(
                image=image_id,
                tags=tags,
                processing_time=proc_time
            )
            results.append(result)

        # Save as standardized JSON
        output = PredictionOutput(metadata=run_metadata, results=results)
        save_path = self.output_dir / 'predictions_standardized.json'
        output.save(save_path)

        logger.info(f"Saved standardized predictions to {save_path}")
    
    def _save_predictions(self, predictions: torch.Tensor, targets: torch.Tensor, metadata: List):
        """Save raw predictions to file"""
        logger.info("Saving predictions...")
        
        save_dict = {
            'predictions': predictions.numpy(),
            'targets': targets.numpy(),
            'metadata': metadata,
            'tag_names': [self.vocab.get_tag_from_index(i) for i in range(len(self.vocab.tag_to_index))],
            'threshold': self.config.prediction_threshold
        }
        
        # Save as pickle for easy loading
        save_path = self.output_dir / 'predictions.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Saved predictions to {save_path}")
    
    def _save_per_image_results(self, predictions: torch.Tensor, targets: torch.Tensor, metadata: List):
        """Save per-image results to CSV"""
        logger.info("Saving per-image results...")
        
        csv_path = self.output_dir / 'per_image_results.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['image_id', 'num_predicted', 'num_actual', 'precision', 'recall', 'f1']
            writer.writerow(header)
            
            # Process each image
            for i in range(len(predictions)):
                pred_binary = predictions[i] > self.config.prediction_threshold
                target_binary = targets[i] > 0.5
                
                # Count predictions and actuals
                num_pred = pred_binary.sum().item()
                num_actual = target_binary.sum().item()
                
                # Calculate metrics
                tp = (pred_binary & target_binary).sum().item()
                
                precision = tp / (num_pred + 1e-8)
                recall = tp / (num_actual + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                # Get image ID from metadata
                image_id = metadata[i].get('image_id', f'image_{i}') if i < len(metadata) else f'image_{i}'
                
                writer.writerow([image_id, num_pred, num_actual, precision, recall, f1])
        
        logger.info(f"Saved per-image results to {csv_path}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save validation results to JSON"""
        logger.info("Saving validation results...")
        
        # Convert numpy types to native Python types
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_json_serializable(results)
        
        # Add metadata
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'results': serializable_results
        }
        
        # Save JSON
        json_path = self.output_dir / f'validation_results_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(json_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Saved results to {json_path}")
        
        # Also save a summary text file
        summary_path = self.output_dir / f'validation_summary_{datetime.now():%Y%m%d_%H%M%S}.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Validation Results Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Mode: {self.config.mode}\n")
            f.write(f"Model: {self.config.checkpoint_path or self.config.model_path}\n")
            f.write(f"\n")
            
            if 'summary' in serializable_results:
                f.write(f"Summary Metrics:\n")
                f.write(f"{'-'*30}\n")
                for key, value in serializable_results['summary'].items():
                    f.write(f"  {key}: {value:.4f}\n")
            
            # Overall metrics if available
            for key in ['f1_macro', 'f1_micro', 'mAP']:
                if key in serializable_results:
                    f.write(f"{key}: {serializable_results[key]:.4f}\n")
        
        logger.info(f"Saved summary to {summary_path}")
    

def main():
    """Main entry point for validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validation script for anime tagger')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--model', type=str, help='Path to model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/images')
    parser.add_argument('--json-dir', type=str, default='data/annotations')
    parser.add_argument('--vocab-path', type=str, default=str(DEFAULT_VOCAB_PATH))
    
    # Validation arguments
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'fast', 'tags', 'hierarchical'])
    parser.add_argument('--specific-tags', nargs='+', help='Tags to validate (for tags mode)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-samples', type=int, help='Maximum samples to validate')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./validation_results')
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--save-per-image', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--create-plots', action='store_true', dest='create_plots', help='Create plots')
    group.add_argument('--no-create-plots', action='store_false', dest='create_plots', help='Disable plots')
    parser.set_defaults(create_plots=False)
    
    # Performance arguments
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Create config
    config = ValidationConfig(
        checkpoint_path=args.checkpoint,
        model_path=args.model,
        data_dir=args.data_dir,
        json_dir=args.json_dir,
        vocab_path=args.vocab_path,
        mode=args.mode,
        specific_tags=args.specific_tags,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        save_per_image_results=args.save_per_image,
        create_visualizations=args.create_plots,
        use_amp=not args.no_amp,
        device=args.device
    )
    
    # Run validation
    runner = ValidationRunner(config)
    results = runner.validate()
    
    # Print summary
    print("\nValidation Results:")
    print("=" * 50)
    
    if 'summary' in results:
        for key, value in results['summary'].items():
            print(f"{key}: {value:.4f}")
    
    for key in ['f1_macro', 'f1_micro', 'mAP']:
        if key in results:
            print(f"{key}: {results[key]:.4f}")

    # Ensure cleanup happens even if there's an exception
    try:
        # Explicitly cleanup if not already done
        if hasattr(runner, '_cleanup_logging'):
            runner._cleanup_logging()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")
    finally:
        # Force cleanup of any remaining resources
        import gc
        gc.collect()    

if __name__ == '__main__':
    main()
