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
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from tqdm import tqdm

# Import our modules
from metrics import MetricComputer, MetricConfig, MetricVisualizer, evaluate_model
from tag_vocabulary import TagVocabulary, load_vocabulary_for_training
from model_architecture import create_model
from hdf5_dataloader import create_dataloaders, HDF5DataConfig
from training_utils import DistributedTrainingHelper

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation"""
    # Model and data paths
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    hdf5_dir: str = "/home/user/datasets/teacher_features"
    vocab_dir: str = "/home/user/datasets/vocabulary"
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
        self.device = torch.device(config.device)
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load vocabulary
        self.vocab = load_vocabulary_for_training(Path(config.vocab_dir))
        logger.info(f"Loaded vocabulary with {len(self.vocab.tag_to_index)} tags")
        
        # Load model
        self.model = self._load_model()
        
        # Create metric config
        if config.metric_config:
            self.metric_config = MetricConfig(**config.metric_config)
        else:
            self.metric_config = MetricConfig(
                compute_per_tag_metrics=config.compute_expensive_metrics,
                compute_auc=config.compute_expensive_metrics,
                save_plots=config.create_visualizations,
                plot_dir=config.plot_dir
            )
        
        # Track validation history
        self.validation_history = []
        
    def _setup_logging(self):
        """Setup validation-specific logging"""
        log_file = self.output_dir / f"validation_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        if self.config.checkpoint_path:
            logger.info(f"Loading model from checkpoint: {self.config.checkpoint_path}")
            checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')
            
            # Extract model config
            if 'config' in checkpoint:
                model_config = checkpoint['config']
                # Handle nested configs
                if 'model_config' in model_config:
                    model_config = model_config['model_config']
            else:
                # Default config
                from model_architecture import VisionTransformerConfig
                model_config = VisionTransformerConfig()
            
            # Create model
            model = create_model(**model_config if isinstance(model_config, dict) else asdict(model_config))
            
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
            
        elif self.config.model_path:
            logger.info(f"Loading model from path: {self.config.model_path}")
            model = create_model(pretrained=self.config.model_path)
        else:
            raise ValueError("Either checkpoint_path or model_path must be provided")
        
        model.to(self.device)
        model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")
        
        return model
    
    def create_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        # Data config
        data_config = HDF5DataConfig(
            hdf5_dir=Path(self.config.hdf5_dir),
            vocab_dir=Path(self.config.vocab_dir),
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225),
            augmentation_enabled=False  # No augmentation for validation
        )
        
        # Create dataloader
        _, val_loader = create_dataloaders(
            hdf5_dir=data_config.hdf5_dir,
            vocab_dir=data_config.vocab_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            distributed=self.config.distributed
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
        
        return results
    
    def validate_full(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Complete validation with all metrics"""
        logger.info("Running full validation...")
        
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        all_metadata = []
        
        # Measure inference time
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
                images = batch['image'].to(self.device)
                labels = batch['labels']
                
                # Time inference
                if self.config.measure_inference_time:
                    torch.cuda.synchronize()
                    start_time = time.time()
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                if self.config.measure_inference_time:
                    torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time / images.shape[0])  # Per image
                
                # Handle hierarchical output
                if logits.dim() == 3:
                    batch_size = logits.shape[0]
                    logits = logits.view(batch_size, -1)
                
                # Convert to probabilities
                predictions = torch.sigmoid(logits)
                
                # Collect results
                all_predictions.append(predictions.cpu())
                all_targets.append(labels['binary'].cpu())
                all_metadata.extend(batch['metadata'])
                
                # Memory profiling
                if self.config.profile_memory and batch_idx % 10 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"GPU memory allocated: {allocated:.2f} GB")
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        logger.info(f"Collected predictions for {len(all_predictions)} samples")
        
        # Compute metrics
        logger.info("Computing metrics...")
        metric_computer = MetricComputer(self.metric_config)
        
        # Get tag names and frequencies
        tag_names = [self.vocab.get_tag_from_index(i) for i in range(len(self.vocab.tag_to_index))]
        tag_frequencies = self._compute_tag_frequencies(dataloader) if self.config.analyze_by_frequency else None
        
        metrics = metric_computer.compute_all_metrics(
            all_predictions,
            all_targets,
            tag_names=tag_names,
            tag_frequencies=tag_frequencies
        )
        
        # Add timing information
        if self.config.measure_inference_time:
            metrics['avg_inference_time_ms'] = np.mean(inference_times) * 1000
            metrics['std_inference_time_ms'] = np.std(inference_times) * 1000
            metrics['total_inference_time_s'] = sum(inference_times)
        
        # Create visualizations
        if self.config.create_visualizations:
            self._create_visualizations(all_predictions, all_targets, metrics, tag_names)
        
        # Save predictions if requested
        if self.config.save_predictions:
            self._save_predictions(all_predictions, all_targets, all_metadata)
        
        # Per-image results
        if self.config.save_per_image_results:
            self._save_per_image_results(all_predictions, all_targets, all_metadata)
        
        return metrics
    
    def validate_fast(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Fast validation with basic metrics only"""
        logger.info("Running fast validation...")
        
        # Temporarily disable expensive metrics
        original_config = self.metric_config
        self.metric_config = MetricConfig(
            compute_per_tag_metrics=False,
            compute_confusion_matrix=False,
            compute_auc=False,
            save_plots=False
        )
        
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
        
        fast_loader = DataLoader(
            ListDataset(limited_data),
            batch_size=1,
            shuffle=False
        )
        
        # Run validation
        results = self.validate_full(fast_loader)
        
        # Restore config
        self.metric_config = original_config
        
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
                images = batch['image'].to(self.device)
                labels = batch['labels']['binary']
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Handle hierarchical output
                if logits.dim() == 3:
                    batch_size = logits.shape[0]
                    logits = logits.view(batch_size, -1)
                
                # Get predictions for specific tags only
                predictions = torch.sigmoid(logits[:, tag_indices])
                targets = labels[:, tag_indices]
                
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
            from sklearn.metrics import average_precision_score
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
                images = batch['image'].to(self.device)
                labels = batch['labels']['hierarchical']
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)
                    logits = outputs['logits']
                
                # Should be (batch, num_groups, tags_per_group)
                if logits.dim() != 3:
                    raise ValueError("Model output is not hierarchical")
                
                predictions = torch.sigmoid(logits)
                
                # Collect by group
                for g in range(self.vocab.num_groups):
                    group_predictions[g].append(predictions[:, g, :].cpu())
                    group_targets[g].append(labels[:, g, :].cpu())
        
        # Compute metrics per group
        results = {'groups': {}}
        group_f1_scores = []
        
        for g in range(self.vocab.num_groups):
            if not group_predictions[g]:
                continue
            
            # Concatenate group data
            g_preds = torch.cat(group_predictions[g], dim=0)
            g_targets = torch.cat(group_targets[g], dim=0)
    
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
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from tqdm import tqdm

# Import our modules
from metrics import MetricComputer, MetricConfig, MetricVisualizer, evaluate_model
from tag_vocabulary import TagVocabulary, load_vocabulary_for_training
from model_architecture import create_model
from hdf5_dataloader import create_dataloaders, HDF5DataConfig
from training_utils import DistributedTrainingHelper

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation"""
    # Model and data paths
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    hdf5_dir: str = "/home/user/datasets/teacher_features"
    vocab_dir: str = "/home/user/datasets/vocabulary"
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
        self.device = torch.device(config.device)
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load vocabulary
        self.vocab = load_vocabulary_for_training(Path(config.vocab_dir))
        logger.info(f"Loaded vocabulary with {len(self.vocab.tag_to_index)} tags")
        
        # Load model
        self.model = self._load_model()
        
        # Create metric config
        if config.metric_config:
            self.metric_config = MetricConfig(**config.metric_config)
        else:
            self.metric_config = MetricConfig(
                compute_per_tag_metrics=config.compute_expensive_metrics,
                compute_auc=config.compute_expensive_metrics,
                save_plots=config.create_visualizations,
                plot_dir=config.plot_dir
            )
        
        # Track validation history
        self.validation_history = []
        
    def _setup_logging(self):
        """Setup validation-specific logging"""
        log_file = self.output_dir / f"validation_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        if self.config.checkpoint_path:
            logger.info(f"Loading model from checkpoint: {self.config.checkpoint_path}")
            checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')
            
            # Extract model config
            if 'config' in checkpoint:
                model_config = checkpoint['config']
                # Handle nested configs
                if 'model_config' in model_config:
                    model_config = model_config['model_config']
            else:
                # Default config
                from model_architecture import VisionTransformerConfig
                model_config = VisionTransformerConfig()
            
            # Create model
            model = create_model(**model_config if isinstance(model_config, dict) else asdict(model_config))
            
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
            
        elif self.config.model_path:
            logger.info(f"Loading model from path: {self.config.model_path}")
            model = create_model(pretrained=self.config.model_path)
        else:
            raise ValueError("Either checkpoint_path or model_path must be provided")
        
        model.to(self.device)
        model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")
        
        return model
    
    def create_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        # Data config
        data_config = HDF5DataConfig(
            hdf5_dir=Path(self.config.hdf5_dir),
            vocab_dir=Path(self.config.vocab_dir),
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225),
            augmentation_enabled=False  # No augmentation for validation
        )
        
        # Create dataloader
        _, val_loader = create_dataloaders(
            hdf5_dir=data_config.hdf5_dir,
            vocab_dir=data_config.vocab_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            distributed=self.config.distributed
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
        
        return results
    
    def validate_full(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Complete validation with all metrics"""
        logger.info("Running full validation...")
        
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        all_metadata = []
        
        # Measure inference time
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
                images = batch['image'].to(self.device)
                labels = batch['labels']
                
                # Time inference
                if self.config.measure_inference_time:
                    torch.cuda.synchronize()
                    start_time = time.time()
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                if self.config.measure_inference_time:
                    torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time / images.shape[0])  # Per image
                
                # Handle hierarchical output
                if logits.dim() == 3:
                    batch_size = logits.shape[0]
                    logits = logits.view(batch_size, -1)
                
                # Convert to probabilities
                predictions = torch.sigmoid(logits)
                
                # Collect results
                all_predictions.append(predictions.cpu())
                all_targets.append(labels['binary'].cpu())
                all_metadata.extend(batch['metadata'])
                
                # Memory profiling
                if self.config.profile_memory and batch_idx % 10 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"GPU memory allocated: {allocated:.2f} GB")
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        logger.info(f"Collected predictions for {len(all_predictions)} samples")
        
        # Compute metrics
        logger.info("Computing metrics...")
        metric_computer = MetricComputer(self.metric_config)
        
        # Get tag names and frequencies
        tag_names = [self.vocab.get_tag_from_index(i) for i in range(len(self.vocab.tag_to_index))]
        tag_frequencies = self._compute_tag_frequencies(dataloader) if self.config.analyze_by_frequency else None
        
        metrics = metric_computer.compute_all_metrics(
            all_predictions,
            all_targets,
            tag_names=tag_names,
            tag_frequencies=tag_frequencies
        )
        
        # Add timing information
        if self.config.measure_inference_time:
            metrics['avg_inference_time_ms'] = np.mean(inference_times) * 1000
            metrics['std_inference_time_ms'] = np.std(inference_times) * 1000
            metrics['total_inference_time_s'] = sum(inference_times)
        
        # Create visualizations
        if self.config.create_visualizations:
            self._create_visualizations(all_predictions, all_targets, metrics, tag_names)
        
        # Save predictions if requested
        if self.config.save_predictions:
            self._save_predictions(all_predictions, all_targets, all_metadata)
        
        # Per-image results
        if self.config.save_per_image_results:
            self._save_per_image_results(all_predictions, all_targets, all_metadata)
        
        return metrics
    
    def validate_fast(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Fast validation with basic metrics only"""
        logger.info("Running fast validation...")
        
        # Temporarily disable expensive metrics
        original_config = self.metric_config
        self.metric_config = MetricConfig(
            compute_per_tag_metrics=False,
            compute_confusion_matrix=False,
            compute_auc=False,
            save_plots=False
        )
        
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
        
        fast_loader = DataLoader(
            ListDataset(limited_data),
            batch_size=1,
            shuffle=False
        )
        
        # Run validation
        results = self.validate_full(fast_loader)
        
        # Restore config
        self.metric_config = original_config
        
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
                images = batch['image'].to(self.device)
                labels = batch['labels']['binary']
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Handle hierarchical output
                if logits.dim() == 3:
                    batch_size = logits.shape[0]
                    logits = logits.view(batch_size, -1)
                
                # Get predictions for specific tags only
                predictions = torch.sigmoid(logits[:, tag_indices])
                targets = labels[:, tag_indices]
                
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
            from sklearn.metrics import average_precision_score
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
                images = batch['image'].to(self.device)
                labels = batch['labels']['hierarchical']
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)
                    logits = outputs['logits']
                
                # Should be (batch, num_groups, tags_per_group)
                if logits.dim() != 3:
                    raise ValueError("Model output is not hierarchical")
                
                predictions = torch.sigmoid(logits)
                
                # Collect by group
                for g in range(self.vocab.num_groups):
                    group_predictions[g].append(predictions[:, g, :].cpu())
                    group_targets[g].append(labels[:, g, :].cpu())
        
        # Compute metrics per group
        results = {'groups': {}}
        group_f1_scores = []
        
        for g in range(self.vocab.num_groups):
            if not group_predictions[g]:
                continue
            
            # Concatenate group data
            g_preds = torch.cat(group_predictions[g], dim=0)
            g_targets = torch.cat(group_targets[g], dim=0)
    
