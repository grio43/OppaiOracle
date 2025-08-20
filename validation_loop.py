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
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from tqdm import tqdm

# Scientific computing imports
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from Evaluation_Metrics import MetricComputer, MetricConfig
from Inference_Engine import load_vocabulary_for_training
from HDF5_loader import create_dataloaders, SimplifiedDataConfig
from training_utils import DistributedTrainingHelper
from model_architecture import create_model, VisionTransformerConfig


logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation"""
    # Model and data paths
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    data_dir: str = "data/images"
    json_dir: str = "data/annotations"
    vocab_path: str = "vocabulary.json"
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
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting directory
        self.plot_dir = Path(config.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load vocabulary
        self.vocab = load_vocabulary_for_training(Path(config.vocab_path))
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
        
        # Default frequency bins if not provided
        if config.frequency_bins is None:
            self.config.frequency_bins = [0, 10, 100, 1000, 10000, float('inf')]
    
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
            model = torch.load(self.config.model_path, map_location='cpu')
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
        data_config = SimplifiedDataConfig(
            data_dir=Path(self.config.data_dir),
            json_dir=Path(self.config.json_dir),
            vocab_path=Path(self.config.vocab_path),
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225),
            augmentation_enabled=False  # No augmentation for validation
        )

        # Create dataloader
        _, val_loader = create_dataloaders(
            data_dir=data_config.data_dir,
            json_dir=data_config.json_dir,
            vocab_path=data_config.vocab_path,
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
                if self.config.measure_inference_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_time = time.time()
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.use_amp and torch.cuda.is_available()):
                    outputs = self.model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                if self.config.measure_inference_time and torch.cuda.is_available():
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
                if self.config.profile_memory and batch_idx % 10 == 0 and torch.cuda.is_available():
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
        tag_frequencies = self._compute_tag_frequencies(all_targets) if self.config.analyze_by_frequency else None
        
        metrics = metric_computer.compute_all_metrics(
            all_predictions,
            all_targets,
            tag_names=tag_names,
            tag_frequencies=tag_frequencies
        )
        
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
                with torch.cuda.amp.autocast(enabled=self.config.use_amp and torch.cuda.is_available()):
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
                with torch.cuda.amp.autocast(enabled=self.config.use_amp and torch.cuda.is_available()):
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
            group_name = self.vocab.get_group_name(g) if hasattr(self.vocab, 'get_group_name') else f"Group_{g}"
            
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
        overall_metrics = ['precision', 'recall', 'f1', 'mAP']
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
            for key in ['precision', 'recall', 'f1', 'mAP']:
                if key in serializable_results:
                    f.write(f"{key}: {serializable_results[key]:.4f}\n")
        
        logger.info(f"Saved summary to {summary_path}")
    
    def compare_checkpoints(self, checkpoint_paths: List[str]) -> Dict[str, Any]:
        """Compare multiple checkpoints"""
        comparison_results = {}
        
        for checkpoint_path in checkpoint_paths:
            logger.info(f"Validating checkpoint: {checkpoint_path}")
            
            # Update config
            self.config.checkpoint_path = checkpoint_path
            
            # Reload model
            self.model = self._load_model()
            
            # Run validation
            results = self.validate()
            
            # Store results
            checkpoint_name = Path(checkpoint_path).stem
            comparison_results[checkpoint_name] = results
        
        # Create comparison plots
        self._create_comparison_plots(comparison_results)
        
        return comparison_results
    
    def _create_comparison_plots(self, comparison_results: Dict[str, Any]):
        """Create plots comparing multiple checkpoints"""
        checkpoints = list(comparison_results.keys())
        metrics = ['precision', 'recall', 'f1', 'mAP']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [comparison_results[cp].get(metric, 0) for cp in checkpoints]
            axes[i].bar(range(len(checkpoints)), values)
            axes[i].set_xticks(range(len(checkpoints)))
            axes[i].set_xticklabels(checkpoints, rotation=45, ha='right')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylim(0, 1)
            
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'checkpoint_comparison.png')
        plt.close()


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
    parser.add_argument('--vocab-path', type=str, default='vocabulary.json')
    
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
    parser.add_argument('--create-plots', action='store_true', default=True)
    
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
    
    for key in ['precision', 'recall', 'f1', 'mAP']:
        if key in results:
            print(f"{key}: {results[key]:.4f}")


if __name__ == '__main__':
    main()