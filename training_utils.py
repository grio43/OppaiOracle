#!/usr/bin/env python3
"""
Training Utilities for Anime Image Tagger
Comprehensive training helpers including schedulers, checkpointing, and distributed training
"""

import os
import sys
import json
import logging
import math
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
import random
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Maintains complete training state"""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    
    # Loss tracking
    train_loss: float = 0.0
    val_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    
    # Metric tracking
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    learning_rates: List[float] = field(default_factory=list)
    
    # Early stopping
    patience_counter: int = 0
    should_stop: bool = False
    
    # Gradient accumulation
    accumulation_steps: int = 0
    effective_batch_size: int = 0
    
    # Training time
    total_training_time: float = 0.0
    epoch_times: List[float] = field(default_factory=list)
    
    # Checkpoint info
    last_checkpoint_step: int = 0
    checkpoints_saved: List[str] = field(default_factory=list)
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics history"""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        # Convert to regular dict to avoid serialization issues
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingState':
        """Load from dictionary"""
        return cls(**data)
    
    def get_summary(self) -> str:
        """Get training state summary"""
        summary = f"Epoch: {self.epoch}, Step: {self.global_step}\n"
        summary += f"Best Metric: {self.best_metric:.4f} (Epoch {self.best_epoch})\n"
        summary += f"Train Loss: {self.train_loss:.4f}, Val Loss: {self.val_loss:.4f}\n"
        summary += f"Patience: {self.patience_counter}, Should Stop: {self.should_stop}"
        return summary


class DistributedTrainingHelper:
    """Helper for distributed training setup and management"""
    
    def __init__(self, local_rank: int = -1, backend: str = 'nccl'):
        self.local_rank = local_rank
        self.backend = backend
        self.is_distributed = False
        self.world_size = 1
        self.rank = 0
        self.device = None
        
    def setup(self) -> torch.device:
        """Set up distributed training and return device for this process.

        Returns:
            torch.device: The GPU or CPU assigned to the current process.
        """
        if 'WORLD_SIZE' in os.environ:
            self.is_distributed = True
            
            # Get distributed parameters
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            
            if self.local_rank == -1:
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    init_method='env://'
                )
            
            # Setup device
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            
            logger.info(f"Distributed training: Rank {self.rank}/{self.world_size}, "
                       f"Local rank {self.local_rank}")
        else:
            # Single GPU or CPU training
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Single device training on {self.device}")
        
        return self.device
    
    def wrap_model(self, model: nn.Module, sync_bn: bool = True) -> nn.Module:
        """Wrap model for distributed training"""
        if self.is_distributed:
            # Convert BatchNorm to SyncBatchNorm
            if sync_bn and torch.cuda.is_available():
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
            # Wrap with DDP
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            logger.info("Model wrapped with DistributedDataParallel")
        
        return model
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_distributed:
            dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.rank == 0
    
    def barrier(self):
        """Synchronize all processes"""
        if self.is_distributed:
            dist.barrier()
    
    def reduce_tensor(self, tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
        """Reduce tensor across all processes"""
        if not self.is_distributed:
            return tensor
        
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        
        if average:
            rt /= self.world_size
        
        return rt
    
    def gather_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all processes"""
        if not self.is_distributed:
            return [tensor]
        
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        
        return gathered


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing with warm restarts and linear warmup"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0
        
        self.base_max_lr = max_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            # Linear warmup
            return [(self.max_lr - self.min_lr) * self.step_in_cycle / self.warmup_steps + self.min_lr
                    for _ in self.base_lrs]
        else:
            # Cosine annealing
            return [self.min_lr + (self.max_lr - self.min_lr) * 
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                    (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for _ in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            self.step_in_cycle += 1
            
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
                self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cycle = n
                    self.cur_cycle_steps = int(self.first_cycle_steps * self.cycle_mult ** n)
                    self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
            else:
                self.step_in_cycle = epoch
        
        super().step(epoch)


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup followed by cosine annealing"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * 
                    self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]


class GradientAccumulator:
    """Helper for gradient accumulation"""
    
    def __init__(
        self,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True
    ):
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        
        if use_amp and torch.cuda.is_available():
            self.scaler = GradScaler(device='cuda')
        else:
            self.scaler = None
        self.current_step = 0
        
    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """Backward pass with gradient accumulation"""
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)
    
    def step(self, optimizer: optim.Optimizer, model: nn.Module = None) -> bool:
        """Optimizer step with gradient accumulation
        
        Returns:
            True if optimizer step was taken, False otherwise
        """
        self.current_step += 1
        
        if self.current_step % self.accumulation_steps == 0:
            # Clip gradients
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(optimizer)
            
            if self.max_grad_norm > 0:
                if model is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                else:
                    # Clip all parameter gradients in optimizer
                    params = []
                    for group in optimizer.param_groups:
                        params.extend(group['params'])
                    torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            
            # Optimizer step
            if self.use_amp and self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            return True
        
        return False
    
    def zero_grad(self, optimizer: optim.Optimizer):
        """Zero gradients if accumulation complete"""
        if self.current_step % self.accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing"""
        state = {
            'current_step': self.current_step,
            'accumulation_steps': self.accumulation_steps,
            'max_grad_norm': self.max_grad_norm,
            'use_amp': self.use_amp
        }
        if self.scaler is not None:
            state['scaler_state'] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint"""
        self.current_step = state_dict.get('current_step', 0)
        if 'scaler_state' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler_state'])


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if mode == 'min' else -np.Inf
        
    def __call__(self, val_score: float, model: nn.Module = None) -> bool:
        """Check if should stop training
        
        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_score: float, model: nn.Module = None):
        """Saves model when validation score improves"""
        if self.verbose:
            delta = val_score - self.val_score_min if self.mode == 'min' else self.val_score_min - val_score
            logger.info(f'Validation score improved ({self.val_score_min:.6f} --> {val_score:.6f})')
        
        if self.mode == 'min':
            self.val_score_min = val_score
        else:
            self.val_score_min = val_score
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if self.mode == 'min' else -np.Inf


class CheckpointManager:
    """Manages model checkpoints"""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        keep_best: bool = True,
        save_frequency: int = 1,
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.save_frequency = save_frequency
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = float('inf')
        
        # Load existing checkpoints
        self._scan_checkpoints()
    
    def _scan_checkpoints(self):
        """Scan directory for existing checkpoints"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        self.checkpoints = checkpoint_files
        
        # Find best checkpoint
        best_file = self.checkpoint_dir / "best_model.pt"
        if best_file.exists():
            self.best_checkpoint = best_file
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        training_state: TrainingState,
        is_best: bool = False,
        config: Optional[Dict] = None
    ) -> Path:
        """Save a checkpoint"""
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'metrics': metrics,
            'training_state': training_state.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if self.save_scheduler and scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if config is not None:
            checkpoint['config'] = config

        # Save vocabulary info (but not full tag_names to avoid placeholders)
        if hasattr(model, 'module'):
            model_to_check = model.module
        else:
            model_to_check = model

        if hasattr(model_to_check, 'config'):
            checkpoint['vocabulary_info'] = {
                'num_tags': getattr(model_to_check.config, 'num_tags', None),
                'vocab_path': 'vocabulary.json',
                'has_vocabulary': True
            }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if applicable
        if is_best and self.keep_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)
            self.best_checkpoint = best_path
            logger.info(f"Saved best model to {best_path}")
        
        # Manage checkpoint limit
        self._cleanup_old_checkpoints()
        
        # Update training state
        training_state.checkpoints_saved.append(str(checkpoint_path))
        training_state.last_checkpoint_step = step
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding limit"""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by modification time
            self.checkpoints.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest checkpoints
            to_remove = self.checkpoints[:-self.max_checkpoints]
            
            for checkpoint in to_remove:
                if checkpoint != self.best_checkpoint:
                    checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint}")
            
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = torch.device('cpu')
    ) -> Dict:
        """Load a checkpoint"""
        
        if checkpoint_path is None:
            # Load latest checkpoint
            if self.checkpoints:
                checkpoint_path = self.checkpoints[-1]
            else:
                raise ValueError("No checkpoints found")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if model is not None:
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            # Check if optimizer has state before trying to move it
            if hasattr(optimizer, 'state') and optimizer.state:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Move optimizer state to device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            else:
                # Some optimizers might not have state yet
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        return self.best_checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        if self.checkpoints:
            return self.checkpoints[-1]
        return None


class LearningRateSchedulerFactory:
    """Factory for creating learning rate schedulers"""
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str,
        num_epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 0,
        warmup_steps: int = 0,
        min_lr: float = 1e-8,
        **kwargs
    ) -> _LRScheduler:
        """Create a learning rate scheduler
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler (cosine, linear, exponential, etc.)
            num_epochs: Total number of epochs
            steps_per_epoch: Number of steps per epoch
            warmup_epochs: Number of warmup epochs
            warmup_steps: Number of warmup steps (overrides warmup_epochs)
            min_lr: Minimum learning rate
            **kwargs: Additional scheduler-specific arguments
        """
        
        # Calculate warmup steps if epochs specified
        if warmup_epochs > 0 and warmup_steps == 0:
            warmup_steps = warmup_epochs * steps_per_epoch
        
        if scheduler_type == 'cosine':
            return LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=num_epochs,
                warmup_start_lr=kwargs.get('warmup_start_lr', min_lr),
                eta_min=min_lr
            )
        
        elif scheduler_type == 'cosine_restarts':
            return CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=kwargs.get('first_cycle_steps', steps_per_epoch * num_epochs),
                cycle_mult=kwargs.get('cycle_mult', 1.0),
                max_lr=kwargs.get('max_lr', optimizer.defaults['lr']),
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                gamma=kwargs.get('gamma', 1.0)
            )
        
        elif scheduler_type == 'linear':
            return optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=kwargs.get('start_factor', 1.0),
                end_factor=kwargs.get('end_factor', 0.01),
                total_iters=num_epochs
            )
        
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        
        elif scheduler_type == 'multistep':
            return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=kwargs.get('milestones', [30, 60, 90]),
                gamma=kwargs.get('gamma', 0.1)
            )
        
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.1),
                patience=kwargs.get('patience', 10),
                threshold=kwargs.get('threshold', 0.0001),
                min_lr=min_lr
            )
        
        elif scheduler_type == 'constant':
            return optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=kwargs.get('factor', 1.0),
                total_iters=kwargs.get('total_iters', 1)
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class MixedPrecisionTrainer:
    """Helper for mixed precision training"""
    
    def __init__(
        self,
        use_amp: bool = True,
        amp_dtype: str = 'float16',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Determine AMP dtype
        if amp_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Create gradient scaler for FP16 with new API
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler = GradScaler(device='cuda')
        else:
            self.scaler = None
    
    def train_step(
        self,
        model: nn.Module,
        batch: Dict,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        accumulation_step: int
    ) -> Tuple[torch.Tensor, Dict]:
        """Execute a single training step with mixed precision
        
        Returns:
            Loss value and additional outputs
        """
        # Forward pass with autocast
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with autocast(device_type=device_type, enabled=self.use_amp, dtype=self.amp_dtype):
            outputs = model(batch['image'], labels=batch.get('labels'))
            
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = criterion(outputs, batch['labels'])
            
            # Scale loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping and optimizer step
        if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)
            
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
            if self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        return loss.item() * self.gradient_accumulation_steps, outputs
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing"""
        state = {
            'use_amp': self.use_amp,
            'amp_dtype': str(self.amp_dtype),
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_grad_norm': self.max_grad_norm
        }
        if self.scaler is not None:
            state['scaler_state'] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint"""
        if 'scaler_state' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler_state'])


class TrainingMetricsTracker:
    """Tracks and aggregates training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.epoch_metrics = defaultdict(list)
        self.global_step = 0
        self._lock = threading.RLock()  # Use RLock to allow recursive locking
        
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Update metrics"""
        with self._lock:
            if step is not None:
                self.global_step = step
            
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.metrics[key].append(value)
    
    def update_epoch(self, metrics: Dict[str, float]):
        """Update epoch-level metrics"""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.epoch_metrics[key].append(value)
    
    def get_average(self, metric_name: str) -> float:
        with self._lock:
            if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
                # Create a copy to avoid issues if deque is modified during mean calculation
                values = list(self.metrics[metric_name])
                return np.mean(values)
            return 0.0
    
    def get_last(self, metric_name: str) -> float:
        """Get last value of metric"""
        with self._lock:
            if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
                return self.metrics[metric_name][-1]
            return 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        with self._lock:
            summary = {}
            for key, values in self.metrics.items():
                if len(values) > 0:
                    # Create a copy to avoid issues during statistical calculations
                    values_copy = list(values)
                    summary[f'{key}_avg'] = np.mean(values_copy)
                    summary[f'{key}_std'] = np.std(values_copy)
                    summary[f'{key}_last'] = values_copy[-1]
            return summary
    
    def reset(self):
        """Reset metrics"""
        self.metrics.clear()
    
    def reset_epoch(self):
        """Reset epoch metrics"""
        self.epoch_metrics.clear()


class TrainingUtils:
    """Static utility functions for training"""
    
    @staticmethod
    def set_random_seed(seed: int, deterministic: bool = False):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'total_mb': total * 4 / 1024 / 1024,  # Assuming float32
            'trainable_mb': trainable * 4 / 1024 / 1024
        }
    
    @staticmethod
    def get_optimizer(
        model: nn.Module,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float = 0.01,
        **kwargs
    ) -> optim.Optimizer:
        """Create optimizer"""
        
        # Get parameters with weight decay exclusion
        params = TrainingUtils.get_parameter_groups(model, weight_decay)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(
                params,
                lr=learning_rate,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=learning_rate,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(
                params,
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0.9),
                nesterov=kwargs.get('nesterov', True)
            )
        
        elif optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(
                params,
                lr=learning_rate,
                alpha=kwargs.get('alpha', 0.99),
                eps=kwargs.get('eps', 1e-8)
            )
        
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def get_parameter_groups(
        model: nn.Module,
        weight_decay: float = 0.01,
        layer_decay: Optional[float] = None
    ) -> List[Dict]:
        """Get parameter groups with proper weight decay and layer-wise learning rate decay"""
        
        # Parameters that should not have weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'layer_norm', 'ln']
        
        if layer_decay is None or layer_decay == 1.0:
            # Standard parameter groups
            params = [
                {
                    'params': [p for n, p in model.named_parameters() 
                              if not any(nd in n for nd in no_decay) and p.requires_grad],
                    'weight_decay': weight_decay
                },
                {
                    'params': [p for n, p in model.named_parameters() 
                              if any(nd in n for nd in no_decay) and p.requires_grad],
                    'weight_decay': 0.0
                }
            ]
        else:
            # Layer-wise learning rate decay
            params = TrainingUtils._get_layer_wise_params(model, weight_decay, layer_decay, no_decay)
        
        return params
    
    @staticmethod
    def _get_layer_wise_params(
        model: nn.Module,
        weight_decay: float,
        layer_decay: float,
        no_decay: List[str]
    ) -> List[Dict]:
        """Get parameters with layer-wise learning rate decay"""
        
        # Get depth of model
        def get_layer_id(name):
            if 'blocks' in name or 'layers' in name:
                # Extract layer number
                import re
                match = re.search(r'\.(\d+)\.', name)
                if match:
                    return int(match.group(1))
            return 0
        
        # Group parameters by layer
        layer_params = defaultdict(lambda: {'decay': [], 'no_decay': []})
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            layer_id = get_layer_id(name)
            
            if any(nd in name for nd in no_decay):
                layer_params[layer_id]['no_decay'].append(param)
            else:
                layer_params[layer_id]['decay'].append(param)
        
        # Create parameter groups with layer-wise decay
        max_layer = max(layer_params.keys()) if layer_params else 0
        params = []
        
        for layer_id in sorted(layer_params.keys()):
            layer_scale = layer_decay ** (max_layer - layer_id)
            
            if layer_params[layer_id]['decay']:
                params.append({
                    'params': layer_params[layer_id]['decay'],
                    'weight_decay': weight_decay,
                    'lr_scale': layer_scale
                })
            
            if layer_params[layer_id]['no_decay']:
                params.append({
                    'params': layer_params[layer_id]['no_decay'],
                    'weight_decay': 0.0,
                    'lr_scale': layer_scale
                })
        
        return params
    
    @staticmethod
    def compute_effective_batch_size(
        batch_size: int,
        accumulation_steps: int,
        world_size: int = 1
    ) -> int:
        """Compute effective batch size"""
        return batch_size * accumulation_steps * world_size
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in seconds to readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def save_training_config(config: Dict, output_dir: Path):
        """Save training configuration"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Saved training config to {config_path}")


if __name__ == "__main__":
    # Test the utilities
    print("Testing Training Utilities...")
    
    # Test distributed helper
    dist_helper = DistributedTrainingHelper()
    device = dist_helper.setup()
    print(f"Device: {device}")
    print(f"Is distributed: {dist_helper.is_distributed}")
    
    # Test training state
    state = TrainingState()
    state.epoch = 5
    state.global_step = 1000
    state.update_metrics({'loss': 0.5, 'accuracy': 0.95})
    print(f"\nTraining State Summary:\n{state.get_summary()}")
    
    # Test checkpoint manager
    checkpoint_manager = CheckpointManager("./test_checkpoints")
    print(f"\nCheckpoint directory: {checkpoint_manager.checkpoint_dir}")
    
    # Test scheduler factory
    import torch.optim as optim
    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = LearningRateSchedulerFactory.create_scheduler(
        optimizer,
        scheduler_type='cosine',
        num_epochs=100,
        steps_per_epoch=100,
        warmup_epochs=5
    )
    print(f"\nCreated scheduler: {type(scheduler).__name__}")
    
    # Test metrics tracker
    tracker = TrainingMetricsTracker()
    for i in range(10):
        tracker.update({'loss': 0.5 - i*0.01, 'accuracy': 0.8 + i*0.01})
    
    summary = tracker.get_summary()
    print(f"\nMetrics summary: {summary}")
    
    print("\n✓ All utilities tested successfully!")