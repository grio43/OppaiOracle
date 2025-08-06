#!/usr/bin/env python3
"""
Monitoring & Logging System for Anime Image Tagger
Comprehensive monitoring for training, inference, and system resources
"""

import os
import sys
import json
import logging
import psutil
import GPUtil
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
import traceback
import socket
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
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
    
    # Alerts
    enable_alerts: bool = True
    alert_on_gpu_memory_threshold: float = 0.9  # 90% usage
    alert_on_cpu_memory_threshold: float = 0.9
    alert_on_disk_space_threshold: float = 0.95
    alert_on_training_stuck_minutes: int = 30
    alert_on_loss_explosion: float = 10.0
    alert_on_nan_loss: bool = True
    
    # Performance profiling
    enable_profiling: bool = False
    profile_interval_steps: int = 1000
    profile_duration_steps: int = 10
    profile_memory: bool = True
    profile_shapes: bool = True
    
    # Data pipeline monitoring
    monitor_data_pipeline: bool = True
    data_pipeline_stats_interval: int = 100  # batches
    
    # Remote monitoring
    enable_prometheus: bool = False
    prometheus_port: int = 8080
    
    # History
    max_history_size: int = 10000
    history_save_interval: int = 100
    
    # Distributed training
    distributed: bool = False
    rank: int = 0
    world_size: int = 1


class MetricsTracker:
    """Tracks and aggregates various metrics"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.metrics = defaultdict(lambda: deque(maxlen=config.max_history_size))
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.current_timers = {}
        
        # Prometheus metrics
        if config.enable_prometheus and PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.prom_counters = {
            'batches_processed': Counter('anime_tagger_batches_processed', 'Total batches processed'),
            'images_processed': Counter('anime_tagger_images_processed', 'Total images processed'),
            'errors': Counter('anime_tagger_errors', 'Total errors', ['error_type'])
        }
        
        self.prom_gauges = {
            'loss': Gauge('anime_tagger_loss', 'Current loss'),
            'learning_rate': Gauge('anime_tagger_learning_rate', 'Current learning rate'),
            'gpu_memory_used': Gauge('anime_tagger_gpu_memory_used_gb', 'GPU memory used (GB)', ['gpu_id']),
            'gpu_utilization': Gauge('anime_tagger_gpu_utilization_percent', 'GPU utilization (%)', ['gpu_id'])
        }
        
        self.prom_histograms = {
            'batch_time': Histogram('anime_tagger_batch_time_seconds', 'Batch processing time'),
            'data_load_time': Histogram('anime_tagger_data_load_time_seconds', 'Data loading time')
        }
    
    def add_metric(self, name: str, value: float, step: Optional[int] = None):
        """Add a metric value"""
        timestamp = time.time()
        self.metrics[name].append({
            'value': value,
            'step': step,
            'timestamp': timestamp
        })
        
        # Update Prometheus
        if self.config.enable_prometheus and PROMETHEUS_AVAILABLE:
            if name in self.prom_gauges:
                self.prom_gauges[name].set(value)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        self.counters[name] += value
        
        # Update Prometheus
        if self.config.enable_prometheus and PROMETHEUS_AVAILABLE:
            if name in self.prom_counters:
                self.prom_counters[name].inc(value)
    
    def start_timer(self, name: str):
        """Start a timer"""
        self.current_timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a timer and record duration"""
        if name not in self.current_timers:
            return 0.0
        
        duration = time.time() - self.current_timers[name]
        self.timers[name].append(duration)
        del self.current_timers[name]
        
        # Update Prometheus
        if self.config.enable_prometheus and PROMETHEUS_AVAILABLE:
            if name in self.prom_histograms:
                self.prom_histograms[name].observe(duration)
        
        return duration
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {}
        
        values = [m['value'] for m in self.metrics[name]]
        return {
            'current': values[-1],
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a timer"""
        if name not in self.timers or len(self.timers[name]) == 0:
            return {}
        
        durations = self.timers[name]
        return {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations),
            'total': np.sum(durations),
            'count': len(durations)
        }


class SystemMonitor:
    """Monitors system resources"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.running = False
        self.thread = None
        self.metrics_queue = queue.Queue()
        
        # Initialize GPU monitoring
        self.gpus = []
        if config.track_gpu_metrics:
            try:
                self.gpus = GPUtil.getGPUs()
                logger.info(f"Found {len(self.gpus)} GPUs for monitoring")
            except:
                logger.warning("GPU monitoring not available")
    
    def start(self):
        """Start system monitoring"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("System monitoring started")
    
    def stop(self):
        """Stop system monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.metrics_queue.put(metrics)
                time.sleep(self.config.system_metrics_interval)
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu': {},
            'memory': {},
            'disk': {},
            'gpu': [],
            'network': {}
        }
        
        # CPU metrics
        metrics['cpu']['percent'] = psutil.cpu_percent(interval=1)
        metrics['cpu']['count'] = psutil.cpu_count()
        metrics['cpu']['freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        
        # Memory metrics
        mem = psutil.virtual_memory()
        metrics['memory']['total_gb'] = mem.total / (1024**3)
        metrics['memory']['used_gb'] = mem.used / (1024**3)
        metrics['memory']['percent'] = mem.percent
        
        # Disk metrics
        if self.config.track_disk_io:
            disk = psutil.disk_usage('/')
            metrics['disk']['total_gb'] = disk.total / (1024**3)
            metrics['disk']['used_gb'] = disk.used / (1024**3)
            metrics['disk']['percent'] = disk.percent
            
            # Disk I/O
            io = psutil.disk_io_counters()
            if io:
                metrics['disk']['read_mb'] = io.read_bytes / (1024**2)
                metrics['disk']['write_mb'] = io.write_bytes / (1024**2)
        
        # GPU metrics
        if self.config.track_gpu_metrics and self.gpus:
            for gpu in self.gpus:
                gpu_metrics = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total_gb': gpu.memoryTotal / 1024,
                    'memory_used_gb': gpu.memoryUsed / 1024,
                    'memory_percent': gpu.memoryUtil * 100,
                    'utilization': gpu.load * 100,
                    'temperature': gpu.temperature
                }
                metrics['gpu'].append(gpu_metrics)
        
        # Network metrics
        if self.config.track_network_io:
            net = psutil.net_io_counters()
            metrics['network']['sent_mb'] = net.bytes_sent / (1024**2)
            metrics['network']['recv_mb'] = net.bytes_recv / (1024**2)
        
        return metrics
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest system metrics"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None


class TrainingMonitor:
    """Main monitoring class for training"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.start_time = time.time()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.metrics = MetricsTracker(config)
        self.system_monitor = SystemMonitor(config)
        
        # Setup visualization backends
        self.writer = None
        if config.use_tensorboard:
            self.writer = SummaryWriter(config.tensorboard_dir)
            logger.info(f"TensorBoard logging to {config.tensorboard_dir}")
        
        self.wandb_run = None
        if config.use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.__dict__
            )
            logger.info("Weights & Biases logging initialized")
        
        # Alert system
        self.alerts = AlertSystem(config) if config.enable_alerts else None
        
        # Profiler
        self.profiler = None
        if config.enable_profiling:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=config.profile_shapes,
                profile_memory=config.profile_memory,
                with_stack=True
            )
        
        # Training state
        self.last_step_time = time.time()
        self.last_loss = None
        self.steps_without_improvement = 0
        
        # Data pipeline stats
        self.data_stats = defaultdict(list)
        
        # Start system monitoring
        if config.track_system_metrics:
            self.system_monitor.start()
        
        # Prometheus server
        if config.enable_prometheus and PROMETHEUS_AVAILABLE:
            start_http_server(config.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {config.prometheus_port}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Create formatter
        formatter = logging.Formatter(self.config.log_format)
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            logging.getLogger().addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logging.getLogger().addHandler(file_handler)
            
            # Also create a symlink to latest log
            latest_log = log_dir / "latest.log"
            if latest_log.exists():
                latest_log.unlink()
            latest_log.symlink_to(log_file.name)
        
        # Set overall log level
        logging.getLogger().setLevel(log_level)
    
    def log_step(
        self,
        step: int,
        loss: float,
        metrics: Dict[str, float],
        learning_rate: float,
        batch_size: int
    ):
        """Log training step metrics"""
        # Update metrics
        self.metrics.add_metric('loss', loss, step)
        self.metrics.add_metric('learning_rate', learning_rate, step)
        
        for name, value in metrics.items():
            self.metrics.add_metric(name, value, step)
        
        # Track counters
        self.metrics.increment_counter('batches_processed')
        self.metrics.increment_counter('images_processed', batch_size)
        
        # Check for issues
        if self.alerts:
            # NaN loss
            if np.isnan(loss) and self.config.alert_on_nan_loss:
                self.alerts.send_alert(
                    "NaN Loss Detected",
                    f"Loss became NaN at step {step}",
                    severity="critical"
                )
            
            # Loss explosion
            if loss > self.config.alert_on_loss_explosion:
                self.alerts.send_alert(
                    "Loss Explosion",
                    f"Loss exceeded threshold: {loss:.4f} at step {step}",
                    severity="warning"
                )
            
            # Training stuck
            if self.last_loss is not None and abs(loss - self.last_loss) < 1e-6:
                self.steps_without_improvement += 1
                if self.steps_without_improvement > 100:  # Arbitrary threshold
                    minutes_stuck = (time.time() - self.last_step_time) / 60
                    if minutes_stuck > self.config.alert_on_training_stuck_minutes:
                        self.alerts.send_alert(
                            "Training Stuck",
                            f"No improvement for {minutes_stuck:.1f} minutes",
                            severity="warning"
                        )
            else:
                self.steps_without_improvement = 0
        
        # Update state
        self.last_loss = loss
        self.last_step_time = time.time()
        
        # Log to visualization backends
        if self.writer:
            self.writer.add_scalar('train/loss', loss, step)
            self.writer.add_scalar('train/learning_rate', learning_rate, step)
            for name, value in metrics.items():
                self.writer.add_scalar(f'train/{name}', value, step)
        
        if self.wandb_run:
            wandb.log({
                'train/loss': loss,
                'train/learning_rate': learning_rate,
                **{f'train/{name}': value for name, value in metrics.items()},
                'step': step
            })
        
        # System metrics
        if self.config.track_system_metrics:
            sys_metrics = self.system_monitor.get_latest_metrics()
            if sys_metrics:
                self._log_system_metrics(sys_metrics, step)
    
    def log_validation(
        self,
        step: int,
        metrics: Dict[str, float]
    ):
        """Log validation metrics"""
        for name, value in metrics.items():
            self.metrics.add_metric(f'val_{name}', value, step)
            
            if self.writer:
                self.writer.add_scalar(f'val/{name}', value, step)
            
            if self.wandb_run:
                wandb.log({f'val/{name}': value, 'step': step})
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        duration: float
    ):
        """Log epoch summary"""
        logger.info(f"Epoch {epoch} completed in {duration:.2f}s")
        logger.info(f"Train metrics: {train_metrics}")
        logger.info(f"Val metrics: {val_metrics}")
        
        # Log to backends
        if self.writer:
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'epoch/train_{name}', value, epoch)
            for name, value in val_metrics.items():
                self.writer.add_scalar(f'epoch/val_{name}', value, epoch)
            self.writer.add_scalar('epoch/duration', duration, epoch)
        
        if self.wandb_run:
            wandb.log({
                **{f'epoch/train_{name}': value for name, value in train_metrics.items()},
                **{f'epoch/val_{name}': value for name, value in val_metrics.items()},
                'epoch/duration': duration,
                'epoch': epoch
            })
    
   def log_images(
    self,
    step: int,
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    tag_names: List[str],
    num_images: int = 4
):
    """Log example images with predictions"""
    if self.writer:
        # Select random images
        indices = torch.randperm(len(images))[:num_images]
        
        for i, idx in enumerate(indices):
            img = images[idx]
            pred = predictions[idx]
            target = targets[idx]
            
            # Get top predictions
            top_k = 10
            pred_scores, pred_indices = torch.topk(pred, top_k)
            
            # Create text overlay
            pred_tags = [f"{tag_names[j]}: {pred_scores[k]:.2f}" 
                        for k, j in enumerate(pred_indices)]
            
            true_indices = torch.where(target > 0)[0][:top_k]
            true_tags = [tag_names[j] for j in true_indices]
            
            # Log image with caption
            caption = (
                f"Predicted: {', '.join(pred_tags[:5])}\n"
                f"True: {', '.join(true_tags[:5])}"
            )
            
            # Add to tensorboard
            self.writer.add_image(f'val/image_{i}', img, step)
            self.writer.add_text(f'val/caption_{i}', caption, step)
        
        # Log to wandb if enabled
        if self.wandb_run:
            import wandb
            wandb_images = []
            for i, idx in enumerate(indices[:min(4, len(indices))]):
                img = images[idx].cpu().numpy().transpose(1, 2, 0)
                pred_tags = [tag_names[j] for j in torch.topk(predictions[idx], 5)[1]]
                true_tags = [tag_names[j] for j in torch.where(targets[idx] > 0)[0][:5]]
                
                wandb_images.append(wandb.Image(
                    img,
                    caption=f"Pred: {', '.join(pred_tags)} | True: {', '.join(true_tags)}"
                ))
            
            wandb.log({"val/images": wandb_images, "step": step})