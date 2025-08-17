#!/usr/bin/env python3
"""
Monitoring & Logging System for Anime Image Tagger
Comprehensive monitoring for training, inference, and system resources
Enhanced with proper error handling, thread safety, and complete implementations
"""

import os
import sys
import json
import logging
import psutil
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
from contextlib import contextmanager
import atexit
import signal

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Optional imports with proper handling
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    warnings.warn("GPUtil not available. GPU monitoring disabled.")

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
    pd = None

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
    wandb_run_name: Optional[str] = None
    
    # Alerts
    enable_alerts: bool = True
    alert_on_gpu_memory_threshold: float = 0.9  # 90% usage
    alert_on_cpu_memory_threshold: float = 0.9
    alert_on_disk_space_threshold: float = 0.95
    alert_on_training_stuck_minutes: int = 30
    alert_on_loss_explosion: float = 10.0
    alert_on_nan_loss: bool = True
    alert_webhook_url: Optional[str] = None  # For Slack/Discord alerts
    
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
    
    # Remote monitoring
    enable_prometheus: bool = False
    prometheus_port: int = 8080
    
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


class AlertSystem:
    """System for sending alerts and notifications"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.alert_history = deque(maxlen=100)
        self.alert_counts = defaultdict(int)
        self.last_alert_time = defaultdict(float)
        self.min_alert_interval = 300  # 5 minutes between same alerts
        
    def send_alert(self, title: str, message: str, severity: str = "info"):
        """Send an alert through configured channels"""
        # Check if we should suppress this alert
        alert_key = f"{title}_{severity}"
        current_time = time.time()
        
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.min_alert_interval:
                return  # Suppress duplicate alerts
        
        self.last_alert_time[alert_key] = current_time
        self.alert_counts[alert_key] += 1
        
        # Record alert
        alert = {
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'message': message,
            'severity': severity,
            'count': self.alert_counts[alert_key]
        }
        self.alert_history.append(alert)
        
        # Log alert
        log_method = getattr(logger, severity.lower(), logger.info)
        log_method(f"ALERT: {title} - {message}")
        
        # Send to webhook if configured
        if self.config.alert_webhook_url:
            self._send_webhook_alert(alert)
        
        # Send to console with color
        self._print_colored_alert(alert)
    
    def _send_webhook_alert(self, alert: dict):
        """Send alert to webhook (Slack/Discord compatible)"""
        try:
            import requests
            
            # Format for common webhook formats
            payload = {
                'text': f"**{alert['severity'].upper()}**: {alert['title']}",
                'attachments': [{
                    'color': self._get_severity_color(alert['severity']),
                    'fields': [
                        {'title': 'Message', 'value': alert['message']},
                        {'title': 'Time', 'value': alert['timestamp']},
                        {'title': 'Occurrence', 'value': str(alert['count'])}
                    ]
                }]
            }
            
            requests.post(self.config.alert_webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _print_colored_alert(self, alert: dict):
        """Print colored alert to console"""
        colors = {
            'critical': '\033[91m',  # Red
            'error': '\033[91m',     # Red
            'warning': '\033[93m',   # Yellow
            'info': '\033[94m',      # Blue
            'success': '\033[92m'    # Green
        }
        reset = '\033[0m'
        
        color = colors.get(alert['severity'], '')
        print(f"{color}[{alert['severity'].upper()}] {alert['title']}: {alert['message']}{reset}")
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color code for severity level"""
        colors = {
            'critical': '#FF0000',
            'error': '#FF6B6B',
            'warning': '#FFA500',
            'info': '#0099CC',
            'success': '#00AA00'
        }
        return colors.get(severity, '#808080')


class ThreadSafeMetricsTracker:
    """Thread-safe metrics tracker"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.metrics = defaultdict(lambda: deque(maxlen=config.max_history_size))
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.current_timers = {}
        self.lock = threading.RLock()
        
        # Prometheus metrics
        self.prom_metrics = {}
        if config.enable_prometheus and PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        try:
            self.prom_metrics['counters'] = {
                'batches_processed': Counter('anime_tagger_batches_processed', 'Total batches processed'),
                'images_processed': Counter('anime_tagger_images_processed', 'Total images processed'),
                'errors': Counter('anime_tagger_errors', 'Total errors', ['error_type'])
            }
            
            self.prom_metrics['gauges'] = {
                'loss': Gauge('anime_tagger_loss', 'Current loss'),
                'learning_rate': Gauge('anime_tagger_learning_rate', 'Current learning rate'),
                'gpu_memory_used': Gauge('anime_tagger_gpu_memory_used_gb', 'GPU memory used (GB)', ['gpu_id']),
                'gpu_utilization': Gauge('anime_tagger_gpu_utilization_percent', 'GPU utilization (%)', ['gpu_id'])
            }
            
            self.prom_metrics['histograms'] = {
                'batch_time': Histogram('anime_tagger_batch_time_seconds', 'Batch processing time'),
                'data_load_time': Histogram('anime_tagger_data_load_time_seconds', 'Data loading time')
            }
        except Exception as e:
            logger.error(f"Failed to setup Prometheus metrics: {e}")
            self.prom_metrics = {}
    
    def add_metric(self, name: str, value: float, step: Optional[int] = None):
        """Add a metric value (thread-safe)"""
        with self.lock:
            timestamp = time.time()
            self.metrics[name].append({
                'value': value,
                'step': step,
                'timestamp': timestamp
            })
            
            # Update Prometheus
            if self.prom_metrics and 'gauges' in self.prom_metrics:
                if name in self.prom_metrics['gauges']:
                    try:
                        self.prom_metrics['gauges'][name].set(value)
                    except Exception as e:
                        logger.debug(f"Failed to update Prometheus gauge {name}: {e}")
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter (thread-safe)"""
        with self.lock:
            self.counters[name] += value
            
            # Update Prometheus
            if self.prom_metrics and 'counters' in self.prom_metrics:
                if name in self.prom_metrics['counters']:
                    try:
                        self.prom_metrics['counters'][name].inc(value)
                    except Exception as e:
                        logger.debug(f"Failed to update Prometheus counter {name}: {e}")
    
    def start_timer(self, name: str):
        """Start a timer (thread-safe)"""
        with self.lock:
            self.current_timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a timer and record duration (thread-safe)"""
        with self.lock:
            if name not in self.current_timers:
                return 0.0
            
            duration = time.time() - self.current_timers[name]
            self.timers[name].append(duration)
            del self.current_timers[name]
            
            # Update Prometheus
            if self.prom_metrics and 'histograms' in self.prom_metrics:
                if name in self.prom_metrics['histograms']:
                    try:
                        self.prom_metrics['histograms'][name].observe(duration)
                    except Exception as e:
                        logger.debug(f"Failed to update Prometheus histogram {name}: {e}")
            
            return duration
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric (thread-safe)"""
        with self.lock:
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
        """Get statistics for a timer (thread-safe)"""
        with self.lock:
            if name not in self.timers or len(self.timers[name]) == 0:
                return {}
            
            durations = self.timers[name][-100:]  # Last 100 for efficiency
            return {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'total': np.sum(self.timers[name]),
                'count': len(self.timers[name])
            }
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        with self.lock:
            try:
                data = {
                    'metrics': {k: list(v) for k, v in self.metrics.items()},
                    'counters': dict(self.counters),
                    'timer_stats': {k: self.get_timer_stats(k) for k in self.timers}
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
                logger.info(f"Metrics saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save metrics: {e}")


class SystemMonitor:
    """Monitors system resources with proper cleanup"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.running = False
        self.thread = None
        self._lock = threading.Lock()  # Add lock for thread safety
        self.metrics_queue = queue.Queue(maxsize=100)
        self.error_count = 0
        self.max_errors = 10
        
        self._shutdown_event = threading.Event()
        # Initialize GPU monitoring
        self.gpu_available = False
        if config.track_gpu_metrics and GPUTIL_AVAILABLE:
            try:
                # Test GPUtil functionality, not just presence
                test_gpus = GPUtil.getGPUs()
                self.gpu_available = test_gpus is not None and len(test_gpus) > 0
                if self.gpu_available:
                    logger.info(f"GPU monitoring enabled for {len(test_gpus)} GPU(s)")
                else:
                    logger.info("No GPUs detected for monitoring")
            except (Exception, ImportError) as e:
                logger.warning(f"GPU monitoring not available: {e}")
                self.gpu_available = False
    
    def start(self):
        """Start system monitoring"""
        with self._lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True, name="SystemMonitor")
            self.thread.start()
            logger.info("System monitoring started")
    
    def stop(self):
        """Stop system monitoring gracefully"""
        with self._lock:
            if not self.running:
                return
            
            self.running = False
            thread_to_join = self.thread
            
        if thread_to_join and thread_to_join.is_alive():
            thread_to_join.join(timeout=5)
            if thread_to_join.is_alive():
                logger.warning("System monitor thread did not stop gracefully")
            # Set shutdown event to interrupt any blocking operations
            self._shutdown_event.set()
            # Give it one more chance to stop
            thread_to_join.join(timeout=2)
            if thread_to_join.is_alive():
                logger.error("System monitor thread is unresponsive and may remain as zombie thread")
                # Force cleanup of thread reference
                self.thread = None
            
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop with error handling"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                
                # Try to put metrics in queue (non-blocking)
                try:
                    self.metrics_queue.put_nowait(metrics)
                except queue.Full:
                    # Remove oldest item and add new one
                    try:
                        self.metrics_queue.get_nowait()
                        self.metrics_queue.put_nowait(metrics)
                    except:
                        pass
                
                # Reset error count on success
                self.error_count = 0
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in system monitoring: {e}")
                
                # Stop monitoring if too many errors
                if self.error_count >= self.max_errors:
                    logger.critical("Too many errors in system monitoring. Stopping.")
                    self.running = False
                    break
            
            # Sleep with interruptible check
            sleep_iterations = int(self.config.system_metrics_interval * 10)
            for _ in range(sleep_iterations):
                if not self.running:
                    break
                if self._shutdown_event.is_set():
                    break
                time.sleep(0.1)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics with error handling"""
        metrics = {
            'timestamp': time.time(),
            'cpu': {},
            'memory': {},
            'disk': {},
            'gpu': [],
            'network': {}
        }
        
        try:
            # CPU metrics
            metrics['cpu']['percent'] = psutil.cpu_percent(interval=0.1)
            metrics['cpu']['count'] = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            metrics['cpu']['freq'] = cpu_freq.current if cpu_freq else 0
            
            # Per-CPU metrics
            metrics['cpu']['per_cpu'] = psutil.cpu_percent(interval=0.1, percpu=True)
        except Exception as e:
            logger.debug(f"Failed to collect CPU metrics: {e}")

        if not self.running or self._shutdown_event.is_set():
            return metrics
                
        try:
            # Memory metrics
            mem = psutil.virtual_memory()
            metrics['memory']['total_gb'] = mem.total / (1024**3)
            metrics['memory']['used_gb'] = mem.used / (1024**3)
            metrics['memory']['available_gb'] = mem.available / (1024**3)
            metrics['memory']['percent'] = mem.percent
            
            # Swap memory
            swap = psutil.swap_memory()
            metrics['memory']['swap_used_gb'] = swap.used / (1024**3)
            metrics['memory']['swap_percent'] = swap.percent
        except Exception as e:
            logger.debug(f"Failed to collect memory metrics: {e}")

        if not self.running or self._shutdown_event.is_set():
            return metrics
        
        try:
            # Disk metrics
            if self.config.track_disk_io:
                disk = psutil.disk_usage('/')
                metrics['disk']['total_gb'] = disk.total / (1024**3)
                metrics['disk']['used_gb'] = disk.used / (1024**3)
                metrics['disk']['free_gb'] = disk.free / (1024**3)
                metrics['disk']['percent'] = disk.percent
                
                # Disk I/O
                io = psutil.disk_io_counters()
                if io:
                    metrics['disk']['read_mb'] = io.read_bytes / (1024**2)
                    metrics['disk']['write_mb'] = io.write_bytes / (1024**2)
                    metrics['disk']['read_count'] = io.read_count
                    metrics['disk']['write_count'] = io.write_count
        except Exception as e:
            logger.debug(f"Failed to collect disk metrics: {e}")
        
        if not self.running or self._shutdown_event.is_set():
            return metrics
           
        try:
            # GPU metrics (with fresh query)
            if self.gpu_available and GPUTIL_AVAILABLE:
                try:
                    # Double-check GPUtil is still available and working
                    gpus = GPUtil.getGPUs()
                    if gpus is not None:  # Additional safety check
                        for gpu in gpus:
                            if gpu is not None:  # Check each GPU object
                                # Safely get all values with proper null handling
                                mem_total = getattr(gpu, 'memoryTotal', 0)
                                mem_used = getattr(gpu, 'memoryUsed', 0)
                                mem_free = getattr(gpu, 'memoryFree', 0)
                                gpu_load = getattr(gpu, 'load', 0)
                                
                                # Ensure numeric values
                                mem_total = float(mem_total) if mem_total is not None else 0
                                mem_used = float(mem_used) if mem_used is not None else 0
                                mem_free = float(mem_free) if mem_free is not None else 0
                                gpu_load = float(gpu_load) if gpu_load is not None else 0
                                
                                gpu_metrics = {
                                    'id': getattr(gpu, 'id', -1),
                                    'name': getattr(gpu, 'name', 'Unknown'),
                                    'memory_total_gb': mem_total / 1024 if mem_total > 0 else 0,
                                    'memory_used_gb': mem_used / 1024 if mem_used > 0 else 0,
                                    'memory_free_gb': mem_free / 1024 if mem_free > 0 else 0,
                                    'memory_percent': (mem_used / mem_total * 100) if mem_total > 0 else 0,
                                    'utilization': gpu_load * 100 if gpu_load >= 0 else 0,
                                    'temperature': getattr(gpu, 'temperature', 0) or 0
                                }
                                metrics['gpu'].append(gpu_metrics)
                except (AttributeError, ImportError, RuntimeError) as e:
                    # GPUtil might become unavailable, disable it
                    self.gpu_available = False
                    logger.warning(f"GPUtil became unavailable, disabling GPU monitoring: {e}")
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")
                    
        try:
            # Network metrics
            if self.config.track_network_io:
                net = psutil.net_io_counters()
                metrics['network']['sent_mb'] = net.bytes_sent / (1024**2)
                metrics['network']['recv_mb'] = net.bytes_recv / (1024**2)
                metrics['network']['packets_sent'] = net.packets_sent
                metrics['network']['packets_recv'] = net.packets_recv
                metrics['network']['errors'] = net.errin + net.errout
                metrics['network']['drops'] = net.dropin + net.dropout
        except Exception as e:
            logger.debug(f"Failed to collect network metrics: {e}")
        
        return metrics
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest system metrics (non-blocking)"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None


class TrainingMonitor:
    """Main monitoring class for training with complete implementation"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.start_time = time.time()
        self.shutdown_handlers = []
        
        # Setup logging first
        self._setup_logging()
        
        # Initialize components
        self.metrics = ThreadSafeMetricsTracker(config)
        self.system_monitor = SystemMonitor(config)
        self.alerts = AlertSystem(config) if config.enable_alerts else None
        
        # Setup visualization backends
        self.writer = None
        if config.use_tensorboard:
            try:
                self.writer = SummaryWriter(config.tensorboard_dir)
                logger.info(f"TensorBoard logging to {config.tensorboard_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize TensorBoard: {e}")
        
        self.wandb_run = None
        if config.use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=config.wandb_run_name,
                    config=asdict(config)
                )
                logger.info("Weights & Biases logging initialized")
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {e}")
        
        # Profiler
        self.profiler = None
        self.profiler_step = 0
        if config.enable_profiling:
            self._setup_profiler()
        
        # Training state
        self.last_step_time = time.time()
        self.last_loss = None
        self.steps_without_improvement = 0
        self.best_val_metric = float('inf')
        
        # Data pipeline stats
        self.data_stats = {
            'load_times': deque(maxlen=1000),
            'batch_sizes': deque(maxlen=1000),
            'augmentation_times': deque(maxlen=1000)
        }
        
        # Start system monitoring
        if config.track_system_metrics:
            self.system_monitor.start()
        
        # Prometheus server
        self.prometheus_server = None
        if config.enable_prometheus and PROMETHEUS_AVAILABLE:
            try:
                from prometheus_client import start_http_server
                self.prometheus_server = start_http_server(config.prometheus_port)
                logger.info(f"Prometheus metrics server started on port {config.prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")
        
        # Register cleanup handlers
        self._register_cleanup()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Remove existing handlers to avoid duplicates
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(self.config.log_format)
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            try:
                log_dir = Path(self.config.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                
                log_file = log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                file_handler.setLevel(log_level)
                root_logger.addHandler(file_handler)
                
                # Create symlink to latest log
                latest_log = log_dir / "latest.log"
                if latest_log.exists() or latest_log.is_symlink():
                    latest_log.unlink()
                latest_log.symlink_to(log_file.name)
                
            except Exception as e:
                print(f"Failed to setup file logging: {e}")
        
        # Set overall log level
        root_logger.setLevel(log_level)
    
    def _setup_profiler(self):
        """Setup PyTorch profiler"""
        try:
            profile_dir = Path(self.config.profile_output_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
                record_shapes=self.config.profile_shapes,
                profile_memory=self.config.profile_memory,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir))
            )
            logger.info(f"Profiler initialized, output to {profile_dir}")
        except Exception as e:
            logger.error(f"Failed to setup profiler: {e}")
            self.profiler = None
    
    def _register_cleanup(self):
        """Register cleanup handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            """Signal handler function"""
            logger.info(f"Received signal {signum}, shutting down...")
            if hasattr(self, 'system_monitor'):
                self.system_monitor.stop()
            if hasattr(self, 'writer') and self.writer:
                try:
                    self.writer.close()
                except:
                    pass
            sys.exit(0)

        # Only register signals that are available on the platform
        signal_names = {signal.SIGINT: 'SIGINT', signal.SIGTERM: 'SIGTERM'}
        for sig, sig_name in signal_names.items():
            try:
                signal.signal(sig, signal_handler)
                logger.debug(f"Registered signal handler for {sig_name}")
            except (OSError, ValueError, AttributeError) as e:
                logger.debug(f"Could not register signal {sig_name}: {e}")

        def cleanup():
            """Cleanup function for atexit"""
            if hasattr(self, 'system_monitor'):
                self.system_monitor.stop()
            if hasattr(self, 'writer') and self.writer:
                try:
                    self.writer.close()
                except:
                    pass
                
        # Register atexit handler
        atexit.register(cleanup)
    
    def log_step(
        self,
        step: int,
        loss: float,
        metrics: Dict[str, float] = None,
        learning_rate: float = 0.0,
        batch_size: int = 1
    ):
        """Log training step metrics with comprehensive monitoring"""
        metrics = metrics or {}
        
        # Update metrics
        self.metrics.add_metric('loss', loss, step)
        self.metrics.add_metric('learning_rate', learning_rate, step)
        
        for name, value in metrics.items():
            self.metrics.add_metric(name, value, step)
        
        # Track counters
        self.metrics.increment_counter('batches_processed')
        self.metrics.increment_counter('images_processed', batch_size)
        
        # Calculate step time
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.metrics.add_metric('step_time', step_time, step)
        self.last_step_time = current_time
        
        # Check for issues and send alerts
        if self.alerts:
            self._check_training_health(step, loss, step_time)
        
        # Update training state
        if self.last_loss is not None and abs(loss - self.last_loss) < 1e-7:
            self.steps_without_improvement += 1
        else:
            self.steps_without_improvement = 0
        self.last_loss = loss
        
        # Log to visualization backends
        self._log_to_backends(step, {
            'train/loss': loss,
            'train/learning_rate': learning_rate,
            'train/step_time': step_time,
            **{f'train/{k}': v for k, v in metrics.items()}
        })
        
        # System metrics
        if self.config.track_system_metrics and step % 10 == 0:
            sys_metrics = self.system_monitor.get_latest_metrics()
            if sys_metrics:
                self._log_system_metrics(sys_metrics, step)
        
        # Update profiler
        if self.profiler:
            try:
                self.profiler.step()
                self.profiler_step += 1
                
                # Start/stop profiling at intervals
                if self.profiler_step == self.config.profile_interval_steps:
                    self.profiler.start()
                elif self.profiler_step == self.config.profile_interval_steps + self.config.profile_duration_steps:
                    self.profiler.stop()
                    self.profiler_step = 0
            except RuntimeError as e:
                logger.warning(f"Profiler error (may be already stopped): {e}")
                self.profiler = None  # Disable profiler on error
            
    def log_validation(
        self,
        step: int,
        metrics: Dict[str, float]
    ):
        """Log validation metrics"""
        for name, value in metrics.items():
            self.metrics.add_metric(f'val_{name}', value, step)
        
        # Track best model
        if 'loss' in metrics and metrics['loss'] < self.best_val_metric:
            self.best_val_metric = metrics['loss']
            if self.alerts:
                self.alerts.send_alert(
                    "New Best Model",
                    f"Validation loss improved to {metrics['loss']:.4f}",
                    severity="success"
                )
        
        # Log to backends
        self._log_to_backends(step, {f'val/{k}': v for k, v in metrics.items()})
    
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
        epoch_metrics = {
            **{f'epoch/train_{k}': v for k, v in train_metrics.items()},
            **{f'epoch/val_{k}': v for k, v in val_metrics.items()},
            'epoch/duration': duration
        }
        self._log_to_backends(epoch, epoch_metrics, use_epoch=True)
        
        # Save metrics checkpoint
        if self.config.checkpoint_metrics:
            self._save_metrics_checkpoint(epoch)
    
    def log_images(
        self,
        step: int,
        images: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        tag_names: List[str],
        num_images: int = 4,
        prefix: str = "val"
    ):
        """Log example images with predictions"""
        try:
            # Ensure we're on CPU and in the right format
            images = images.cpu()
            predictions = predictions.cpu()
            targets = targets.cpu()
            
            # Select random subset
            batch_size = min(len(images), num_images)
            indices = torch.randperm(len(images))[:batch_size]
            
            # Log to TensorBoard
            if self.writer:
                for i, idx in enumerate(indices):
                    img = images[idx]
                    pred = predictions[idx]
                    target = targets[idx]
                    
                    # Ensure image is in correct format (C, H, W) with values in [0, 1]
                    if img.dim() == 2:
                        img = img.unsqueeze(0)
                    if img.max() > 1:
                        img = img / 255.0
                    
                    # Get top predictions
                    top_k = min(10, len(pred))
                    pred_scores, pred_indices = torch.topk(pred, top_k)
                    
                    # Create text caption
                    pred_tags = [f"{tag_names[j.item()]}: {pred_scores[k].item():.2f}" 
                                for k, j in enumerate(pred_indices) if j.item() < len(tag_names)]
                    
                    true_indices = torch.where(target > 0.5)[0][:top_k]
                    true_tags = [tag_names[j.item()] for j in true_indices if j.item() < len(tag_names)]
                    
                    caption = (
                        f"Predicted: {', '.join(pred_tags[:5])}\n"
                        f"True: {', '.join(true_tags[:5])}"
                    )
                    
                    # Log to tensorboard
                    self.writer.add_image(f'{prefix}/image_{i}', img, step)
                    self.writer.add_text(f'{prefix}/caption_{i}', caption, step)
            
            # Log to wandb
            if self.wandb_run and WANDB_AVAILABLE:
                wandb_images = []
                for i, idx in enumerate(indices[:min(4, len(indices))]):
                    img = images[idx]
                    
                    # Convert to numpy and proper shape (H, W, C)
                    if img.dim() == 3:
                        img_np = img.permute(1, 2, 0).numpy()
                    else:
                        img_np = img.numpy()
                    
                    # Normalize to [0, 1]
                    if img_np.max() > 1:
                        img_np = img_np / 255.0
                    
                    # Get predictions and targets
                    pred_indices = torch.topk(predictions[idx], 5)[1]
                    pred_tags = [tag_names[j.item()] for j in pred_indices if j.item() < len(tag_names)]
                    
                    true_indices = torch.where(targets[idx] > 0.5)[0][:5]
                    true_tags = [tag_names[j.item()] for j in true_indices if j.item() < len(tag_names)]
                    
                    wandb_images.append(wandb.Image(
                        img_np,
                        caption=f"Pred: {', '.join(pred_tags)} | True: {', '.join(true_tags)}"
                    ))
                
                wandb.log({f"{prefix}/images": wandb_images, "step": step})
                
        except Exception as e:
            logger.error(f"Failed to log images: {e}")
            if self.config.safe_mode:
                logger.debug(traceback.format_exc())
    
    def log_data_pipeline_stats(self, load_time: float, batch_size: int, augmentation_time: float = 0):
        """Log data pipeline statistics"""
        self.data_stats['load_times'].append(load_time)
        self.data_stats['batch_sizes'].append(batch_size)
        self.data_stats['augmentation_times'].append(augmentation_time)
        
        # Log aggregated stats periodically
        if len(self.data_stats['load_times']) >= self.config.data_pipeline_stats_interval:
            avg_load_time = np.mean(self.data_stats['load_times'])
            avg_batch_size = np.mean(self.data_stats['batch_sizes'])
            avg_aug_time = np.mean(self.data_stats['augmentation_times'])
            
            self.metrics.add_metric('data_load_time_avg', avg_load_time)
            self.metrics.add_metric('batch_size_avg', avg_batch_size)
            self.metrics.add_metric('augmentation_time_avg', avg_aug_time)
            
            logger.debug(f"Data pipeline stats - Load: {avg_load_time:.3f}s, "
                        f"Batch: {avg_batch_size:.1f}, Aug: {avg_aug_time:.3f}s")
    
    def _check_training_health(self, step: int, loss: float, step_time: float):
        """Check training health and send alerts if needed"""
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
        if self.steps_without_improvement > 100:
            minutes_stuck = self.steps_without_improvement * step_time / 60
            if minutes_stuck > self.config.alert_on_training_stuck_minutes:
                self.alerts.send_alert(
                    "Training Stuck",
                    f"No improvement for {minutes_stuck:.1f} minutes",
                    severity="warning"
                )
        
        # Slow training
        if step_time > 60:  # More than 1 minute per step
            self.alerts.send_alert(
                "Slow Training",
                f"Step time is {step_time:.1f}s at step {step}",
                severity="warning"
            )
    
    def _log_system_metrics(self, metrics: Dict[str, Any], step: int):
        """Log system metrics to backends"""
        system_metrics = {}
        
        # CPU metrics
        if 'cpu' in metrics:
            system_metrics['system/cpu_percent'] = metrics['cpu'].get('percent', 0)
            system_metrics['system/cpu_freq_mhz'] = metrics['cpu'].get('freq', 0)
        
        # Memory metrics  
        if 'memory' in metrics:
            system_metrics['system/memory_used_gb'] = metrics['memory'].get('used_gb', 0)
            system_metrics['system/memory_percent'] = metrics['memory'].get('percent', 0)
            
            # Check memory threshold
            if (self.alerts and 
                metrics['memory'].get('percent', 0) > self.config.alert_on_cpu_memory_threshold * 100):
                self.alerts.send_alert(
                    "High Memory Usage",
                    f"Memory usage is {metrics['memory']['percent']:.1f}%",
                    severity="warning"
                )
        
        # Disk metrics
        if 'disk' in metrics:
            system_metrics['system/disk_used_gb'] = metrics['disk'].get('used_gb', 0)
            system_metrics['system/disk_percent'] = metrics['disk'].get('percent', 0)
            
            # Check disk threshold
            if (self.alerts and 
                metrics['disk'].get('percent', 0) > self.config.alert_on_disk_space_threshold * 100):
                self.alerts.send_alert(
                    "Low Disk Space",
                    f"Disk usage is {metrics['disk']['percent']:.1f}%",
                    severity="critical"
                )
        
        # GPU metrics
        if 'gpu' in metrics and metrics['gpu']:
            for gpu in metrics['gpu']:
                gpu_id = gpu['id']
                system_metrics[f'system/gpu{gpu_id}_memory_gb'] = gpu.get('memory_used_gb', 0)
                system_metrics[f'system/gpu{gpu_id}_memory_percent'] = gpu.get('memory_percent', 0)
                system_metrics[f'system/gpu{gpu_id}_utilization'] = gpu.get('utilization', 0)
                system_metrics[f'system/gpu{gpu_id}_temperature'] = gpu.get('temperature', 0)
                
                # Check GPU memory threshold
                if (self.alerts and 
                    gpu.get('memory_percent', 0) > self.config.alert_on_gpu_memory_threshold * 100):
                    self.alerts.send_alert(
                        f"High GPU {gpu_id} Memory",
                        f"GPU {gpu_id} memory usage is {gpu['memory_percent']:.1f}%",
                        severity="warning"
                    )
                
                # Update Prometheus GPU metrics
                if self.metrics.prom_metrics and 'gauges' in self.metrics.prom_metrics:
                    if 'gpu_memory_used' in self.metrics.prom_metrics['gauges']:
                        try:
                            self.metrics.prom_metrics['gauges']['gpu_memory_used'].labels(
                                gpu_id=str(gpu_id)
                            ).set(gpu.get('memory_used_gb', 0))
                        except:
                            pass
        
        # Log to backends
        self._log_to_backends(step, system_metrics)
    
    def _log_to_backends(self, step: int, metrics: Dict[str, float], use_epoch: bool = False):
        """Log metrics to all configured backends"""
        # TensorBoard
        if self.writer:
            try:
                for name, value in metrics.items():
                    if use_epoch:
                        self.writer.add_scalar(name, value, step)
                    else:
                        self.writer.add_scalar(name, value, step)
            except Exception as e:
                logger.debug(f"Failed to log to TensorBoard: {e}")
        
        # Weights & Biases
        if self.wandb_run and WANDB_AVAILABLE:
            try:
                log_dict = dict(metrics)
                log_dict['step' if not use_epoch else 'epoch'] = step
                wandb.log(log_dict)
            except Exception as e:
                logger.debug(f"Failed to log to wandb: {e}")
    
    def _save_metrics_checkpoint(self, epoch: int):
        """Save metrics checkpoint to file"""
        try:
            checkpoint_dir = Path(self.config.log_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_file = checkpoint_dir / f"metrics_epoch_{epoch}.json"
            self.metrics.save_metrics(str(checkpoint_file))
            
            # Keep only last N checkpoints
            checkpoints = sorted(checkpoint_dir.glob("metrics_epoch_*.json"))
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    old_checkpoint.unlink()
                    
        except Exception as e:
            logger.error(f"Failed to save metrics checkpoint: {e}")
    
    @contextmanager
    def profile_section(self, name: str):
        """Context manager for profiling code sections"""
        self.metrics.start_timer(name)
        try:
            yield
        finally:
            duration = self.metrics.end_timer(name)
            logger.debug(f"Section '{name}' took {duration:.3f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        summary = {
            'duration_hours': (time.time() - self.start_time) / 3600,
            'batches_processed': self.metrics.counters.get('batches_processed', 0),
            'images_processed': self.metrics.counters.get('images_processed', 0),
            'best_val_metric': self.best_val_metric,
            'metrics': {}
        }
        
        # Add metric statistics
        for metric_name in ['loss', 'learning_rate', 'val_loss']:
            stats = self.metrics.get_metric_stats(metric_name)
            if stats:
                summary['metrics'][metric_name] = stats
        
        # Add timer statistics
        for timer_name in ['batch_time', 'data_load_time']:
            stats = self.metrics.get_timer_stats(timer_name)
            if stats:
                summary['metrics'][f'{timer_name}_stats'] = stats
        
        return summary
    
    def close(self):
        """Cleanup and close all resources"""
        # Prevent multiple cleanup calls
        if hasattr(self, '_closed') and self._closed:
            return
        self._closed = True
        
        logger.info("Closing training monitor...")
        
        # Stop system monitor
        if self.system_monitor:
            self.system_monitor.stop()
            
        # Close profiler
        if self.profiler:
            try:
                self.profiler.stop()
            except:
                pass
        
        # Save final metrics
        try:
            final_metrics_file = Path(self.config.log_dir) / "final_metrics.json"
            self.metrics.save_metrics(str(final_metrics_file))
            
            # Save summary
            summary_file = Path(self.config.log_dir) / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.get_summary(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save final metrics: {e}")
        
        # Close TensorBoard writer
        if self.writer:
            try:
                self.writer.close()
            except:
                pass
        
        # Close wandb run
        if self.wandb_run and WANDB_AVAILABLE:
            try:
                wandb.finish()
            except:
                pass
        
        logger.info("Training monitor closed")


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = MonitorConfig(
        log_level="INFO",
        use_tensorboard=True,
        use_wandb=False,
        track_gpu_metrics=True,
        enable_alerts=True,
        enable_profiling=False,
        enable_prometheus=False
    )
    
    # Create monitor
    monitor = TrainingMonitor(config)
    
    try:
        # Simulate training loop
        for step in range(100):
            # Simulate metrics
            loss = np.random.random() * 2
            metrics = {
                'accuracy': np.random.random(),
                'precision': np.random.random(),
                'recall': np.random.random()
            }
            
            # Log step
            monitor.log_step(
                step=step,
                loss=loss,
                metrics=metrics,
                learning_rate=0.001,
                batch_size=32
            )
            
            # Simulate validation every 10 steps
            if step % 10 == 0:
                val_metrics = {
                    'loss': np.random.random() * 2,
                    'accuracy': np.random.random()
                }
                monitor.log_validation(step, val_metrics)
            
            time.sleep(0.1)  # Simulate training time
            
    finally:
        # Always cleanup
        monitor.close()