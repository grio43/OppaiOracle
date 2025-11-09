#!/usr/bin/env python3
"""
Monitoring & Logging System for Anime Image Tagger
Comprehensive monitoring for training, inference, and system resources
Enhanced with proper error handling, thread safety, and complete implementations
Primary monitoring module used by training and inference components
"""

import os
import sys
import json
import logging
import psutil
import time
import threading
import queue
import concurrent.futures
import re
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
import torchvision
import matplotlib.pyplot as plt
from utils.logging_sanitize import sanitize_metrics
import seaborn as sns
from tqdm import tqdm

# Load sensitive configuration values
try:
    from sensitive_config import ALERT_WEBHOOK_URL as _DEFAULT_WEBHOOK
except ImportError:  # pragma: no cover - fallback when file missing
    _DEFAULT_WEBHOOK = None
import os
from urllib.parse import urlparse


def _validate_webhook_url(url: Optional[str]) -> Optional[str]:
    """Validate and sanitize webhook URL.

    Returns:
        Validated URL or None if invalid
    """
    if not url:
        return None

    if not isinstance(url, str):
        logger.error("Webhook URL must be a string")
        return None

    # Remove whitespace
    url = url.strip()

    # Basic format validation
    if not url.startswith(('http://', 'https://')):
        logger.error("Webhook URL must start with http:// or https://")
        return None

    try:
        parsed = urlparse(url)
    except Exception as e:
        logger.error(f"Invalid webhook URL format: {e}")
        return None

    # Check hostname is present
    if not parsed.netloc:
        logger.error("Webhook URL missing hostname")
        return None

    # Whitelist known webhook services
    allowed_domains = [
        'discord.com',
        'discordapp.com',
        'slack.com',
        'hooks.slack.com',
        # Add other trusted webhook services as needed
    ]

    hostname = parsed.netloc.lower()
    # Remove port if present
    hostname = hostname.split(':')[0]

    # Check if domain or subdomain is allowed
    is_allowed = False
    for allowed_domain in allowed_domains:
        if hostname == allowed_domain or hostname.endswith('.' + allowed_domain):
            is_allowed = True
            break

    if not is_allowed:
        logger.error(f"Webhook URL domain '{hostname}' is not in allowed list: {allowed_domains}")
        return None

    # Ensure HTTPS for production
    if parsed.scheme != 'https':
        logger.warning(f"Webhook URL should use HTTPS, got {parsed.scheme}")

    return url


def _resolve_webhook_url(cfg_value: Optional[str]) -> Optional[str]:
    """Resolve and validate webhook URL from multiple sources."""
    # Try sources in order of priority
    url = cfg_value or os.getenv("OPPAI_ALERT_WEBHOOK") or _DEFAULT_WEBHOOK

    # Validate before returning
    return _validate_webhook_url(url)

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

from Configuration_System import MonitorConfig

logger = logging.getLogger(__name__)


# Discord Webhook constants
MAX_CONTENT = 2000
MAX_TITLE = 256
MAX_DESC = 4096
MAX_FIELDS = 25
MAX_FIELD_NAME = 256
MAX_FIELD_VALUE = 1024

def trim(s: str, n: int) -> str:
    """Trim a string to a max length, adding an ellipsis if truncated."""
    if s is None:
        return None
    s = str(s)
    return s if len(s) <= n else s[:n-1] + '…'


class ImageLogger:
    def __init__(self, writer=None, use_matplotlib: bool = False):
        self.writer = writer
        self.use_matplotlib = use_matplotlib
        self.in_training_loop = True

    def log_images(self, images, predictions=None, labels=None, step=None):
        from torchvision.utils import make_grid

        grid = make_grid(images, nrow=8, normalize=True)
        if self.writer:
            self.writer.add_image('images/batch', grid, step)

        if self.use_matplotlib and not self.in_training_loop:
            self._create_composite_plot(images, predictions, labels)

    def _create_composite_plot(self, images, predictions, labels):
        """Create composite plots offline or during validation only"""
        pass


class GradientHistogramLogger:
    def __init__(self, config, writer=None):
        self.writer = writer
        self.enabled = (
            getattr(config, 'log_histograms', False)
            or getattr(config, 'log_param_histograms', False)
            or getattr(config, 'log_grad_histograms', False)
        )
        self.log_every_n_steps = getattr(
            config,
            'histogram_log_interval',
            max(
                getattr(config, 'param_hist_interval_steps', 500),
                getattr(config, 'grad_hist_interval_steps', 500),
            ),
        )
        self.param_whitelist = getattr(config, 'histogram_params', None)

    def log_histograms(self, model, step):
        """Conditionally log parameter/gradient histograms"""
        if not self.enabled or self.writer is None:
            return

        if step % self.log_every_n_steps != 0:
            return

        for name, param in model.named_parameters():
            if self.param_whitelist and name not in self.param_whitelist:
                continue

            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)
                self.writer.add_histogram(f'parameters/{name}', param, step)

    @staticmethod
    def warn_performance():
        """Warning for users about performance impact"""
        print(
            "WARNING: Histogram logging enabled. This may significantly impact "
            "training performance on large models. Consider disabling or reducing "
            "frequency via 'histogram_log_interval' config."
        )


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
        webhook_url = _resolve_webhook_url(self.config.alert_webhook_url)
        if webhook_url:
            self._send_webhook_alert(alert, webhook_url)
        
        # Send to console with color
        self._print_colored_alert(alert)
    
    def _send_webhook_alert(self, alert: dict, webhook_url: str):
        """Spawns a background thread to send a webhook alert to avoid blocking."""
        thread = threading.Thread(
            target=self._execute_webhook_in_thread,
            args=(alert, webhook_url),
            daemon=True
        )
        thread.start()

    def _sanitize_webhook_content(self, content: str, max_length: int) -> str:
        """Sanitize content for webhook transmission.

        Args:
            content: Raw content to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized and truncated content
        """
        if content is None:
            return ""

        content = str(content)

        # Remove potentially sensitive information
        # Redact file paths
        content = re.sub(r'[A-Za-z]:\\[^\\:\*\?"<>\|]+', '[REDACTED_PATH]', content)
        content = re.sub(r'/[/\w\-\.]+/[/\w\-\.]+', '[REDACTED_PATH]', content)

        # Redact potential API keys or tokens
        content = re.sub(r'[A-Za-z0-9]{32,}', '[REDACTED_TOKEN]', content)

        # Remove control characters that could break JSON
        content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\t')

        # Escape special characters for JSON
        content = content.replace('\\', '\\\\')
        content = content.replace('"', '\\"')
        content = content.replace('\n', '\\n')
        content = content.replace('\r', '\\r')
        content = content.replace('\t', '\\t')

        # Truncate to max length
        if len(content) > max_length:
            content = content[:max_length - 1] + '…'

        return content

    def _execute_webhook_in_thread(self, alert: dict, webhook_url: str):
        """Send alert to webhook (Discord optimized) in a thread."""
        try:
            import requests

            # Sanitize all content
            title = f"[{alert['severity'].upper()}] {alert['title']}"
            title = self._sanitize_webhook_content(title, MAX_TITLE)

            description = self._sanitize_webhook_content(alert['message'], MAX_DESC)

            timestamp = alert['timestamp']
            if not isinstance(timestamp, str):
                timestamp = str(timestamp)

            fields = [
                {"name": "Time", "value": self._sanitize_webhook_content(timestamp, MAX_FIELD_VALUE), "inline": True},
                {"name": "Occurrence", "value": str(alert['count']), "inline": True},
            ]

            embed = {
                "title": title,
                "description": description,
                "color": int(self._get_severity_color(alert['severity']).lstrip('#'), 16),
                "fields": [
                    {
                        "name": self._sanitize_webhook_content(f["name"], MAX_FIELD_NAME),
                        "value": self._sanitize_webhook_content(f["value"], MAX_FIELD_VALUE),
                        "inline": bool(f.get("inline", False)),
                    }
                    for f in fields
                ][:MAX_FIELDS],
                "timestamp": datetime.now().isoformat()
            }

            payload = {
                "content": self._sanitize_webhook_content(f"Training Alert: {alert['title']}", MAX_CONTENT),
                "embeds": [embed],
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send webhook alert for '{alert['title']}': {e}")
    
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
                # Add queue metrics if available
                data['queue_metrics'] = self.get_queue_metrics()

                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
                logger.info(f"Metrics saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save metrics: {e}")

    def get_queue_metrics(self) -> Dict[str, Any]:
        """Get logging queue metrics if available."""
        metrics = {}
        try:
            # Try to get metrics from the logging queue if it's a BoundedLevelAwareQueue
            import logging
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if hasattr(handler, 'queue') and hasattr(handler.queue, 'get_drop_stats'):
                    drop_stats = handler.queue.get_drop_stats()
                    metrics['log_queue_drops'] = drop_stats
                    metrics['log_queue_size'] = len(handler.queue.queue)
                    break
        except Exception as e:
            logger.debug(f"Could not get queue metrics: {e}")
        
        # Add host pinned memory metrics if available
        metrics['host_pinned_bytes'] = self.counters.get('host_pinned_bytes', 0)
        
        return metrics

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
        # Track GPU monitoring failures to disable after persistent errors
        self.gpu_failure_count = 0
        self.max_gpu_failures = 5

        # Use ThreadPoolExecutor for timeout handling
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="SystemMetrics")        
        
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

            # 1. Signal shutdown first - this will cause loop to exit
            self._shutdown_event.set()
            self.running = False
            thread_to_join = self.thread
            self.thread = None

        # 2. Wait for thread to exit (outside lock to avoid deadlock)
        if thread_to_join and thread_to_join.is_alive():
            thread_to_join.join(timeout=5)  # Increased timeout
            if thread_to_join.is_alive():
                logger.warning("System monitor thread did not stop within timeout")

        # 3. Shutdown executor with proper error handling
        if hasattr(self, '_executor'):
            try:
                # Cancel any pending futures
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 doesn't have cancel_futures parameter
                self._executor.shutdown(wait=False)
            except Exception as e:
                logger.debug(f"Error shutting down executor: {e}")

        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop with error handling"""
        while self.running and not self._shutdown_event.is_set():
            try:
                # Check shutdown before submitting
                if self._shutdown_event.is_set():
                    break

                # Double-check executor is still available
                if not hasattr(self, '_executor'):
                    logger.warning("Executor not available, exiting monitor loop")
                    break

                # Use executor with timeout for metric collection
                try:
                    future = self._executor.submit(self._collect_metrics_safe)
                except RuntimeError:
                    # Executor shut down, exit cleanly
                    logger.debug("Executor unavailable, exiting monitor loop")
                    break

                try:
                    metrics = future.result(timeout=self.config.system_metrics_interval * 0.8)
                except concurrent.futures.TimeoutError:
                    logger.warning("Metric collection timed out, skipping this cycle")
                    future.cancel()
                    continue
                except Exception as e:
                    logger.error(f"Error in metric collection: {e}")
                    continue
                
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
            
            # Use Event.wait() for interruptible sleeping
            if self._shutdown_event.wait(timeout=self.config.system_metrics_interval):
                break  # Shutdown event was set
    
    def _collect_metrics_safe(self) -> Dict[str, Any]:
        """Wrapper for metric collection with frequent shutdown checks"""
        if self._shutdown_event.is_set():
            return {}
        return self._collect_metrics()
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics with error handling"""
        # Early return if shutdown requested
        if self._shutdown_event.is_set():
            return {}

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
            # Use shorter interval to be more responsive to shutdown
            if self._shutdown_event.is_set():
                return metrics
            metrics['cpu']['percent'] = psutil.cpu_percent(interval=0.01)
  
            metrics['cpu']['count'] = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            metrics['cpu']['freq'] = cpu_freq.current if cpu_freq else 0
            
            # Per-CPU metrics
            if not self._shutdown_event.is_set():
                metrics['cpu']['per_cpu'] = psutil.cpu_percent(interval=0.01, percpu=True)
        except Exception as e:
            logger.debug(f"Failed to collect CPU metrics: {e}")

        # Check shutdown more frequently
        if self._shutdown_event.is_set():            
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
            if self._shutdown_event.is_set():
                return metrics
                        
        
        try:
            # Disk metrics
            if self.config.track_disk_io:
                if self._shutdown_event.is_set():
                    return metrics                
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
        
        if self._shutdown_event.is_set():
            return metrics
           
        try:
            # GPU metrics (with fresh query)
            if self.gpu_available and GPUTIL_AVAILABLE:
                try:
                    if self._shutdown_event.is_set():
                        return metrics                    
                    # Double-check GPUtil is still available and working
                    gpus = GPUtil.getGPUs()
                    if gpus is not None:  # Additional safety check
                        for gpu in gpus:
                            if gpu is not None:  # Check each GPU object
                                try:
                                    # Safely get all values with proper null handling
                                    mem_total = getattr(gpu, 'memoryTotal', None)
                                    mem_used = getattr(gpu, 'memoryUsed', None)
                                    mem_free = getattr(gpu, 'memoryFree', None)
                                    gpu_load = getattr(gpu, 'load', None)

                                    # Ensure numeric values with proper None handling
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
                                        'utilization': gpu_load * 100,  # Convert to percentage
                                        'temperature': getattr(gpu, 'temperature', 0) or 0
                                    }
                                    metrics['gpu'].append(gpu_metrics)
                                    # Reset failure count on success
                                    self.gpu_failure_count = 0
                                except Exception as gpu_err:
                                    logger.debug(f"Failed to process GPU {getattr(gpu, 'id', '?')}: {gpu_err}")
                except (AttributeError, ImportError, RuntimeError, Exception) as e:
                    # Increment failure count and possibly disable GPU monitoring
                    self.gpu_failure_count += 1
                    logger.warning(
                        f"GPU monitoring error (failure {self.gpu_failure_count}/{self.max_gpu_failures}): {e}"
                    )
                    # Disable GPU monitoring after persistent failures
                    if self.gpu_failure_count >= self.max_gpu_failures:
                        self.gpu_available = False
                        logger.error(
                            f"GPU monitoring disabled after {self.max_gpu_failures} failures"
                        )
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")
                    
        try:
            # Network metrics
            if self.config.track_network_io:
                if self._shutdown_event.is_set():
                    return metrics                
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
    
    def __init__(self, config, *args, **kwargs):
        super().__init__() if hasattr(super(), "__init__") else None
        self.config = config
        self.writer = None
        # Preserve existing logger if present
        self.logger = getattr(self, "logger", None)

        # Core state
        self.start_time = time.time()
        self.metrics = ThreadSafeMetricsTracker(config)
        self.last_step_time = self.start_time
        self.last_loss = None
        self.steps_without_improvement = 0
        self.best_val_metric = float('inf')
        self._graph_logged = False

        # Determine primary rank with proper error handling
        try:
            # Check if distributed is available (compiled with distributed support)
            if not dist.is_available():
                self.is_primary = True
                logger.debug("Distributed training not available, assuming primary rank")
            elif not dist.is_initialized():
                # Distributed available but not initialized - assume primary
                # This is the common case for single-GPU or non-distributed training
                self.is_primary = True
                logger.debug("Distributed training not initialized, assuming primary rank")
            else:
                # Distributed is initialized, get actual rank
                rank = dist.get_rank()
                self.is_primary = (rank == 0)
                logger.debug(f"Distributed training initialized, rank={rank}, is_primary={self.is_primary}")
        except RuntimeError as e:
            # Distributed may raise RuntimeError if not properly set up
            logger.warning(f"Error checking distributed rank: {e}, assuming primary")
            self.is_primary = True
        except ValueError as e:
            # Invalid rank or world size
            logger.warning(f"Invalid distributed configuration: {e}, assuming primary")
            self.is_primary = True
        except Exception as e:
            # Unexpected error - log it but continue
            logger.error(f"Unexpected error determining rank: {e}, assuming primary", exc_info=True)
            self.is_primary = True

        # Optional components
        self.alerts = AlertSystem(config) if config.enable_alerts else None
        # Run system metrics on primary rank only
        self.system_monitor = SystemMonitor(config) if (config.track_system_metrics and self.is_primary) else None
        if self.system_monitor:
            try:
                self.system_monitor.start()
            except Exception as e:
                logger.warning(f"System monitor failed to start: {e}")

        # Profiling and logging helpers
        self.profiler = None
        self.profiler_step = 0
        self.data_stats = {
            'load_times': [],
            'batch_sizes': [],
            'augmentation_times': []
        }
        self.last_aug_log_step = 0
        self.wandb_run = None

        use_tb = bool(getattr(self.config, "use_tensorboard", False))
        if use_tb and self.is_primary:
            from datetime import datetime
            import socket
            from pathlib import Path
            from torch.utils.tensorboard import SummaryWriter

            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            host = socket.gethostname().split('.')[0]
            seed = getattr(self.config, "seed", None)
            suffix = f"{run_name}-{host}" + (f"-s{seed}" if seed is not None else "")

            tb_root = Path(getattr(self.config, "tensorboard_dir", "runs"))
            tb_dir = tb_root / suffix
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_dir = str(tb_dir)

            # Tighter flush and bounded queue for durability and memory
            self.writer = SummaryWriter(log_dir=self._tb_dir, flush_secs=30, max_queue=1000)
            if self.logger: self.logger.info(f"TensorBoard logging to {self._tb_dir}")

            # Optional: small curated dashboard panels (best-effort)
            try:
                self.writer.add_custom_scalars({
                    "Loss": {"Train vs Val": ["Multiline", ["train/loss", "val/loss"]]},
                    "Accuracy": {"Acc": ["Multiline", ["train/acc", "val/acc"]]},
                    "LR": {"Schedule": ["Multiline", ["lr"]]}
                })
            except Exception:
                pass
        # Initialize auxiliary loggers
        self.image_logger = ImageLogger(writer=self.writer, use_matplotlib=getattr(self.config, 'use_matplotlib', False))
        self.grad_hist_logger = GradientHistogramLogger(self.config, writer=self.writer)
        if self.grad_hist_logger.enabled:
            self.grad_hist_logger.warn_performance()

        # Final setup
        self._setup_logging()
        if self.config.enable_profiling:
            self._setup_profiler()
        self._register_cleanup()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        module_logger = logging.getLogger(__name__)
        # Always console
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(self.config.log_format))
        module_logger.addHandler(ch)
        # Only rank 0 writes files
        if getattr(self, "is_primary", True) and self.config.log_to_file:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "training_{:%Y%m%d_%H%M%S}.log".format(datetime.now()))
            fh.setFormatter(logging.Formatter(self.config.log_format))
            module_logger.addHandler(fh)
        module_logger.setLevel(getattr(logging, self.config.log_level.upper()))
    
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
            if getattr(self, 'system_monitor', None):
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
            if getattr(self, 'system_monitor', None):
                self.system_monitor.stop()
            if hasattr(self, 'writer') and self.writer:
                try:
                    self.writer.close()
                except:
                    pass
                
        # Register atexit handler
        atexit.register(cleanup)

    def log_model_graph(self, model: torch.nn.Module, example_images: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """Write the model graph to TensorBoard once (safe in DDP)."""
        if not hasattr(self, "_graph_logged"):
            # Older monitor instances may not define this flag
            self._graph_logged = False

        if not self.writer or self._graph_logged:
            return
        try:
            class _GraphWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x, padding_mask=None):
                    out = self.m(x, padding_mask=padding_mask)
                    return out['tag_logits'] if isinstance(out, dict) else out

            wrapper = _GraphWrapper(model).eval()
            with torch.no_grad():
                if padding_mask is not None:
                    self.writer.add_graph(wrapper, (example_images, padding_mask))
                else:
                    self.writer.add_graph(wrapper, (example_images,))
            self._graph_logged = True
            logger.info("Wrote model graph to TensorBoard")
        except Exception as e:
            logger.debug(f"Skipping add_graph: {e}")

    def log_param_and_grad_histograms(self, model, step: int):
        if getattr(self, "grad_hist_logger", None) is None:
            return

        self.grad_hist_logger.writer = self.writer
        self.grad_hist_logger.log_histograms(model, step)

    def flush(self):
        if getattr(self, "writer", None):
            try:
                self.writer.flush()
            except Exception:
                pass
    
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

        # Ensure metrics tracker exists (older versions may lack it)
        if not hasattr(self, "metrics"):
            self.metrics = ThreadSafeMetricsTracker(self.config)

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

    def log_hyperparameters(self, hparams: dict, metrics: dict):
        if getattr(self, "writer", None) is None:
            return
        try:
            hp_clean: Dict[str, Any] = {}

            def _flatten(prefix: str, obj: Any):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        new_key = f"{prefix}.{k}" if prefix else k
                        _flatten(new_key, v)
                elif isinstance(obj, (str, int, float, bool)):
                    hp_clean[prefix] = obj
                else:
                    hp_clean[prefix] = str(obj)

            _flatten("", hparams)
            self.writer.add_hparams(hp_clean, metrics)
        except Exception as e:
            if getattr(self, "logger", None):
                self.logger.warning(f"TensorBoard add_hparams failed: {e}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Logs a single scalar value to the configured backends."""
        if self.writer:
            try:
                self.writer.add_scalar(tag, value, step)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"TensorBoard add_scalar failed for tag={tag}: {e}")

    def log_images(
        self,
        images: torch.Tensor,
        step: int,
        prefix: str = "images",
        num_images: int = 4,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        tag_names: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        """Unified image logging entry point"""
        if getattr(self, "writer", None) is None or getattr(self, "image_logger", None) is None:
            return

        self.image_logger.writer = self.writer
        self.image_logger.in_training_loop = True
        self.image_logger.log_images(
            images[:num_images], predictions=predictions, labels=targets, step=step
        )

    def log_predictions(
        self,
        *,
        step: int,
        images: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        tag_names: List[str],
        prefix: str = "val",
        max_images: int = 4,
        topk: int = 15,
    ):
        """Log per-sample images and top-k predictions/targets as TensorBoard entries."""
        if getattr(self, "writer", None) is None:
            return

        images = images.detach().cpu()
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        num_samples = min(images.shape[0], max_images)
        for i in range(num_samples):
            img = images[i]
            if torch.is_floating_point(img):
                img = img.clamp(0, 1)
                img = (img * 255).round().to(torch.uint8)
            else:
                img = img.to(torch.uint8)

            if img.dim() == 3:
                if img.shape[0] in [1, 3, 4]:
                    img = img.permute(1, 2, 0)
                else:
                    img = img.squeeze()
            elif img.dim() == 2:
                pass  # already HxW

            img_np = img.numpy()
            self.writer.add_image(
                f"{prefix}/sample_{i}/image",
                img_np,
                step,
                dataformats="HWC",
            )

            probs = predictions[i]
            tgt = targets[i]
            k = min(topk, probs.numel())
            topk_indices = torch.topk(probs, k=k).indices.tolist()

            lines = ["| tag | prob | target |", "| --- | --- | --- |"]
            for idx in topk_indices:
                tag = tag_names[idx] if idx < len(tag_names) else str(idx)
                prob = probs[idx].item()
                target_val = tgt[idx].item()
                lines.append(f"| {tag} | {prob:.4f} | {int(target_val)} |")

            markdown = "\n".join(lines)
            self.writer.add_text(
                f"{prefix}/sample_{i}/topk",
                markdown,
                step,
            )

    def log_composite_grid(
        self,
        *,
        step: int,
        images: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        tag_names: List[str],
        prefix: str = "val",
        max_images: int = 4,
        topk: int = 15,
        dpi: int = 220,
    ):
        """Optional helper to log a composite matplotlib panel."""
        if getattr(self, "writer", None) is None:
            return

        images = images.detach().cpu()
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        num_samples = min(images.shape[0], max_images)
        for i in range(num_samples):
            img = images[i]
            if torch.is_floating_point(img):
                img = img.clamp(0, 1)
                img = img.permute(1, 2, 0).numpy()
            else:
                img = img.permute(1, 2, 0).numpy()

            fig, ax = plt.subplots(figsize=(10, 12), dpi=dpi)
            ax.imshow(img)
            ax.axis("off")

            probs = predictions[i]
            tgt = targets[i]
            k = min(topk, probs.numel())
            topk_indices = torch.topk(probs, k=k).indices.tolist()
            lines = [f"{tag_names[idx]}: {probs[idx]:.3f} ({int(tgt[idx])})" for idx in topk_indices]
            fig.text(0.01, 0.01, "\n".join(lines), fontsize=8, va="bottom")

            fig.tight_layout()
            fig.canvas.draw()
            plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            self.writer.add_image(
                f"{prefix}/sample_{i}/composite",
                plot_img,
                step,
                dataformats="HWC",
            )
    
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
    
    def log_augmentations(self, step: int, stats: Any):
        """Log augmentation statistics to TensorBoard.
        
        Args:
            step: Current training step
            stats: AugmentationStats object or dict with augmentation statistics
        """
        # Only log at intervals to avoid overwhelming TensorBoard
        if step - self.last_aug_log_step < self.config.augmentation_stats_interval:
            return
            
        self.last_aug_log_step = step
        
        # Extract stats (handle both object and dict)
        if hasattr(stats, '__dict__'):
            stats = stats.__dict__
        
        # Log flip statistics
        flip_total = stats.get('flip_total', 0)
        if flip_total > 0:
            flip_rate = stats.get('flip_safe', 0) / flip_total
            self._log_to_backends(step, {
                'data/flip/total': flip_total,
                'data/flip/safe': stats.get('flip_safe', 0),
                'data/flip/skipped_text': stats.get('flip_skipped_text', 0),
                'data/flip/skipped_unmapped': stats.get('flip_skipped_unmapped', 0),
                'data/flip/blocked_safety': stats.get('flip_blocked_safety', 0),
                'data/flip/rate': flip_rate,
            })
        
        # Log color jitter statistics
        jitter_applied = stats.get('jitter_applied', 0)
        image_count = stats.get('image_count', 1)
        if jitter_applied > 0:
            jitter_rate = jitter_applied / image_count
            self._log_to_backends(step, {
                'data/color_jitter/applied_rate': jitter_rate,
            })
            
            # Log histograms if enabled
            if self.config.log_augmentation_histograms and self.writer:
                brightness_factors = stats.get('jitter_brightness_factors', [])
                if brightness_factors:
                    self.writer.add_histogram('data/color_jitter/brightness_factor', 
                                             np.array(brightness_factors), step)
                    
                contrast_factors = stats.get('jitter_contrast_factors', [])
                if contrast_factors:
                    self.writer.add_histogram('data/color_jitter/contrast_factor',
                                             np.array(contrast_factors), step)
        
        # Log crop statistics
        crop_applied = stats.get('crop_applied', 0)
        if crop_applied > 0:
            crop_rate = crop_applied / image_count
            self._log_to_backends(step, {
                'data/crop/applied_rate': crop_rate,
            })
            
            if self.config.log_augmentation_histograms and self.writer:
                crop_scales = stats.get('crop_scales', [])
                if crop_scales:
                    self.writer.add_histogram('data/crop/scale', np.array(crop_scales), step)
                    
                crop_aspects = stats.get('crop_aspects', [])
                if crop_aspects:
                    self.writer.add_histogram('data/crop/aspect_ratio', np.array(crop_aspects), step)
        
        # Log resize/letterbox statistics
        resize_scales = stats.get('resize_scales', [])
        if resize_scales and self.config.log_augmentation_histograms and self.writer:
            self.writer.add_histogram('data/resize/scale_r', np.array(resize_scales), step)
            
        resize_pad_pixels = stats.get('resize_pad_pixels', [])
        if resize_pad_pixels and self.config.log_augmentation_histograms and self.writer:
            self.writer.add_histogram('data/resize/pad_total_px', np.array(resize_pad_pixels), step)
        
        # Log timing metrics
        self._log_to_backends(step, {
            'data/aug/batch_time_ms': 0,  # Would need to track this
            'data/aug/images_per_sec': image_count / max(1, self.config.augmentation_stats_interval),
        })
    
    def log_augmentation_images(self, step: int, original_images: torch.Tensor, 
                               augmented_images: torch.Tensor, num_images: int = 4):
        """Log before/after augmentation image grids.
        
        Args:
            step: Current training step  
            original_images: Batch of original images
            augmented_images: Batch of augmented images
            num_images: Number of images to log
        """
        if not self.config.log_augmentation_images:
            return
            
        if step % self.config.augmentation_image_interval != 0:
            return
            
        try:
            # Select subset of images
            num_images = min(num_images, len(original_images))
            indices = torch.randperm(len(original_images))[:num_images]
            
            if self.writer:
                # Create grid of original images
                orig_grid = torchvision.utils.make_grid(
                    original_images[indices].cpu(),
                    nrow=num_images,
                    normalize=True
                )
                self.writer.add_image('data/preview/original', orig_grid, step)
                
                # Create grid of augmented images
                aug_grid = torchvision.utils.make_grid(
                    augmented_images[indices].cpu(),
                    nrow=num_images,
                    normalize=True
                )
                self.writer.add_image('data/preview/augmented', aug_grid, step)
                
        except Exception as e:
            logger.error(f"Failed to log augmentation images: {e}")
    
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
    
    def _log_to_backends(self, step: int, metrics: dict, use_epoch: bool = False):
        if getattr(self, "writer", None):
            safe_metrics = sanitize_metrics(metrics)
            for name, value in safe_metrics.items():
                tag = f"{name}{'/epoch' if use_epoch else ''}"
                try:
                    self.writer.add_scalar(tag, value, step)
                except Exception as e:
                    if getattr(self, "logger", None):
                        self.logger.warning(f"TensorBoard add_scalar failed for tag={tag}: {e}")

            if self.wandb_run is not None:
                try:
                    # Note: include step so W&B aligns correctly
                    self.wandb_run.log({**safe_metrics, "step": step})
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"W&B log failed: {e}")
    
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

        # Add queue metrics
        queue_metrics = self.metrics.get_queue_metrics()
        if queue_metrics:
            summary['queue_metrics'] = queue_metrics
        
        # Add prefetch controller metrics if available
        try:
            if hasattr(self, 'prefetch_controller'):
                summary['prefetch_metrics'] = self.prefetch_controller.get_metrics()
        except:
            pass

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
        # Check if already closed
        if getattr(self, '_closed', False):
            return

        logger.info("Closing training monitor...")
        cleanup_errors = []

        # Stop system monitor first (includes executor shutdown)
        system_monitor = getattr(self, 'system_monitor', None)
        if system_monitor:
            try:
                system_monitor.stop()
            except Exception as e:
                logger.error(f"Error stopping system monitor: {e}")
                cleanup_errors.append(('system_monitor', e))

        # Close profiler
        profiler = getattr(self, 'profiler', None)
        if profiler:
            try:
                profiler.stop()
            except Exception as e:
                logger.debug(f"Error stopping profiler: {e}")
                cleanup_errors.append(('profiler', e))

        # Save final metrics
        config = getattr(self, 'config', None)
        metrics = getattr(self, 'metrics', None)
        if config and metrics:
            try:
                final_metrics_file = Path(config.log_dir) / "final_metrics.json"
                metrics.save_metrics(str(final_metrics_file))

                # Save summary
                summary_file = Path(config.log_dir) / "training_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(self.get_summary(), f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to save final metrics: {e}")
                cleanup_errors.append(('save_metrics', e))

        # Close TensorBoard writer
        writer = getattr(self, 'writer', None)
        if writer:
            try:
                writer.flush()
                writer.close()
            except Exception as e:
                logger.debug(f"Error closing TensorBoard writer: {e}")
                cleanup_errors.append(('tensorboard', e))

        # Close wandb run
        wandb_run = getattr(self, 'wandb_run', None)
        if wandb_run and WANDB_AVAILABLE:
            try:
                wandb.finish()
            except Exception as e:
                logger.debug(f"Error finishing wandb: {e}")
                cleanup_errors.append(('wandb', e))

        # Mark as closed only after attempting all cleanup
        self._closed = True

        if cleanup_errors:
            logger.warning(f"Training monitor closed with {len(cleanup_errors)} errors: {cleanup_errors}")
        else:
            logger.info("Training monitor closed successfully")


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
