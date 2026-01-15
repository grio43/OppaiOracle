"""Memory monitoring for training.

Tracks RAM usage and alerts when approaching limits to prevent OOM errors.
"""
import psutil
import logging
import os
import time
from typing import Dict


class MemoryMonitor:
    """Monitor system and process memory usage."""

    def __init__(self, warn_threshold_gb: float = None, critical_threshold_gb: float = None):
        """Initialize memory monitor.

        Args:
            warn_threshold_gb: System memory threshold for warning.
                Can be overridden via MEMORY_WARN_THRESHOLD_GB env var.
                Default: 90% of system memory
            critical_threshold_gb: System memory threshold for critical alert.
                Can be overridden via MEMORY_CRITICAL_THRESHOLD_GB env var.
                Default: 95% of system memory
        """
        # Get system memory for auto-scaling defaults
        total_memory_gb = psutil.virtual_memory().total / (1024**3)

        # Auto-scale defaults based on system memory (90% warn, 95% critical)
        default_warn = total_memory_gb * 0.90
        default_critical = total_memory_gb * 0.95

        # Allow env var override for different system configurations
        if warn_threshold_gb is None:
            warn_threshold_gb = float(os.environ.get("MEMORY_WARN_THRESHOLD_GB", str(default_warn)))
        if critical_threshold_gb is None:
            critical_threshold_gb = float(os.environ.get("MEMORY_CRITICAL_THRESHOLD_GB", str(default_critical)))

        # Validate thresholds
        if warn_threshold_gb <= 0:
            raise ValueError(f"warn_threshold_gb must be positive, got {warn_threshold_gb}")
        if critical_threshold_gb <= 0:
            raise ValueError(f"critical_threshold_gb must be positive, got {critical_threshold_gb}")
        if warn_threshold_gb >= critical_threshold_gb:
            logging.warning(
                f"warn_threshold_gb ({warn_threshold_gb:.1f}) >= critical_threshold_gb ({critical_threshold_gb:.1f}), "
                f"adjusting warn to 90% of critical"
            )
            warn_threshold_gb = critical_threshold_gb * 0.90

        self.warn_threshold_gb = warn_threshold_gb
        self.critical_threshold_gb = critical_threshold_gb
        self.total_memory_gb = total_memory_gb
        self.last_warning = 0
        self.warning_interval = 300  # Warn every 5 minutes max

        # Cache for expensive children process enumeration
        self._children_mem_cache = 0.0
        self._children_cache_time = 0.0
        self._children_cache_interval = 30.0  # Refresh children mem every 30 seconds

        logging.debug(
            f"MemoryMonitor initialized: system={total_memory_gb:.1f}GB, "
            f"warn={warn_threshold_gb:.1f}GB, critical={critical_threshold_gb:.1f}GB"
        )

    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage and alert if needed.

        Returns:
            Dictionary containing memory statistics:
                - system_used_gb: Total system memory used
                - system_available_gb: Available system memory
                - system_percent: Percentage of system memory used
                - process_gb: Memory used by main process
                - workers_gb: Memory used by child processes (DataLoader workers)
                - total_process_gb: Total memory used by process tree
        """
        # System memory
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        available_gb = mem.available / (1024**3)
        percent = mem.percent

        # Process memory
        process = psutil.Process(os.getpid())
        process_mem_gb = process.memory_info().rss / (1024**3)

        # Children memory (DataLoader workers, L2 writer, etc.)
        # Use cached value if recent enough to avoid expensive process tree traversal
        now = time.time()
        if now - self._children_cache_time > self._children_cache_interval:
            children_mem_gb = 0
            try:
                for child in process.children(recursive=True):
                    try:
                        children_mem_gb += child.memory_info().rss / (1024**3)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Child may have died between iteration and memory query
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self._children_mem_cache = children_mem_gb
            self._children_cache_time = now
        else:
            children_mem_gb = self._children_mem_cache

        total_process_gb = process_mem_gb + children_mem_gb

        # Alert if needed (reuse 'now' from cache check above)
        if used_gb > self.critical_threshold_gb:
            if now - self.last_warning > self.warning_interval:
                logging.error(
                    f"CRITICAL: System memory at {used_gb:.1f} GB / {mem.total / (1024**3):.1f} GB "
                    f"({percent:.1f}%) - approaching limit! "
                    f"Process tree: {total_process_gb:.1f} GB "
                    f"(main: {process_mem_gb:.1f} GB + workers: {children_mem_gb:.1f} GB)"
                )
                self.last_warning = now
        elif used_gb > self.warn_threshold_gb:
            if now - self.last_warning > self.warning_interval:
                logging.warning(
                    f"WARNING: System memory at {used_gb:.1f} GB / {mem.total / (1024**3):.1f} GB "
                    f"({percent:.1f}%) - getting high. "
                    f"Process tree: {total_process_gb:.1f} GB"
                )
                self.last_warning = now

        return {
            'system_used_gb': used_gb,
            'system_available_gb': available_gb,
            'system_percent': percent,
            'process_gb': process_mem_gb,
            'workers_gb': children_mem_gb,
            'total_process_gb': total_process_gb,
        }
