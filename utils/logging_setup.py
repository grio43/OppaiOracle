import logging
import logging.handlers
import sys
import os
import json
import uuid
import subprocess
import socket
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any
from HDF5_loader import CompressingRotatingFileHandler

# --- Globals for context ---
_GLOBAL_CONTEXT = {
    "run_id": str(uuid.uuid4()),
    "rank": 0,
    "world_size": 1,
    "host": socket.gethostname(),
    "git_rev": "unknown",
    "dirty": False,
}

def _get_git_info():
    """Fetches git revision and dirty status."""
    try:
        git_rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.PIPE).strip().decode()
        dirty_status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.PIPE).strip()
        return git_rev, bool(dirty_status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False

_GLOBAL_CONTEXT["git_rev"], _GLOBAL_CONTEXT["dirty"] = _get_git_info()


class ContextFilter(logging.Filter):
    """Injects contextual information into the log record."""
    def filter(self, record):
        record.run_id = _GLOBAL_CONTEXT["run_id"]
        record.rank = _GLOBAL_CONTEXT["rank"]
        record.world_size = _GLOBAL_CONTEXT["world_size"]
        record.host = _GLOBAL_CONTEXT["host"]
        record.git_rev = _GLOBAL_CONTEXT["git_rev"]
        record.dirty = _GLOBAL_CONTEXT["dirty"]
        return True

class JsonFormatter(logging.Formatter):
    """Formats log records as JSON."""
    def format(self, record):
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "run_id": getattr(record, 'run_id', 'unknown'),
            "rank": getattr(record, 'rank', 0),
            "world_size": getattr(record, 'world_size', 1),
            "host": getattr(record, 'host', 'unknown'),
            "git_rev": getattr(record, 'git_rev', 'unknown'),
            "dirty": getattr(record, 'dirty', False),
        }
        if record.exc_info:
            log_object['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_object)

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    json_console: bool = True,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Set up centralized logging for the application.

    Args:
        log_level: The minimum logging level (e.g., "INFO", "DEBUG").
        log_dir: The directory to store log files.
        log_to_file: Whether to log to a file (only for rank 0).
        json_console: Whether to use JSON format for console logs.
        rank: The process rank in a distributed setup.
        world_size: The total number of processes in a distributed setup.
    """
    _GLOBAL_CONTEXT["rank"] = rank
    _GLOBAL_CONTEXT["world_size"] = world_size

    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addFilter(ContextFilter())

    # Remove all existing handlers from the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_console:
        console_formatter = JsonFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - (%(rank)d/%(world_size)d) - %(message)s'
        )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    handlers = [console_handler]

    # File handler (only for rank 0)
    if log_to_file and rank == 0:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = CompressingRotatingFileHandler(
            log_path / "training.log",
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5,
            compress=True
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Queue for multiprocessing-safe logging
    log_queue = multiprocessing.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)

    # The listener will pull from the queue and send to the actual handlers
    listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=True)

    root_logger.addHandler(queue_handler)

    listener.start()

    # Return listener so it can be stopped gracefully
    return listener
