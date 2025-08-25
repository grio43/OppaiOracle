#!/usr/bin/env python3
"""Utility helpers for consistent logging configuration."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    logger_name: Optional[str] = None,
    config_path: str | Path = "configs/logging.yaml",
    log_file_name: Optional[str] = None,
) -> logging.Logger:
    """Configure logging using a YAML config file.

    Falls back to a basic configuration if the YAML file cannot be read. When
    file logging is enabled in the configuration, a :class:`RotatingFileHandler`
    is attached to the root logger.

    Args:
        logger_name: Name of the logger to return. ``None`` returns the root
            logger.
        config_path: Path to the YAML logging configuration.
        log_file_name: Optional filename for the rotating log file. If not
            provided, the ``filename`` from the YAML configuration is used.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(logger_name)

    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
    fmt = cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=fmt)

    file_cfg = cfg.get("file_logging", {}) or {}
    if file_cfg.get("enabled"):
        log_dir = Path(file_cfg.get("dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        rot = cfg.get("rotation", {}) or {}
        filename = log_file_name or file_cfg.get("filename", "app.log")
        handler = RotatingFileHandler(
            log_dir / filename,
            maxBytes=int(rot.get("max_bytes", 10 * 1024 * 1024)),
            backupCount=int(rot.get("backups", 5)),
        )
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel(level)
        logging.getLogger().addHandler(handler)

    return logging.getLogger(logger_name)
