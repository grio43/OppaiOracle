import os
import json
import logging
import logging.handlers
import queue
import time
from threading import Thread

from pathlib import Path
import multiprocessing as mp
from typing import Optional

import torch
from torch.utils.data import Dataset, get_worker_info, DataLoader
from PIL import Image

from l2_cache import LMDBReader, start_l2_writer, _tensor_to_bytes, _tensor_from_bytes
from vocabulary import load_vocabulary_for_training


class DatasetLoader(Dataset):
    def __init__(
        self,
        annotations_path,
        image_dir,
        transform=None,
        max_retries=3,
        num_classes=None,
        # L2 cache plumbing
        l2_enabled: bool = False,
        l2_cache_path: Optional[str] = None,
        l2_map_size_bytes: int = 0,
        l2_max_readers: int = 4096,
        l2_writer_queue: Optional[mp.Queue] = None,
    ):
        """
        Dataset loader for images and JSON metadata.
        Note: Despite legacy naming, this does NOT handle HDF5 files.
        """
        self.annotations = self._load_annotations(annotations_path)
        self.image_dir = image_dir
        self.transform = transform
        self.max_retries = max_retries
        self.num_classes = num_classes
        self.retry_counts = {}
        self.failed_samples = set()
        self.logger = logging.getLogger(__name__)

        # Properly initialise background validator
        self.validator = BackgroundValidator(self)
        self.validator.start()

        # --- L2 cache (read-only in workers) ---
        self._l2_enabled = bool(l2_enabled and l2_cache_path)
        self._l2_path = l2_cache_path
        self._l2_map_size = int(l2_map_size_bytes or 0)
        self._l2_max_readers = int(l2_max_readers or 4096)
        self._l2_reader: Optional[LMDBReader] = None
        self._l2_writer_q: Optional[mp.Queue] = l2_writer_queue

    def _ensure_l2_reader(self):
        if not self._l2_enabled:
            return
        # Open env lazily **inside** the worker process to avoid fork-related handle reuse
        if self._l2_reader is None:
            assert self._l2_path and self._l2_map_size > 0
            self._l2_reader = LMDBReader(self._l2_path, self._l2_map_size, max_readers=self._l2_max_readers)

    def _load_annotations(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # HL002 Fix: Return error sample immediately on failure, don't bias distribution
        if idx in self.failed_samples:
            return self._create_error_sample(idx, "Previously failed sample")

        if idx not in self.retry_counts:
            self.retry_counts[idx] = 0

        try:
            annotation = self.annotations[idx]
            image_id = str(annotation['image_id'])
            image_key = image_id.encode('utf-8')

            # --- L2 READ PATH ---
            if self._l2_enabled:
                self._ensure_l2_reader()
                payload = self._l2_reader.get(image_key) if self._l2_reader else None
                if payload is not None:
                    t = _tensor_from_bytes(payload)
                    return {
                        "image": t,
                        "labels": torch.tensor(annotation["labels"]),
                        "image_id": image_id,
                        "cached": True,
                    }

            # --- Cache miss: load + transform ---
            image = Image.open(f"{self.image_dir}/{image_id}.jpg")

            if self.transform:
                image = self.transform(image)

            # Reset retry count on success
            self.retry_counts[idx] = 0

            sample = {
                "image": image,
                "labels": torch.tensor(annotation["labels"]),
                "image_id": image_id,
            }

            # Enqueue write but never block __getitem__
            if self._l2_enabled and self._l2_writer_q is not None:
                try:
                    self._l2_writer_q.put_nowait((image_key, _tensor_to_bytes(image)))
                except queue.Full:
                    # Drop silently; training must not stall on cache IO
                    pass
            return sample

        except Exception as e:
            self.retry_counts[idx] += 1
            self.logger.warning(f"Failed to load sample {idx}: {e}")

            if self.retry_counts[idx] >= self.max_retries:
                self.failed_samples.add(idx)
                self.logger.error(f"Sample {idx} exceeded max retries, marking as failed")
                return self._create_error_sample(idx, str(e))

            # Return error sample instead of silently advancing to next index
            return self._create_error_sample(idx, f"Temporary failure: {e}")

    def _create_error_sample(self, idx, reason):
        """Create a clearly marked error sample"""
        return {
            "image": torch.zeros((3, 224, 224)),  # Placeholder tensor
            "labels": torch.tensor([-1]),  # Error label
            "image_id": f"error_{idx}",
            "error": True,
            "error_reason": reason,
        }

    def get_failure_statistics(self):
        """Return statistics about failed samples for logging"""
        return {
            "total_failed": len(self.failed_samples),
            "failed_indices": list(self.failed_samples),
            "retry_counts": self.retry_counts,
        }


class BackgroundValidator(Thread):
    # HL003 Fix: Implement actual validation logic
    def __init__(self, dataset_loader):
        super().__init__(daemon=True)
        self.dataset_loader = dataset_loader
        self.validation_queue = queue.Queue()
        self.running = True

    def run(self):
        """Background validation loop"""
        while self.running:
            try:
                if not self.validation_queue.empty():
                    item_idx = self.validation_queue.get(timeout=1.0)
                    self.validate_item(item_idx)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Validation error: {e}")

    def validate_item(self, idx):
        """Perform actual validation of dataset items"""
        try:
            annotation = self.dataset_loader.annotations[idx]

            # Check if image file exists
            image_path = f"{self.dataset_loader.image_dir}/{annotation['image_id']}.jpg"
            if not os.path.exists(image_path):
                logging.warning(f"Missing image file: {image_path}")
                return False

            # Validate image can be opened
            try:
                with Image.open(image_path) as img:
                    if img.mode not in ["RGB", "L"]:
                        logging.warning(f"Unexpected image mode {img.mode} for {image_path}")
            except Exception as e:
                logging.warning(f"Cannot open image {image_path}: {e}")
                return False

            # Validate labels are within expected range
            if "labels" in annotation and self.dataset_loader.num_classes is not None:
                labels = annotation["labels"]
                if not all(0 <= label < self.dataset_loader.num_classes for label in labels):
                    logging.warning(f"Invalid labels for item {idx}: {labels}")
                    return False

            return True

        except Exception as e:
            logging.error(f"Validation failed for item {idx}: {e}")
            return False

    def stop(self):
        """Stop the validation thread"""
        self.running = False


class AugmentationStats:
    """Placeholder class for augmentation statistics."""
    pass


def validate_dataset(*args, **kwargs):
    """Placeholder dataset validation function."""
    return {}


def create_dataloaders(
    data_config,
    validation_config,
    vocab_path,
    active_data_path,
    distributed=False,
    rank=-1,
    world_size=1,
    seed=42,
    debug_config=None,
    **kwargs,
):
    logger = logging.getLogger(__name__)

    # Map size for LMDB (bytes)
    map_size_bytes = int(getattr(data_config, "l2_max_size_gb", 0) * (1024 ** 3))

    # Optional writer process
    if getattr(data_config, "l2_cache_enabled", False):
        q, _proc = start_l2_writer(data_config.l2_cache_path, map_size_bytes)
        logger.info(
            "L2 cache enabled at %s (map_size_bytes=%d)",
            data_config.l2_cache_path,
            map_size_bytes,
        )
    else:
        q, _proc = None, None
        logger.info("L2 cache disabled; proceeding without LMDB writer")

    # Build datasets
    train_ds = DatasetLoader(
        annotations_path=str(Path(active_data_path) / "train.json"),
        image_dir=str(Path(active_data_path) / "images"),
        transform=None,
        num_classes=None,
        l2_enabled=bool(getattr(data_config, "l2_cache_enabled", False)),
        l2_cache_path=getattr(data_config, "l2_cache_path", None),
        l2_map_size_bytes=map_size_bytes,
        l2_max_readers=getattr(data_config, "l2_max_readers", 4096),
        l2_writer_queue=q,
    )

    val_ds = DatasetLoader(
        annotations_path=str(Path(active_data_path) / "val.json"),
        image_dir=str(Path(active_data_path) / "images"),
        transform=None,
        num_classes=None,
        l2_enabled=bool(getattr(data_config, "l2_cache_enabled", False)),
        l2_cache_path=getattr(data_config, "l2_cache_path", None),
        l2_map_size_bytes=map_size_bytes,
        l2_max_readers=getattr(data_config, "l2_max_readers", 4096),
        l2_writer_queue=q,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor,
        persistent_workers=data_config.persistent_workers,
        drop_last=data_config.drop_last,
        shuffle=True,
    )

    val_batch = (
        validation_config.dataloader.batch_size
        if hasattr(validation_config, "dataloader")
        else data_config.batch_size
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor,
        persistent_workers=data_config.persistent_workers,
        drop_last=False,
        shuffle=False,
    )

    vocab = load_vocabulary_for_training(Path(vocab_path))

    return train_loader, val_loader, vocab


class CompressingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating file handler with optional compression (placeholder)."""

    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0,
                 encoding=None, delay=False, compress=False):
        super().__init__(filename, mode=mode, maxBytes=maxBytes,
                         backupCount=backupCount, encoding=encoding, delay=delay)
        self.compress = compress

    def doRollover(self):
        super().doRollover()
        # Compression could be implemented here if needed
