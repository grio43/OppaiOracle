import os
import json
import logging
import logging.handlers
import queue
import time
from threading import Thread

import torch
from torch.utils.data import Dataset
from PIL import Image


class DatasetLoader(Dataset):
    def __init__(self, annotations_path, image_dir, transform=None, max_retries=3, num_classes=None):
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
            image = Image.open(f"{self.image_dir}/{annotation['image_id']}.jpg")

            if self.transform:
                image = self.transform(image)

            # Reset retry count on success
            self.retry_counts[idx] = 0

            return {
                "image": image,
                "labels": torch.tensor(annotation["labels"]),
                "image_id": annotation["image_id"],
            }

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


def create_dataloaders(*args, **kwargs):
    """Placeholder dataloader creation function."""
    raise NotImplementedError("create_dataloaders is not implemented")


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
