import os
import json
import logging
import logging.handlers
import queue
import time
from threading import Thread

from pathlib import Path
import multiprocessing as mp
from typing import Optional, List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, get_worker_info, DataLoader
from PIL import Image
from torchvision import transforms
from utils.path_utils import sanitize_identifier, validate_image_path, resolve_and_confine
from utils.metadata_ingestion import parse_tags_field

from l2_cache import LMDBReader, start_l2_writer, _tensor_to_bytes, _tensor_from_bytes
from vocabulary import load_vocabulary_for_training, TagVocabulary


class DatasetLoader(Dataset):
    def __init__(
        self,
        annotations_path,
        image_dir,
        transform=None,
        max_retries=3,
        num_classes=None,
        # Image pipeline params
        image_size: int = 640,
        pad_color: Tuple[int, int, int] = (114, 114, 114),
        normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
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

        # Image pipeline settings
        self.image_size = int(image_size)
        self.pad_color: Tuple[int, int, int] = (
            int(pad_color[0]), int(pad_color[1]), int(pad_color[2])
        ) if isinstance(pad_color, (list, tuple)) else (114, 114, 114)
        self.normalize_mean: Tuple[float, float, float] = tuple(normalize_mean)
        self.normalize_std: Tuple[float, float, float] = tuple(normalize_std)

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
            # Enforce allowlist and strip any sneaky path components
            image_id = sanitize_identifier(str(annotation['image_id']))
            image_key = image_id.encode('utf-8')

            # --- L2 READ PATH ---
            if self._l2_enabled:
                self._ensure_l2_reader()
                payload = self._l2_reader.get(image_key) if self._l2_reader else None
                if payload is not None:
                    try:
                        t = _tensor_from_bytes(payload)

                        # Attempt to reconstruct padding mask by detecting normalized pad color
                        pad_vec = torch.tensor([c / 255.0 for c in self.pad_color], dtype=t.dtype, device=t.device)
                        mean = torch.tensor(self.normalize_mean, dtype=t.dtype, device=t.device)
                        std = torch.tensor(self.normalize_std, dtype=t.dtype, device=t.device)
                        pad_norm = ((pad_vec - mean) / std).view(3, 1, 1)
                        with torch.no_grad():
                            pmask = torch.isclose(t, pad_norm, atol=1e-6).all(dim=0)

                        # Prepare labels for training loop compatibility
                        tag_indices = annotation.get("labels") or []
                        if (
                            isinstance(tag_indices, list) and len(tag_indices) > 0 and isinstance(tag_indices[0], (int, float))
                            and self.num_classes
                        ):
                            tag_vec = torch.zeros(self.num_classes, dtype=torch.float32)
                            tag_vec.scatter_(
                                0,
                                torch.tensor(
                                    [int(i) for i in tag_indices if 0 <= int(i) < self.num_classes], dtype=torch.long
                                ),
                                1.0,
                            )
                        else:
                            tag_vec = torch.zeros(self.num_classes or 1, dtype=torch.float32)

                        rating = annotation.get("rating", "unknown")
                        rating_idx = 4  # default 'unknown'
                        if isinstance(rating, int):
                            rating_idx = int(rating)
                        elif isinstance(rating, str):
                            r = rating.strip().lower()
                            mapping = {
                                "g": 0, "general": 0, "safe": 0,
                                "sensitive": 1,
                                "q": 2, "questionable": 2,
                                "e": 3, "explicit": 3,
                                "u": 4, "unknown": 4,
                            }
                            rating_idx = mapping.get(r, 4)

                        return {
                            "images": t,
                            "padding_mask": pmask.to(torch.bool),
                            "tag_labels": tag_vec,
                            "rating_labels": torch.tensor(rating_idx, dtype=torch.long),
                            "image_id": image_id,
                            "cached": True,
                        }
                    except Exception as e:
                        # Treat as cache miss; skip bad/tampered records safely
                        self.logger.warning(
                            f"L2 cache decode failed for {image_id}: {e} (treating as miss)"
                        )

            # --- Cache miss: load + transform (confined path) ---
            img_path = validate_image_path(Path(self.image_dir), image_id)
            with Image.open(img_path) as pil_img:
                img = pil_img

            # 1) Load & composite transparency onto gray
            if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
                bg = Image.new("RGBA", img.size, (*self.pad_color, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg.convert("RGB")
            else:
                img = img.convert("RGB")

            # 2) Keep aspect ratio via letterbox to square + build padding mask (True = PAD)
            target = int(self.image_size)
            w, h = img.size
            scale = min(target / float(w), target / float(h)) if (w > 0 and h > 0) else 1.0
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = img.resize((max(1, nw), max(1, nh)), Image.BILINEAR)

            canvas = Image.new("RGB", (target, target), tuple(self.pad_color))
            left = (target - resized.size[0]) // 2
            top = (target - resized.size[1]) // 2
            canvas.paste(resized, (left, top))

            pmask = torch.ones(target, target, dtype=torch.bool)
            pmask[top:top + resized.size[1], left:left + resized.size[0]] = False

            # 3) To tensor + normalize (optionally allow external augmentations if provided)
            if self.transform:
                try:
                    tmp = self.transform(canvas)
                    if isinstance(tmp, torch.Tensor):
                        t = tmp
                    else:
                        canvas = tmp
                        t = transforms.ToTensor()(canvas)
                        t = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(t)
                except Exception:
                    t = transforms.ToTensor()(canvas)
                    t = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(t)
            else:
                t = transforms.ToTensor()(canvas)
                t = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(t)

            # Reset retry count on success
            self.retry_counts[idx] = 0

            # Prepare labels for training loop compatibility
            tag_indices = annotation.get("labels") or []
            if isinstance(tag_indices, list) and len(tag_indices) > 0 and isinstance(tag_indices[0], (int, float)) and self.num_classes:
                tag_vec = torch.zeros(self.num_classes, dtype=torch.float32)
                tag_vec.scatter_(0, torch.tensor([int(i) for i in tag_indices if 0 <= int(i) < self.num_classes], dtype=torch.long), 1.0)
            else:
                # Fallback if no labels provided
                tag_vec = torch.zeros(self.num_classes or 1, dtype=torch.float32)

            rating = annotation.get("rating", "unknown")
            rating_idx = 4  # default 'unknown'
            if isinstance(rating, int):
                rating_idx = int(rating)
            elif isinstance(rating, str):
                r = rating.strip().lower()
                mapping = {
                    "g": 0, "general": 0, "safe": 0,
                    "sensitive": 1,
                    "q": 2, "questionable": 2,
                    "e": 3, "explicit": 3,
                    "u": 4, "unknown": 4,
                }
                rating_idx = mapping.get(r, 4)

            sample = {
                "images": t,
                "padding_mask": pmask,
                "tag_labels": tag_vec,
                "rating_labels": torch.tensor(rating_idx, dtype=torch.long),
                "image_id": image_id,
            }

            # Enqueue write but never block __getitem__
            if self._l2_enabled and self._l2_writer_q is not None:
                try:
                    self._l2_writer_q.put_nowait((image_key, _tensor_to_bytes(sample["images"])))
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
        # Default to a common square size when transform is unknown
        sz = int(getattr(self, "image_size", 224) or 224)
        return {
            "images": torch.zeros((3, sz, sz)),  # Placeholder tensor
            "padding_mask": torch.ones((sz, sz), dtype=torch.bool),
            "tag_labels": torch.zeros(self.num_classes or 1, dtype=torch.float32),
            "rating_labels": torch.tensor(4, dtype=torch.long),  # unknown
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

            # Confine and locate image file safely
            try:
                image_id = sanitize_identifier(str(annotation["image_id"]))
                image_path = validate_image_path(Path(self.dataset_loader.image_dir), image_id)
            except Exception as e:
                logging.warning(f"Invalid image_id for item {idx}: {e}")
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
                try:
                    if not all(0 <= int(label) < int(self.dataset_loader.num_classes) for label in labels):
                        logging.warning(f"Invalid labels for item {idx}: {labels}")
                        return False
                except Exception:
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


class SidecarJsonDataset(Dataset):
    """Dataset that reads per-image JSON sidecars in the same folder as images.

    Each JSON is expected to contain at least:
      - filename: image file name (e.g., "12345.jpg")
      - tags: space-delimited string or list of tags
      - rating: optional rating string or int (safe/general/questionable/explicit/unknown)
    """

    def __init__(
        self,
        root_dir: Path,
        json_files: List[Path],
        vocab: TagVocabulary,
        transform=None,
        max_retries: int = 3,
        # Image pipeline params
        image_size: int = 640,
        pad_color: Tuple[int, int, int] = (114, 114, 114),
        normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        # L2 cache plumbing
        l2_enabled: bool = False,
        l2_cache_path: Optional[str] = None,
        l2_map_size_bytes: int = 0,
        l2_max_readers: int = 4096,
        l2_writer_queue: Optional[mp.Queue] = None,
    ):
        self.root = Path(root_dir)
        self.json_files = list(json_files)
        self.vocab = vocab
        self.transform = transform
        self.max_retries = max_retries
        self.retry_counts: Dict[int, int] = {}
        self.failed_samples = set()
        self.logger = logging.getLogger(__name__)

        # Image pipeline settings
        self.image_size = int(image_size)
        self.pad_color: Tuple[int, int, int] = (
            int(pad_color[0]), int(pad_color[1]), int(pad_color[2])
        ) if isinstance(pad_color, (list, tuple)) else (114, 114, 114)
        self.normalize_mean: Tuple[float, float, float] = tuple(normalize_mean)
        self.normalize_std: Tuple[float, float, float] = tuple(normalize_std)

        # L2 cache
        self._l2_enabled = bool(l2_enabled and l2_cache_path)
        self._l2_path = l2_cache_path
        self._l2_map_size = int(l2_map_size_bytes or 0)
        self._l2_max_readers = int(l2_max_readers or 4096)
        self._l2_reader: Optional[LMDBReader] = None
        self._l2_writer_q: Optional[mp.Queue] = l2_writer_queue

        # Pre-parse minimal fields for speed
        self.items: List[Dict[str, Any]] = []
        for jp in self.json_files:
            try:
                data = json.loads(Path(jp).read_text(encoding="utf-8"))
                fname = str(data.get("filename") or jp.with_suffix(".png").name)
                image_id = sanitize_identifier(Path(fname).stem)
                tags_raw = data.get("tags")
                tags_list = parse_tags_field(tags_raw)
                rating = data.get("rating", "unknown")
                self.items.append({
                    "image_id": image_id,
                    "tags": tags_list,
                    "rating": rating,
                })
            except Exception as e:
                self.logger.warning(f"Failed to parse {jp}: {e}")

    def _ensure_l2_reader(self):
        if not self._l2_enabled:
            return
        if self._l2_reader is None:
            assert self._l2_path and self._l2_map_size > 0
            self._l2_reader = LMDBReader(self._l2_path, self._l2_map_size, max_readers=self._l2_max_readers)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self.failed_samples:
            return self._error_sample(idx, "Previously failed sample")

        if idx not in self.retry_counts:
            self.retry_counts[idx] = 0

        try:
            ann = self.items[idx]
            image_id = ann["image_id"]
            image_key = image_id.encode("utf-8")

            # Try L2 cache
            if self._l2_enabled:
                self._ensure_l2_reader()
                payload = self._l2_reader.get(image_key) if self._l2_reader else None
                if payload is not None:
                    try:
                        img_t = _tensor_from_bytes(payload)
                    except Exception as e:
                        self.logger.warning(f"L2 cache decode failed for {image_id}: {e}")
                        img_t = None
                    if img_t is not None:
                        tag_vec = self.vocab.encode_tags(ann["tags"])  # (V,)
                        rating_idx = _map_rating(ann.get("rating", "unknown"))
                        # Reconstruct padding mask by detecting normalized pad color
                        pad_vec = torch.tensor([c / 255.0 for c in self.pad_color], dtype=img_t.dtype, device=img_t.device)
                        mean = torch.tensor(self.normalize_mean, dtype=img_t.dtype, device=img_t.device)
                        std = torch.tensor(self.normalize_std, dtype=img_t.dtype, device=img_t.device)
                        pad_norm = ((pad_vec - mean) / std).view(3, 1, 1)
                        with torch.no_grad():
                            pmask = torch.isclose(img_t, pad_norm, atol=1e-6).all(dim=0)
                        return {
                            "images": img_t,
                            "padding_mask": pmask.to(torch.bool),
                            "tag_labels": tag_vec,
                            "rating_labels": torch.tensor(rating_idx, dtype=torch.long),
                            "image_id": image_id,
                        }

            # Cache miss: load from disk
            img_path = validate_image_path(self.root, image_id)
            with Image.open(img_path) as pil_img:
                pil = pil_img
            # 1) Alpha composite onto gray
            if pil.mode in ("RGBA", "LA") or ("transparency" in pil.info):
                bg = Image.new("RGBA", pil.size, (*self.pad_color, 255))
                bg.paste(pil, mask=pil.split()[-1])
                pil = bg.convert("RGB")
            else:
                pil = pil.convert("RGB")
            # 2) Letterbox to square and padding mask
            target = int(self.image_size)
            w, h = pil.size
            scale = min(target / float(w), target / float(h)) if (w > 0 and h > 0) else 1.0
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = pil.resize((max(1, nw), max(1, nh)), Image.BILINEAR)
            canvas = Image.new("RGB", (target, target), tuple(self.pad_color))
            left = (target - resized.size[0]) // 2
            top = (target - resized.size[1]) // 2
            canvas.paste(resized, (left, top))
            pmask = torch.ones(target, target, dtype=torch.bool)
            pmask[top:top + resized.size[1], left:left + resized.size[0]] = False
            # 3) To tensor + normalize with optional augmentations support
            if self.transform:
                try:
                    tmp = self.transform(canvas)
                    if isinstance(tmp, torch.Tensor):
                        img = tmp
                    else:
                        canvas = tmp
                        img = transforms.ToTensor()(canvas)
                        img = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img)
                except Exception:
                    img = transforms.ToTensor()(canvas)
                    img = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img)
            else:
                img = transforms.ToTensor()(canvas)
                img = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img)

            # Encode labels
            tag_vec = self.vocab.encode_tags(ann["tags"])  # (V,)
            rating_idx = _map_rating(ann.get("rating", "unknown"))

            # Enqueue write (non-blocking)
            if self._l2_enabled and self._l2_writer_q is not None:
                try:
                    self._l2_writer_q.put_nowait((image_key, _tensor_to_bytes(img)))
                except queue.Full:
                    pass

            self.retry_counts[idx] = 0
            return {
                "images": img,
                "padding_mask": pmask,
                "tag_labels": tag_vec,
                "rating_labels": torch.tensor(rating_idx, dtype=torch.long),
                "image_id": image_id,
            }

        except Exception as e:
            self.retry_counts[idx] += 1
            self.logger.warning(f"Failed to load sample {idx}: {e}")
            if self.retry_counts[idx] >= self.max_retries:
                self.failed_samples.add(idx)
                self.logger.error(f"Sample {idx} exceeded max retries, marking as failed")
                return self._error_sample(idx, str(e))
            return self._error_sample(idx, f"Temporary failure: {e}")

    def _error_sample(self, idx: int, reason: str) -> Dict[str, Any]:
        sz = getattr(self.transform.transforms[0], "size", 224) if hasattr(self, "transform") and self.transform else getattr(self, "image_size", 224)
        if isinstance(sz, (tuple, list)):
            sz = sz[0]
        return {
            "images": torch.zeros((3, int(sz), int(sz))),
            "padding_mask": torch.ones((int(sz), int(sz)), dtype=torch.bool),
            "tag_labels": torch.zeros(len(self.vocab.tag_to_index), dtype=torch.float32),
            "rating_labels": torch.tensor(4, dtype=torch.long),
            "image_id": f"error_{idx}",
            "error": True,
            "error_reason": reason,
        }


def _map_rating(rating: Any) -> int:
    """Map dataset rating field to fixed indices used by the model.

    Mapping: general/safe->0, sensitive->1, questionable->2, explicit->3, unknown->4
    """
    if isinstance(rating, int):
        return int(rating)
    r = str(rating).strip().lower()
    mapping = {
        "g": 0, "general": 0, "safe": 0,
        "sensitive": 1,
        "q": 2, "questionable": 2,
        "e": 3, "explicit": 3,
        "u": 4, "unknown": 4,
    }
    return mapping.get(r, 4)


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

    # Load vocabulary once (needed for sidecar mode and to determine num classes)
    vocab = load_vocabulary_for_training(Path(vocab_path))
    num_tags = len(vocab.tag_to_index)

    # Dataset performs letterbox + tensor + normalize; use None unless you add PIL-only augmentations
    image_size = int(getattr(data_config, "image_size", 640))
    mean = tuple(getattr(data_config, "normalize_mean", [0.5, 0.5, 0.5]))
    std = tuple(getattr(data_config, "normalize_std", [0.5, 0.5, 0.5]))
    pad_color = tuple(getattr(data_config, "pad_color", [114, 114, 114]))
    transform = None

    # Determine dataset mode
    root = Path(active_data_path)
    manifest_train = root / "train.json"
    manifest_val = root / "val.json"
    images_dir = root / "images"

    if manifest_train.exists() and manifest_val.exists() and images_dir.exists():
        # Manifest mode (back-compat)
        train_ds = DatasetLoader(
            annotations_path=str(manifest_train),
            image_dir=str(images_dir),
            transform=transform,
            num_classes=num_tags,
            image_size=image_size,
            pad_color=pad_color,
            normalize_mean=mean,
            normalize_std=std,
            l2_enabled=bool(getattr(data_config, "l2_cache_enabled", False)),
            l2_cache_path=getattr(data_config, "l2_cache_path", None),
            l2_map_size_bytes=map_size_bytes,
            l2_max_readers=getattr(data_config, "l2_max_readers", 4096),
            l2_writer_queue=q,
        )

        val_ds = DatasetLoader(
            annotations_path=str(manifest_val),
            image_dir=str(images_dir),
            transform=transform,
            num_classes=num_tags,
            image_size=image_size,
            pad_color=pad_color,
            normalize_mean=mean,
            normalize_std=std,
            l2_enabled=bool(getattr(data_config, "l2_cache_enabled", False)),
            l2_cache_path=getattr(data_config, "l2_cache_path", None),
            l2_map_size_bytes=map_size_bytes,
            l2_max_readers=getattr(data_config, "l2_max_readers", 4096),
            l2_writer_queue=q,
        )
    else:
        # Sidecar JSON mode: scan *.json in the same folder as images
        logger.info("Manifest not found; entering sidecar JSON mode (scanning .json next to images)")

        all_jsons = sorted([p for p in root.iterdir() if p.suffix.lower() == ".json"]) if root.exists() else []
        if not all_jsons:
            raise FileNotFoundError(
                f"No annotation JSON files found under {root}. Expected per-image JSON sidecars."
            )

        # Deterministic split
        import random as _random
        rng = _random.Random(int(seed))
        rng.shuffle(all_jsons)
        split_ratio = 0.95
        n_train = max(1, int(len(all_jsons) * split_ratio))
        train_list = all_jsons[:n_train]
        val_list = all_jsons[n_train:] if n_train < len(all_jsons) else all_jsons[-max(1, len(all_jsons)//20):]

        train_ds = SidecarJsonDataset(
            root_dir=root,
            json_files=train_list,
            vocab=vocab,
            transform=transform,
            image_size=image_size,
            pad_color=pad_color,
            normalize_mean=mean,
            normalize_std=std,
            l2_enabled=bool(getattr(data_config, "l2_cache_enabled", False)),
            l2_cache_path=getattr(data_config, "l2_cache_path", None),
            l2_map_size_bytes=map_size_bytes,
            l2_max_readers=getattr(data_config, "l2_max_readers", 4096),
            l2_writer_queue=q,
        )

        val_ds = SidecarJsonDataset(
            root_dir=root,
            json_files=val_list,
            vocab=vocab,
            transform=transform,
            image_size=image_size,
            pad_color=pad_color,
            normalize_mean=mean,
            normalize_std=std,
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
