import os
import json
import logging
from torch.utils.data.distributed import DistributedSampler
import logging.handlers
import queue
import time
from threading import Thread

from pathlib import Path
import multiprocessing as mp
from typing import Optional, List, Dict, Any, Tuple, Set
import hashlib

import torch
from torch.utils.data import Dataset, get_worker_info, DataLoader as _TorchDataLoader
from PIL import Image, ImageOps, ImageFile
# Make torchvision optional at import time; raise only when actually used.
try:
    from torchvision import transforms  # type: ignore
except Exception:
    transforms = None  # resolved lazily
# Torchvision v2 joint transforms (optional)
try:
    from torchvision.transforms import v2 as T
    from torchvision import tv_tensors
except Exception:  # keep backward compatible
    T = None
    tv_tensors = None
from vocabulary import load_vocabulary_for_training, TagVocabulary

# Orientation-aware flipping (optional; keeps file usable in legacy setups)
try:
    from orientation_handler import OrientationHandler  # noqa: F401
except Exception:  # pragma: no cover
    OrientationHandler = None  # type: ignore
from utils.path_utils import sanitize_identifier, validate_image_path, resolve_and_confine
from utils.metadata_ingestion import parse_tags_field

from l2_cache import LMDBReader, start_l2_writer, _tensor_to_bytes, _tensor_from_bytes
from l1_cache import build_l1_cache, encode_l1_image_01, decode_l1_image_01

# Pillow resampling compatibility and truncated image handling
try:  # Pillow ≥10
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
except AttributeError:  # Pillow <10
    RESAMPLE_BILINEAR = Image.BILINEAR

# Optionally allow loading truncated/corrupt files (opt-in via env)
ALLOW_TRUNCATED = bool(int(os.environ.get("OO_ALLOW_TRUNCATED", "0")))
if ALLOW_TRUNCATED:
    ImageFile.LOAD_TRUNCATED_IMAGES = True


# Single place to control floating tolerance when reconstructing padding masks
PAD_MASK_ATOL = 1e-3

# Guarded DataLoader wrapper:
# - If num_workers == 0, drop prefetch_factor and force persistent_workers=False.
#   This avoids ValueError in PyTorch when setting multiprocessing-only args with zero workers.
class DataLoader(_TorchDataLoader):  # keep public name the same
    def __init__(self, *args, **kwargs):
        num_workers = int(kwargs.get("num_workers", 0) or 0)
        if num_workers == 0:
            # Disallow multiprocessing-only knobs in single-process mode
            kwargs.pop("prefetch_factor", None)
            kwargs["persistent_workers"] = False
        super().__init__(*args, **kwargs)


def _make_worker_init(log_queue):
    """Create a worker_init_fn that attaches a QueueHandler for logging."""
    if log_queue is None:
        return None
    def _init(_worker_id: int):
        logger = logging.getLogger()
        # Ensure a single QueueHandler per worker
        for h in list(logger.handlers):
            try:
                from logging.handlers import QueueHandler  # local import to avoid import-time dependency
                if isinstance(h, QueueHandler):
                    logger.removeHandler(h)
            except Exception:
                # Fallback: check class name to avoid hard import
                if getattr(h, "__class__", None) and h.__class__.__name__ == "QueueHandler":
                    logger.removeHandler(h)
        try:
            from logging.handlers import QueueHandler
            logger.addHandler(QueueHandler(log_queue))
        except Exception:
            pass
    return _init


class DatasetLoader(Dataset):
    def __init__(
        self,
        annotations_path,
        image_dir,
        transform=None,
        joint_transforms=None,  # NEW: torchvision v2 transforms applied to (image, mask) together
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
        l2_max_readers: int = 512,
        l2_writer_queue: Optional[mp.Queue] = None,
        # --- L1 (in-memory) cache ---
        use_memory_cache: bool = True,
        l1_per_worker_mb: int = 256,
        canonical_cache_dtype: str = "uint8",
        preload_files: int = 0,
        # Background validator control
        enable_background_validator: Optional[bool] = None,
    ):
        """
        Dataset loader for images and JSON metadata.
        Note: Despite legacy naming, this does NOT handle HDF5 files.
        """
        self.annotations = self._load_annotations(annotations_path)
        self.image_dir = image_dir
        self.transform = transform
        self.joint_transforms = joint_transforms
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

        # Compute a hash of preprocessing parameters.  Include image size, pad
        # colour, normalization, cache dtype and a schema version so that any
        # change invalidates stale cache entries.
        try:
            cfg_fields = {
                "image_size": self.image_size,
                "pad_color": self.pad_color,
                "normalize_mean": self.normalize_mean,
                "normalize_std": self.normalize_std,
                "cache_storage_dtype": getattr(self, "canonical_cache_dtype", getattr(self, "_l1_dtype_str", "float32")),
                "schema_version": os.getenv("CACHE_SCHEMA_VERSION", "v1"),
            }
            cfg_str = "|".join(f"{k}={v}" for k, v in cfg_fields.items())
            self._l2_cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:8]
        except Exception:
            self._l2_cfg_hash = "00000000"

        # --- Compute a hash of preprocessing parameters for L2 cache versioning ---
        try:
            cfg_fields = {
                "image_size": self.image_size,
                "pad_color": self.pad_color,
                "normalize_mean": self.normalize_mean,
                "normalize_std": self.normalize_std,
                "cache_storage_dtype": getattr(self, "canonical_cache_dtype", getattr(self, "_l1_dtype_str", "float32")),
                "schema_version": os.getenv("CACHE_SCHEMA_VERSION", "v1"),
            }
            cfg_str = "|".join(f"{k}={v}" for k, v in cfg_fields.items())
            self._l2_cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:8]
        except Exception:
            self._l2_cfg_hash = "00000000"

        # --- Compute a hash of preprocessing parameters for L2 cache versioning ---
        try:
            cfg_fields = {
                "image_size": self.image_size,
                "pad_color": self.pad_color,
                "normalize_mean": self.normalize_mean,
                "normalize_std": self.normalize_std,
                "cache_storage_dtype": getattr(self, "canonical_cache_dtype", getattr(self, "_l1_dtype_str", "float32")),
                "schema_version": os.getenv("CACHE_SCHEMA_VERSION", "v1"),
            }
            cfg_str = "|".join(f"{k}={v}" for k, v in cfg_fields.items())
            self._l2_cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:8]
        except Exception:
            self._l2_cfg_hash = "00000000"

        # Properly initialise background validator (opt-out via env or param)
        if enable_background_validator is None:
            enable_background_validator = os.getenv("DATASET_BACKGROUND_VALIDATOR", "1") != "0"
        self.validator = None
        if enable_background_validator:
            self.validator = BackgroundValidator(self)
            self.validator.start()

        # --- L2 cache (read-only in workers) ---
        self._l2_enabled = bool(l2_enabled and l2_cache_path)
        self._l2_path = l2_cache_path
        self._l2_map_size = int(l2_map_size_bytes or 0)
        self._l2_max_readers = int(l2_max_readers or 4096)
        self._l2_reader: Optional[LMDBReader] = None
        self._l2_writer_q: Optional[mp.Queue] = l2_writer_queue
        self._last_qfull_warn: float = 0.0

        # Compute a hash of preprocessing parameters for L2 cache versioning.
        try:
            cfg_fields = {
                "image_size": self.image_size,
                "pad_color": self.pad_color,
                "normalize_mean": self.normalize_mean,
                "normalize_std": self.normalize_std,
                "cache_storage_dtype": getattr(self, "canonical_cache_dtype", getattr(self, "_l1_dtype_str", "float32")),
                "schema_version": os.getenv("CACHE_SCHEMA_VERSION", "v1"),
            }
            cfg_str = "|".join(f"{k}={v}" for k, v in cfg_fields.items())
            self._l2_cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:8]
        except Exception:
            self._l2_cfg_hash = "00000000"

        # --- L1 cache (per-worker; created lazily in worker) ---
        self._use_l1 = bool(use_memory_cache)
        self._l1_mb = int(l1_per_worker_mb or 0)
        self._l1_dtype_str = str(canonical_cache_dtype or "uint8").lower()
        self._l1 = None  # created lazily per worker
        self._preload_n = int(preload_files or 0)

    def _ensure_l2_reader(self):
        """Create the L2 LMDB reader if (and only if) the path is set and size is positive."""
        if not getattr(self, "_l2_enabled", False):
            return
        # If the path is unset OR empty, treat L2 as disabled.
        if not getattr(self, "_l2_path", None) or not str(self._l2_path):
            return
        # If size missing or non-positive, warn once and treat as disabled.
        if not getattr(self, "_l2_map_size", None) or int(self._l2_map_size) <= 0:
            logging.warning("L2 cache enabled but l2_map_size_bytes<=0 or missing; disabling L2 reads.")
            return
        # Open env lazily **inside** the worker process to avoid fork-related handle reuse
        if self._l2_reader is None:
            self._l2_reader = LMDBReader(self._l2_path, self._l2_map_size, max_readers=self._l2_max_readers)

    # ---------- L1 ----------
    def _ensure_l1(self):
        if not self._use_l1 or self._l1 is not None or self._l1_mb <= 0:
            return
        self._l1 = build_l1_cache(self._l1_mb, self._l1_dtype_str)

    # Helpers for L2 keys: incorporate preprocessing config and flip status
    def _l2_key(self, image_id: str, *, flipped: bool) -> bytes:
        """
        Build a unique key for the L2 cache that includes the image_id, a hash of
        the preprocessing configuration, and a flip bit.  The flip bit is always
        false for DatasetLoader since this dataset does not support flipping.
        """
        flip = "1" if flipped else "0"
        return f"{image_id}|cfg{self._l2_cfg_hash}|flip{flip}".encode("utf-8")

    def _l2_mask_key(self, image_id: str, *, flipped: bool) -> bytes:
        """
        Build a unique key for the L2 cache for the padding mask.

        The mask key appends '|m' to the standard L2 key. Storing an explicit
        mask alongside the image avoids heuristic reconstruction based on pad
        colour and normalization.
        """
        flip = "1" if flipped else "0"
        return f"{image_id}|cfg{self._l2_cfg_hash}|flip{flip}|m".encode("utf-8")

    def _l1_keys(self, image_key: bytes, *, flipped: bool) -> tuple[bytes, bytes]:
        sz = str(int(self.image_size)).encode("utf-8")
        flip = b"1" if flipped else b"0"
        base = image_key + b"|sz" + sz + b"|flip" + flip
        return base + b"|raw", base + b"|m"

    def preload_first_n(self, n: int):
        n = int(max(0, n))
        if n == 0 or len(self) == 0:
            return
        for i in range(min(n, len(self))):
            _ = self[i]

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
            raw_image_id = sanitize_identifier(str(annotation['image_id']))
            # L1 keys are based on the raw image id; L2 keys include config + flip bit
            image_key_base = raw_image_id.encode("utf-8")

            flipped = False  # this Dataset has no flip logic
            l2_key = self._l2_key(raw_image_id, flipped=flipped)

            # --- L1 READ PATH (precedes L2) ---
            if self._use_l1 and self._l1_mb > 0:
                self._ensure_l1()
                if self._l1 is not None:
                    raw_key, mask_key = self._l1_keys(image_key_base, flipped=flipped)
                    raw_stored = self._l1.get(raw_key)
                    if raw_stored is not None:
                        if transforms is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        img_01 = decode_l1_image_01(raw_stored)
                        t = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img_01)
                        m = self._l1.get(mask_key)
                        if m is not None:
                            pmask = m.to(torch.bool)
                        else:
                            # Legacy entries may lack a stored mask. Reconstruct by comparing
                            # the pre-normalised 0–1 image to the pad colour. Without this
                            # fallback the model would incorrectly treat all tokens as valid.
                            pad_vec = torch.tensor(
                                [c / 255.0 for c in self.pad_color],
                                dtype=img_01.dtype,
                                device=img_01.device,
                            ).view(3, 1, 1)
                            with torch.no_grad():
                                pmask = torch.isclose(
                                    img_01,
                                    pad_vec,
                                    atol=PAD_MASK_ATOL,
                                    rtol=0.0,
                                ).all(dim=0)

                        # labels (same as L2 path)
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
                        rating_idx = 4
                        if isinstance(rating, int):
                            rating_idx = int(rating)
                        elif isinstance(rating, str):
                            r = rating.strip().lower()
                            rating_idx = {"g":0,"general":0,"safe":0,"sensitive":1,"q":2,"questionable":2,"e":3,"explicit":3,"u":4,"unknown":4}.get(r, 4)

                        return {
                            "images": t,
                            "padding_mask": pmask.to(torch.bool),
                            "tag_labels": tag_vec,
                            "rating_labels": torch.tensor(rating_idx, dtype=torch.long),
                            "image_id": raw_image_id,
                            "cached": True,
                            "error": False,
                            "error_reason": "",
                        }

            # --- L2 READ PATH ---
            if self._l2_enabled:
                self._ensure_l2_reader()
                payload = self._l2_reader.get(l2_key) if self._l2_reader else None
                mask_payload = self._l2_reader.get(self._l2_mask_key(raw_image_id, flipped=False)) if self._l2_reader else None
                if payload is not None:
                    try:
                        t = _tensor_from_bytes(payload)
                        # Require channel‑first tensors with 3 channels; reject HWC or wrong channel counts
                        if (
                            t.dim() != 3
                            or t.shape[0] != 3
                            or t.shape[1] != int(self.image_size)
                            or t.shape[2] != int(self.image_size)
                        ):
                            raise ValueError(
                                f"L2 cached tensor shape {t.shape} does not match expected (3, {self.image_size}, {self.image_size})"
                            )

                        # Try explicit mask first; fall back to heuristic reconstruction for legacy caches
                        pmask: torch.Tensor
                        if mask_payload is not None:
                            try:
                                m = _tensor_from_bytes(mask_payload)
                                # Validate mask geometry matches current config
                                if m.dim() != 2 or m.shape[0] != int(self.image_size) or m.shape[1] != int(self.image_size):
                                    mask_payload = None  # fall back to reconstruction
                                else:
                                    pmask = m.to(torch.bool)
                            except Exception:
                                mask_payload = None
                        if mask_payload is None:
                            pad_vec = torch.tensor(
                                [c / 255.0 for c in self.pad_color],
                                dtype=t.dtype,
                                device=t.device,
                            )
                            mean = torch.tensor(self.normalize_mean, dtype=t.dtype, device=t.device)
                            std = torch.tensor(self.normalize_std, dtype=t.dtype, device=t.device)
                            pad_norm = ((pad_vec - mean) / std).view(3, 1, 1)
                            with torch.no_grad():
                                pmask = torch.isclose(
                                    t,
                                    pad_norm,
                                    atol=PAD_MASK_ATOL,
                                    rtol=0.0,
                                ).all(dim=0)

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
                            "image_id": raw_image_id,
                            "cached": True,
                            "error": False,
                            "error_reason": "",
                        }
                    except Exception as e:
                        # Treat as cache miss; skip bad/tampered records safely
                        self.logger.warning(
                            f"L2 cache decode failed for {raw_image_id}: {e} (treating as miss)"
                        )

            # --- Cache miss: load + transform (confined path) ---
            img_path = validate_image_path(Path(self.image_dir), image_id)
            # Fully decode while file is open; fix EXIF rotations.
            with Image.open(img_path) as pil_img:
                pil_img.load()
                pil_img = ImageOps.exif_transpose(pil_img)

                if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
                    rgba = pil_img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, tuple(self.pad_color))
                    alpha = rgba.getchannel("A")
                    bg.paste(rgba, mask=alpha)
                    img = bg
                else:
                    img = pil_img.convert("RGB")

            # 2) Keep aspect ratio via letterbox to square + build padding mask (True = PAD)
            target = int(self.image_size)
            w, h = img.size
            # Downscale-only letterbox: preserve aspect, never upscale
            ratio = min(target / float(w), target / float(h)) if (w > 0 and h > 0) else 1.0
            scale = min(1.0, ratio)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            # (Optional) modern Pillow name to avoid deprecation noise:
            # from PIL import Image; resample = Image.Resampling.BILINEAR
            resized = img.resize((max(1, nw), max(1, nh)), RESAMPLE_BILINEAR)

            canvas = Image.new("RGB", (target, target), tuple(self.pad_color))
            left = (target - resized.size[0]) // 2
            top = (target - resized.size[1]) // 2
            canvas.paste(resized, (left, top))

            pmask = torch.ones(target, target, dtype=torch.bool)
            pmask[top:top + resized.size[1], left:left + resized.size[0]] = False

            # If provided, run joint v2 transforms to keep image & mask aligned
            if self.joint_transforms is not None and T is not None and tv_tensors is not None:
                img_tv = tv_tensors.Image(canvas)
                mask_tv = tv_tensors.Mask(pmask.to(torch.uint8))  # 1=PAD, 0=valid
                # v2 ops automatically use NEAREST for Mask; geometry stays in sync
                img_tv, mask_tv = self.joint_transforms(img_tv, mask_tv)
                # Pre-norm 0..1 tensor for L1; then normalize for model
                img_01 = T.ToTensor()(img_tv)  # 0..1 float
                if transforms is None:
                    raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                t = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img_01)
                pmask = mask_tv.to(torch.bool)
            else:
                # Fallback: color-only transforms ok; any geometry here would desync pmask
                if self.transform:
                    try:
                        transformed = self.transform(canvas)
                        if transforms is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        # Ensure we can derive 0..1 image for L1 regardless of transform type
                        img_01 = transformed if isinstance(transformed, torch.Tensor) else transforms.ToTensor()(transformed)
                        if isinstance(img_01, torch.Tensor) and img_01.dtype != torch.float32:
                            img_01 = img_01.to(torch.float32)
                        t = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img_01)
                    except Exception:
                        if transforms is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        img_01 = transforms.ToTensor()(canvas)
                        t = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img_01)
                else:
                    if transforms is None:
                        raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                    img_01 = transforms.ToTensor()(canvas)
                    t = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img_01)

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
                "image_id": raw_image_id,
                "cached": False,
                "error": False,
                "error_reason": "",
            }

            # Enqueue write but never block __getitem__
            if self._l2_enabled and self._l2_writer_q is not None:
                try:
                    # Write normalized image and explicit padding mask
                    self._l2_writer_q.put_nowait((l2_key, _tensor_to_bytes(sample["images"])))
                    self._l2_writer_q.put_nowait((self._l2_mask_key(raw_image_id, flipped=False), _tensor_to_bytes(sample["padding_mask"].to(torch.uint8))))
                except queue.Full:
                    # Drop, but surface a rate-limited warning for visibility.
                    now = time.time()
                    if (now - self._last_qfull_warn) > 5.0:
                        self._last_qfull_warn = now
                        self.logger.warning(
                            "L2 writer queue full; dropping cache write (rate-limited)"
                        )
            # L1: write pre-norm image (and mask) in canonical dtype; never block
            if self._use_l1 and self._l1_mb > 0:
                self._ensure_l1()
                if self._l1 is not None:
                    try:
                        raw_key, mask_key = self._l1_keys(image_key_base, flipped=False)
                        self._l1.put(raw_key, encode_l1_image_01(img_01, dtype_str=self._l1_dtype_str))
                        self._l1.put(mask_key, pmask.to(torch.uint8))
                    except Exception:
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
            "cached": False,
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
        joint_transforms=None,  # NEW
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
        l2_max_readers: int = 512,
        l2_writer_queue: Optional[mp.Queue] = None,
        # --- Orientation / flipping ---
        random_flip_prob: float = 0.0,
        orientation_handler: Optional["OrientationHandler"] = None,
        flip_overrides_path: Optional[str] = None,   # JSON with {"force_flip":[ids], "never_flip":[ids]} (also accepts {"flip":[...]} or a bare list)
        respect_flip_list: bool = True,
        stats_queue: Optional[mp.Queue] = None,
    ):
        self.root = Path(root_dir)
        self.json_files = list(json_files)
        self.vocab = vocab
        self.transform = transform
        self.joint_transforms = joint_transforms
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
        self._last_qfull_warn: float = 0.0

        # Compute a hash of preprocessing parameters for L2 cache versioning.
        try:
            cfg_fields = {
                "image_size": self.image_size,
                "pad_color": self.pad_color,
                "normalize_mean": self.normalize_mean,
                "normalize_std": self.normalize_std,
                # Sidecar dataset does not use L1 dtype, but keep field for parity
                "cache_storage_dtype": getattr(self, "canonical_cache_dtype", "float32"),
                "schema_version": os.getenv("CACHE_SCHEMA_VERSION", "v1"),
            }
            cfg_str = "|".join(f"{k}={v}" for k, v in cfg_fields.items())
            self._l2_cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:8]
        except Exception:
            self._l2_cfg_hash = "00000000"

        # --- Orientation / flipping state ---
        self.random_flip_prob = float(random_flip_prob or 0.0)
        self.orientation_handler = orientation_handler
        self.respect_flip_list = bool(respect_flip_list)
        self._force_flip_ids: Set[str] = set()
        self._never_flip_ids: Set[str] = set()
        if flip_overrides_path:
            try:
                path = Path(flip_overrides_path)
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        force = data.get("force_flip") or data.get("flip") or []
                        never = data.get("never_flip") or data.get("no_flip") or []
                        self._force_flip_ids = {sanitize_identifier(str(x)) for x in force}
                        self._never_flip_ids = {sanitize_identifier(str(x)) for x in never}
                    elif isinstance(data, list):
                        self._force_flip_ids = {sanitize_identifier(str(x)) for x in data}
            except Exception as e:
                self.logger.warning(f"Failed to load flip_overrides from {flip_overrides_path}: {e}")

        # Telemetry (optional): push orientation stats periodically
        self._stats_queue = stats_queue
        self._samples_seen = 0

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
                # Remember the shard folder this pair lives in for image resolution
                self.items.append({
                    "image_id": image_id,
                    "tags": tags_list,
                    "rating": rating,
                    "dir": Path(jp).parent,
                })
            except Exception as e:
                self.logger.warning(f"Failed to parse {jp}: {e}")

    def _ensure_l2_reader(self):
        """Create the L2 LMDB reader if (and only if) the path is set and size is positive."""
        if not getattr(self, "_l2_enabled", False):
            return
        # If the path is unset OR empty, treat L2 as disabled.
        if not getattr(self, "_l2_path", None) or not str(self._l2_path):
            return
        # If size missing or non-positive, warn once and treat as disabled.
        if not getattr(self, "_l2_map_size", None) or int(self._l2_map_size) <= 0:
            logging.warning("L2 cache enabled but l2_map_size_bytes<=0 or missing; disabling L2 reads.")
            return
        if self._l2_reader is None:
            self._l2_reader = LMDBReader(self._l2_path, self._l2_map_size, max_readers=self._l2_max_readers)

    def __len__(self) -> int:
        return len(self.items)

    # Helpers for L2 keys: incorporate preprocessing config and flip status
    def _l2_key(self, image_id: str, *, flipped: bool) -> bytes:
        """
        Build a unique key for the L2 cache that includes the image_id, a hash of the
        preprocessing configuration, and a flip bit.
        """
        flip_bit = "1" if flipped else "0"
        return f"{image_id}|cfg{self._l2_cfg_hash}|flip{flip_bit}".encode("utf-8")

    def _l2_mask_key(self, image_id: str, *, flipped: bool) -> bytes:
        """
        Build a unique key for the L2 cache for the padding mask. Appends '|m' to the standard L2 key.
        Storing explicit masks avoids brittle reconstruction.
        """
        flip_bit = "1" if flipped else "0"
        return f"{image_id}|cfg{self._l2_cfg_hash}|flip{flip_bit}|m".encode("utf-8")

    def _deterministic_coin(self, image_id: str) -> bool:
        """Stable per-image coin flip based on SHA256(image_id)."""
        if self.random_flip_prob <= 0:
            return False
        h = hashlib.sha256(image_id.encode("utf-8")).digest()
        v = int.from_bytes(h[:4], byteorder="big") / 2**32  # [0,1)
        return v < float(self.random_flip_prob)

    def _decide_flip_mode(self, image_id: str, tags: List[str]) -> str:
        """
        Decide flipping policy: 'none' | 'random' | 'force'
        Respects flip list first; then applies safety veto; then p-coin.
        """
        if self.respect_flip_list:
            if image_id in self._never_flip_ids:
                return "none"
            if image_id in self._force_flip_ids:
                return "force"
        if self.random_flip_prob <= 0:
            return "none"
        if self.orientation_handler is not None:
            try:
                if self.orientation_handler.should_skip_flip(tags):
                    return "none"
            except Exception:
                pass
        return "random" if self._deterministic_coin(image_id) else "none"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self.failed_samples:
            return self._error_sample(idx, "Previously failed sample")

        if idx not in self.retry_counts:
            self.retry_counts[idx] = 0

        try:
            ann = self.items[idx]
            image_id = ann["image_id"]
            # Work on a copy of the tag list so we can safely modify it
            original_tags: List[str] = list(ann["tags"])
            tags_now: List[str] = original_tags
            # Decide whether to flip and adjust tags accordingly
            mode = self._decide_flip_mode(image_id, original_tags)
            flip_bit = False
            if mode != "none" and self.orientation_handler is not None:
                if mode == "force":
                    tags_now = [self.orientation_handler.swap_tag(t) for t in original_tags]
                    flip_bit = True
                else:
                    # Avoid double safety checks here; decision already made by _decide_flip_mode
                    tags_now, flipped = self.orientation_handler.swap_tags(original_tags, skip_safety_check=True)
                    flip_bit = bool(flipped)
            # Build the L2 key and mask key using the config hash and flip bit
            l2_key = self._l2_key(image_id, flipped=flip_bit)
            mask_key = self._l2_mask_key(image_id, flipped=flip_bit)

            # Try L2 cache first
            if self._l2_enabled:
                self._ensure_l2_reader()
                img_payload = self._l2_reader.get(l2_key) if self._l2_reader else None
                mask_payload = self._l2_reader.get(mask_key) if self._l2_reader else None
                if img_payload is not None:
                    try:
                        img_t = _tensor_from_bytes(img_payload)
                        # Verify cached shape matches expected resolution
                        if img_t.dim() != 3 or img_t.shape[-2] != int(self.image_size) or img_t.shape[-1] != int(self.image_size):
                            raise ValueError(
                                f"L2 cached tensor shape {img_t.shape} does not match expected {(3, self.image_size, self.image_size)}"
                            )
                    except Exception as e:
                        self.logger.warning(f"L2 cache decode failed for {image_id}: {e}")
                        img_t = None
                    if img_t is not None:
                        # Try to load an explicit mask; fall back to heuristic reconstruction
                        if mask_payload is not None:
                            try:
                                m = _tensor_from_bytes(mask_payload)
                                if m.dim() != 2 or m.shape[0] != int(self.image_size) or m.shape[1] != int(self.image_size):
                                    mask_payload = None
                                else:
                                    pmask = m.to(torch.bool)
                            except Exception:
                                mask_payload = None
                        if mask_payload is None:
                            pad_vec = torch.tensor(
                                [c / 255.0 for c in self.pad_color],
                                dtype=img_t.dtype,
                                device=img_t.device,
                            )
                            mean = torch.tensor(self.normalize_mean, dtype=img_t.dtype, device=img_t.device)
                            std = torch.tensor(self.normalize_std, dtype=img_t.dtype, device=img_t.device)
                            pad_norm = ((pad_vec - mean) / std).view(3, 1, 1)
                            with torch.no_grad():
                                pmask = torch.isclose(
                                    img_t,
                                    pad_norm,
                                    atol=PAD_MASK_ATOL,
                                    rtol=0.0,
                                ).all(dim=0)
                        # Use tags that already reflect the flip decision
                        tag_vec = self.vocab.encode_tags(tags_now)
                        rating_idx = _map_rating(ann.get("rating", "unknown"))
                        return {
                            "images": img_t,
                            "padding_mask": pmask.to(torch.bool),
                            "tag_labels": tag_vec,
                            "rating_labels": torch.tensor(rating_idx, dtype=torch.long),
                            "image_id": image_id,
                            "cached": True,
                            "error": False,
                            "error_reason": "",
                        }

            # Cache miss: load from disk (resolve under the JSON's shard folder)
            img_root = ann.get("dir", self.root)
            img_path = validate_image_path(Path(img_root), image_id)
            # Fully decode and correct EXIF while file is open
            with Image.open(img_path) as pil_img:
                pil_img.load()
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
                    rgba = pil_img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, tuple(self.pad_color))
                    alpha = rgba.getchannel("A")
                    bg.paste(rgba, mask=alpha)
                    pil = bg
                else:
                    pil = pil_img.convert("RGB")
            # 2) Letterbox to square and padding mask
            target = int(self.image_size)
            w, h = pil.size
            # Downscale-only letterbox: preserve aspect, never upscale
            ratio = min(target / float(w), target / float(h)) if (w > 0 and h > 0) else 1.0
            scale = min(1.0, ratio)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = pil.resize((max(1, nw), max(1, nh)), RESAMPLE_BILINEAR)
            canvas = Image.new("RGB", (target, target), tuple(self.pad_color))
            left = (target - resized.size[0]) // 2
            top = (target - resized.size[1]) // 2
            canvas.paste(resized, (left, top))
            pmask = torch.ones(target, target, dtype=torch.bool)
            pmask[top:top + resized.size[1], left:left + resized.size[0]] = False
            # Apply horizontal flip based on the pre-decided flip_bit
            if flip_bit:
                canvas = ImageOps.mirror(canvas)
                pmask = torch.flip(pmask, dims=[1])
            # Joint v2 transforms keep geometry aligned with mask when used
            if self.joint_transforms is not None and T is not None and tv_tensors is not None:
                img_tv = tv_tensors.Image(canvas)
                mask_tv = tv_tensors.Mask(pmask.to(torch.uint8))
                img_tv, mask_tv = self.joint_transforms(img_tv, mask_tv)
                img = T.ToTensor()(img_tv)
                if transforms is None:
                    raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                img = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img)
                pmask = mask_tv.to(torch.bool)
            else:
                # Fallback: color-only transforms permitted
                if self.transform:
                    try:
                        transformed = self.transform(canvas)
                        if transforms is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        img = transformed if isinstance(transformed, torch.Tensor) else transforms.ToTensor()(transformed)
                        img = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img)
                    except Exception:
                        if transforms is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        img = transforms.ToTensor()(canvas)
                        img = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img)
                else:
                    if transforms is None:
                        raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                    img = transforms.ToTensor()(canvas)
                    img = transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(img)

            # Encode labels (tags already account for flipping)
            tag_vec = self.vocab.encode_tags(tags_now)  # (V,)
            rating_idx = _map_rating(ann.get("rating", "unknown"))

            # Enqueue write (non-blocking)
            if self._l2_enabled and self._l2_writer_q is not None:
                try:
                    # Store the normalized image and explicit padding mask
                    self._l2_writer_q.put_nowait((l2_key, _tensor_to_bytes(img)))
                    self._l2_writer_q.put_nowait((mask_key, _tensor_to_bytes(pmask.to(torch.uint8))))
                except queue.Full:
                    now = time.time()
                    if (now - self._last_qfull_warn) > 5.0:
                        self._last_qfull_warn = now
                        self.logger.warning(
                            "L2 writer queue full; dropping cache write (rate-limited)"
                        )

            self.retry_counts[idx] = 0
            return {
                "images": img,
                "padding_mask": pmask,
                "tag_labels": tag_vec,
                "rating_labels": torch.tensor(rating_idx, dtype=torch.long),
                "image_id": image_id,
                "cached": False,
                "error": False,
                "error_reason": "",
            }

        except Exception as e:
            self.retry_counts[idx] += 1
            self.logger.warning(f"Failed to load sample {idx}: {e}")
            if self.retry_counts[idx] >= self.max_retries:
                self.failed_samples.add(idx)
                self.logger.error(f"Sample {idx} exceeded max retries, marking as failed")
                return self._error_sample(idx, str(e))
            return self._error_sample(idx, f"Temporary failure: {e}")
        finally:
            # Opportunistic telemetry: push orientation stats every 128 samples
            try:
                if self._stats_queue is not None and self.orientation_handler is not None:
                    self._samples_seen += 1
                    if (self._samples_seen & 127) == 0:
                        stats = self.orientation_handler.get_statistics()
                        payload = {
                            "flip_total": int(stats.get("total_flips", 0)),
                            "flip_safe": int(stats.get("safe_flips", 0)),
                            "flip_skipped_text": int(stats.get("blocked_by_text", 0)),
                            "flip_skipped_unmapped": int(stats.get("skipped_flips", 0)),
                            "flip_blocked_safety": int(stats.get("blocked_by_safety", 0)),
                        }
                        self._stats_queue.put_nowait(payload)
            except Exception:
                pass

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
            "cached": False,
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

    # ---- L2 fail-fast guard --------------------------------------------
    if bool(getattr(data_config, "l2_cache_enabled", False)):
        max_gb = getattr(data_config, "l2_max_size_gb", None)
        max_bytes = getattr(data_config, "l2_map_size_bytes", None)
        ok = False
        try:
            ok = (max_gb is not None and float(max_gb) > 0.0) or (max_bytes is not None and int(max_bytes) > 0)
        except Exception:
            ok = False
        if not ok:
            raise ValueError("data.l2_cache_enabled=true requires a positive l2_max_size_gb or l2_map_size_bytes.")

    # Map size for LMDB (bytes) — accept either GB or direct bytes
    _size_bytes_cfg = getattr(data_config, "l2_map_size_bytes", None)
    if _size_bytes_cfg is not None and int(_size_bytes_cfg) > 0:
        map_size_bytes = int(_size_bytes_cfg)
    else:
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

    # --- Orientation handler / flip list wiring ---
    random_flip_prob = float(getattr(data_config, "random_flip_prob", 0.0))
    orientation_map_path = getattr(data_config, "orientation_map_path", None)
    if isinstance(orientation_map_path, str) and orientation_map_path:
        orientation_map_path = Path(orientation_map_path)
    flip_overrides_path = getattr(data_config, "flip_overrides_path", None)
    _handler = None
    try:
        if random_flip_prob > 0 and OrientationHandler is not None:
            _handler = OrientationHandler(
                mapping_file=orientation_map_path if orientation_map_path else None,
                random_flip_prob=random_flip_prob,
                strict_mode=bool(getattr(data_config, "strict_orientation_validation", False)),
                safety_mode=str(getattr(data_config, "orientation_safety_mode", "conservative")),
                skip_unmapped=bool(getattr(data_config, "skip_unmapped", False)),
            )
    except Exception as e:
        logger.warning(f"OrientationHandler init failed; flips disabled: {e}")

    if manifest_train.exists() and manifest_val.exists() and images_dir.exists():
        # Manifest mode (back-compat)
        # Manifest datasets are not orientation-aware. If flips are enabled, warn and disable them.
        if float(getattr(data_config, "random_flip_prob", 0.0) or 0.0) > 0.0:
            logger.warning(
                "random_flip_prob > 0 with manifest dataset; disabling orientation-aware flips (manifest is non-orientation-aware)."
            )
            try:
                setattr(data_config, "random_flip_prob", 0.0)
            except Exception:
                pass
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
        # Sidecar JSON mode: scan per-image *.json recursively (shard-aware)
        logger.info("Manifest not found; entering sidecar JSON mode (scanning .json next to images)")

        all_jsons = sorted(root.rglob("*.json")) if root.exists() else []
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
            random_flip_prob=random_flip_prob,
            orientation_handler=_handler,
            flip_overrides_path=flip_overrides_path,
            stats_queue=getattr(data_config, "stats_queue", None),
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
            random_flip_prob=0.0,          # keep val deterministic
            orientation_handler=_handler,  # still needed to encode swapped tags if you ever TTA
            flip_overrides_path=None,
            stats_queue=getattr(data_config, "stats_queue", None),
        )

    # ---- Samplers for distributed --------------------------------------
    train_sampler = None
    val_sampler = None
    if distributed:
        # NOTE: when sampler is set, DataLoader.shuffle must be False.
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=True,
            drop_last=bool(getattr(data_config, "drop_last", False)),
            seed=int(seed) if seed is not None else 0,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=False,
            drop_last=False,
        )
    # --------------------------------------------------------------------

    # DataLoaders
    def _dl_kwargs(cfg, *, shuffle: bool, drop_last: bool):
        kw = dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=getattr(cfg, "pin_memory", False),
            drop_last=drop_last,
            shuffle=shuffle,
        )
        # Only use multiprocessing knobs when workers > 0
        if int(getattr(cfg, "num_workers", 0) or 0) > 0:
            if getattr(cfg, "prefetch_factor", None) is not None:
                kw["prefetch_factor"] = cfg.prefetch_factor
            kw["persistent_workers"] = bool(getattr(cfg, "persistent_workers", False))
        return kw

    _train_kw = _dl_kwargs(
        data_config,
        shuffle=(train_sampler is None),
        drop_last=bool(getattr(data_config, "drop_last", False)),
    )
    if train_sampler is not None:
        _train_kw["sampler"] = train_sampler
    # Attach logging QueueHandler in workers if a queue is provided
    log_queue = kwargs.get("log_queue")
    if log_queue is not None:
        _train_kw["worker_init_fn"] = _make_worker_init(log_queue)
    train_loader = DataLoader(train_ds, **_train_kw)

    val_batch = (
        validation_config.dataloader.batch_size
        if hasattr(validation_config, "dataloader")
        else data_config.batch_size
    )
    # Build kwargs for val loader separately to honor val batch size
    _val_kw = _dl_kwargs(data_config, shuffle=False, drop_last=False)
    _val_kw["batch_size"] = val_batch
    if val_sampler is not None:
        _val_kw["sampler"] = val_sampler
    if log_queue is not None:
        _val_kw["worker_init_fn"] = _make_worker_init(log_queue)
    val_loader = DataLoader(val_ds, **_val_kw)

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
