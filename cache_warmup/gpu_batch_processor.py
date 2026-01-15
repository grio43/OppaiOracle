"""
GPU-accelerated batch processor for sidecar cache warmup.

This module provides GPU-accelerated image preprocessing with dynamic batch sizing
that automatically scales to utilize target VRAM usage (default: 90% of available).
"""

import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class VRAMMonitor:
    """Monitor and track GPU VRAM usage."""

    def __init__(self, device_id: int = 0, target_vram_gb: float = 31.5, target_util: float = 0.9):
        """
        Initialize VRAM monitor.

        Args:
            device_id: CUDA device ID
            target_vram_gb: Target VRAM capacity in GB
            target_util: Target utilization (0.9 = 90%)
        """
        self.device = torch.device(f'cuda:{device_id}')
        self.device_id = device_id
        self.target_vram_bytes = int(target_vram_gb * (1024**3))
        self.target_util = target_util
        self.target_vram_used = int(self.target_vram_bytes * target_util)

        # Verify GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Get actual GPU memory
        props = torch.cuda.get_device_properties(device_id)
        self.actual_vram_bytes = props.total_memory

        # Validate and clamp target VRAM to actual if needed (prevents OOM)
        if self.target_vram_bytes > self.actual_vram_bytes:
            logger.warning(
                f"target_vram_gb ({target_vram_gb:.1f}GB) exceeds actual GPU memory "
                f"({self.actual_vram_bytes / (1024**3):.1f}GB). Clamping to actual memory."
            )
            self.target_vram_bytes = self.actual_vram_bytes
            self.target_vram_used = int(self.target_vram_bytes * target_util)

        # Effective limit is the smaller of target and actual (safety measure)
        self.effective_vram_bytes = min(self.target_vram_bytes, self.actual_vram_bytes)

        logger.info(f"VRAM Monitor initialized:")
        logger.info(f"  Target VRAM: {target_vram_gb:.2f} GB")
        logger.info(f"  Actual VRAM: {self.actual_vram_bytes / (1024**3):.2f} GB")
        logger.info(f"  Effective VRAM: {self.effective_vram_bytes / (1024**3):.2f} GB")
        logger.info(f"  Target utilization: {target_util * 100:.1f}%")
        logger.info(f"  Target used: {self.target_vram_used / (1024**3):.2f} GB")

    def get_vram_stats(self) -> Dict[str, float]:
        """Get current VRAM statistics."""
        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        # Use effective VRAM (min of target and actual) to avoid false sense of headroom
        free = self.effective_vram_bytes - reserved
        utilization = reserved / self.effective_vram_bytes

        return {
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'free_gb': free / (1024**3),
            'utilization': utilization,
            'effective_vram_gb': self.effective_vram_bytes / (1024**3),
            'target_free_gb': (self.effective_vram_bytes * (1 - self.target_util)) / (1024**3)
        }

    def get_utilization(self) -> float:
        """Get current VRAM utilization (0.0 to 1.0)."""
        reserved = torch.cuda.memory_reserved(self.device_id)
        # Use effective VRAM (min of target and actual) for accurate utilization
        return reserved / self.effective_vram_bytes

    def has_room_for_increase(self) -> bool:
        """Check if we have room to increase batch size."""
        stats = self.get_vram_stats()
        # We have room if we're below 80% of target
        return stats['utilization'] < (self.target_util * 0.8)

    def is_over_target(self) -> bool:
        """Check if we're over target utilization."""
        stats = self.get_vram_stats()
        # Over target if we exceed 95% of target
        return stats['utilization'] > (self.target_util * 0.95)

    def clear_cache(self):
        """Clear GPU cache."""
        torch.cuda.empty_cache()


class DynamicBatchSizer:
    """Dynamically adjust batch size based on VRAM usage."""

    def __init__(self, initial_batch_size: int = 32, min_batch_size: int = 8, max_batch_size: int = 2048):
        """
        Initialize dynamic batch sizer.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size (default: 2048 for RTX 5090)
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.stable_count = 0  # Count of stable batches before increasing
        self.increase_threshold = 3  # Process 3 stable batches before increasing

        logger.info(f"Dynamic Batch Sizer initialized:")
        logger.info(f"  Initial batch size: {initial_batch_size}")
        logger.info(f"  Range: [{min_batch_size}, {max_batch_size}]")

    def adjust(self, vram_monitor: VRAMMonitor) -> int:
        """
        Adjust batch size based on current VRAM usage.

        Args:
            vram_monitor: VRAM monitor instance

        Returns:
            New batch size
        """
        old_size = self.current_batch_size

        if vram_monitor.is_over_target():
            # Over target - reduce batch size immediately
            self.current_batch_size = max(
                int(self.current_batch_size * 0.5),
                self.min_batch_size
            )
            self.stable_count = 0
            logger.warning(f"VRAM over target - reducing batch size: {old_size} -> {self.current_batch_size}")

        elif vram_monitor.has_room_for_increase():
            # Has room - increase batch size gradually
            self.stable_count += 1

            if self.stable_count >= self.increase_threshold:
                new_size = min(
                    int(self.current_batch_size * 1.25),
                    self.max_batch_size
                )
                # Only log and update if we actually increased
                if new_size > self.current_batch_size:
                    self.current_batch_size = new_size
                    logger.info(f"VRAM has room - increasing batch size: {old_size} -> {self.current_batch_size}")
                self.stable_count = 0
        else:
            # In target range - keep stable
            self.stable_count += 1

        return self.current_batch_size

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size


class AsyncImagePreloader:
    """
    Async image preloader for GPU batch warmup.

    Uses ThreadPoolExecutor to load the NEXT batch of images while the
    current batch is being processed on GPU and written to disk.

    Memory management:
    - Tracks estimated RAM usage per preloaded batch
    - Dynamically adjusts queue depth based on available RAM
    - Automatically clears completed futures to prevent leaks
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_queue_depth: int = 4,  # Increased from 2 to maintain GPU saturation
        ram_headroom_gb: float = 8.0,
        image_size: int = 1024,
        pad_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Initialize async preloader.

        Args:
            max_workers: Number of parallel loading threads
            max_queue_depth: Maximum number of batches to preload ahead (default 4 for GPU saturation)
            ram_headroom_gb: Minimum free RAM to maintain (safety margin)
            image_size: Target image size for memory estimation
            pad_color: RGB tuple for transparency compositing
        """
        self.max_workers = max_workers
        self.max_queue_depth = max_queue_depth
        self.ram_headroom_gb = ram_headroom_gb
        self.image_size = image_size
        self.pad_color_int = pad_color

        # Thread pool for parallel image loading
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="preload"
        )

        # Pending futures: batch_id -> List[(idx, Future)]
        self.futures: Dict[int, List[Tuple[int, Future]]] = {}
        self._lock = threading.Lock()
        self._shutdown = False

        logger.info(f"AsyncImagePreloader initialized:")
        logger.info(f"  max_workers: {max_workers}")
        logger.info(f"  max_queue_depth: {max_queue_depth}")
        logger.info(f"  ram_headroom_gb: {ram_headroom_gb}")

    def _load_single_image(
        self,
        path: str,
        idx: int,
    ) -> Tuple[int, Optional[torch.Tensor], Optional[Tuple[int, int]]]:
        """
        Load a single image (runs in thread pool).

        Returns:
            (idx, tensor or None, (w, h) or None)
        """
        img = None
        background = None
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)

            # Handle transparency (matches dataset_loader.py for consistency)
            if img.mode in ('RGBA', 'LA') or ('transparency' in img.info):
                background = Image.new('RGB', img.size, self.pad_color_int)
                # Convert indexed/grayscale+alpha to RGBA for proper compositing
                if img.mode not in ('RGBA',):
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1])
                # Close original image, use background
                img.close()
                img = background
                background = None  # Prevent double-close
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            original_size = img.size

            # Convert to tensor [0, 1] - same as load_and_prepare_images()
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW

            return (idx, img_tensor, original_size)

        except Exception as e:
            logger.error(f"Failed to preload image {path}: {e}")
            return (idx, None, None)
        finally:
            # Explicitly close PIL Images to release file handles
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass
            if background is not None:
                try:
                    background.close()
                except Exception:
                    pass

    def estimate_batch_memory_mb(self, batch_size: int) -> float:
        """
        Estimate RAM usage for a preloaded batch.

        Returns memory in MB based on:
        - image_size x image_size x 3 channels x float32 (4 bytes)
        - ~20% overhead for PIL Image objects and numpy arrays
        """
        bytes_per_image = self.image_size * self.image_size * 3 * 4
        return (batch_size * bytes_per_image * 1.2) / (1024 * 1024)

    def get_available_ram_gb(self) -> float:
        """Get available system RAM in GB using psutil."""
        if not HAS_PSUTIL:
            # Conservative fallback: assume 8GB available
            return 8.0
        return psutil.virtual_memory().available / (1024**3)

    def can_queue_more(self, batch_size: int) -> bool:
        """
        Check if we have RAM headroom to queue another batch.

        Returns True if:
        - pending_batches < max_queue_depth
        - Estimated memory for pending + 1 batch < available RAM - headroom
        """
        with self._lock:
            pending = len(self.futures)

        if pending >= self.max_queue_depth:
            return False

        # Check RAM availability
        available = self.get_available_ram_gb()
        estimated_mb = self.estimate_batch_memory_mb(batch_size) * (pending + 1)
        estimated_gb = estimated_mb / 1024

        return available - estimated_gb > self.ram_headroom_gb

    def submit_batch(self, batch_id: int, image_paths: List[str]) -> None:
        """
        Submit a batch for async preloading.

        Args:
            batch_id: Unique batch identifier (for ordering)
            image_paths: Paths to preload
        """
        with self._lock:
            if self._shutdown:
                return

            if batch_id in self.futures:
                return  # Already submitted

            # Submit individual image loading tasks (not _load_batch as a whole)
            # This avoids thread starvation from nested executor submissions
            image_futures = []
            for idx, path in enumerate(image_paths):
                img_future = self.executor.submit(self._load_single_image, path, idx)
                image_futures.append((idx, img_future))

            # Store futures list for later collection
            self.futures[batch_id] = image_futures

    def get_preloaded_batch(
        self,
        batch_id: int,
        timeout: float = 10.0,  # Reduced from 60s - fail fast on slow images
        per_image_timeout: float = 5.0,  # Individual image timeout
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], List[int]]:
        """
        Get preloaded batch, waiting if necessary.

        OPTIMIZED: Uses concurrent.futures.wait() for parallel completion checking
        instead of sequential blocking. Reduced timeouts for faster failure recovery.

        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum total wait time for all images (default 10s)
            per_image_timeout: Timeout for individual slow images (default 5s)

        Returns:
            Same format as load_and_prepare_images():
            (images: List[Tensor], original_sizes: List[(w,h)], failed_indices: List[int])
        """
        from concurrent.futures import wait, FIRST_COMPLETED, TimeoutError as FuturesTimeoutError

        # Check for shutdown before proceeding
        if self._shutdown:
            raise RuntimeError("Preloader has been shut down")

        with self._lock:
            image_futures = self.futures.pop(batch_id, None)

        if image_futures is None:
            raise ValueError(f"Batch {batch_id} was not submitted for preloading")

        # Collect results from individual image futures
        batch_size = len(image_futures)
        images: List[Optional[torch.Tensor]] = [None] * batch_size
        original_sizes: List[Optional[Tuple[int, int]]] = [None] * batch_size
        failed_indices: List[int] = []

        # Build mapping of future -> (idx, original_idx)
        future_to_idx = {future: (i, idx) for i, (idx, future) in enumerate(image_futures)}
        pending = set(future_to_idx.keys())
        start_time = time.time()

        # Process futures as they complete (parallel waiting)
        while pending and (time.time() - start_time) < timeout and not self._shutdown:
            remaining_time = max(0.1, timeout - (time.time() - start_time))
            done, pending = wait(pending, timeout=min(remaining_time, per_image_timeout), return_when=FIRST_COMPLETED)

            for future in done:
                i, idx = future_to_idx[future]
                try:
                    result_idx, tensor, size = future.result(timeout=0)  # Already done, no wait
                    if tensor is not None:
                        images[result_idx] = tensor
                        original_sizes[result_idx] = size
                    else:
                        failed_indices.append(result_idx)
                        images[result_idx] = torch.zeros(3, self.image_size, self.image_size)
                        original_sizes[result_idx] = (self.image_size, self.image_size)
                except Exception as e:
                    logger.warning(f"Preload failed for image {idx}: {e}")
                    failed_indices.append(idx)
                    images[idx] = torch.zeros(3, self.image_size, self.image_size)
                    original_sizes[idx] = (self.image_size, self.image_size)

        # Handle any remaining pending futures (timeout)
        for future in pending:
            i, idx = future_to_idx[future]
            logger.warning(f"Image {idx} preload timed out after {timeout}s")
            failed_indices.append(idx)
            images[idx] = torch.zeros(3, self.image_size, self.image_size)
            original_sizes[idx] = (self.image_size, self.image_size)
            # Cancel the future to free resources
            future.cancel()

        return images, original_sizes, failed_indices

    def get_pending_count(self) -> int:
        """Return number of batches currently being preloaded."""
        with self._lock:
            return len(self.futures)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown executor and cleanup resources.

        Thread-safe shutdown that coordinates with get_preloaded_batch().
        """
        # Set shutdown flag first to signal any waiting operations
        self._shutdown = True

        # Cancel any pending futures to unblock waiting threads
        with self._lock:
            for batch_futures in self.futures.values():
                for idx, future in batch_futures:
                    future.cancel()
            self.futures.clear()

        # Now shutdown executor (wait for any in-flight operations)
        self.executor.shutdown(wait=wait)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup on exceptions."""
        self.shutdown(wait=True)
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor fallback for cleanup."""
        if not self._shutdown:
            try:
                self.shutdown(wait=False)
            except (RuntimeError, TypeError):
                pass  # Interpreter may be shutting down


class GPUBatchPreprocessor:
    """GPU-accelerated batch image preprocessing."""

    def __init__(
        self,
        image_size: int,
        pad_color: Tuple[int, int, int],
        normalize_mean: Tuple[float, float, float],
        normalize_std: Tuple[float, float, float],
        device_id: int = 0,
        cpu_bf16_cache_pipeline: bool = True
    ):
        """
        Initialize GPU batch preprocessor.

        Args:
            image_size: Target image size (square)
            pad_color: RGB padding color
            normalize_mean: Normalization mean
            normalize_std: Normalization std
            device_id: CUDA device ID
            cpu_bf16_cache_pipeline: Convert to bfloat16
        """
        self.image_size = image_size
        self.pad_color_int = tuple(pad_color)  # Keep original int values for PIL compositing
        self.pad_color = torch.tensor(pad_color, dtype=torch.float32) / 255.0
        self.device = torch.device(f'cuda:{device_id}')
        self.cpu_bf16 = cpu_bf16_cache_pipeline
        # Check compute capability instead of torch.cuda.is_bf16_supported() for compatibility
        props = torch.cuda.get_device_properties(device_id)
        if self.cpu_bf16 and props.major < 8:
            raise RuntimeError("bfloat16 requires Ampere GPU or newer (compute capability 8.0+)")
        self.gpu_dtype = torch.bfloat16 if self.cpu_bf16 else torch.float32

        # Create normalization transform on GPU
        self.normalize = transforms.Normalize(
            mean=normalize_mean,
            std=normalize_std
        )

        # Note: pad_color is applied BEFORE normalization, so the entire tensor
        # (including pad regions) gets normalized uniformly at line 338.
        # This matches the CPU path in dataset_loader.py.

        logger.info(f"GPU Batch Preprocessor initialized:")
        logger.info(f"  Image size: {image_size}x{image_size}")
        logger.info(f"  Pad color: {pad_color}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  BF16 cache: {cpu_bf16_cache_pipeline}")

    def load_and_prepare_images(
        self,
        image_paths: List[str]
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], List[int]]:
        """
        Load images from disk and prepare for GPU processing.

        Args:
            image_paths: List of image file paths

        Returns:
            Tuple of (list of image tensors, list of original sizes, list of failed indices)
            Failed indices indicate which images in the batch failed to load and have dummy data.
        """
        images = []
        original_sizes = []
        failed_indices = []

        for idx, path in enumerate(image_paths):
            img = None
            background = None
            try:
                # Load image with EXIF correction
                img = Image.open(path)
                img = ImageOps.exif_transpose(img)

                # Handle transparency (matches dataset_loader.py for consistency)
                if img.mode in ('RGBA', 'LA') or ('transparency' in img.info):
                    # Use original int pad_color to avoid float→int precision loss
                    background = Image.new('RGB', img.size, self.pad_color_int)
                    # Convert indexed/grayscale+alpha to RGBA for proper compositing
                    if img.mode not in ('RGBA',):
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1])
                    # Close original image, use background
                    img.close()
                    img = background
                    background = None  # Prevent double-close
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Store original size
                original_sizes.append(img.size)

                # Convert to tensor [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
                images.append(img_tensor)

            except FileNotFoundError:
                logger.error(f"Image file not found: {path}")
                images.append(torch.zeros(3, self.image_size, self.image_size))
                original_sizes.append((self.image_size, self.image_size))
                failed_indices.append(idx)
            except Image.DecompressionBombError:
                logger.error(f"Image too large (decompression bomb protection): {path}")
                images.append(torch.zeros(3, self.image_size, self.image_size))
                original_sizes.append((self.image_size, self.image_size))
                failed_indices.append(idx)
            except (IOError, OSError) as e:
                logger.error(f"I/O error loading image {path}: {e}")
                images.append(torch.zeros(3, self.image_size, self.image_size))
                original_sizes.append((self.image_size, self.image_size))
                failed_indices.append(idx)
            except Exception as e:
                # Catch-all for unexpected errors (e.g., corrupted images, memory issues)
                logger.error(f"Unexpected error loading image {path}: {type(e).__name__}: {e}")
                images.append(torch.zeros(3, self.image_size, self.image_size))
                original_sizes.append((self.image_size, self.image_size))
                failed_indices.append(idx)
            finally:
                # Explicitly close PIL Images to release file handles
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass
                if background is not None:
                    try:
                        background.close()
                    except Exception:
                        pass

        return images, original_sizes, failed_indices

    def letterbox_resize_gpu(
        self,
        images: List[torch.Tensor],
        original_sizes: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Apply letterbox resize on GPU in batch.

        OPTIMIZED: Uses batched H2D transfer instead of per-image transfers.
        Images are padded to uniform size on CPU, transferred as a single batch,
        then processed on GPU. This reduces H2D transfer overhead significantly.

        Args:
            images: List of image tensors [C, H, W]
            original_sizes: List of (width, height) tuples

        Returns:
            Tuple of (resized images [B, C, H, W], padding masks [B, H, W], invalid_indices)
            invalid_indices contains batch indices where image dimensions were invalid
        """
        batch_size = len(images)
        target_size = self.image_size
        invalid_indices: List[int] = []

        # Pre-calculate resize parameters and validate dimensions
        resize_params: List[Optional[Tuple[int, int, int, int, int, int]]] = []
        max_h, max_w = 0, 0

        for i, (orig_w, orig_h) in enumerate(original_sizes):
            if orig_w <= 0 or orig_h <= 0:
                logger.warning(f"Invalid image dimensions (w={orig_w}, h={orig_h}) at index {i}, marking as failed")
                invalid_indices.append(i)
                resize_params.append(None)
                continue

            # Calculate resize dimensions (downscale only, preserve aspect ratio)
            scale = min(target_size / orig_w, target_size / orig_h, 1.0)
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            pad_top = (target_size - new_h) // 2
            pad_left = (target_size - new_w) // 2

            resize_params.append((new_w, new_h, pad_top, pad_left, orig_w, orig_h))
            max_h = max(max_h, orig_h)
            max_w = max(max_w, orig_w)

        # Create output tensors on GPU
        output_images = torch.zeros(
            batch_size, 3, target_size, target_size,
            dtype=self.gpu_dtype,
            device=self.device
        )

        # Fill with pad color (vectorized)
        for c in range(3):
            output_images[:, c, :, :] = self.pad_color[c]

        # Padding mask convention (matches dataset_loader.py for model compatibility):
        #   True  = padding pixel (not real image content)
        #   False = real image content
        padding_masks = torch.ones(
            batch_size, target_size, target_size,
            dtype=torch.bool,
            device=self.device
        )

        # Early exit if all images are invalid
        if max_h == 0 or max_w == 0:
            return output_images, padding_masks, invalid_indices

        # OPTIMIZATION: Pad all images to uniform size on CPU, then batch transfer
        # This replaces N separate H2D transfers with 1 batched transfer
        padded_batch = torch.zeros(batch_size, 3, max_h, max_w, dtype=torch.float32)

        # Fill padded batch with images (CPU operation)
        for i, (img, params) in enumerate(zip(images, resize_params)):
            if params is None:
                continue  # Invalid image, leave as zeros
            h, w = img.shape[1], img.shape[2]
            padded_batch[i, :, :h, :w] = img

        # Single batched H2D transfer (major optimization)
        batch_gpu = padded_batch.to(device=self.device, dtype=self.gpu_dtype, non_blocking=True)

        # Process each image on GPU (resize and place in output)
        for i, params in enumerate(resize_params):
            if params is None:
                continue  # Invalid image

            new_w, new_h, pad_top, pad_left, orig_w, orig_h = params

            # Extract original image region from padded batch
            img_gpu = batch_gpu[i:i+1, :, :orig_h, :orig_w]  # [1, C, H, W]

            # Resize if needed
            if (new_h, new_w) != (orig_h, orig_w):
                img_resized = F.interpolate(
                    img_gpu,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                img_resized = img_gpu

            # Place resized image in center of output
            output_images[i, :, pad_top:pad_top+new_h, pad_left:pad_left+new_w] = img_resized.squeeze(0)

            # Mark non-padded region in mask (False = real content)
            padding_masks[i, pad_top:pad_top+new_h, pad_left:pad_left+new_w] = False

        # Free intermediate GPU memory
        del batch_gpu

        return output_images, padding_masks, invalid_indices

    def process_batch(
        self,
        image_paths: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Process a batch of images on GPU.

        Args:
            image_paths: List of image file paths

        Returns:
            Tuple of (processed images [B, C, H, W], padding masks [B, H, W], failed_indices) on CPU
            failed_indices contains batch indices where image loading or processing failed
        """
        # Load images on CPU
        images, original_sizes, failed_indices = self.load_and_prepare_images(image_paths)

        # Apply letterbox resize on GPU
        images_gpu, masks_gpu, invalid_indices = self.letterbox_resize_gpu(images, original_sizes)

        # Merge invalid dimension indices with load failures (avoid duplicates)
        all_failed = set(failed_indices)
        all_failed.update(invalid_indices)
        failed_indices = sorted(all_failed)

        # Apply normalization on GPU (tensor is already in self.gpu_dtype from letterbox_resize_gpu)
        images_normalized = self.normalize(images_gpu)

        # Move back to CPU for cache writing
        images_cpu = images_normalized.cpu()
        masks_cpu = masks_gpu.cpu()

        return images_cpu, masks_cpu, failed_indices

    def process_batch_from_preloaded(
        self,
        images: List[torch.Tensor],
        original_sizes: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        GPU processing for pre-loaded images (skips load_and_prepare_images).

        Args:
            images: Pre-loaded image tensors [C, H, W] from AsyncImagePreloader
            original_sizes: Original (width, height) for each image

        Returns:
            Tuple of (processed images [B, C, H, W], padding masks [B, H, W], invalid_indices) on CPU
            invalid_indices contains batch indices where image dimensions were invalid
        """
        # Apply letterbox resize on GPU
        images_gpu, masks_gpu, invalid_indices = self.letterbox_resize_gpu(images, original_sizes)

        # Apply normalization on GPU
        images_normalized = self.normalize(images_gpu)

        # Move back to CPU for cache writing
        images_cpu = images_normalized.cpu()
        masks_cpu = masks_gpu.cpu()

        return images_cpu, masks_cpu, invalid_indices


class GPUBatchProcessor:
    """Main GPU batch processor with dynamic sizing and VRAM monitoring."""

    def __init__(
        self,
        image_size: int,
        pad_color: Tuple[int, int, int],
        normalize_mean: Tuple[float, float, float],
        normalize_std: Tuple[float, float, float],
        device_id: int = 0,
        target_vram_gb: float = 31.5,
        target_vram_util: float = 0.9,
        initial_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 2048,
        cpu_bf16_cache_pipeline: bool = True,
        # Preload settings
        enable_preload: bool = True,
        preload_workers: int = 4,
        preload_queue_depth: int = 2,
        preload_ram_headroom_gb: float = 8.0,
    ):
        """
        Initialize GPU batch processor.

        Args:
            image_size: Target image size
            pad_color: RGB padding color
            normalize_mean: Normalization mean
            normalize_std: Normalization std
            device_id: CUDA device ID
            target_vram_gb: Target VRAM capacity
            target_vram_util: Target utilization
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size (default: 8)
            max_batch_size: Maximum batch size (default: 2048 for RTX 5090)
            cpu_bf16_cache_pipeline: Convert to bfloat16
            enable_preload: Enable async image preloading
            preload_workers: Number of threads for async preloading
            preload_queue_depth: Max batches to preload ahead
            preload_ram_headroom_gb: Minimum free RAM to maintain
        """
        self.vram_monitor = VRAMMonitor(device_id, target_vram_gb, target_vram_util)
        self.batch_sizer = DynamicBatchSizer(initial_batch_size, min_batch_size=min_batch_size, max_batch_size=max_batch_size)
        self.preprocessor = GPUBatchPreprocessor(
            image_size, pad_color, normalize_mean, normalize_std,
            device_id, cpu_bf16_cache_pipeline
        )

        # Initialize async preloader if enabled
        self.preloader: Optional[AsyncImagePreloader] = None
        if enable_preload:
            self.preloader = AsyncImagePreloader(
                max_workers=preload_workers,
                max_queue_depth=preload_queue_depth,
                ram_headroom_gb=preload_ram_headroom_gb,
                image_size=image_size,
                pad_color=pad_color,
            )

        logger.info("GPU Batch Processor fully initialized")
        if enable_preload:
            logger.info("  Async preloading: ENABLED")

    def get_current_batch_size(self) -> int:
        """Get current batch size."""
        return self.batch_sizer.get_batch_size()

    def get_vram_stats(self) -> Dict[str, float]:
        """Get current VRAM stats."""
        return self.vram_monitor.get_vram_stats()

    def adjust_batch_size(self):
        """Adjust batch size based on VRAM usage."""
        return self.batch_sizer.adjust(self.vram_monitor)

    def process_batch(
        self,
        image_paths: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Process a batch of images.

        Args:
            image_paths: List of image paths

        Returns:
            Tuple of (images, masks, failed_indices) on CPU
            failed_indices contains batch indices where image loading failed (dummy data inserted)
        """
        return self.preprocessor.process_batch(image_paths)

    def has_preloader(self) -> bool:
        """Check if async preloader is enabled."""
        return self.preloader is not None

    def submit_preload(self, batch_id: int, image_paths: List[str]) -> bool:
        """
        Submit a batch for async preloading.

        Args:
            batch_id: Unique batch identifier
            image_paths: Paths to preload

        Returns:
            True if batch was queued, False if preloader disabled or at capacity
        """
        if self.preloader is None:
            return False
        if not self.preloader.can_queue_more(len(image_paths)):
            return False
        self.preloader.submit_batch(batch_id, image_paths)
        return True

    def process_batch_preloaded(
        self,
        batch_id: int,
        timeout: float = 60.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Process a batch using pre-loaded images.

        Args:
            batch_id: Batch ID that was previously submitted via submit_preload()
            timeout: Max seconds to wait for preload to complete

        Returns:
            Same as process_batch(): (images, masks, failed_indices)
        """
        if self.preloader is None:
            raise RuntimeError("Preloader not enabled")

        images, original_sizes, preload_failed = self.preloader.get_preloaded_batch(
            batch_id, timeout
        )

        images_cpu, masks_cpu, invalid_indices = self.preprocessor.process_batch_from_preloaded(
            images, original_sizes
        )

        # Merge preload failures with invalid dimension failures
        all_failed = set(preload_failed)
        all_failed.update(invalid_indices)
        failed_indices = sorted(all_failed)

        return images_cpu, masks_cpu, failed_indices

    def get_preload_pending_count(self) -> int:
        """Get number of batches currently being preloaded."""
        if self.preloader is None:
            return 0
        return self.preloader.get_pending_count()

    def clear_cache(self):
        """Clear GPU cache and shutdown preloader.

        Performs thorough cleanup of GPU resources including:
        - Shutting down async preloader threads
        - Clearing CUDA memory cache
        - Logging VRAM stats before cleanup for debugging
        """
        import gc

        # Log VRAM stats before cleanup for debugging
        try:
            stats = self.get_vram_stats()
            logger.debug(
                f"GPU cleanup: VRAM before clear - "
                f"allocated={stats['allocated_gb']:.2f}GB, "
                f"reserved={stats['reserved_gb']:.2f}GB"
            )
        except Exception:
            pass  # Don't fail cleanup if stats fail

        # Shutdown preloader first to stop any pending operations
        if self.preloader is not None:
            self.preloader.shutdown(wait=False)
            self.preloader = None

        # Force garbage collection before clearing CUDA cache
        gc.collect()

        # Clear CUDA cache
        self.vram_monitor.clear_cache()

        # Synchronize to ensure all GPU operations complete
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                # Log sync failures - these can indicate GPU driver issues or memory corruption
                logger.warning(f"CUDA synchronize failed during cache clear: {e}. GPU may be in inconsistent state.")
