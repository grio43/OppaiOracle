"""
GPU-accelerated batch processor for sidecar cache warmup.

This module provides GPU-accelerated image preprocessing with dynamic batch sizing
that automatically scales to utilize target VRAM usage (default: 90% of available).
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

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

        logger.info(f"VRAM Monitor initialized:")
        logger.info(f"  Target VRAM: {target_vram_gb:.2f} GB")
        logger.info(f"  Actual VRAM: {self.actual_vram_bytes / (1024**3):.2f} GB")
        logger.info(f"  Target utilization: {target_util * 100:.1f}%")
        logger.info(f"  Target used: {self.target_vram_used / (1024**3):.2f} GB")

    def get_vram_stats(self) -> Dict[str, float]:
        """Get current VRAM statistics."""
        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        free = self.target_vram_bytes - reserved
        utilization = reserved / self.target_vram_bytes

        return {
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'free_gb': free / (1024**3),
            'utilization': utilization,
            'target_free_gb': (self.target_vram_bytes * (1 - self.target_util)) / (1024**3)
        }

    def get_utilization(self) -> float:
        """Get current VRAM utilization (0.0 to 1.0)."""
        reserved = torch.cuda.memory_reserved(self.device_id)
        return reserved / self.target_vram_bytes

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
        if self.cpu_bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 GPU preprocessing requested but CUDA device does not support bf16.")
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
            try:
                # Load image with EXIF correction
                img = Image.open(path)
                img = ImageOps.exif_transpose(img)

                # Handle transparency
                if img.mode == 'RGBA' or img.mode == 'LA':
                    # Use original int pad_color to avoid float→int precision loss
                    background = Image.new('RGB', img.size, self.pad_color_int)
                    if img.mode == 'LA':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Store original size
                original_sizes.append(img.size)

                # Convert to tensor [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
                images.append(img_tensor)

            except Exception as e:
                logger.error(f"Failed to load image {path}: {e}")
                # Create dummy image with correct target size
                images.append(torch.zeros(3, self.image_size, self.image_size))
                original_sizes.append((self.image_size, self.image_size))
                failed_indices.append(idx)

        return images, original_sizes, failed_indices

    def letterbox_resize_gpu(
        self,
        images: List[torch.Tensor],
        original_sizes: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply letterbox resize on GPU in batch.

        Args:
            images: List of image tensors [C, H, W]
            original_sizes: List of (width, height) tuples

        Returns:
            Tuple of (resized images [B, C, H, W], padding masks [B, H, W])
        """
        batch_size = len(images)
        target_size = self.image_size

        # Create output tensors on GPU
        output_images = torch.zeros(
            batch_size, 3, target_size, target_size,
            dtype=self.gpu_dtype,
            device=self.device
        )

        # Fill with pad color
        for c in range(3):
            output_images[:, c, :, :] = self.pad_color[c]

        # Create padding masks (True = PAD, False = real content)
        # Must match dataset_loader.py convention for model compatibility
        padding_masks = torch.ones(
            batch_size, target_size, target_size,
            dtype=torch.bool,
            device=self.device
        )

        for i, (img, (orig_w, orig_h)) in enumerate(zip(images, original_sizes)):
            # Calculate resize dimensions (downscale only, preserve aspect ratio)
            scale = min(target_size / orig_w, target_size / orig_h, 1.0)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            # Resize image
            img_gpu = img.unsqueeze(0).to(device=self.device, dtype=self.gpu_dtype)  # Add batch dim and move to GPU

            if (new_h, new_w) != (orig_h, orig_w):
                img_resized = F.interpolate(
                    img_gpu,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                img_resized = img_gpu

            # Calculate padding to center the image
            pad_top = (target_size - new_h) // 2
            pad_left = (target_size - new_w) // 2

            # Place resized image in center
            output_images[i, :, pad_top:pad_top+new_h, pad_left:pad_left+new_w] = img_resized.squeeze(0)

            # Mark non-padded region in mask (False = real content)
            padding_masks[i, pad_top:pad_top+new_h, pad_left:pad_left+new_w] = False

        return output_images, padding_masks

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
            failed_indices contains batch indices where image loading failed (dummy data inserted)
        """
        # Load images on CPU
        images, original_sizes, failed_indices = self.load_and_prepare_images(image_paths)

        # Apply letterbox resize on GPU
        images_gpu, masks_gpu = self.letterbox_resize_gpu(images, original_sizes)

        # Apply normalization on GPU (tensor is already in self.gpu_dtype from letterbox_resize_gpu)
        images_normalized = self.normalize(images_gpu)

        # Move back to CPU for cache writing
        images_cpu = images_normalized.cpu()
        masks_cpu = masks_gpu.cpu()

        return images_cpu, masks_cpu, failed_indices


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
        max_batch_size: int = 2048,
        cpu_bf16_cache_pipeline: bool = True
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
            max_batch_size: Maximum batch size (default: 2048 for RTX 5090)
            cpu_bf16_cache_pipeline: Convert to bfloat16
        """
        self.vram_monitor = VRAMMonitor(device_id, target_vram_gb, target_vram_util)
        self.batch_sizer = DynamicBatchSizer(initial_batch_size, max_batch_size=max_batch_size)
        self.preprocessor = GPUBatchPreprocessor(
            image_size, pad_color, normalize_mean, normalize_std,
            device_id, cpu_bf16_cache_pipeline
        )

        logger.info("GPU Batch Processor fully initialized")

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

    def clear_cache(self):
        """Clear GPU cache."""
        self.vram_monitor.clear_cache()
