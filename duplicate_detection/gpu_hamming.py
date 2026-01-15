"""
GPU-accelerated Hamming distance computation for near-duplicate detection.

Uses tiled processing to efficiently compare large sets of perceptual hashes
on GPU while maintaining bounded memory usage.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BucketResult:
    """Result from processing a single bucket."""
    pairs: List[Tuple[int, int]]  # (global_idx_a, global_idx_b) pairs
    comparison_count: int
    was_sampled: bool


class GPUHammingProcessor:
    """
    GPU-accelerated Hamming distance computation with tiled processing.

    Designed for comparing perceptual hashes (dHash, pHash) stored as uint8 byte arrays.
    Uses tiled computation to limit VRAM usage while maximizing throughput.

    Args:
        device_id: CUDA device index (default: 0)
        tile_size: Size of comparison tiles (default: 1024 for 31GB+ VRAM)
        max_distance: Maximum Hamming distance to consider as duplicate
        batch_size: Number of buckets to process in parallel (default: 32)
    """

    def __init__(
        self,
        device_id: int = 0,
        tile_size: int = 1024,
        max_distance: int = 164,
        batch_size: int = 32,
    ):
        self.device = torch.device(f'cuda:{device_id}')
        self.tile_size = tile_size
        self.max_distance = max_distance
        self.batch_size = batch_size

        # Pre-computed popcount lookup table on GPU (256 bytes)
        self.popcount = torch.tensor(
            [bin(i).count('1') for i in range(256)],
            dtype=torch.uint8,
            device=self.device,
        )

        # Statistics
        self.total_comparisons = 0
        self.total_pairs_found = 0

    def find_similar_pairs(
        self,
        bucket_indices: np.ndarray,
        all_hashes_gpu: torch.Tensor,
        max_bucket_size: int = 8000,
    ) -> BucketResult:
        """
        Find all pairs within a bucket with Hamming distance <= max_distance.

        Args:
            bucket_indices: Array of global indices into all_hashes_gpu
            all_hashes_gpu: (N, hash_bytes) uint8 tensor on GPU
            max_bucket_size: Sample if bucket exceeds this size

        Returns:
            BucketResult with pairs, comparison count, and sampling flag
        """
        n = len(bucket_indices)
        if n < 2:
            return BucketResult(pairs=[], comparison_count=0, was_sampled=False)

        sampled = False
        working_indices = bucket_indices

        # Distance-aware sampling for very large buckets
        if n > max_bucket_size:
            sampled = True
            working_indices = self._distance_aware_sample(
                bucket_indices, all_hashes_gpu, max_bucket_size
            )
            n = len(working_indices)

        # Extract bucket hashes (view, no copy)
        indices_tensor = torch.from_numpy(working_indices).to(self.device, non_blocking=True)
        bucket_hashes = all_hashes_gpu[indices_tensor]

        # GPU tiled comparison
        local_pairs = self._find_pairs_tiled(bucket_hashes)

        # Map local indices back to global indices
        pairs = [
            (int(working_indices[a]), int(working_indices[b]))
            for a, b in local_pairs
        ]

        comparison_count = n * (n - 1) // 2
        self.total_comparisons += comparison_count
        self.total_pairs_found += len(pairs)

        return BucketResult(
            pairs=pairs,
            comparison_count=comparison_count,
            was_sampled=sampled,
        )

    def find_similar_pairs_batch(
        self,
        bucket_list: List[np.ndarray],
        all_hashes_gpu: torch.Tensor,
        max_bucket_size: int = 8000,
    ) -> List[BucketResult]:
        """
        Process multiple buckets efficiently.

        For small buckets, combines them into larger batches for better GPU utilization.
        For large buckets, processes individually with tiling.

        Args:
            bucket_list: List of index arrays, one per bucket
            all_hashes_gpu: (N, hash_bytes) uint8 tensor on GPU
            max_bucket_size: Sample if bucket exceeds this size

        Returns:
            List of BucketResult, one per input bucket
        """
        results = []

        # Separate small and large buckets
        small_buckets = []
        small_bucket_indices = []

        for i, indices in enumerate(bucket_list):
            if len(indices) < 2:
                results.append(BucketResult(pairs=[], comparison_count=0, was_sampled=False))
            elif len(indices) <= 256:  # Small bucket - batch together
                small_buckets.append(indices)
                small_bucket_indices.append(i)
                results.append(None)  # Placeholder
            else:  # Large bucket - process individually
                result = self.find_similar_pairs(indices, all_hashes_gpu, max_bucket_size)
                results.append(result)

        # Process small buckets in batches
        if small_buckets:
            small_results = self._process_small_buckets_batch(
                small_buckets, all_hashes_gpu
            )
            for idx, result in zip(small_bucket_indices, small_results):
                results[idx] = result

        return results

    def _distance_aware_sample(
        self,
        indices: np.ndarray,
        all_hashes_gpu: torch.Tensor,
        max_size: int,
    ) -> np.ndarray:
        """Sample indices closest to centroid for better recall."""
        indices_tensor = torch.from_numpy(indices).to(self.device, non_blocking=True)
        bucket_hashes = all_hashes_gpu[indices_tensor]

        # Use first 4 bytes as quick distance proxy
        first_bytes = bucket_hashes[:, :4].float()
        centroid = first_bytes.mean(dim=0)
        quick_distances = (first_bytes - centroid).abs().sum(dim=1)

        # Keep closest to centroid
        keep_idx = torch.argsort(quick_distances)[:max_size]
        return indices[keep_idx.cpu().numpy()]

    def _find_pairs_tiled(
        self,
        hashes: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """
        Find similar pairs using tiled computation.

        Processes the pairwise distance matrix in tiles to bound memory usage.
        Only computes upper triangle (i < j) to avoid duplicate pairs.

        Args:
            hashes: (N, hash_bytes) uint8 tensor on GPU

        Returns:
            List of (local_idx_i, local_idx_j) pairs where i < j
        """
        N = hashes.shape[0]
        pairs = []

        for i_start in range(0, N, self.tile_size):
            i_end = min(i_start + self.tile_size, N)

            # For upper triangle, j starts at i_start (not 0)
            for j_start in range(i_start, N, self.tile_size):
                j_end = min(j_start + self.tile_size, N)

                tile_pairs = self._process_tile(
                    hashes, i_start, i_end, j_start, j_end
                )
                pairs.extend(tile_pairs)

        return pairs

    def _process_tile(
        self,
        hashes: torch.Tensor,
        i_start: int,
        i_end: int,
        j_start: int,
        j_end: int,
    ) -> List[Tuple[int, int]]:
        """
        Process a single tile of the pairwise distance matrix.

        Memory usage: tile_i_size * tile_j_size * hash_bytes bytes
        For 1024x1024 tiles with 32-byte hashes: 32MB
        """
        tile_i = hashes[i_start:i_end]  # (Ti, B)
        tile_j = hashes[j_start:j_end]  # (Tj, B)

        # Vectorized XOR: (Ti, 1, B) ^ (1, Tj, B) -> (Ti, Tj, B)
        xor = tile_i.unsqueeze(1) ^ tile_j.unsqueeze(0)

        # Popcount via lookup table
        # Use int32 for indexing (sufficient for 0-255, saves 2.2 GB vs int64)
        flat = xor.reshape(-1).int()
        counts = self.popcount[flat].view(xor.shape)

        # Explicit cleanup to reduce peak VRAM
        del flat, xor

        distances = counts.sum(dim=-1, dtype=torch.int16)
        del counts

        # Find pairs below threshold
        mask = distances <= self.max_distance

        # Handle diagonal tile (i_start == j_start): only keep upper triangle
        if i_start == j_start:
            # Create upper triangle mask
            Ti = i_end - i_start
            triu_mask = torch.triu(torch.ones(Ti, Ti, dtype=torch.bool, device=self.device), diagonal=1)
            mask = mask & triu_mask

        # Extract matching pairs
        local_i, local_j = torch.where(mask)

        if len(local_i) == 0:
            return []

        # Convert to global indices
        global_i = (local_i + i_start).cpu().numpy()
        global_j = (local_j + j_start).cpu().numpy()

        return list(zip(global_i.tolist(), global_j.tolist()))

    def _process_small_buckets_batch(
        self,
        buckets: List[np.ndarray],
        all_hashes_gpu: torch.Tensor,
    ) -> List[BucketResult]:
        """
        Process multiple small buckets by padding and batching.

        Pads small buckets to uniform size and processes as a single batch
        for better GPU utilization.
        """
        results = []

        # For simplicity, process small buckets individually but with minimal overhead
        # A more sophisticated implementation could pad and batch them
        for indices in buckets:
            if len(indices) < 2:
                results.append(BucketResult(pairs=[], comparison_count=0, was_sampled=False))
                continue

            indices_tensor = torch.from_numpy(indices).to(self.device, non_blocking=True)
            bucket_hashes = all_hashes_gpu[indices_tensor]

            n = len(indices)

            # For small buckets, compute full pairwise matrix directly (no tiling needed)
            xor = bucket_hashes.unsqueeze(1) ^ bucket_hashes.unsqueeze(0)
            flat = xor.reshape(-1).int()  # Use int32 instead of int64
            counts = self.popcount[flat].view(n, n, -1)
            del flat, xor  # Explicit cleanup
            distances = counts.sum(dim=-1)
            del counts

            # Upper triangle only
            i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=self.device)
            mask = distances[i_idx, j_idx] <= self.max_distance

            matched_i = i_idx[mask].cpu().numpy()
            matched_j = j_idx[mask].cpu().numpy()

            pairs = [
                (int(indices[i]), int(indices[j]))
                for i, j in zip(matched_i, matched_j)
            ]

            results.append(BucketResult(
                pairs=pairs,
                comparison_count=n * (n - 1) // 2,
                was_sampled=False,
            ))

        return results

    def reset_stats(self):
        """Reset comparison statistics."""
        self.total_comparisons = 0
        self.total_pairs_found = 0

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            'total_comparisons': self.total_comparisons,
            'total_pairs_found': self.total_pairs_found,
        }


def check_gpu_available() -> Tuple[bool, Optional[str]]:
    """
    Check if CUDA GPU is available and return device info.

    Returns:
        (is_available, device_name or error message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    try:
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / (1024 ** 3)
        return True, f"{props.name} ({vram_gb:.1f} GB VRAM)"
    except Exception as e:
        return False, str(e)


def create_gpu_processor(
    max_distance: int,
    vram_gb: float = None,
) -> Optional[GPUHammingProcessor]:
    """
    Create a GPUHammingProcessor with optimal settings for available VRAM.

    Args:
        max_distance: Maximum Hamming distance threshold
        vram_gb: Override VRAM detection (for testing)

    Returns:
        GPUHammingProcessor instance or None if GPU unavailable
    """
    available, info = check_gpu_available()
    if not available:
        print(f"GPU not available: {info}")
        return None

    if vram_gb is None:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)

    # Choose tile size based on VRAM
    # Memory per tile: tile_size^2 * hash_bytes * 2 (for XOR and counts)
    # For 128-byte hashes (hash_size=32): 1024^2 * 128 * 2 = 256MB per tile
    # Conservative settings to target ~20GB peak on 31.5GB GPUs
    if vram_gb >= 24:
        tile_size = 1024  # Conservative for 20GB target (was 2048)
    elif vram_gb >= 12:
        tile_size = 768   # Balanced for mid-range GPUs
    elif vram_gb >= 6:
        tile_size = 512   # 16MB per tile
    else:
        tile_size = 256   # 4MB per tile

    print(f"GPU detected: {info}")
    print(f"Using tile size: {tile_size} (optimized for {vram_gb:.1f} GB VRAM)")

    return GPUHammingProcessor(
        device_id=0,
        tile_size=tile_size,
        max_distance=max_distance,
    )
