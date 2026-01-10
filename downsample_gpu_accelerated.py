#!/usr/bin/env python3
"""
RAM-Based CPU-Only Image Downsampling

Major optimizations:
1. Load ENTIRE shard to RAM first (eliminates disk read during processing)
2. Parallel CPU processing with multiprocessing (no GPU - handles variable sizes)
3. Efficient I/O pattern: Read All -> Process in RAM -> Write All
4. Sequential writes for optimal throughput
5. Pure PIL processing on CPU
6. Low-priority background operation (won't lag games or foreground apps)

Expected performance: 50-100+ images/sec with 6 CPU workers (background-friendly)

Optimization Strategy:
- PHASE 1: Read entire shard into RAM as bytes (single sequential read pass)
- PHASE 2: Process all images in parallel on CPU (disk idle, all data in RAM)
- PHASE 3: Write all results back sequentially (single sequential write pass)
- This completely eliminates read/write thrashing

Usage:
    python downsample_gpu_accelerated.py --yes
    python downsample_gpu_accelerated.py --target=640 --workers=12 --yes  # Override worker count
    python downsample_gpu_accelerated.py --shard=shard_00010  # Dry run single shard
    python downsample_gpu_accelerated.py --shard=shard_00010 --yes  # Execute single shard
    python downsample_gpu_accelerated.py --resume-from=50 --yes  # Resume from shard 50 onwards
    python downsample_gpu_accelerated.py --resume-from=shard_00050 --yes  # Same as above
    python downsample_gpu_accelerated.py --yes --direct-write  # Direct write mode (faster)
    python downsample_gpu_accelerated.py --yes --no-low-priority  # Run at normal priority (not recommended while gaming)
"""

import sys
import json
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing as mp

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    HAS_PIL = True
except ImportError:
    print("ERROR: PIL/Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_TARGET_SIZE = 512
CPU_WORKERS = 6  # Fixed at 6 workers for minimal game impact (can override with --workers=N)
JPEG_QUALITY = 95  # High quality JPEG encoding
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# File size limits (safety checks)
MAX_FILE_SIZE_MB = 500  # Maximum file size to load into RAM (in MB)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# PNG to JPEG compression estimate (used in dry-run calculations)
PNG_TO_JPEG_COMPRESSION_RATIO = 7  # Typical PNG files compress ~7x when converted to JPEG

# Retry settings for file operations
MAX_WRITE_RETRIES = 3  # Number of retries for write operations
WRITE_RETRY_DELAY = 0.1  # Initial retry delay in seconds

# I/O optimization
READ_BUFFER_SIZE = 16 * 1024 * 1024  # 16MB read buffer for better read performance
WRITE_BUFFER_SIZE = 8 * 1024 * 1024  # 8MB write buffer for better write performance

# Command line parsing
DRY_RUN = '--yes' not in sys.argv
VERBOSE = '-v' in sys.argv or '--verbose' in sys.argv
FORCE = '--force' in sys.argv  # Skip confirmation prompt
SKIP_CLEANUP = '--skip-cleanup' in sys.argv  # Skip temp file cleanup
DIRECT_WRITE = '--direct-write' in sys.argv  # Skip temp file, write directly (faster but less safe)
LOW_PRIORITY = '--no-low-priority' not in sys.argv  # Run at low priority by default (disable with --no-low-priority)

TARGET_SIZE = DEFAULT_TARGET_SIZE
SINGLE_SHARD = None  # Process only this shard if specified
RESUME_FROM_SHARD = None  # Resume processing from this shard onwards (e.g., "shard_00050" or just "50")

# Parse and validate command line arguments
for arg in sys.argv:
    if arg.startswith('--target='):
        try:
            TARGET_SIZE = int(arg.split('=')[1])
            if TARGET_SIZE <= 0:
                print(f"ERROR: --target must be positive, got {TARGET_SIZE}")
                sys.exit(1)
            if TARGET_SIZE > 8192:
                print(f"WARNING: --target={TARGET_SIZE} is unusually large")
        except ValueError:
            print(f"ERROR: Invalid --target value: {arg.split('=')[1]}")
            sys.exit(1)
    elif arg.startswith('--workers='):
        try:
            CPU_WORKERS = int(arg.split('=')[1])
            if CPU_WORKERS <= 0:
                print(f"ERROR: --workers must be positive, got {CPU_WORKERS}")
                sys.exit(1)
            if CPU_WORKERS > mp.cpu_count():
                print(f"WARNING: --workers={CPU_WORKERS} exceeds CPU count ({mp.cpu_count()})")
        except ValueError:
            print(f"ERROR: Invalid --workers value: {arg.split('=')[1]}")
            sys.exit(1)
    elif arg.startswith('--shard='):
        SINGLE_SHARD = arg.split('=')[1]
    elif arg.startswith('--resume-from='):
        resume_value = arg.split('=')[1]
        # Accept either "50" or "shard_00050" format
        if resume_value.startswith('shard_'):
            RESUME_FROM_SHARD = resume_value
        else:
            # Convert number to shard format
            try:
                shard_num = int(resume_value)
                RESUME_FROM_SHARD = f'shard_{shard_num:05d}'
            except ValueError:
                print(f"ERROR: Invalid --resume-from value: {resume_value}")
                print("Usage: --resume-from=50 or --resume-from=shard_00050")
                sys.exit(1)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ImageJob:
    """Image to process with data already loaded in RAM."""
    path: Path
    shard_name: str
    image_bytes: bytes  # Raw file bytes loaded into RAM


@dataclass
class ProcessedImage:
    """Processed image result."""
    path: Path
    shard_name: str
    original_bytes: int
    new_bytes: int
    action: str
    image_data: Optional[bytes] = None
    output_path: Optional[Path] = None
    should_delete_original: bool = False


# ============================================================================
# CPU Processing (Pure PIL)
# ============================================================================

def process_single_image(job: ImageJob, target_size: int, dry_run: bool, verbose: bool = False) -> ProcessedImage:
    """
    Process a single image with PIL on CPU from in-memory bytes.

    This function runs in a separate process via ProcessPoolExecutor.
    Image bytes are already in RAM, no disk I/O during processing.
    Resizes if needed, converts to JPEG, returns bytes.
    """
    try:
        original_bytes = len(job.image_bytes)

        # Open image from in-memory bytes
        with Image.open(BytesIO(job.image_bytes)) as img:
            w, h = img.size
            is_png = job.path.suffix.upper() == '.PNG'
            longest = max(w, h)
            needs_resize = longest > target_size

            # Skip if no work needed
            if not needs_resize and not is_png:
                return ProcessedImage(
                    path=job.path,
                    shard_name=job.shard_name,
                    original_bytes=original_bytes,
                    new_bytes=original_bytes,
                    action="skipped"
                )

            # Dry run - just estimate
            if dry_run:
                if needs_resize:
                    scale = target_size / longest
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    estimated_bytes = int(original_bytes * (new_h * new_w) / (w * h))
                else:
                    estimated_bytes = original_bytes // PNG_TO_JPEG_COMPRESSION_RATIO

                action = "would_downsample" if needs_resize else "would_convert_to_jpeg"
                return ProcessedImage(
                    path=job.path,
                    shard_name=job.shard_name,
                    original_bytes=original_bytes,
                    new_bytes=estimated_bytes,
                    action=action
                )

            # REAL MODE: Process the image
            # Handle transparency
            if img.mode in ('RGBA', 'LA'):
                # Images with explicit alpha channel
                bg = Image.new('RGB', img.size, (114, 114, 114))
                bg.paste(img, mask=img.getchannel('A'))
                img = bg
            elif img.mode == 'P':
                # Palette mode - check if it has transparency
                if 'transparency' in img.info:
                    # Has transparency - convert to RGBA first, then composite
                    img = img.convert('RGBA')
                    bg = Image.new('RGB', img.size, (114, 114, 114))
                    bg.paste(img, mask=img.getchannel('A'))
                    img = bg
                else:
                    # No transparency - direct conversion to RGB
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if needed
            if needs_resize:
                scale = target_size / longest
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Encode to JPEG
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True, subsampling=0)
            image_bytes = buffer.getvalue()

            output_path = job.path.with_suffix('.jpg')
            should_delete = (output_path != job.path)

            # Determine action
            if needs_resize and is_png:
                action = "downsampled+converted"
            elif needs_resize:
                action = "downsampled"
            else:
                action = "converted"

            return ProcessedImage(
                path=job.path,
                shard_name=job.shard_name,
                original_bytes=original_bytes,
                new_bytes=len(image_bytes),
                action=action,
                image_data=image_bytes,
                output_path=output_path,
                should_delete_original=should_delete
            )

    except Exception as e:
        if verbose:
            print(f"  [ERROR] Failed to process {job.path.name}: {type(e).__name__}: {e}")
        return ProcessedImage(
            path=job.path,
            shard_name=job.shard_name,
            original_bytes=len(job.image_bytes),
            new_bytes=0,
            action="error"
        )


def process_batch_parallel(jobs: List[ImageJob], target_size: int, num_workers: int, dry_run: bool, verbose: bool = False) -> List[ProcessedImage]:
    """
    Process batch of images in parallel using ProcessPoolExecutor.

    Each worker processes one image at a time on CPU with PIL.
    All images loaded to RAM and processed in parallel.
    """
    results = []

    # Process in parallel using multiple CPU cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        futures = {executor.submit(process_single_image, job, target_size, dry_run, verbose): job
                   for job in jobs}

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                job = futures[future]
                if verbose:
                    print(f"  [WORKER ERROR] Worker failed for {job.path.name}: {type(e).__name__}: {e}")
                results.append(ProcessedImage(
                    path=job.path,
                    shard_name=job.shard_name,
                    original_bytes=0,
                    new_bytes=0,
                    action="error"
                ))

    return results


# ============================================================================
# I/O Functions
# ============================================================================

def load_shard_to_ram(shard_path: Path, shard_name: str, verbose: bool = False) -> List[ImageJob]:
    """
    Load all images from a shard into RAM with optimized I/O.

    Returns list of ImageJob objects with image bytes loaded.
    This is PHASE 1 of the pipeline - sequential read pass.

    Optimizations:
    - Large 16MB read buffers to reduce I/O operations
    - Batch stat() operations upfront to avoid per-file syscalls
    - Sequential reading with sorted paths for optimal throughput
    """
    jobs = []

    print(f"  [PHASE 1] Loading {shard_name} to RAM...")

    # Collect all image paths
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(shard_path.glob(f'*{ext}'))

    if not image_paths:
        return jobs

    # Sort paths for sequential disk access
    image_paths = sorted(image_paths)

    # Batch stat operations upfront and filter by size
    stat_start = time.time()
    print(f"  [PHASE 1A] Checking file sizes for {len(image_paths)} images...")
    valid_paths = []
    skipped_oversized = 0

    for img_path in image_paths:
        try:
            file_size = img_path.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                if verbose:
                    print(f"  [SKIP] {img_path.name}: File too large ({file_size / 1024**2:.1f} MB > {MAX_FILE_SIZE_MB} MB)")
                skipped_oversized += 1
            else:
                valid_paths.append(img_path)
        except Exception as e:
            if verbose:
                print(f"  [STAT ERROR] Failed to stat {img_path.name}: {type(e).__name__}: {e}")

    stat_time = time.time() - stat_start
    stat_rate = len(image_paths) / stat_time if stat_time > 0 else 0
    print(f"  [PHASE 1A] File size check complete in {stat_time:.2f}s ({stat_rate:.1f} files/sec)")

    if not valid_paths:
        if skipped_oversized > 0:
            print(f"  [PHASE 1] No valid images to load (skipped {skipped_oversized} oversized files)")
        return jobs

    # Load all files to RAM with large buffers
    read_start = time.time()
    print(f"  [PHASE 1B] Reading {len(valid_paths)} files to RAM (using {READ_BUFFER_SIZE // 1024**2}MB buffers)...")
    total_bytes = 0

    if HAS_TQDM:
        pbar = tqdm(valid_paths, desc=f"  Loading {shard_name}", unit="img")
    else:
        pbar = valid_paths

    for img_path in pbar:
        try:
            # Read with large buffer for better I/O performance
            with open(img_path, 'rb', buffering=READ_BUFFER_SIZE) as f:
                image_bytes = f.read()
                jobs.append(ImageJob(
                    path=img_path,
                    shard_name=shard_name,
                    image_bytes=image_bytes
                ))
                total_bytes += len(image_bytes)
        except Exception as e:
            if verbose:
                print(f"  [LOAD ERROR] Failed to load {img_path.name}: {type(e).__name__}: {e}")

    if HAS_TQDM and hasattr(pbar, 'close'):
        pbar.close()

    read_time = time.time() - read_start
    read_throughput_mb = (total_bytes / 1024**2) / read_time if read_time > 0 else 0
    read_throughput_files = len(jobs) / read_time if read_time > 0 else 0

    if skipped_oversized > 0:
        print(f"  [PHASE 1B] Read complete: {len(jobs)} images, {total_bytes / 1024**2:.1f} MB in {read_time:.2f}s")
        print(f"  [PHASE 1B] Read throughput: {read_throughput_mb:.1f} MB/s, {read_throughput_files:.1f} files/sec")
        print(f"  [PHASE 1] Loaded {len(jobs)} images ({total_bytes / 1024**2:.1f} MB) into RAM (skipped {skipped_oversized} oversized files)")
    else:
        print(f"  [PHASE 1B] Read complete: {len(jobs)} images, {total_bytes / 1024**2:.1f} MB in {read_time:.2f}s")
        print(f"  [PHASE 1B] Read throughput: {read_throughput_mb:.1f} MB/s, {read_throughput_files:.1f} files/sec")
        print(f"  [PHASE 1] Loaded {len(jobs)} images ({total_bytes / 1024**2:.1f} MB) into RAM")

    return jobs


def write_results_sequential(results: List[ProcessedImage], dry_run: bool, verbose: bool = False) -> int:
    """
    Write all processed images sequentially to disk with batched I/O.

    This is PHASE 3 of the pipeline - sequential write pass.
    Sorts results by output path for optimal throughput.

    Optimizations:
    - Uses large write buffers (8MB) to reduce I/O operations
    - Defers all delete operations to end (batch metadata updates)
    - Minimizes file system metadata churn during writes

    NOTE: Atomic operations on network drives:
    - Path.replace() may not be truly atomic on network drives
    - Power loss or interruption during rename can corrupt files
    - Use --direct-write for faster (but less safe) direct writes

    Returns: Number of files written
    """
    if dry_run:
        return 0

    # Sort by output path for sequential writes
    results_to_write = [r for r in results if r.image_data and r.output_path]
    results_to_write = sorted(results_to_write, key=lambda r: str(r.output_path))

    print(f"  [PHASE 3] Writing {len(results_to_write)} images...")

    written = 0
    files_to_delete = []  # Defer deletes to end for better I/O performance

    if HAS_TQDM:
        pbar = tqdm(results_to_write, desc="  Writing", unit="img")
    else:
        pbar = results_to_write

    for result in pbar:
        # Retry logic for file locks
        for attempt in range(MAX_WRITE_RETRIES):
            try:
                if DIRECT_WRITE:
                    # Direct write mode - faster but less safe
                    # Write directly to final location with large buffer
                    with open(result.output_path, 'wb', buffering=WRITE_BUFFER_SIZE) as f:
                        f.write(result.image_data)

                    # Defer delete - collect files to delete
                    if result.should_delete_original and result.path != result.output_path:
                        if result.path.exists():
                            files_to_delete.append(result.path)
                else:
                    # Safe mode: Write to temporary file first with large buffer
                    temp_path = result.output_path.with_suffix('.tmp')

                    with open(temp_path, 'wb', buffering=WRITE_BUFFER_SIZE) as f:
                        f.write(result.image_data)

                    # Atomic rename (NOTE: May not be truly atomic on network drives)
                    temp_path.replace(result.output_path)

                    # Defer delete - collect files to delete
                    if result.should_delete_original and result.path != result.output_path:
                        if result.path.exists():
                            files_to_delete.append(result.path)

                written += 1
                break

            except (PermissionError, OSError) as e:
                if attempt < MAX_WRITE_RETRIES - 1:
                    if verbose:
                        print(f"  [WRITE RETRY] Attempt {attempt+1}/{MAX_WRITE_RETRIES} failed for {result.output_path.name}: {type(e).__name__}: {e}")
                    time.sleep(WRITE_RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    if verbose:
                        print(f"  [WRITE ERROR] Failed to write {result.output_path.name} after {MAX_WRITE_RETRIES} attempts: {type(e).__name__}: {e}")
            except Exception as e:
                if verbose:
                    print(f"  [WRITE ERROR] Unexpected error writing {result.output_path.name}: {type(e).__name__}: {e}")
                break

    if HAS_TQDM and hasattr(pbar, 'close'):
        pbar.close()

    print(f"  [PHASE 3] Wrote {written} images")

    # PHASE 3B: Batch delete all original files (deferred metadata operations)
    if files_to_delete:
        print(f"  [PHASE 3B] Cleaning up {len(files_to_delete)} original files...")
        deleted = 0
        failed_deletes = 0

        if HAS_TQDM:
            del_pbar = tqdm(files_to_delete, desc="  Deleting", unit="file")
        else:
            del_pbar = files_to_delete

        for file_path in del_pbar:
            try:
                file_path.unlink()
                deleted += 1
            except Exception as e:
                failed_deletes += 1
                if verbose:
                    print(f"  [DELETE ERROR] Failed to delete {file_path.name}: {type(e).__name__}: {e}")

        if HAS_TQDM and hasattr(del_pbar, 'close'):
            del_pbar.close()

        if failed_deletes > 0:
            print(f"  [PHASE 3B] Deleted {deleted} files ({failed_deletes} failed)")
        else:
            print(f"  [PHASE 3B] Deleted {deleted} files")

    return written


def process_shard_pipeline(shard_path: Path, shard_name: str, target_size: int, num_workers: int, dry_run: bool, verbose: bool = False) -> List[ProcessedImage]:
    """
    Process an entire shard with 3-phase pipeline.

    PHASE 1: Load entire shard to RAM (sequential read, no disk thrashing)
    PHASE 2: Process all images in parallel on CPU (disk completely idle)
    PHASE 3: Write all results sequentially (sequential write, no disk thrashing)

    This approach completely eliminates read/write interleaving for optimal performance.
    """
    phase_times = {}

    # PHASE 1: LOAD all images to RAM
    load_start = time.time()
    jobs = load_shard_to_ram(shard_path, shard_name, verbose)
    phase_times['load'] = time.time() - load_start

    if not jobs:
        print(f"  No images to process in {shard_name}")
        return []

    # PHASE 2: PROCESS in parallel on CPU (all data in RAM, no disk I/O)
    print(f"  [PHASE 2] Processing {len(jobs)} images on {num_workers} CPU workers...")
    process_start = time.time()

    if HAS_TQDM:
        # Process with progress bar
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_image, job, target_size, dry_run, verbose): job
                       for job in jobs}

            with tqdm(total=len(jobs), desc=f"  Processing", unit="img") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        job = futures[future]
                        if verbose:
                            print(f"  [PROCESS ERROR] Failed to process {job.path.name}: {type(e).__name__}: {e}")
    else:
        results = process_batch_parallel(jobs, target_size, num_workers, dry_run, verbose)

    phase_times['process'] = time.time() - process_start
    print(f"  [PHASE 2] Processing complete ({phase_times['process']:.1f}s)")

    # PHASE 3: WRITE sequentially
    write_start = time.time()
    written = write_results_sequential(results, dry_run, verbose)
    phase_times['write'] = time.time() - write_start

    # Summary
    total_time = sum(phase_times.values())
    print(f"\n  [TIMING] Load: {phase_times['load']:.1f}s | Process: {phase_times['process']:.1f}s | Write: {phase_times['write']:.1f}s | Total: {total_time:.1f}s")

    return results


# ============================================================================
# Main
# ============================================================================

def set_low_priority() -> bool:
    """
    Set process priority to low (background task) to avoid impacting foreground apps like games.

    Returns True if priority was successfully set, False otherwise.
    """
    if not HAS_PSUTIL:
        return False

    try:
        proc = psutil.Process()
        if sys.platform == 'win32':
            # On Windows, use BELOW_NORMAL priority class
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("[PRIORITY] Process priority set to BELOW_NORMAL (background task)")
        else:
            # On Unix-like systems, use nice value of 10 (lower priority)
            proc.nice(10)
            print("[PRIORITY] Process priority set to low (nice +10)")
        return True
    except Exception as e:
        print(f"[PRIORITY] Warning: Could not set low priority: {e}")
        return False


def cleanup_temp_files(database_path: Path, shards_to_clean: Optional[List[Path]] = None, verbose: bool = False) -> int:
    """Remove any leftover .tmp files from previous failed runs."""
    if shards_to_clean is None:
        all_shards = [d for d in database_path.iterdir()
                      if d.is_dir() and d.name.startswith('shard_')]
    else:
        all_shards = shards_to_clean

    if len(all_shards) == 1:
        print(f"Cleaning up temporary files from {all_shards[0].name}...")
    else:
        print(f"Cleaning up temporary files from {len(all_shards)} shards...")

    tmp_count = 0
    for shard in all_shards:
        for tmp_file in shard.glob('*.tmp'):
            try:
                tmp_file.unlink()
                tmp_count += 1
            except Exception as e:
                if verbose:
                    print(f"  [CLEANUP ERROR] Failed to remove temporary file {tmp_file.name}: {type(e).__name__}: {e}")

    if tmp_count > 0:
        print(f"[CLEANUP] Removed {tmp_count} leftover .tmp files")

    return tmp_count


def main() -> None:
    """Main execution with RAM-based parallel CPU processing."""

    # Validate mutually exclusive options
    if SINGLE_SHARD and RESUME_FROM_SHARD:
        print("ERROR: Cannot use --shard and --resume-from together")
        print("Use --shard for single shard processing, or --resume-from to process multiple shards starting from a specific point")
        return

    # Set low priority to avoid impacting foreground apps (games, etc.)
    if LOW_PRIORITY:
        set_low_priority()

    # Allow custom database path via --database= parameter
    database_path = Path(r'L:\Dab\Dab')
    for arg in sys.argv:
        if arg.startswith('--database='):
            db_path_str = arg[11:]  # Remove '--database=' prefix
            database_path = Path(db_path_str)
            break

    if not database_path.exists():
        print(f"ERROR: Database not found: {database_path}")
        return

    all_shards = sorted([d for d in database_path.iterdir()
                        if d.is_dir() and d.name.startswith('shard_')])

    # Filter to single shard if specified
    if SINGLE_SHARD:
        shard_path = database_path / SINGLE_SHARD
        if not shard_path.exists() or not shard_path.is_dir():
            print(f"ERROR: Shard not found: {shard_path}")
            print(f"Available shards: {', '.join([s.name for s in all_shards[:10]])}...")
            return
        all_shards = [shard_path]
        print(f"\n{'='*70}")
        print(f"RAM-BASED PARALLEL IMAGE DOWNSAMPLING (SINGLE SHARD MODE)")
        print(f"{'='*70}")
    elif RESUME_FROM_SHARD:
        # Filter shards to only process from resume point onwards
        original_count = len(all_shards)
        all_shards = [s for s in all_shards if s.name >= RESUME_FROM_SHARD]

        if not all_shards:
            print(f"ERROR: No shards found at or after {RESUME_FROM_SHARD}")
            available = sorted([d for d in database_path.iterdir() if d.is_dir() and d.name.startswith('shard_')])
            if available:
                print(f"Available shards range: {available[0].name} to {available[-1].name}")
            else:
                print("No shards found in database")
            return

        skipped_count = original_count - len(all_shards)
        print(f"\n{'='*70}")
        print(f"RAM-BASED PARALLEL IMAGE DOWNSAMPLING (RESUME MODE)")
        print(f"{'='*70}")
        print(f"Resuming from: {RESUME_FROM_SHARD}")
        print(f"Skipping {skipped_count} shards (already processed)")
        print(f"Processing {len(all_shards)} remaining shards")
    else:
        print(f"\n{'='*70}")
        print(f"RAM-BASED PARALLEL IMAGE DOWNSAMPLING")
        print(f"{'='*70}")

    # Clean up any leftover temp files
    if not SKIP_CLEANUP:
        cleanup_temp_files(database_path, all_shards, VERBOSE)

    print(f"Target size: {TARGET_SIZE}px")
    print(f"Database: {database_path}")
    if SINGLE_SHARD:
        print(f"Single shard mode: {SINGLE_SHARD}")
    elif RESUME_FROM_SHARD:
        print(f"Resume mode: Starting from {RESUME_FROM_SHARD}")
        print(f"Shards to process: {len(all_shards)}")
    else:
        print(f"Total shards: {len(all_shards)}")

    print(f"\nConfiguration:")
    print(f"  CPU workers: {CPU_WORKERS} (system has {mp.cpu_count()} cores)")
    print(f"  Priority: {'LOW (background task, won\'t lag games)' if LOW_PRIORITY else 'NORMAL (may impact foreground apps)'}")
    print(f"  Write mode: {'DIRECT (faster, less safe)' if DIRECT_WRITE else 'SAFE (atomic renames)'}")
    print(f"  Processing: Pure CPU with PIL (no GPU)")

    print(f"\nOptimization Strategy:")
    print(f"  1. Load ENTIRE shard to RAM (sequential read)")
    print(f"  2. Process all images in parallel on {CPU_WORKERS} CPU cores (disk idle)")
    print(f"  3. Write all results sequentially (sequential write)")
    print(f"  4. Repeat for next shard")
    print(f"\n  This eliminates read/write thrashing!")

    mode = "DRY RUN" if DRY_RUN else "EXECUTE"
    print(f"\nMode: {mode}")

    if not DRY_RUN and not FORCE:
        print(f"{'='*70}")
        print(f"[WARNING] FILES WILL BE MODIFIED")
        print(f"{'='*70}")
        response = input("Type 'YES' to proceed: ").strip()
        if response != 'YES':
            print("Aborted")
            return
    elif not DRY_RUN and FORCE:
        print(f"{'='*70}")
        print(f"[WARNING] FILES WILL BE MODIFIED (--force used)")
        print(f"{'='*70}")

    print(f"\nProcessing {len(all_shards)} shards...\n")
    start_time = time.time()

    # Statistics
    shard_stats = {}
    total_images_processed = 0

    # Process one shard at a time
    for shard_idx, shard_path in enumerate(all_shards, 1):
        shard_name = shard_path.name

        print(f"\n{'='*70}")
        print(f"[SHARD {shard_idx}/{len(all_shards)}] {shard_name}")
        print(f"{'='*70}")

        # Process entire shard with 3-phase pipeline
        results = process_shard_pipeline(shard_path, shard_name, TARGET_SIZE, CPU_WORKERS, DRY_RUN, VERBOSE)

        if not results:
            continue

        # Update stats
        if shard_name not in shard_stats:
            shard_stats[shard_name] = {
                'total': 0,
                'skipped': 0,
                'downsampled': 0,
                'original_bytes': 0,
                'final_bytes': 0
            }

        s = shard_stats[shard_name]
        for result in results:
            s['total'] += 1
            s['original_bytes'] += result.original_bytes
            s['final_bytes'] += result.new_bytes

            if result.action == 'skipped':
                s['skipped'] += 1
            elif 'downsample' in result.action or 'convert' in result.action:
                s['downsampled'] += 1

        total_images_processed += len(results)

        # Print shard summary
        shard_original = shard_stats[shard_name]['original_bytes']
        shard_final = shard_stats[shard_name]['final_bytes']
        shard_saved = shard_original - shard_final
        print(f"\n[SHARD {shard_idx}/{len(all_shards)}] {shard_name} complete!")
        print(f"  Processed: {shard_stats[shard_name]['downsampled']} images")
        print(f"  Saved: {shard_saved / 1024**2:.1f} MB")

    elapsed = time.time() - start_time

    # Calculate totals
    total_original = sum(s['original_bytes'] for s in shard_stats.values())
    total_final = sum(s['final_bytes'] for s in shard_stats.values())
    total_downsampled = sum(s['downsampled'] for s in shard_stats.values())

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'DRY RUN ' if DRY_RUN else ''}COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total images: {total_images_processed}")
    print(f"Speed: {total_images_processed/elapsed:.1f} images/sec")
    print(f"\nProcessed: {total_downsampled}")
    print(f"Original: {total_original / 1024**3:.2f} GB")
    print(f"Final: {total_final / 1024**3:.2f} GB")
    print(f"Saved: {(total_original - total_final) / 1024**3:.2f} GB")

    if total_original > 0:
        compression = 100 * (1 - total_final / total_original)
        print(f"Compression: {compression:.1f}%")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
