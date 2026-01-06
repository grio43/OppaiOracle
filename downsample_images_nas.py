#!/usr/bin/env python3
"""
Optimized Image Downsampling for OppaiOracle

Downsizes high-resolution images to training resolution (default 512px longest side).
Converts large PNGs to JPEG, handles transparency, and auto-tags grey background.
Resumable with progress tracking and multiprocessing support.

Usage:
    python downsample_images_nas.py           # Dry run
    python downsample_images_nas.py --yes     # Execute
    python downsample_images_nas.py --target=640 --yes  # Custom size
"""

import sys
import warnings
import json
from io import BytesIO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple
import multiprocessing as mp

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # Allow large images
    HAS_PIL = True
except ImportError:
    print("ERROR: PIL/Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================================
# Configuration
# ============================================================================

# Training resolution from OppaiOracle config
TRAINING_SIZE = 512

# Target size for downsampling (matches training resolution exactly)
# Images with longest side <= TARGET_SIZE will not be touched
DEFAULT_TARGET_SIZE = 512

# File size threshold for PNG→JPEG conversion (in bytes)
PNG_TO_JPEG_THRESHOLD = 1 * 1024 * 1024  # 1MB
# NOTE: For 512px target, max uncompressed PNG size is ~786KB (512×512×3)
# So this threshold will rarely trigger at 512px. Consider 500KB for 512px training.

# JPEG quality for converted files (95 = high quality, minimal artifacts)
JPEG_QUALITY = 95

# Batch size for efficient I/O
BATCH_SIZE = 100

# Number of parallel workers
NUM_WORKERS = 3

# Common image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif'}

# Command line arguments
DRY_RUN = '--dry-run' not in sys.argv and '--yes' not in sys.argv  # Default to dry run
EXECUTE = '--yes' in sys.argv
VERBOSE = '--verbose' in sys.argv or '-v' in sys.argv

# Parse custom target size
TARGET_SIZE = DEFAULT_TARGET_SIZE
for arg in sys.argv:
    if arg.startswith('--target='):
        try:
            TARGET_SIZE = int(arg.split('=')[1])
            print(f"Custom target size: {TARGET_SIZE}px")
        except ValueError:
            print(f"Invalid target size: {arg}")
            sys.exit(1)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ImageStats:
    """Statistics for a single image processing operation."""
    path: Path
    original_size: Tuple[int, int]
    original_bytes: int
    original_format: str
    new_size: Optional[Tuple[int, int]] = None
    new_bytes: Optional[int] = None
    new_format: Optional[str] = None
    action: str = "skipped"  # skipped, downsampled, converted, error
    error: Optional[str] = None
    gray_background_applied: bool = False  # Track if gray background was added

    @property
    def space_saved(self) -> int:
        """Calculate space saved in bytes."""
        if self.new_bytes is not None:
            return self.original_bytes - self.new_bytes
        return 0

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (new/original)."""
        if self.new_bytes is not None and self.original_bytes > 0:
            return self.new_bytes / self.original_bytes
        return 1.0


@dataclass
class ShardStats:
    """Aggregate statistics for a shard."""
    shard_name: str
    total_images: int = 0
    skipped: int = 0
    downsampled: int = 0
    converted: int = 0
    errors: int = 0
    gray_backgrounds: int = 0  # Count images with gray background applied
    original_bytes: int = 0
    final_bytes: int = 0

    @property
    def space_saved(self) -> int:
        return self.original_bytes - self.final_bytes

    @property
    def space_saved_mb(self) -> float:
        return self.space_saved / (1024 * 1024)

    @property
    def compression_ratio(self) -> float:
        if self.original_bytes > 0:
            return self.final_bytes / self.original_bytes
        return 1.0


# ============================================================================
# Progress Tracking
# ============================================================================

def load_progress(progress_file: Path) -> set:
    """Load completed shards from progress file."""
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            print(f"Warning: Could not load progress: {e}")
    return set()


def save_progress(progress_file: Path, shard_name: str):
    """Append completed shard to progress file."""
    try:
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{shard_name}\n")
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")


# ============================================================================
# JSON Metadata Handling
# ============================================================================

def add_tag_to_json(json_path: Path, tag: str):
    """
    Add a tag to the JSON metadata file.

    Args:
        json_path: Path to JSON file
        tag: Tag to add (e.g., 'grey_background')
    """
    try:
        # Read existing JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get existing tags
        tags_field = data.get('tags')

        # Parse tags based on type
        if tags_field is None:
            # No tags exist, create list with single tag
            data['tags'] = tag
        elif isinstance(tags_field, str):
            # Comma-separated string
            existing_tags = [t.strip() for t in tags_field.split(',') if t.strip()]
            if tag not in existing_tags:
                existing_tags.append(tag)
                data['tags'] = ', '.join(existing_tags)
        elif isinstance(tags_field, list):
            # Already a list
            if tag not in tags_field:
                tags_field.append(tag)
                data['tags'] = tags_field
        else:
            # Unknown format, create new field
            data['tags'] = tag

        # Write back to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        # Don't fail the entire operation if JSON update fails
        print(f"Warning: Could not update JSON {json_path}: {e}")


# ============================================================================
# Image Processing
# ============================================================================

def should_process_image(img_path: Path, target_size: int) -> Tuple[bool, str]:
    """
    Determine if image needs processing.
    Returns: (should_process, reason)
    """
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            longest_side = max(width, height)

            # Skip images already at or below target size
            if longest_side <= target_size:
                return False, f"already_optimal_{width}x{height}"

            return True, f"oversized_{width}x{height}"

    except Exception as e:
        return False, f"error_reading: {e}"


def calculate_target_dimensions(original_size: Tuple[int, int], target_size: int) -> Tuple[int, int]:
    """
    Calculate target dimensions preserving aspect ratio.
    Ensures longest side = target_size.
    """
    width, height = original_size
    longest_side = max(width, height)

    if longest_side <= target_size:
        return original_size

    scale = target_size / longest_side
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    return (max(1, new_width), max(1, new_height))


def process_image(img_path: Path, target_size: int, dry_run: bool = True) -> Tuple[ImageStats, Optional[bytes], Optional[Path], bool]:
    """
    Process a single image: downsample and optionally convert format.

    Args:
        img_path: Path to image file
        target_size: Target size for longest side
        dry_run: If True, don't actually process image data

    Returns:
        Tuple of (ImageStats, image_bytes, output_path, should_delete_original)
        - image_bytes: Processed image data ready to write (None if skipped/error/dry_run)
        - output_path: Path where image should be written (None if skipped/error)
        - should_delete_original: True if original file should be deleted before writing
    """
    try:
        # Get original file info
        original_bytes = img_path.stat().st_size

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(img_path) as img:
                original_size = img.size
                original_format = img.format or 'UNKNOWN'

                # Check if processing needed
                should_process, reason = should_process_image(img_path, target_size)

                if not should_process:
                    stats = ImageStats(
                        path=img_path,
                        original_size=original_size,
                        original_bytes=original_bytes,
                        original_format=original_format,
                        action="skipped",
                        error=reason if "error" in reason else None
                    )
                    return (stats, None, None, False)

                # Calculate new dimensions
                new_size = calculate_target_dimensions(original_size, target_size)

                # Convert to RGB if needed (handles RGBA, LA, P modes)
                gray_bg_applied = False
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create gray background for transparency (matches training pad_color)
                    bg = Image.new('RGB', img.size, (114, 114, 114))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if 'A' in img.mode:
                        bg.paste(img, mask=img.getchannel('A'))
                        img = bg
                        gray_bg_applied = True  # Track that we applied gray background
                    else:
                        img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Perform downsampling
                resized = img.resize(new_size, Image.Resampling.LANCZOS)

                if dry_run:
                    # Estimate size without actually processing
                    # Use original format and compression to estimate
                    estimated_bytes = int(original_bytes * (new_size[0] * new_size[1]) / (original_size[0] * original_size[1]))

                    action = "would_downsample"
                    new_format = original_format

                    # Check if PNG→JPEG conversion would be beneficial
                    if original_format == 'PNG' and estimated_bytes > PNG_TO_JPEG_THRESHOLD:
                        # JPEG typically 5-10x smaller than PNG for photos
                        estimated_bytes = estimated_bytes // 7
                        action = "would_convert_to_jpeg"
                        new_format = "JPEG"

                    stats = ImageStats(
                        path=img_path,
                        original_size=original_size,
                        original_bytes=original_bytes,
                        original_format=original_format,
                        new_size=new_size,
                        new_bytes=estimated_bytes,
                        new_format=new_format,
                        action=action,
                        gray_background_applied=gray_bg_applied
                    )
                    return (stats, None, None, False)

                else:
                    # Process the image in memory and determine output format
                    # First, check PNG size to decide on conversion
                    should_convert_to_jpeg = False
                    output_format = original_format

                    if original_format == 'PNG':
                        # Use BytesIO to check PNG size without writing to disk
                        buffer = BytesIO()
                        resized.save(buffer, format='PNG', optimize=True)
                        png_size = buffer.tell()

                        if png_size > PNG_TO_JPEG_THRESHOLD:
                            should_convert_to_jpeg = True
                            output_format = 'JPEG'
                        else:
                            # Keep as PNG
                            output_format = 'PNG'
                            image_bytes = buffer.getvalue()

                    # Generate final image bytes based on determined format
                    if should_convert_to_jpeg or (original_format in ('JPEG', 'JPG')):
                        # Convert to JPEG
                        buffer = BytesIO()
                        resized.save(
                            buffer,
                            format='JPEG',
                            quality=JPEG_QUALITY,
                            optimize=True,
                            subsampling=0  # 4:4:4 chroma subsampling for best quality
                        )
                        image_bytes = buffer.getvalue()
                        output_format = 'JPEG'
                        action = 'converted' if should_convert_to_jpeg else 'downsampled'
                        output_path = img_path.with_suffix('.jpg')

                    elif original_format == 'WebP':
                        # Keep as WebP
                        buffer = BytesIO()
                        resized.save(buffer, format='WebP', quality=95, method=6)
                        image_bytes = buffer.getvalue()
                        output_format = 'WebP'
                        action = 'downsampled'
                        output_path = img_path

                    elif original_format == 'PNG' and not should_convert_to_jpeg:
                        # Already generated PNG bytes above
                        action = 'downsampled'
                        output_path = img_path

                    else:
                        # Other formats: keep original format
                        buffer = BytesIO()
                        resized.save(buffer, format=original_format, optimize=True)
                        image_bytes = buffer.getvalue()
                        output_format = original_format
                        action = 'downsampled'
                        output_path = img_path

                    # Determine if we need to delete the original file
                    # Delete if format changed (e.g., PNG -> JPEG means different extension)
                    should_delete_original = (output_path != img_path)

                    stats = ImageStats(
                        path=img_path,
                        original_size=original_size,
                        original_bytes=original_bytes,
                        original_format=original_format,
                        new_size=new_size,
                        new_bytes=len(image_bytes),
                        new_format=output_format,
                        action=action,
                        gray_background_applied=gray_bg_applied
                    )

                    return (stats, image_bytes, output_path, should_delete_original)

    except Exception as e:
        stats = ImageStats(
            path=img_path,
            original_size=(0, 0),
            original_bytes=img_path.stat().st_size if img_path.exists() else 0,
            original_format="UNKNOWN",
            action="error",
            error=str(e)
        )
        return (stats, None, None, False)


# ============================================================================
# Shard Processing
# ============================================================================

def process_shard_worker(args) -> Tuple[str, ShardStats, list]:
    """
    Worker function to process a single shard with batched I/O.
    Returns: (shard_name, ShardStats, list[ImageStats])
    """
    shard_path, target_size, dry_run = args
    shard_name = shard_path.name

    stats = ShardStats(shard_name=shard_name)
    image_stats_list = []

    try:
        # Collect all image files
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(shard_path.glob(f'*{ext}'))

        if not image_files:
            return (shard_name, stats, [])

        stats.total_images = len(image_files)

        print(f"[START] {shard_name}: Processing {len(image_files)} images...")

        # Process in batches for efficient I/O
        for i in range(0, len(image_files), BATCH_SIZE):
            batch = image_files[i:i + BATCH_SIZE]

            if i > 0 and VERBOSE:
                print(f"  [{shard_name}] Progress: {i}/{len(image_files)} files...")

            # Batch 1: Process all images in memory
            batch_results = []
            for img_file in batch:
                result = process_image(img_file, target_size, dry_run)
                batch_results.append(result)

            # Batch 2: Write all processed images to disk (if not dry run)
            if not dry_run:
                for img_stat, image_bytes, output_path, should_delete_original in batch_results:
                    # Update JSON metadata if gray background was applied
                    if img_stat.gray_background_applied:
                        json_path = img_stat.path.with_suffix('.json')
                        if json_path.exists():
                            add_tag_to_json(json_path, 'grey_background')

                    # Write the image if we have bytes to write
                    if image_bytes is not None and output_path is not None:
                        # Delete original if format changed
                        if should_delete_original and img_stat.path.exists():
                            img_stat.path.unlink()

                        # Write new image (overwrite if same path, create new if different)
                        with open(output_path, 'wb') as f:
                            f.write(image_bytes)

            # Batch 3: Update statistics
            for img_stat, _, _, _ in batch_results:
                image_stats_list.append(img_stat)

                # Update shard stats
                stats.original_bytes += img_stat.original_bytes
                stats.final_bytes += img_stat.new_bytes if img_stat.new_bytes else img_stat.original_bytes

                if img_stat.action == "skipped":
                    stats.skipped += 1
                elif img_stat.action in ("downsampled", "would_downsample"):
                    stats.downsampled += 1
                elif img_stat.action in ("converted", "would_convert_to_jpeg"):
                    stats.converted += 1
                elif img_stat.action == "error":
                    stats.errors += 1

                # Track gray background applications
                if img_stat.gray_background_applied:
                    stats.gray_backgrounds += 1

        print(f"[DONE] {shard_name}: Saved {stats.space_saved_mb:.1f} MB")
        return (shard_name, stats, image_stats_list)

    except Exception as e:
        print(f"Error processing shard {shard_name}: {e}")
        return (shard_name, stats, [])


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution function."""

    database_path = Path(r'L:\Dab\Dab')
    progress_file = Path(__file__).parent / 'downsample_progress.txt'

    if not database_path.exists():
        print(f"ERROR: Database path not found: {database_path}")
        return

    # Load progress
    completed_shards = load_progress(progress_file)

    if completed_shards:
        print(f"[OK] Resuming: {len(completed_shards)} shards already processed")

    # Get all shard directories
    all_shards = sorted([d for d in database_path.iterdir()
                        if d.is_dir() and d.name.startswith('shard_')])

    remaining_shards = [d for d in all_shards if d.name not in completed_shards]

    # Print configuration
    print(f"\n{'='*70}")
    print(f"OPPAI ORACLE - IMAGE DOWNSAMPLING FOR TRAINING")
    print(f"{'='*70}")
    print(f"Training Resolution: {TRAINING_SIZE}x{TRAINING_SIZE} (letterbox)")
    print(f"Target Size: {TARGET_SIZE}px (longest side)")
    if TARGET_SIZE == TRAINING_SIZE:
        print(f"Strategy: Perfect match to training (zero resize overhead)")
    else:
        print(f"Quality Headroom: {TARGET_SIZE/TRAINING_SIZE:.1f}x training resolution")
    print(f"\nOptimization Strategy:")
    print(f"  - Images ≤ {TARGET_SIZE}px: No changes (already optimal)")
    print(f"  - Images > {TARGET_SIZE}px: Downsample to {TARGET_SIZE}px")
    print(f"  - Large PNGs (>{PNG_TO_JPEG_THRESHOLD//1024//1024}MB): Convert to JPEG (quality={JPEG_QUALITY})")
    print(f"\nDataset: {database_path}")
    print(f"Total shards: {len(all_shards)}")
    print(f"Already processed: {len(completed_shards)}")
    print(f"Will process: {len(remaining_shards)}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Batch size: {BATCH_SIZE} images")

    if not remaining_shards:
        print("\n[OK] All shards already processed!")
        return

    # Show mode
    mode = "DRY RUN" if DRY_RUN else "EXECUTE"
    print(f"\nMode: {mode}")

    if DRY_RUN:
        print(f"{'='*70}")
        print(f"DRY RUN MODE - Estimating space savings")
        print(f"{'='*70}")
        print(f"No files will be modified. Run with --yes to execute.")
    else:
        print(f"{'='*70}")
        print(f"[WARNING] FILES WILL BE PERMANENTLY MODIFIED")
        print(f"{'='*70}")
        print(f"This will downsample and potentially convert {len(remaining_shards)} shards.")
        print(f"Original files will be REPLACED. Ensure you have backups if needed!")
        response = input("\nContinue? Type 'YES' to proceed: ").strip()

        if response != 'YES':
            print("Aborted by user.")
            return

    print(f"\nStarting processing with {NUM_WORKERS} parallel workers...\n")

    # Global statistics
    global_stats = ShardStats(shard_name="GLOBAL")
    completed_in_run = []

    # Process shards in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_shard = {
            executor.submit(process_shard_worker, (shard, TARGET_SIZE, DRY_RUN)): shard
            for shard in remaining_shards
        }

        print(f"Submitted {len(remaining_shards)} shards to worker pool\n")

        # Setup progress bar
        if HAS_TQDM:
            pbar = tqdm(total=len(remaining_shards), desc="Processing shards")

        # Collect results
        for future in as_completed(future_to_shard):
            shard_path = future_to_shard[future]
            try:
                shard_name, shard_stats, _ = future.result()

                # Aggregate global stats
                global_stats.total_images += shard_stats.total_images
                global_stats.skipped += shard_stats.skipped
                global_stats.downsampled += shard_stats.downsampled
                global_stats.converted += shard_stats.converted
                global_stats.errors += shard_stats.errors
                global_stats.gray_backgrounds += shard_stats.gray_backgrounds
                global_stats.original_bytes += shard_stats.original_bytes
                global_stats.final_bytes += shard_stats.final_bytes

                completed_in_run.append(shard_name)

                # Save progress
                if not DRY_RUN:
                    save_progress(progress_file, shard_name)

                # Progress message
                action_verb = "Would save" if DRY_RUN else "Saved"
                completion_msg = (
                    f"[{len(completed_in_run)}/{len(remaining_shards)}] {shard_name} | "
                    f"{action_verb} {shard_stats.space_saved_mb:.1f} MB | "
                    f"Processed: {shard_stats.downsampled + shard_stats.converted}/{shard_stats.total_images}"
                )

                if HAS_TQDM:
                    pbar.update(1)
                    tqdm.write(completion_msg)
                else:
                    print(completion_msg)

            except Exception as e:
                print(f"Error getting result for {shard_path.name}: {e}")

        if HAS_TQDM:
            pbar.close()

    # Final summary
    print(f"\n{'='*70}")
    print(f"{'DRY RUN ' if DRY_RUN else ''}PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Shards processed: {len(completed_in_run)}")
    print(f"Total images: {global_stats.total_images}")
    print(f"\nActions:")
    action_prefix = "Would be " if DRY_RUN else ""
    print(f"  - {action_prefix}Skipped (already optimal): {global_stats.skipped}")
    print(f"  - {action_prefix}Downsampled: {global_stats.downsampled}")
    print(f"  - {action_prefix}Converted to JPEG: {global_stats.converted}")
    if global_stats.gray_backgrounds > 0:
        json_action = "Would be tagged" if DRY_RUN else "Tagged"
        print(f"  - Gray backgrounds applied: {global_stats.gray_backgrounds} ({json_action} in JSON)")
    print(f"  - Errors: {global_stats.errors}")

    print(f"\nStorage Impact:")
    print(f"  - Original size: {global_stats.original_bytes / (1024**3):.2f} GB")
    print(f"  - Final size: {global_stats.final_bytes / (1024**3):.2f} GB")
    print(f"  - Space saved: {global_stats.space_saved / (1024**3):.2f} GB")
    print(f"  - Compression ratio: {global_stats.compression_ratio:.1%}")

    if DRY_RUN:
        print(f"\n{'='*70}")
        print(f"This was a DRY RUN - no files were modified.")
        print(f"Run with --yes to execute the downsampling.")
        print(f"{'='*70}")
    else:
        print(f"\nProgress saved to: {progress_file}")

    print(f"{'='*70}")


if __name__ == '__main__':
    mp.freeze_support()
    main()
