"""
Ultra-fast low-tag image filtering.

Key optimizations:
- Multiprocessing for parallel shard processing
- Large file batches (500 files at once)
- Progress tracking for resume capability
- Efficient I/O patterns

Usage:
    python filter_low_tag_images_turbo.py --dry-run           # Preview
    python filter_low_tag_images_turbo.py --yes               # Skip confirmation
    python filter_low_tag_images_turbo.py --min-tags 25       # Custom threshold
    python filter_low_tag_images_turbo.py --shard-range 0-10  # Test range
"""

import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Try to use orjson for faster JSON parsing (3-5x faster than stdlib json)
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Configuration
DEFAULT_MIN_TAG_COUNT = 26
BATCH_SIZE = 500  # Process files in batches for better performance
NUM_WORKERS = 8   # Parallel shard processing
FILE_WORKERS = 4  # Parallel file processing within each shard
DATASET_BASE = r"L:\Dab\Dab"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.gif']

# Parse command line args
DRY_RUN = '--dry-run' in sys.argv
SKIP_CONFIRM = '--yes' in sys.argv or '-y' in sys.argv

def get_arg_value(arg_name, default):
    """Get command line argument value."""
    for i, arg in enumerate(sys.argv):
        if arg == arg_name and i + 1 < len(sys.argv):
            try:
                return type(default)(sys.argv[i + 1])
            except (ValueError, TypeError, IndexError):
                return default
    return default

MIN_TAG_COUNT = get_arg_value('--min-tags', DEFAULT_MIN_TAG_COUNT)

def load_progress(progress_file):
    """Load completed shards from progress file."""
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            print(f"Warning: Could not load progress: {e}")
    return set()

def save_progress(progress_file, shard_name):
    """Append completed shard to progress file."""
    try:
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{shard_name}\n")
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")

def parse_tags_fast(tags_str):
    """Fast tag parsing using string operations."""
    if not tags_str or not isinstance(tags_str, str):
        return 0
    # Count commas + 1 for comma-separated tags
    # Or split by spaces for space-separated tags
    if ',' in tags_str:
        return len([t for t in tags_str.split(',') if t.strip()])
    else:
        return len([t for t in tags_str.split() if t.strip()])

def find_image_for_json(json_path):
    """Find the corresponding image file for a JSON file."""
    base_name = json_path.stem
    parent = json_path.parent

    for ext in IMAGE_EXTENSIONS:
        image_path = parent / f"{base_name}{ext}"
        if image_path.exists():
            return image_path
    return None

def analyze_json_file(json_path):
    """
    Analyze a single JSON file for tag count.
    Returns: (json_path, tag_count, image_path, should_delete)
    """
    try:
        # Use orjson if available (3-5x faster for batch analysis)
        if HAS_ORJSON:
            data = orjson.loads(Path(json_path).read_bytes())
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        tags_str = data.get('tags', '')
        tag_count = parse_tags_fast(tags_str)

        if tag_count < MIN_TAG_COUNT:
            image_path = find_image_for_json(json_path)
            return (json_path, tag_count, image_path, True)
        else:
            return (json_path, tag_count, None, False)

    except Exception as e:
        return (json_path, 0, None, False)

def delete_file_pair(json_path, image_path):
    """Delete a JSON/image file pair."""
    deleted = {'json': False, 'image': False}

    try:
        if json_path and json_path.exists():
            json_path.unlink()
            deleted['json'] = True
    except Exception as e:
        pass

    try:
        if image_path and image_path.exists():
            image_path.unlink()
            deleted['image'] = True
    except Exception as e:
        pass

    return deleted

def process_shard_worker(args):
    """
    Worker function to process a single shard.
    Returns: (shard_name, stats_dict)
    """
    shard_path, dry_run = args
    shard_name = shard_path.name

    stats = {
        'total_json': 0,
        'to_delete': 0,
        'deleted_json': 0,
        'deleted_images': 0,
        'missing_images': 0,
        'errors': 0,
        'tag_distribution': defaultdict(int),
        'sample_deletions': []  # Store sample files with details for dry run
    }

    try:
        # Get all JSON files
        json_files = list(shard_path.glob('*.json'))
        stats['total_json'] = len(json_files)

        # Analyze all files first
        files_to_delete = []

        # Process in batches with parallel I/O for better performance
        for i in range(0, len(json_files), BATCH_SIZE):
            batch = json_files[i:i + BATCH_SIZE]

            # Use ThreadPoolExecutor for parallel file I/O
            with ThreadPoolExecutor(max_workers=FILE_WORKERS) as executor:
                futures = [executor.submit(analyze_json_file, json_file) for json_file in batch]

                for future in as_completed(futures):
                    json_path, tag_count, image_path, should_delete = future.result()

                    stats['tag_distribution'][tag_count] += 1

                    if should_delete:
                        stats['to_delete'] += 1
                        if not image_path:
                            stats['missing_images'] += 1
                        files_to_delete.append((json_path, image_path))

                        # Store sample deletions for dry run (first 5 from this shard)
                        if dry_run and len(stats['sample_deletions']) < 5:
                            stats['sample_deletions'].append({
                                'json': json_path,
                                'image': image_path,
                                'tag_count': tag_count,
                                'json_exists': json_path.exists(),
                                'image_exists': image_path.exists() if image_path else False
                            })

        # Delete files if not dry run (also parallelize deletions)
        if not dry_run and files_to_delete:
            with ThreadPoolExecutor(max_workers=FILE_WORKERS) as executor:
                futures = [executor.submit(delete_file_pair, json_path, image_path)
                          for json_path, image_path in files_to_delete]

                for future in as_completed(futures):
                    deleted = future.result()
                    if deleted['json']:
                        stats['deleted_json'] += 1
                    if deleted['image']:
                        stats['deleted_images'] += 1

        return (shard_name, stats)

    except Exception as e:
        print(f"Error processing shard {shard_name}: {e}")
        return (shard_name, stats)

def parse_shard_range(range_str):
    """Parse shard range string like '0-10' into (start, end)."""
    try:
        start, end = map(int, range_str.split('-'))
        return (start, end)
    except (ValueError, AttributeError):
        return None

def main():
    """Main function with multiprocessing."""
    base_path = Path(DATASET_BASE)
    progress_file = Path(__file__).parent / 'filter_progress.txt'

    if not base_path.exists():
        print(f"Error: Dataset path not found: {DATASET_BASE}")
        return

    # Parse shard range if provided
    shard_range = None
    for i, arg in enumerate(sys.argv):
        if arg == '--shard-range' and i + 1 < len(sys.argv):
            shard_range = parse_shard_range(sys.argv[i + 1])
            break

    # Load progress
    completed_shards = load_progress(progress_file)

    # Get all shard directories
    all_shards = sorted([d for d in base_path.iterdir()
                        if d.is_dir() and d.name.startswith('shard_')])

    # Apply shard range filter
    if shard_range:
        start, end = shard_range
        all_shards = [s for s in all_shards
                     if start <= int(s.name.split('_')[1]) <= end]

    # Filter out completed shards
    remaining_shards = [s for s in all_shards if s.name not in completed_shards]

    # Header
    print(f"{'='*70}")
    print(f"FILTER LOW-TAG IMAGES - {'DRY RUN' if DRY_RUN else 'DELETION MODE'}")
    print(f"{'='*70}")
    print(f"Minimum tag count: {MIN_TAG_COUNT}")
    print(f"Dataset path: {DATASET_BASE}")
    print(f"Shard workers: {NUM_WORKERS} (parallel shards)")
    print(f"File workers: {FILE_WORKERS} (parallel I/O per shard)")
    print(f"Batch size: {BATCH_SIZE} files")
    print(f"\nTotal shards: {len(all_shards)}")
    print(f"Already completed: {len(completed_shards)}")
    print(f"Will process: {len(remaining_shards)}")

    if not remaining_shards:
        print("\n[OK] All shards already processed!")
        return

    if shard_range:
        print(f"Shard range: {shard_range[0]}-{shard_range[1]}")

    print(f"Progress file: {progress_file}")
    print(f"{'='*70}\n")

    # Dry run info
    if DRY_RUN:
        print("[DRY RUN] No files will be deleted\n")

    # Confirmation
    if not DRY_RUN and not SKIP_CONFIRM:
        print(f"[WARNING] About to DELETE files from {len(remaining_shards)} shards")
        print(f"[WARNING] This action is PERMANENT and cannot be undone!")
        response = input("\nType 'yes' to continue: ").strip().lower()

        if response != 'yes':
            print("Aborted.")
            return

    print("\nStarting processing...\n")

    # Aggregate statistics
    total_stats = {
        'total_json': 0,
        'to_delete': 0,
        'deleted_json': 0,
        'deleted_images': 0,
        'missing_images': 0,
        'errors': 0,
        'tag_distribution': defaultdict(int),
        'shards_processed': 0,
        'sample_deletions': []  # Collect sample deletions for dry run
    }

    # Process shards in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_shard = {
            executor.submit(process_shard_worker, (shard, DRY_RUN)): shard
            for shard in remaining_shards
        }

        # Setup progress bar
        if HAS_TQDM:
            pbar = tqdm(total=len(remaining_shards), desc="Processing shards", unit="shard")

        # Process results as they complete
        for future in as_completed(future_to_shard):
            shard_path = future_to_shard[future]
            try:
                shard_name, stats = future.result()

                # Aggregate stats
                total_stats['total_json'] += stats['total_json']
                total_stats['to_delete'] += stats['to_delete']
                total_stats['deleted_json'] += stats['deleted_json']
                total_stats['deleted_images'] += stats['deleted_images']
                total_stats['missing_images'] += stats['missing_images']
                total_stats['errors'] += stats['errors']
                total_stats['shards_processed'] += 1

                for tag_count, count in stats['tag_distribution'].items():
                    total_stats['tag_distribution'][tag_count] += count

                # Collect sample deletions for dry run (limit to 15 total samples)
                if DRY_RUN and len(total_stats['sample_deletions']) < 15:
                    for sample in stats.get('sample_deletions', []):
                        if len(total_stats['sample_deletions']) < 15:
                            total_stats['sample_deletions'].append(sample)

                # Save progress
                if not DRY_RUN:
                    save_progress(progress_file, shard_name)

                # Update progress
                if HAS_TQDM:
                    pbar.update(1)
                    if stats['to_delete'] > 0:
                        action = "marked" if DRY_RUN else "deleted"
                        tqdm.write(f"{shard_name}: {stats['to_delete']} {action} (of {stats['total_json']})")
                else:
                    action = "marked" if DRY_RUN else "deleted"
                    print(f"[{total_stats['shards_processed']}/{len(remaining_shards)}] "
                          f"{shard_name}: {stats['to_delete']} {action} (of {stats['total_json']})")

            except Exception as e:
                print(f"Error getting result for {shard_path.name}: {e}")

        if HAS_TQDM:
            pbar.close()

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Shards processed: {total_stats['shards_processed']}")
    print(f"Total JSON files: {total_stats['total_json']:,}")
    print(f"Files with <{MIN_TAG_COUNT} tags: {total_stats['to_delete']:,} "
          f"({total_stats['to_delete']/total_stats['total_json']*100:.2f}%)")

    if DRY_RUN:
        print(f"\nWould delete:")
        print(f"  - JSON files: {total_stats['to_delete']:,}")
        print(f"  - Image files: {total_stats['to_delete'] - total_stats['missing_images']:,}")
        if total_stats['missing_images'] > 0:
            print(f"  - Missing images: {total_stats['missing_images']:,}")
    else:
        print(f"\nDeleted:")
        print(f"  - JSON files: {total_stats['deleted_json']:,}")
        print(f"  - Image files: {total_stats['deleted_images']:,}")

    # Calculate retention
    retained = total_stats['total_json'] - total_stats['to_delete']
    print(f"\nRetained: {retained:,} images ({retained/total_stats['total_json']*100:.2f}%)")

    # Show sample deletions for dry run verification
    if DRY_RUN and total_stats['sample_deletions']:
        print(f"\n{'='*70}")
        print("SAMPLE FILES FLAGGED FOR DELETION (Verification)")
        print(f"{'='*70}")
        for i, sample in enumerate(total_stats['sample_deletions'], 1):
            print(f"\nSample #{i}:")
            print(f"  Tag count: {sample['tag_count']} tags (< {MIN_TAG_COUNT} threshold)")
            print(f"  JSON: {sample['json']}")
            print(f"    - Exists: {'YES' if sample['json_exists'] else 'NO'}")
            if sample['image']:
                print(f"  Image: {sample['image']}")
                print(f"    - Exists: {'YES' if sample['image_exists'] else 'NO'}")
            else:
                print(f"  Image: MISSING (no matching image file found)")

            # Verify tag count by re-reading the JSON
            if sample['json_exists']:
                try:
                    # Use orjson if available (3-5x faster)
                    if HAS_ORJSON:
                        data = orjson.loads(Path(sample['json']).read_bytes())
                    else:
                        with open(sample['json'], 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    tags_str = data.get('tags', '')
                    verified_count = parse_tags_fast(tags_str)
                    print(f"  Verified tag count: {verified_count} tags")
                    if verified_count != sample['tag_count']:
                        print(f"    [WARNING] Count mismatch!")
                except Exception as e:
                    print(f"  Could not verify: {e}")

    # Show tag distribution for deleted files
    print(f"\n{'='*70}")
    print("TAG DISTRIBUTION (files to delete)")
    print(f"{'='*70}")

    deleted_tags = [(tc, cnt) for tc, cnt in total_stats['tag_distribution'].items()
                    if tc < MIN_TAG_COUNT]
    deleted_tags.sort()

    for tag_count, count in deleted_tags[:20]:  # Show top 20
        pct = count / total_stats['total_json'] * 100
        print(f"  {tag_count:3d} tags: {count:6,} images ({pct:5.2f}%)")

    if len(deleted_tags) > 20:
        print(f"  ... and {len(deleted_tags) - 20} more tag counts")

    print(f"{'='*70}")

    if DRY_RUN:
        print("\n[DRY RUN] No files were deleted.")
        print("Run without --dry-run to actually delete files.")
    else:
        print(f"\nDeletion complete! Progress saved to: {progress_file}")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
