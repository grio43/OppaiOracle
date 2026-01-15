r"""
Script to identify and clean up orphan JSON files and orphan images from L:\Dab\Dab
Orphan JSON: JSON file without corresponding image
Orphan Image: Image file without corresponding JSON

Optimized for large datasets (millions of files):
- Uses os.scandir() for faster directory iteration
- Parallel size calculation and deletion
- Configurable thread count via --threads N
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

DEFAULT_THREADS = 32
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

def process_shard(shard_dir, shard_idx, total_shards, verbose=False):
    """Process a single shard directory to find orphans using os.scandir()."""
    if verbose:
        print(f"[Thread {threading.current_thread().name}] Processing shard {shard_idx}/{total_shards}: {shard_dir.name}")

    orphan_jsons = []
    orphan_images = []
    total_size = 0

    # Use os.scandir() - much faster than listdir as it provides file info directly
    try:
        # Build stem -> extensions mapping using dicts for O(1) lookups
        stems_with_json = {}  # stem -> (full_path, size)
        stems_with_image = {}  # stem -> list of (full_path, size)

        with os.scandir(shard_dir) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                name = entry.name
                dot_idx = name.rfind('.')
                if dot_idx == -1:
                    continue
                stem = name[:dot_idx]
                ext = name[dot_idx:].lower()

                if ext == '.json':
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    stems_with_json[stem] = (entry.path, size)
                elif ext in IMAGE_EXTENSIONS:
                    # Cache the stat result for size calculation later
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    if stem not in stems_with_image:
                        stems_with_image[stem] = []
                    stems_with_image[stem].append((entry.path, size))

        # Find orphans - O(n) with set operations
        # Orphan JSONs: have JSON but no image
        for stem, (json_path, size) in stems_with_json.items():
            if stem not in stems_with_image:
                orphan_jsons.append((json_path, size))
                total_size += size

        # Orphan images: have image but no JSON
        for stem, images in stems_with_image.items():
            if stem not in stems_with_json:
                for img_path, size in images:
                    orphan_images.append((img_path, size))
                    total_size += size

    except Exception as e:
        print(f"Error reading {shard_dir}: {e}")
        return orphan_jsons, orphan_images, total_size

    if verbose:
        print(f"[Thread {threading.current_thread().name}] Shard {shard_idx}/{total_shards} done: {len(orphan_jsons)} orphan JSONs, {len(orphan_images)} orphan images")

    return orphan_jsons, orphan_images, total_size

def find_orphans(base_path, num_threads=DEFAULT_THREADS, verbose=False):
    """Find orphan JSON files and orphan images in the dataset using multiple threads."""
    base_path = Path(base_path)

    if not base_path.exists():
        raise ValueError(f"Path does not exist: {base_path}")
    if not base_path.is_dir():
        raise ValueError(f"Path is not a directory: {base_path}")

    orphan_jsons = []
    orphan_images = []
    total_size = 0

    # Process each shard directory
    shard_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    total_shards = len(shard_dirs)

    print(f"Found {total_shards} shard directories")
    print(f"Using {num_threads} threads for processing...")
    print("=" * 80)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all shard processing tasks with index
        futures = {
            executor.submit(process_shard, shard_dir, idx + 1, total_shards, verbose): (shard_dir, idx)
            for idx, shard_dir in enumerate(shard_dirs)
        }

        # Use tqdm if available for progress tracking
        if HAS_TQDM:
            future_iter = tqdm(as_completed(futures), total=len(futures), desc="Scanning shards", unit="shard")
        else:
            future_iter = as_completed(futures)
            processed_count = 0

        # Process results as they complete
        for future in future_iter:
            shard_dir, idx = futures[future]
            try:
                jsons, images, size = future.result()
                orphan_jsons.extend(jsons)
                orphan_images.extend(images)
                total_size += size
                if not HAS_TQDM:
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"  Processed {processed_count}/{total_shards} shards...")
            except Exception as e:
                print(f"Error processing {shard_dir}: {e}")

    print("=" * 80)
    print(f"Scanning complete!")

    return orphan_jsons, orphan_images, total_size

def delete_file(file_path):
    """Delete a single file. Returns (success, path)."""
    try:
        os.unlink(file_path)
        return True, file_path
    except Exception as e:
        return False, f"{file_path}: {e}"


def parallel_delete(file_paths, num_threads=DEFAULT_THREADS):
    """Delete files in parallel using ThreadPoolExecutor."""
    deleted_count = 0
    error_count = 0
    total_files = len(file_paths)

    print(f"\nDeleting {total_files} files using {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(delete_file, path) for path in file_paths]

        if HAS_TQDM:
            future_iter = tqdm(as_completed(futures), total=len(futures), desc="Deleting files", unit="file")
            for future in future_iter:
                success, result = future.result()
                if success:
                    deleted_count += 1
                else:
                    error_count += 1
                    if error_count <= 10:  # Only print first 10 errors
                        print(f"  Error: {result}")
        else:
            for i, future in enumerate(as_completed(futures)):
                success, result = future.result()
                if success:
                    deleted_count += 1
                else:
                    error_count += 1
                    if error_count <= 10:  # Only print first 10 errors
                        print(f"  Error: {result}")
                # Progress update without tqdm
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{total_files} files...")

    return deleted_count, error_count


def _invalidate_training_caches():
    """Invalidate split and Arrow caches after file deletion.

    This ensures training will rescan the filesystem instead of
    using stale file lists that reference deleted files.
    """
    # Find OppaiOracle project root (parent of data_preparation)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    logs_dir = project_root / "logs"

    if not logs_dir.exists():
        return

    deleted_caches = []

    # Invalidate split cache
    splits_dir = logs_dir / "splits"
    if splits_dir.exists():
        for cache_file in splits_dir.glob("*.txt"):
            try:
                cache_file.unlink()
                deleted_caches.append(str(cache_file.name))
            except OSError as e:
                print(f"  Warning: Could not delete {cache_file}: {e}")

    # Invalidate Arrow metadata cache
    metadata_dir = logs_dir / "metadata_cache"
    if metadata_dir.exists():
        for cache_file in metadata_dir.iterdir():
            try:
                if cache_file.is_file():
                    cache_file.unlink()
                    deleted_caches.append(str(cache_file.name))
            except OSError as e:
                print(f"  Warning: Could not delete {cache_file}: {e}")

    if deleted_caches:
        print(f"\nInvalidated {len(deleted_caches)} cache file(s) to force fresh rescan:")
        for f in deleted_caches[:5]:
            print(f"  - {f}")
        if len(deleted_caches) > 5:
            print(f"  ... and {len(deleted_caches) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Find and clean up orphan JSON files and images from dataset"
    )
    parser.add_argument("--delete", action="store_true", help="Delete orphan files")
    parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt when deleting")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help=f"Number of threads to use (default: {DEFAULT_THREADS})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--path", type=str, default=r"L:\Dab\Dab",
                        help="Base path to scan (default: L:\\Dab\\Dab)")
    parser.add_argument("--output", "-o", type=str, default="orphan_files_list.txt",
                        help="Output file path for orphan list (default: orphan_files_list.txt)")
    args = parser.parse_args()

    base_path = args.path
    num_threads = max(1, args.threads)

    print(f"Scanning {base_path} for orphan files...")
    print("=" * 80)

    orphan_jsons, orphan_images, total_size = find_orphans(base_path, num_threads, args.verbose)

    print(f"\nRESULTS:")
    print(f"  Orphan JSON files (no matching image): {len(orphan_jsons)}")
    print(f"  Orphan images (no matching JSON): {len(orphan_images)}")
    print(f"  Total size of orphan files: {total_size / (1024**2):.2f} MB")
    print("=" * 80)

    # Write full list to file
    if orphan_jsons or orphan_images:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("ORPHAN JSON FILES (no matching image):\n")
            f.write("=" * 80 + "\n")
            for file_path, size in orphan_jsons:
                f.write(f"{file_path}\n")

            f.write(f"\n\nORPHAN IMAGES (no matching JSON):\n")
            f.write("=" * 80 + "\n")
            for file_path, size in orphan_images:
                f.write(f"{file_path}\n")

        print(f"\nFull list saved to: {args.output}")

    if orphan_jsons:
        print("\nSample orphan JSON files (first 10):")
        for file_path, size in orphan_jsons[:10]:
            print(f"  {file_path}")
        if len(orphan_jsons) > 10:
            print(f"  ... and {len(orphan_jsons) - 10} more")

    if orphan_images:
        print("\nSample orphan images (first 10):")
        for file_path, size in orphan_images[:10]:
            print(f"  {file_path}")
        if len(orphan_images) > 10:
            print(f"  ... and {len(orphan_images) - 10} more")

    if orphan_jsons or orphan_images:
        if args.delete:
            # Collect all file paths for parallel deletion
            all_paths = [path for path, size in orphan_jsons] + [path for path, size in orphan_images]
            total_count = len(all_paths)

            # Confirm before deletion (unless --force is used)
            print(f"\nAbout to delete {total_count} files ({total_size / (1024**2):.2f} MB)")
            if args.force:
                proceed = True
            elif not sys.stdin.isatty():
                print("Non-interactive mode detected. Use --force to skip confirmation.")
                proceed = False
            else:
                confirm = input("Are you sure you want to proceed? [y/N]: ").strip().lower()
                proceed = (confirm == 'y')

            if not proceed:
                print("Deletion cancelled.")
            else:
                deleted_count, error_count = parallel_delete(all_paths, num_threads)
                print(f"\nSuccessfully deleted {deleted_count} files!")

                # Invalidate training caches after deleting files
                # Without this, training may use stale file lists
                _invalidate_training_caches()

                if error_count > 0:
                    print(f"Failed to delete {error_count} files.")
                    return 1
        else:
            print(f"\nTo delete these files, run: python cleanup_orphans.py --delete")
            print(f"Optional flags: --threads N (default {DEFAULT_THREADS}), --verbose, --force")
    else:
        print("\nNo orphan files found! Dataset is clean.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
