r"""
Script to identify and clean up orphan JSON files and orphan images from L:\Dab\Dab
Orphan JSON: JSON file without corresponding image
Orphan Image: Image file without corresponding JSON
"""

import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

NUM_THREADS = 16

def process_shard(shard_dir, shard_idx, total_shards):
    """Process a single shard directory to find orphans."""
    print(f"[Thread {threading.current_thread().name}] Processing shard {shard_idx}/{total_shards}: {shard_dir.name}")

    orphan_jsons = []
    orphan_images = []

    # Use os.listdir for faster performance
    try:
        filenames = os.listdir(shard_dir)
    except Exception as e:
        print(f"Error reading {shard_dir}: {e}")
        return orphan_jsons, orphan_images

    # Separate by extension - use filename directly instead of Path objects
    files_by_stem = defaultdict(list)
    for filename in filenames:
        stem, ext = os.path.splitext(filename)
        if ext:  # Only process files with extensions
            files_by_stem[stem].append(ext.lower())

    # Check for orphans
    for stem, extensions in files_by_stem.items():
        has_json = '.json' in extensions
        has_image = any(ext in extensions for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif'])

        if has_json and not has_image:
            orphan_jsons.append(shard_dir / f"{stem}.json")
        elif has_image and not has_json:
            # Find the actual image file
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                if ext in extensions:
                    orphan_images.append(shard_dir / f"{stem}{ext}")
                    break

    print(f"[Thread {threading.current_thread().name}] Shard {shard_idx}/{total_shards} done: {len(orphan_jsons)} orphan JSONs, {len(orphan_images)} orphan images")
    return orphan_jsons, orphan_images

def find_orphans(base_path):
    """Find orphan JSON files and orphan images in the dataset using multiple threads."""
    base_path = Path(base_path)

    orphan_jsons = []
    orphan_images = []

    # Process each shard directory
    shard_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    total_shards = len(shard_dirs)

    print(f"Found {total_shards} shard directories")
    print(f"Using {NUM_THREADS} threads for processing...")
    print("=" * 80)

    processed_count = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all shard processing tasks with index
        futures = {
            executor.submit(process_shard, shard_dir, idx + 1, total_shards): (shard_dir, idx)
            for idx, shard_dir in enumerate(shard_dirs)
        }

        # Process results as they complete
        for future in as_completed(futures):
            shard_dir, idx = futures[future]
            try:
                jsons, images = future.result()
                with lock:
                    orphan_jsons.extend(jsons)
                    orphan_images.extend(images)
                    processed_count += 1
            except Exception as e:
                print(f"Error processing {shard_dir}: {e}")

    print("=" * 80)
    print(f"Processed {processed_count}/{total_shards} shards - Complete!")

    return orphan_jsons, orphan_images

def main():
    import sys
    base_path = r"L:\Dab\Dab"
    auto_delete = '--delete' in sys.argv

    print(f"Scanning {base_path} for orphan files...")
    print("=" * 80)

    orphan_jsons, orphan_images = find_orphans(base_path)

    print(f"\nRESULTS:")
    print(f"  Orphan JSON files (no matching image): {len(orphan_jsons)}")
    print(f"  Orphan images (no matching JSON): {len(orphan_images)}")
    print("=" * 80)

    # Write full list to file
    if orphan_jsons or orphan_images:
        with open('orphan_files_list.txt', 'w', encoding='utf-8') as f:
            f.write("ORPHAN JSON FILES (no matching image):\n")
            f.write("=" * 80 + "\n")
            for json_file in orphan_jsons:
                f.write(f"{json_file}\n")

            f.write(f"\n\nORPHAN IMAGES (no matching JSON):\n")
            f.write("=" * 80 + "\n")
            for img_file in orphan_images:
                f.write(f"{img_file}\n")

        print(f"\nFull list saved to: orphan_files_list.txt")

    if orphan_jsons:
        print("\nSample orphan JSON files (first 10):")
        for json_file in orphan_jsons[:10]:
            print(f"  {json_file}")
        if len(orphan_jsons) > 10:
            print(f"  ... and {len(orphan_jsons) - 10} more")

    if orphan_images:
        print("\nSample orphan images (first 10):")
        for img_file in orphan_images[:10]:
            print(f"  {img_file}")
        if len(orphan_images) > 10:
            print(f"  ... and {len(orphan_images) - 10} more")

    # Calculate total size
    if orphan_jsons or orphan_images:
        total_size = 0
        for f in orphan_jsons + orphan_images:
            if f.exists():
                total_size += f.stat().st_size

        print(f"\nTotal size of orphan files: {total_size / (1024**2):.2f} MB")
        print("=" * 80)

        if auto_delete:
            deleted_count = 0
            print("\nAuto-delete enabled. Deleting orphan files...")

            for f in orphan_jsons + orphan_images:
                try:
                    if f.exists():
                        f.unlink()
                        deleted_count += 1
                        if deleted_count % 10 == 0:
                            print(f"  Deleted {deleted_count} files...")
                except Exception as e:
                    print(f"  Error deleting {f}: {e}")

            print(f"\nSuccessfully deleted {deleted_count} orphan files!")
        else:
            print("\nTo delete these files, run: python cleanup_orphans.py --delete")
    else:
        print("\nNo orphan files found! Dataset is clean.")

if __name__ == "__main__":
    main()
