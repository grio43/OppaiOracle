"""
Filter and delete image/JSON pairs with fewer than 26 tags.

This script removes poorly-tagged images from the dataset to improve
training quality and reduce training time.

Usage:
    python filter_low_tag_images.py --dry-run  # Preview what would be deleted
    python filter_low_tag_images.py            # Actually delete files
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

# Configuration
MIN_TAG_COUNT = 26  # Keep images with 26+ tags, delete images with <26 tags
DATASET_BASE = r"L:\Dab\Dab"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.gif']

def parse_tags(tags_str):
    """Parse comma-separated tags, matching dataset loader logic."""
    if not tags_str or not isinstance(tags_str, str):
        return []

    # Split by comma, strip whitespace, filter empty strings
    tags = [tag.strip() for tag in tags_str.split(',')]
    tags = [tag for tag in tags if tag]
    return tags

def find_image_for_json(json_path):
    """Find the corresponding image file for a JSON file."""
    json_file = Path(json_path)
    base_name = json_file.stem  # filename without extension

    # Check for image with same base name
    for ext in IMAGE_EXTENSIONS:
        image_path = json_file.parent / f"{base_name}{ext}"
        if image_path.exists():
            return image_path

    return None

def analyze_shard(shard_path, dry_run=True):
    """Analyze a shard and optionally delete low-tag images."""
    shard_dir = Path(shard_path)

    if not shard_dir.exists():
        return None

    json_files = list(shard_dir.glob("*.json"))

    stats = {
        'total_json': len(json_files),
        'to_delete': 0,
        'deleted_json': 0,
        'deleted_images': 0,
        'missing_images': 0,
        'errors': 0,
        'tag_distribution': defaultdict(int)
    }

    files_to_delete = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tags_str = data.get('tags', '')
            tags = parse_tags(tags_str)
            tag_count = len(tags)

            stats['tag_distribution'][tag_count] += 1

            # If tag count is below threshold, mark for deletion
            if tag_count < MIN_TAG_COUNT:
                image_path = find_image_for_json(json_file)

                if image_path:
                    files_to_delete.append({
                        'json': json_file,
                        'image': image_path,
                        'tags': tag_count
                    })
                    stats['to_delete'] += 1
                else:
                    stats['missing_images'] += 1
                    stats['to_delete'] += 1

        except Exception as e:
            stats['errors'] += 1

    # Delete files if not dry run
    if not dry_run and files_to_delete:
        for file_pair in files_to_delete:
            json_deleted = False
            image_deleted = False

            try:
                # Try to delete image first (if it exists)
                # This order is safer: if JSON exists without image, it's still usable
                # But if image exists without JSON, it's an orphan that's hard to clean up
                if file_pair['image']:
                    try:
                        if file_pair['image'].exists():
                            file_pair['image'].unlink()
                            image_deleted = True
                            stats['deleted_images'] += 1
                    except FileNotFoundError:
                        # Image was already deleted (race condition with another process)
                        image_deleted = True
                    except Exception as e:
                        print(f"Warning: Could not delete image {file_pair['image']}: {e}")
                        # Continue to try deleting JSON anyway

                # Now delete JSON file
                try:
                    if file_pair['json'].exists():
                        file_pair['json'].unlink()
                        json_deleted = True
                        stats['deleted_json'] += 1
                except FileNotFoundError:
                    # JSON was already deleted
                    json_deleted = True

                # Log warning if we have an inconsistent state
                if json_deleted and not image_deleted and file_pair['image']:
                    print(f"Warning: Orphaned image file may exist: {file_pair['image']}")

            except Exception as e:
                print(f"Error deleting {file_pair['json']}: {e}")
                stats['errors'] += 1

    return stats

def process_all_shards(dry_run=True, shard_range=None):
    """Process all shards in the dataset."""
    base_path = Path(DATASET_BASE)

    # Find all shard directories
    all_shards = sorted([d for d in base_path.iterdir()
                        if d.is_dir() and d.name.startswith('shard_')])

    if shard_range:
        start, end = shard_range
        all_shards = [s for s in all_shards
                     if start <= int(s.name.split('_')[1]) <= end]

    print(f"{'='*70}")
    print(f"FILTER LOW-TAG IMAGES - {'DRY RUN' if dry_run else 'DELETION MODE'}")
    print(f"{'='*70}")
    print(f"Minimum tag count: {MIN_TAG_COUNT}")
    print(f"Total shards to process: {len(all_shards)}")
    print(f"Dataset path: {DATASET_BASE}")
    print(f"{'='*70}\n")

    if not dry_run:
        response = input("WARNING: This will permanently delete files. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        print()

    # Aggregate statistics
    total_stats = {
        'total_json': 0,
        'to_delete': 0,
        'deleted_json': 0,
        'deleted_images': 0,
        'missing_images': 0,
        'errors': 0,
        'tag_distribution': defaultdict(int),
        'shards_processed': 0
    }

    # Process each shard
    for i, shard in enumerate(all_shards):
        print(f"Processing {shard.name}... ({i+1}/{len(all_shards)})", end='', flush=True)

        shard_stats = analyze_shard(shard, dry_run=dry_run)

        if shard_stats:
            total_stats['total_json'] += shard_stats['total_json']
            total_stats['to_delete'] += shard_stats['to_delete']
            total_stats['deleted_json'] += shard_stats['deleted_json']
            total_stats['deleted_images'] += shard_stats['deleted_images']
            total_stats['missing_images'] += shard_stats['missing_images']
            total_stats['errors'] += shard_stats['errors']
            total_stats['shards_processed'] += 1

            for tag_count, count in shard_stats['tag_distribution'].items():
                total_stats['tag_distribution'][tag_count] += count

            print(f" {shard_stats['to_delete']} marked" if dry_run
                  else f" {shard_stats['deleted_json']} deleted")
        else:
            print(" SKIPPED (not found)")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Shards processed: {total_stats['shards_processed']}")
    print(f"Total JSON files: {total_stats['total_json']:,}")
    print(f"Files with <{MIN_TAG_COUNT} tags: {total_stats['to_delete']:,} "
          f"({(total_stats['to_delete']/total_stats['total_json']*100) if total_stats['total_json'] > 0 else 0:.2f}%)")

    if dry_run:
        print(f"\nWould delete:")
        print(f"  - JSON files: {total_stats['to_delete']:,}")
        print(f"  - Image files: {total_stats['to_delete'] - total_stats['missing_images']:,}")
        print(f"  - Missing images: {total_stats['missing_images']:,}")
    else:
        print(f"\nDeleted:")
        print(f"  - JSON files: {total_stats['deleted_json']:,}")
        print(f"  - Image files: {total_stats['deleted_images']:,}")

    if total_stats['errors'] > 0:
        print(f"\nErrors encountered: {total_stats['errors']:,}")

    # Calculate retention
    retained = total_stats['total_json'] - total_stats['to_delete']
    print(f"\nRetained: {retained:,} images ({(retained/total_stats['total_json']*100) if total_stats['total_json'] > 0 else 0:.2f}%)")

    # Show tag distribution summary
    print(f"\n{'='*70}")
    print("TAG DISTRIBUTION (images to delete)")
    print(f"{'='*70}")

    for tag_count in sorted(total_stats['tag_distribution'].keys()):
        if tag_count < MIN_TAG_COUNT:
            count = total_stats['tag_distribution'][tag_count]
            pct = (count / total_stats['total_json'] * 100) if total_stats['total_json'] > 0 else 0
            print(f"  {tag_count:3d} tags: {count:6,} images ({pct:5.2f}%)")

    print(f"{'='*70}")

    if dry_run:
        print("\nThis was a DRY RUN. No files were deleted.")
        print("Run without --dry-run to actually delete files.")
    else:
        print("\nDeletion complete!")

def main():
    parser = argparse.ArgumentParser(
        description='Filter and delete image/JSON pairs with fewer than 26 tags'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--shard-range',
        type=str,
        help='Process only specific shard range (e.g., "0-10")'
    )
    parser.add_argument(
        '--min-tags',
        type=int,
        default=26,
        help='Minimum tag count to keep (default: 26)'
    )

    args = parser.parse_args()

    # Update global MIN_TAG_COUNT
    global MIN_TAG_COUNT
    MIN_TAG_COUNT = args.min_tags

    # Parse shard range if provided
    shard_range = None
    if args.shard_range:
        try:
            start, end = map(int, args.shard_range.split('-'))
            shard_range = (start, end)
        except (ValueError, TypeError):
            print(f"Invalid shard range: {args.shard_range}")
            print("Use format: --shard-range 0-10")
            return

    # Process shards
    process_all_shards(dry_run=args.dry_run, shard_range=shard_range)

if __name__ == "__main__":
    main()
