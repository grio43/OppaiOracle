#!/usr/bin/env python3
"""
Fix JSON Metadata Extensions

Scans dataset JSON sidecar files and corrects the `filename` field to match
the actual image file extension on disk. Optimized for large datasets with
millions of files across multiple shards.

Usage:
    # Dry run (preview changes, no modifications):
    python fix_json_extensions.py --data-path "L:/Dab/Dab"

    # Actually apply fixes:
    python fix_json_extensions.py --data-path "L:/Dab/Dab" --apply

    # With backup (creates .json.bak files):
    python fix_json_extensions.py --data-path "L:/Dab/Dab" --apply --backup
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Supported image extensions (in order of preference)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')


def find_actual_image(json_path: Path, claimed_filename: str, dir_files: Set[str]) -> Optional[str]:
    """
    Find the actual image file for a JSON sidecar.

    Args:
        json_path: Path to the JSON file
        claimed_filename: The filename claimed in the JSON
        dir_files: Pre-cached set of files in the directory (for speed)

    Returns:
        Actual filename if found with different extension, None if correct or not found
    """
    # If claimed file exists, no fix needed
    if claimed_filename in dir_files:
        return None

    # Extract base name without extension
    base_name = Path(claimed_filename).stem

    # Try each extension (fast set lookup)
    for ext in IMAGE_EXTENSIONS:
        candidate = f"{base_name}{ext}"
        if candidate in dir_files:
            return candidate

    # No matching image found
    return None


def process_json_file(
    json_path: Path,
    apply: bool,
    backup: bool,
    dir_files: Set[str]
) -> Dict:
    """
    Process a single JSON file and fix extension if needed.

    Args:
        json_path: Path to the JSON file
        apply: Whether to actually modify the file
        backup: Whether to create a backup before modifying
        dir_files: Set of filenames in the directory (from os.scandir)

    Returns:
        Dict with status info: {status, old_filename, new_filename, error}
    """
    result = {
        'path': str(json_path),
        'status': 'ok',
        'old_filename': None,
        'new_filename': None,
        'error': None
    }

    try:
        # Read JSON
        if HAS_ORJSON:
            data = orjson.loads(json_path.read_bytes())
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # Skip non-dict JSON (e.g., manifest files)
        if not isinstance(data, dict):
            result['status'] = 'skipped'
            return result

        # Get current filename
        current_filename = data.get('filename')
        if not current_filename or not isinstance(current_filename, str):
            result['status'] = 'no_filename'
            return result

        # Check if fix is needed (use cached dir listing)
        actual_filename = find_actual_image(json_path, current_filename, dir_files)

        if actual_filename is None:
            # Either correct or image not found at all
            if current_filename in dir_files:
                result['status'] = 'ok'
            else:
                result['status'] = 'missing'
                result['old_filename'] = current_filename
            return result

        # Fix needed
        result['status'] = 'fixed' if apply else 'needs_fix'
        result['old_filename'] = current_filename
        result['new_filename'] = actual_filename

        if apply:
            # Create backup if requested (foo.json -> foo.json.bak)
            if backup:
                backup_path = json_path.with_suffix('.json.bak')
                shutil.copy2(json_path, backup_path)

            # Update and write
            data['filename'] = actual_filename

            if HAS_ORJSON:
                json_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            else:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

        return result

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        return result


def process_directory_batch(
    dir_path: Path,
    json_files: List[Path],
    apply: bool,
    backup: bool
) -> List[Dict]:
    """
    Process all JSON files in a single directory (batch for efficiency).

    Pre-caches the directory listing once and reuses it for all files.
    """
    # Cache directory listing once for all files in this dir (scandir is faster)
    try:
        dir_files = {entry.name for entry in os.scandir(dir_path)}
    except OSError:
        # Directory may have been deleted/moved since discovery
        return [{'path': str(p), 'status': 'error', 'old_filename': None,
                 'new_filename': None, 'error': f'Directory inaccessible: {dir_path}'}
                for p in json_files]

    results = []
    for json_path in json_files:
        result = process_json_file(json_path, apply, backup, dir_files)
        results.append(result)

    return results


def discover_json_files_by_dir(data_path: Path) -> Dict[Path, List[Path]]:
    """
    Discover all JSON files in the dataset, grouped by directory.

    Grouping by directory enables efficient batch processing with shared
    directory listing cache.
    """
    logger.info(f"Discovering JSON files in {data_path}...")

    by_dir: Dict[Path, List[Path]] = {}
    total_files = 0

    for root, dirs, files in os.walk(data_path):
        root_path = Path(root)
        json_files_in_dir = []

        for file in files:
            if file.endswith('.json'):
                json_files_in_dir.append(root_path / file)

        if json_files_in_dir:
            by_dir[root_path] = json_files_in_dir
            total_files += len(json_files_in_dir)

    logger.info(f"Found {total_files:,} JSON files in {len(by_dir):,} directories")
    return by_dir


def main():
    parser = argparse.ArgumentParser(
        description='Fix JSON metadata file extensions to match actual images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to dataset directory'
    )

    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply fixes (default is dry-run)'
    )

    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create .json.bak backup files before modifying'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=64,
        help='Number of parallel workers (64 recommended for SSD, 20 for HDD)'
    )

    parser.add_argument(
        '--limit-dirs',
        type=int,
        default=0,
        help='Limit number of directories to process (0 = all)'
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        return 1

    # Discover files grouped by directory (more efficient batch processing)
    files_by_dir = discover_json_files_by_dir(data_path)

    if not files_by_dir:
        logger.warning("No JSON files found")
        return 0

    # Convert to list of (dir, files) tuples for processing
    dir_batches = list(files_by_dir.items())

    if args.limit_dirs > 0:
        dir_batches = dir_batches[:args.limit_dirs]
        logger.info(f"Limited to {len(dir_batches):,} directories")

    total_files = sum(len(files) for _, files in dir_batches)

    # Process files
    mode = "APPLYING FIXES" if args.apply else "DRY RUN (use --apply to fix)"
    logger.info(f"Processing {total_files:,} files in {len(dir_batches):,} directories - {mode}")
    logger.info(f"Using {args.workers} worker processes (batch processing by directory)")

    start_time = time.time()

    # Statistics
    stats = {
        'ok': 0,
        'fixed': 0,
        'needs_fix': 0,
        'missing': 0,
        'no_filename': 0,
        'skipped': 0,
        'error': 0
    }

    # Sample of fixes for reporting
    fix_samples = []
    missing_samples = []
    error_samples = []

    # Process directories in parallel (each dir is a batch)
    # This is much faster than per-file parallelism due to shared dir listing cache
    processed_files = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_directory_batch, dir_path, json_files, args.apply, args.backup): (dir_path, len(json_files))
            for dir_path, json_files in dir_batches
        }

        pbar = tqdm(total=total_files, desc="Processing", unit="files", mininterval=0.5) if HAS_TQDM else None
        pending_update = 0  # Batch progress updates for efficiency

        try:
            for future in as_completed(futures):
                try:
                    results = future.result()

                    for result in results:
                        status = result['status']
                        stats[status] += 1

                        # Collect samples
                        if status in ('fixed', 'needs_fix') and len(fix_samples) < 10:
                            fix_samples.append(result)
                        elif status == 'missing' and len(missing_samples) < 5:
                            missing_samples.append(result)
                        elif status == 'error' and len(error_samples) < 5:
                            error_samples.append(result)

                    processed_files += len(results)
                    pending_update += len(results)

                    # Batch tqdm updates every 1000 files to reduce overhead
                    if pbar and pending_update >= 1000:
                        pbar.update(pending_update)
                        pending_update = 0

                except Exception as e:
                    dir_path, file_count = futures[future]
                    stats['error'] += file_count
                    if len(error_samples) < 5:
                        error_samples.append({'path': str(dir_path), 'error': str(e)})
        finally:
            # Ensure progress bar is always closed
            if pbar:
                if pending_update > 0:
                    pbar.update(pending_update)
                pbar.close()

    elapsed = time.time() - start_time
    rate = total_files / elapsed if elapsed > 0 else 0

    # Report
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total processed: {total_files:,} files in {elapsed:.1f}s ({rate:,.0f} files/s)")
    logger.info(f"  OK (no change needed): {stats['ok']:,}")
    logger.info(f"  {'Fixed' if args.apply else 'Needs fix'}: {stats.get('fixed', 0) + stats.get('needs_fix', 0):,}")
    logger.info(f"  Missing (no image found): {stats['missing']:,}")
    logger.info(f"  No filename field: {stats['no_filename']:,}")
    logger.info(f"  Skipped (non-dict): {stats['skipped']:,}")
    logger.info(f"  Errors: {stats['error']:,}")

    if fix_samples:
        logger.info("")
        logger.info("Sample fixes:")
        for sample in fix_samples[:5]:
            logger.info(f"  {sample['old_filename']} -> {sample['new_filename']}")

    if missing_samples:
        logger.info("")
        logger.info("Sample missing images:")
        for sample in missing_samples[:3]:
            logger.info(f"  {sample['path']}: {sample['old_filename']}")

    if error_samples:
        logger.info("")
        logger.info("Sample errors:")
        for sample in error_samples[:3]:
            logger.info(f"  {sample.get('path', 'unknown')}: {sample['error']}")

    if not args.apply and (stats.get('needs_fix', 0) > 0):
        logger.info("")
        logger.info("To apply these fixes, run with --apply flag")

    return 1 if stats['error'] > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
