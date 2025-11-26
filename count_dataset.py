#!/usr/bin/env python3
"""Optimized dataset counter for NAS - shard-based processing to minimize thrashing"""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock

# Force immediate output flushing
sys.stdout.reconfigure(line_buffering=True)

dataset_path = Path("Z:/workspace/Dab")

# Thread-safe counter for progress updates
progress_lock = Lock()
global_stats = {'files': 0, 'dirs': 0, 'shards_done': 0}


def count_json_in_shard(shard_path):
    """
    Count all JSON files in an entire shard (folder tree).
    Each worker processes one complete shard at a time for better locality.
    """
    local_count = 0
    local_dirs = 0

    try:
        # Use rglob to recursively find all JSON files in this shard
        # This keeps the worker focused on one area of the NAS
        for json_file in shard_path.rglob("*.json"):
            local_count += 1

        # Also count directories for stats
        for item in shard_path.rglob("*"):
            if item.is_dir():
                local_dirs += 1

    except (PermissionError, OSError) as e:
        print(f"Warning: Could not access {shard_path}: {e}", flush=True)

    return local_count, local_dirs, shard_path.name


def count_json_sharded(root_path, max_workers=4, shard_depth=1):
    """
    Shard-based parallel counting optimized for NAS.
    Each worker processes one complete folder shard at a time.

    Args:
        root_path: Root directory to scan
        max_workers: Number of parallel workers (default 4, one per major folder)
        shard_depth: Depth at which to create shards (1 = immediate subdirs)
    """
    start_time = time.time()

    print(f"Collecting shards at depth {shard_depth}...", flush=True)

    # Collect shard directories (folders at specified depth)
    shards = []

    if shard_depth == 0:
        # Treat root as single shard
        shards = [root_path]
    elif shard_depth == 1:
        # Use immediate subdirectories as shards
        try:
            shards = [d for d in root_path.iterdir() if d.is_dir()]
        except Exception as e:
            print(f"Error collecting shards: {e}", flush=True)
            return 0, 0, 0
    else:
        # Walk to specified depth to find shards
        current_level = [root_path]
        for _ in range(shard_depth):
            next_level = []
            for parent in current_level:
                try:
                    next_level.extend([d for d in parent.iterdir() if d.is_dir()])
                except Exception:
                    continue
            current_level = next_level
        shards = current_level

    if not shards:
        print("No shards found! Falling back to root directory.", flush=True)
        shards = [root_path]

    total_shards = len(shards)
    print(f"Found {total_shards} shards to process", flush=True)
    print(f"Using {max_workers} workers (1 worker per shard at a time)", flush=True)
    print("-" * 70, flush=True)

    # Process shards in parallel, each worker takes one complete shard at a time
    total_files = 0
    total_dirs = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all shard processing tasks
        future_to_shard = {
            executor.submit(count_json_in_shard, shard): shard
            for shard in shards
        }

        # Process results as they complete
        for future in as_completed(future_to_shard):
            shard = future_to_shard[future]
            try:
                file_count, dir_count, shard_name = future.result()

                # Update global stats
                with progress_lock:
                    total_files += file_count
                    total_dirs += dir_count
                    global_stats['files'] = total_files
                    global_stats['dirs'] = total_dirs
                    global_stats['shards_done'] += 1

                    elapsed = time.time() - start_time
                    pct = (global_stats['shards_done'] * 100) // total_shards
                    rate = global_stats['shards_done'] / elapsed if elapsed > 0 else 0

                    print(f"[{global_stats['shards_done']:3d}/{total_shards}] ({pct:3d}%) "
                          f"Shard '{shard_name}': {file_count:,} files, {dir_count:,} dirs | "
                          f"Total: {total_files:,} files | "
                          f"{rate:.2f} shards/sec", flush=True)

            except Exception as e:
                print(f"Error processing shard {shard}: {e}", flush=True)

    elapsed = time.time() - start_time
    return total_files, total_dirs, elapsed


def count_json_simple(root_path):
    """
    Simple single-threaded approach using pathlib.rglob.
    More efficient than os.walk for counting with pattern matching.
    """
    print("Counting JSON files (single-threaded with pattern matching)...", flush=True)
    print("-" * 70, flush=True)
    start_time = time.time()

    # Let pathlib's rglob do the work with pattern matching
    count = 0
    for i, _ in enumerate(root_path.rglob("*.json"), 1):
        count = i
        if count % 5000 == 0:
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            print(f"Found {count:,} JSON files... ({rate:.1f} files/sec)", flush=True)

    elapsed = time.time() - start_time
    return count, elapsed


if __name__ == "__main__":
    print(f"Dataset path: {dataset_path}", flush=True)
    print(f"Exists: {dataset_path.exists()}", flush=True)
    print("=" * 70, flush=True)

    # Parse command line arguments
    mode = "simple"
    workers = 4  # Default: 4 workers for typical NAS with a few main folders
    shard_depth = 1  # Default: use immediate subdirectories as shards

    if len(sys.argv) > 1:
        if sys.argv[1] == "--sharded":
            mode = "sharded"
            if len(sys.argv) > 2:
                workers = int(sys.argv[2])
            if len(sys.argv) > 3:
                shard_depth = int(sys.argv[3])
        elif sys.argv[1] == "--parallel":
            # Legacy support
            mode = "sharded"
            workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    if mode == "sharded":
        print(f"Using SHARDED mode with {workers} workers", flush=True)
        print(f"Shard depth: {shard_depth} (each worker processes complete folder trees)", flush=True)
        print("=" * 70, flush=True)
        try:
            total_files, total_dirs, elapsed = count_json_sharded(
                dataset_path,
                max_workers=workers,
                shard_depth=shard_depth
            )
            print("=" * 70, flush=True)
            print(f"Total directories scanned: {total_dirs:,}")
            print(f"Total JSON files found: {total_files:,}")
            print(f"Time elapsed: {elapsed:.2f} seconds")
            if elapsed > 0:
                print(f"Average rate: {total_files/elapsed:.1f} files/sec")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user", flush=True)
            print(f"Partial results: {global_stats['files']:,} files found", flush=True)
        except Exception as e:
            print(f"\n\nError: {e}", flush=True)
            import traceback
            traceback.print_exc()
    else:
        print("Using SIMPLE mode (single-threaded pattern matching)", flush=True)
        print("Tip: Use --sharded [workers] [depth] for faster scanning", flush=True)
        print("     Example: --sharded 4 1  (4 workers, shard at depth 1)", flush=True)
        print("=" * 70, flush=True)
        try:
            total_files, elapsed = count_json_simple(dataset_path)
            print("=" * 70, flush=True)
            print(f"Total JSON files found: {total_files:,}")
            print(f"Time elapsed: {elapsed:.2f} seconds")
            if elapsed > 0:
                print(f"Average rate: {total_files/elapsed:.1f} files/sec")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user", flush=True)
        except Exception as e:
            print(f"\n\nError: {e}", flush=True)
            import traceback
            traceback.print_exc()
