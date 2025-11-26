#!/usr/bin/env python3
"""
Test runner for downsampling script - uses test directory
"""
import sys
from pathlib import Path

# Modify the database path in the downsample script module
import downsample_images_nas as ds

# Override the database path for testing
original_main = ds.main

def test_main():
    """Modified main for testing"""
    database_path = Path(r'Z:\OppaiOracle\test_downsample')
    progress_file = Path(r'Z:\OppaiOracle\test_downsample\test_progress.txt')

    if not database_path.exists():
        print(f"ERROR: Test database path not found: {database_path}")
        return

    # Load progress
    completed_shards = ds.load_progress(progress_file)

    if completed_shards:
        print(f"[OK] Resuming: {len(completed_shards)} shards already processed")

    # Get all shard directories
    all_shards = sorted([d for d in database_path.iterdir()
                        if d.is_dir() and d.name.startswith('shard_')])

    remaining_shards = [d for d in all_shards if d.name not in completed_shards]

    # Print configuration
    print(f"\n{'='*70}")
    print(f"TEST MODE - IMAGE DOWNSAMPLING")
    print(f"{'='*70}")
    print(f"Training Resolution: {ds.TRAINING_SIZE}x{ds.TRAINING_SIZE} (letterbox)")
    print(f"Target Size: {ds.TARGET_SIZE}px (longest side)")
    if ds.TARGET_SIZE == ds.TRAINING_SIZE:
        print(f"Strategy: Perfect match to training (zero resize overhead)")
    else:
        print(f"Quality Headroom: {ds.TARGET_SIZE/ds.TRAINING_SIZE:.1f}x training resolution")
    print(f"\nOptimization Strategy:")
    print(f"  - Images <= {ds.TARGET_SIZE}px: No changes (already optimal)")
    print(f"  - Images > {ds.TARGET_SIZE}px: Downsample to {ds.TARGET_SIZE}px")
    print(f"  - Large PNGs (>{ds.PNG_TO_JPEG_THRESHOLD//1024//1024}MB): Convert to JPEG (quality={ds.JPEG_QUALITY})")
    print(f"\nTest Dataset: {database_path}")
    print(f"Total shards: {len(all_shards)}")
    print(f"Already processed: {len(completed_shards)}")
    print(f"Will process: {len(remaining_shards)}")
    print(f"Workers: {ds.NUM_WORKERS}")
    print(f"Batch size: {ds.BATCH_SIZE} images")

    if not remaining_shards:
        print("\n[OK] All shards already processed!")
        return

    # Show mode
    mode = "DRY RUN" if ds.DRY_RUN else "EXECUTE"
    print(f"\nMode: {mode}")

    if ds.DRY_RUN:
        print(f"{'='*70}")
        print(f"DRY RUN MODE - Estimating space savings")
        print(f"{'='*70}")
        print(f"No files will be modified. Run with --yes to execute.")
    else:
        print(f"{'='*70}")
        print(f"[WARNING] TEST FILES WILL BE PERMANENTLY MODIFIED")
        print(f"{'='*70}")
        print(f"This will downsample and potentially convert {len(remaining_shards)} test shards.")
        response = input("\nContinue? Type 'YES' to proceed: ").strip()

        if response != 'YES':
            print("Aborted by user.")
            return

    print(f"\nStarting processing with {ds.NUM_WORKERS} parallel workers...\n")

    # Global statistics
    global_stats = ds.ShardStats(shard_name="GLOBAL")
    completed_in_run = []

    # Process shards in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=ds.NUM_WORKERS) as executor:
        future_to_shard = {
            executor.submit(ds.process_shard_worker, (shard, ds.TARGET_SIZE, ds.DRY_RUN)): shard
            for shard in remaining_shards
        }

        print(f"Submitted {len(remaining_shards)} shards to worker pool\n")

        # Setup progress bar
        if ds.HAS_TQDM:
            from tqdm import tqdm
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
                if not ds.DRY_RUN:
                    ds.save_progress(progress_file, shard_name)

                # Progress message
                action_verb = "Would save" if ds.DRY_RUN else "Saved"
                completion_msg = (
                    f"[{len(completed_in_run)}/{len(remaining_shards)}] {shard_name} | "
                    f"{action_verb} {shard_stats.space_saved_mb:.1f} MB | "
                    f"Processed: {shard_stats.downsampled + shard_stats.converted}/{shard_stats.total_images}"
                )

                if ds.HAS_TQDM:
                    tqdm.write(completion_msg)
                    pbar.update(1)
                else:
                    print(completion_msg)

            except Exception as e:
                print(f"Error getting result for {shard_path.name}: {e}")

        if ds.HAS_TQDM:
            pbar.close()

    # Final summary
    print(f"\n{'='*70}")
    print(f"{'DRY RUN ' if ds.DRY_RUN else ''}PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Shards processed: {len(completed_in_run)}")
    print(f"Total images: {global_stats.total_images}")
    print(f"\nActions:")
    action_prefix = "Would be " if ds.DRY_RUN else ""
    print(f"  - {action_prefix}Skipped (already optimal): {global_stats.skipped}")
    print(f"  - {action_prefix}Downsampled: {global_stats.downsampled}")
    print(f"  - {action_prefix}Converted to JPEG: {global_stats.converted}")
    if global_stats.gray_backgrounds > 0:
        json_action = "Would be tagged" if ds.DRY_RUN else "Tagged"
        print(f"  - Gray backgrounds applied: {global_stats.gray_backgrounds} ({json_action} in JSON)")
    print(f"  - Errors: {global_stats.errors}")

    print(f"\nStorage Impact:")
    print(f"  - Original size: {global_stats.original_bytes / (1024**3):.2f} GB")
    print(f"  - Final size: {global_stats.final_bytes / (1024**3):.2f} GB")
    print(f"  - Space saved: {global_stats.space_saved / (1024**3):.2f} GB")
    print(f"  - Compression ratio: {global_stats.compression_ratio:.1%}")

    if ds.DRY_RUN:
        print(f"\n{'='*70}")
        print(f"This was a DRY RUN - no files were modified.")
        print(f"Run with --yes to execute the downsampling.")
        print(f"{'='*70}")
    else:
        print(f"\nProgress saved to: {progress_file}")

    print(f"{'='*70}")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    test_main()
