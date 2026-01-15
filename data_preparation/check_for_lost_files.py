#!/usr/bin/env python3
"""
Check for potential data loss from the lock condition.
This script looks for:
1. Orphaned .tmp files (failed writes)
2. Missing files (deleted originals with no replacement)
3. PNG files that should have been converted but weren't
"""

from pathlib import Path
import json

database_path = Path(r'L:\Dab\Dab')

print("Checking for data loss...")
print("=" * 70)

# Find all shards
all_shards = sorted([d for d in database_path.iterdir()
                     if d.is_dir() and d.name.startswith('shard_')])

print(f"Scanning {len(all_shards)} shards...\n")

# Statistics
tmp_files = []
png_files = []
jpg_files = []
total_images = 0

for shard in all_shards:
    # Count .tmp files (sign of failed writes)
    for tmp in shard.glob('*.tmp'):
        tmp_files.append(tmp)

    # Count PNG files (should have been converted)
    for png in shard.glob('*.png'):
        png_files.append(png)

    # Count JPG files
    for jpg in shard.glob('*.jpg'):
        jpg_files.append(jpg)
    for jpeg in shard.glob('*.jpeg'):
        jpg_files.append(jpeg)

total_images = len(png_files) + len(jpg_files)

print(f"Results:")
print(f"  Total images found: {total_images}")
print(f"  JPG/JPEG files: {len(jpg_files)}")
print(f"  PNG files remaining: {len(png_files)}")
print(f"  Orphaned .tmp files: {len(tmp_files)}")

print("\n" + "=" * 70)

if tmp_files:
    print(f"\n⚠️  FOUND {len(tmp_files)} ORPHANED .TMP FILES")
    print("These indicate failed writes. Original files may have been deleted.")
    print("\nFirst 10 .tmp files:")
    for tmp in tmp_files[:10]:
        print(f"  - {tmp}")
        # Check if there's a corresponding .jpg or original file
        jpg_path = tmp.with_suffix('.jpg')
        png_path = tmp.with_suffix('.png')

        if not jpg_path.exists() and not png_path.exists():
            print(f"    ⚠️  NO CORRESPONDING FILE FOUND - LIKELY DATA LOSS")
        elif jpg_path.exists():
            print(f"    ✓ JPG exists: {jpg_path.name}")
        elif png_path.exists():
            print(f"    ✓ PNG exists: {png_path.name}")
else:
    print("\n✓ No orphaned .tmp files found")

if png_files:
    print(f"\n📌 INFO: {len(png_files)} PNG files remain unconverted")
    print("This is normal if the script hasn't finished or was run in dry-run mode.")

# Summary
print("\n" + "=" * 70)
print("DATA LOSS ASSESSMENT:")

if tmp_files:
    print("  ⚠️  POTENTIAL DATA LOSS: Found orphaned .tmp files")
    print("  → Check the list above for files with no corresponding JPG/PNG")
else:
    print("  ✓ NO SIGNS OF DATA LOSS")
    print("  → No orphaned temp files found")

if total_images == 0:
    print("  ⚠️  WARNING: No images found at all! This could indicate:")
    print("     - Wrong database path")
    print("     - All files were deleted")
    print("     - Storage is not mounted")

print("=" * 70)

# Save detailed report
report = {
    'total_images': total_images,
    'jpg_count': len(jpg_files),
    'png_count': len(png_files),
    'tmp_count': len(tmp_files),
    'tmp_files': [str(f) for f in tmp_files[:100]],  # First 100
    'potential_data_loss': len(tmp_files) > 0
}

with open('data_loss_check_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nDetailed report saved to: data_loss_check_report.json")
