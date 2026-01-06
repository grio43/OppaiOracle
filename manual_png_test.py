#!/usr/bin/env python3
"""Direct test of PNG to JPEG conversion and file deletion."""
from pathlib import Path
import downsample_images_nas as ds

# Temporarily lower threshold
ds.PNG_TO_JPEG_THRESHOLD = 500 * 1024  # 500KB

test_dir = Path(r'L:\Dab\OppaiOracle\test_downsample\shard_test')
test_png = test_dir / 'test_large_png_1.png'

print("Testing PNG to JPEG conversion with file deletion")
print("="*60)
print(f"Test file: {test_png.name}")
print(f"Size before: {test_png.stat().st_size:,} bytes")
print(f"Threshold: 500KB")
print()

# Record before state
png_exists_before = test_png.exists()
jpg_path = test_png.with_suffix('.jpg')
jpg_exists_before = jpg_path.exists()

print("BEFORE processing:")
print(f"  {test_png.name} exists: {png_exists_before}")
print(f"  {jpg_path.name} exists: {jpg_exists_before}")
print()

# Process
print("Processing...")
stats, image_bytes, output_path, should_delete = ds.process_image(
    test_png, ds.TARGET_SIZE, dry_run=False
)

print(f"  Action: {stats.action}")
print(f"  New format: {stats.new_format}")
print(f"  New size: {stats.new_bytes:,} bytes ({stats.new_bytes/1024:.0f} KB)")
print(f"  Output path: {output_path}")
print(f"  Should delete original: {should_delete}")
print()

# Write the result
if image_bytes and output_path:
    if should_delete and test_png.exists():
        print(f"Deleting original: {test_png.name}")
        test_png.unlink()

    print(f"Writing output: {output_path.name}")
    with open(output_path, 'wb') as f:
        f.write(image_bytes)
print()

# Verify results
png_exists_after = test_png.exists()
jpg_exists_after = jpg_path.exists()

print("AFTER processing:")
print(f"  {test_png.name} exists: {png_exists_after}")
print(f"  {jpg_path.name} exists: {jpg_exists_after}")
if jpg_exists_after:
    print(f"  {jpg_path.name} size: {jpg_path.stat().st_size:,} bytes")
print()

print("="*60)
print("RESULT:")
if stats.action == 'converted' and not png_exists_after and jpg_exists_after:
    print("SUCCESS - PNG converted to JPEG and original deleted!")
elif stats.action == 'downsampled' and png_exists_after and not jpg_exists_after:
    print("SUCCESS - PNG kept as PNG (under threshold)")
else:
    print(f"UNEXPECTED - action={stats.action}, PNG exists={png_exists_after}, JPG exists={jpg_exists_after}")
print("="*60)
