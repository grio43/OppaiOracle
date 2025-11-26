#!/usr/bin/env python3
"""
Create a PNG by converting an existing large JPEG and saving as uncompressed PNG.
This should guarantee a large file size.
"""
from PIL import Image
from pathlib import Path

test_dir = Path(r'Z:\OppaiOracle\test_downsample\shard_test')

# Find a large existing image
large_images = []
for img_path in test_dir.glob('*.jpg'):
    size = img_path.stat().st_size
    if size > 500000:  # > 500KB
        large_images.append((img_path, size))

if not large_images:
    print("No large images found!")
    exit(1)

# Take the largest one
source_path, _ = max(large_images, key=lambda x: x[1])

print(f"Converting {source_path.name} to PNG...")

# Load and convert to PNG
with Image.open(source_path) as img:
    original_dims = img.size
    print(f"  Original: {original_dims[0]}x{original_dims[1]}")

    # Save as PNG with no compression (compress_level=0)
    output_path = test_dir / 'test_huge_uncompressed.png'
    img.save(output_path, format='PNG', compress_level=0)

output_size = output_path.stat().st_size
print(f"\nCreated: {output_path.name}")
print(f"  Size: {output_size:,} bytes ({output_size/1024/1024:.2f} MB)")
print(f"  Dimensions: {original_dims[0]}x{original_dims[1]}")

# Calculate what size will be at 512px
longest = max(original_dims)
scale = 512 / longest
new_w = int(original_dims[0] * scale)
new_h = int(original_dims[1] * scale)
estimated_size_512 = int(output_size * (new_w * new_h) / (original_dims[0] * original_dims[1]))

print(f"\nAfter downsampling to {new_w}x{new_h}:")
print(f"  Estimated size: {estimated_size_512:,} bytes ({estimated_size_512/1024:.0f} KB)")

if estimated_size_512 > 1024 * 1024:
    print(f"  ✓ Will exceed 1MB threshold - should convert to JPEG")
else:
    print(f"  ✗ Won't exceed 1MB threshold - won't convert")
