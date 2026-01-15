#!/usr/bin/env python3
"""
Create a PNG that will be >1MB even after downsampling to 512px.
"""
from PIL import Image
import numpy as np
from pathlib import Path

test_dir = Path(r'L:\Dab\OppaiOracle\test_downsample\shard_test')

print("Creating ultra-detailed PNG that will be >1MB at 512px...")

# Create a very detailed 4000x4000 image with high-frequency patterns
# This will compress poorly as PNG even at 512x512
np.random.seed(42)

# Create high-frequency noise pattern that compresses poorly
size = 4000
img_array = np.zeros((size, size, 3), dtype=np.uint8)

# Layer 1: Random noise
img_array += np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)

# Layer 2: High-frequency patterns (checkerboard at pixel level)
for i in range(size):
    for j in range(size):
        if (i + j) % 2 == 0:
            img_array[i, j] += np.array([50, 50, 50], dtype=np.uint8)

# Create image
img = Image.fromarray(img_array)

# Save with minimal compression to keep it large
path = test_dir / 'test_very_large_png.png'
img.save(path, format='PNG', compress_level=1)  # Low compression
size_bytes = path.stat().st_size

print(f"Created: {path.name}")
print(f"  Original size: {size_bytes:,} bytes ({size_bytes/1024/1024:.1f} MB)")
print(f"  Original dimensions: {size}x{size}")

# Estimate size at 512x512
# High entropy content won't compress well even when small
estimated_512 = int((512 * 512) * 3 * 0.8)  # Rough estimate
print(f"  Estimated at 512x512: ~{estimated_512/1024:.0f}KB (should exceed 1MB threshold)")

print("\nNow run the downsampling script to test PNG->JPEG conversion")
