#!/usr/bin/env python3
"""Copy about 100 files from a shard to test folder."""

import shutil
from pathlib import Path

# Source and destination
source_shard = Path(r'L:\Dab\Dab\shard_00000')
dest_shard = Path(r'L:\Dab\OppaiOracle\test_downsample\shard_00000')

# Ensure destination exists
dest_shard.mkdir(parents=True, exist_ok=True)

# Image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# Find image files
all_images = []
for ext in IMAGE_EXTENSIONS:
    all_images.extend(source_shard.glob(f'*{ext}'))

# Take first 100
images_to_copy = all_images[:100]

print(f"Found {len(all_images)} total images in {source_shard.name}")
print(f"Copying {len(images_to_copy)} files to {dest_shard}...")

# Copy files
copied_count = 0
for img_path in images_to_copy:
    try:
        dest_path = dest_shard / img_path.name
        shutil.copy2(img_path, dest_path)
        copied_count += 1
        if copied_count % 10 == 0:
            print(f"  Copied {copied_count}/{len(images_to_copy)}...")
    except Exception as e:
        print(f"  ERROR copying {img_path.name}: {e}")

print(f"\nDone! Copied {copied_count} files successfully.")
print(f"Test folder: {dest_shard}")
