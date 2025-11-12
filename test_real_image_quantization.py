#!/usr/bin/env python3
"""
Test quantization impact on a real image from the dataset.
Shows actual error distribution and visual differences.
"""
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys

def find_sample_image(base_path="./data"):
    """Find a sample image from the dataset."""
    base = Path(base_path)

    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']

    for ext in extensions:
        images = list(base.rglob(ext))
        if images:
            return images[0]

    return None

def load_and_preprocess(img_path, size=640):
    """Load and preprocess image like the dataset loader."""
    from PIL import Image, ImageOps

    # Load image
    with Image.open(img_path) as pil_img:
        pil_img.load()
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert('RGB')

        # Letterbox resize (preserve aspect ratio)
        w, h = pil_img.size
        ratio = min(size / w, size / h)
        scale = min(1.0, ratio)  # Never upscale
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = pil_img.resize((max(1, nw), max(1, nh)), Image.Resampling.BILINEAR)

        # Pad to square
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        left = (size - resized.size[0]) // 2
        top = (size - resized.size[1]) // 2
        canvas.paste(resized, (left, top))

        # Convert to tensor (0-1 range)
        img_np = np.array(canvas).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW

        return img_tensor

def quantize_uint8(img_01: torch.Tensor) -> torch.Tensor:
    """Simulate uint8 storage."""
    quantized = (img_01.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    return quantized.to(torch.float32) / 255.0

def quantize_bfloat16(img_01: torch.Tensor) -> torch.Tensor:
    """Simulate bfloat16 storage."""
    return img_01.to(torch.bfloat16).to(torch.float32)

def normalize_image(img_01: torch.Tensor):
    """Apply normalization (ImageNet stats)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_01 - mean) / std

def analyze_errors(original, uint8_result, bf16_result):
    """Analyze error statistics."""
    uint8_error = (uint8_result - original).abs()
    bf16_error = (bf16_result - original).abs()

    print("\n" + "="*70)
    print("ERROR ANALYSIS (in normalized space, input to model)")
    print("="*70)

    print("\nUINT8 Quantization Error:")
    print(f"  Mean:   {uint8_error.mean():.6f}")
    print(f"  Std:    {uint8_error.std():.6f}")
    print(f"  Max:    {uint8_error.max():.6f}")
    print(f"  Median: {uint8_error.median():.6f}")
    print(f"  P95:    {uint8_error.flatten().quantile(0.95):.6f}")
    print(f"  P99:    {uint8_error.flatten().quantile(0.99):.6f}")

    print("\nBFLOAT16 Quantization Error:")
    print(f"  Mean:   {bf16_error.mean():.6f}")
    print(f"  Std:    {bf16_error.std():.6f}")
    print(f"  Max:    {bf16_error.max():.6f}")
    print(f"  Median: {bf16_error.median():.6f}")
    print(f"  P95:    {bf16_error.flatten().quantile(0.95):.6f}")
    print(f"  P99:    {bf16_error.flatten().quantile(0.99):.6f}")

    print("\n" + "-"*70)
    print("ERROR REDUCTION:")
    print(f"  bfloat16 mean error is {(1 - bf16_error.mean()/uint8_error.mean())*100:.1f}% lower")
    print(f"  bfloat16 max error is {(1 - bf16_error.max()/uint8_error.max())*100:.1f}% lower")
    print("-"*70)

    # Pixel-level analysis
    uint8_big_errors = (uint8_error > 0.01).sum().item()
    bf16_big_errors = (bf16_error > 0.01).sum().item()
    total_pixels = uint8_error.numel()

    print(f"\nPixels with error > 0.01 (significant for model):")
    print(f"  uint8:    {uint8_big_errors:,} / {total_pixels:,} ({uint8_big_errors/total_pixels*100:.2f}%)")
    print(f"  bfloat16: {bf16_big_errors:,} / {total_pixels:,} ({bf16_big_errors/total_pixels*100:.2f}%)")

def main():
    print("="*70)
    print("REAL IMAGE QUANTIZATION TEST")
    print("="*70)

    # Find sample image
    img_path = find_sample_image()
    if img_path is None:
        print("\n[ERROR] No sample image found in ./data directory")
        print("Please specify an image path as argument:")
        print("  python test_real_image_quantization.py path/to/image.jpg")
        if len(sys.argv) > 1:
            img_path = Path(sys.argv[1])
            if not img_path.exists():
                print(f"[ERROR] Image not found: {img_path}")
                return
        else:
            return

    print(f"\nTesting with image: {img_path}")
    print(f"Image size: {img_path.stat().st_size / 1024:.1f} KB")

    # Load and preprocess
    print("\n[1/4] Loading and preprocessing image...")
    img_01 = load_and_preprocess(img_path)
    print(f"  Preprocessed shape: {img_01.shape}")
    print(f"  Value range: [{img_01.min():.3f}, {img_01.max():.3f}]")

    # Simulate caching
    print("\n[2/4] Simulating cache storage...")
    img_uint8_cached = quantize_uint8(img_01)
    img_bf16_cached = quantize_bfloat16(img_01)

    # Cache size comparison
    uint8_size = img_01.shape[0] * img_01.shape[1] * img_01.shape[2] * 1  # 1 byte
    bf16_size = img_01.shape[0] * img_01.shape[1] * img_01.shape[2] * 2   # 2 bytes
    print(f"  uint8 cache size:    {uint8_size / 1024:.1f} KB")
    print(f"  bfloat16 cache size: {bf16_size / 1024:.1f} KB")

    # Normalize (what model sees)
    print("\n[3/4] Applying normalization (model input)...")
    original_normalized = normalize_image(img_01)
    uint8_normalized = normalize_image(img_uint8_cached)
    bf16_normalized = normalize_image(img_bf16_cached)

    # Analyze errors
    print("\n[4/4] Computing error metrics...")
    analyze_errors(original_normalized, uint8_normalized, bf16_normalized)

    print("\n" + "="*70)
    print("INTERPRETATION FOR TAGGING:")
    print("="*70)
    print("""
The model receives normalized tensors (typically in range -3 to +3).
Errors in this range directly affect feature extraction and tag prediction.

- Errors > 0.01: May affect subtle feature detection
- Errors > 0.05: Can impact classification boundaries
- Errors > 0.10: Likely to cause misclassifications

For anime tagging with fine-grained categories (eye colors, styles, etc.),
minimizing quantization error helps preserve subtle visual features that
distinguish between similar tags.
""")

    print("="*70)
    print("CONCLUSION:")
    print("="*70)
    if bf16_error.mean() < uint8_error.mean() * 0.5:
        print("bfloat16 provides SIGNIFICANTLY better accuracy (>50% error reduction)")
        print("RECOMMENDATION: Use bfloat16 for production tagging")
    elif bf16_error.mean() < uint8_error.mean() * 0.8:
        print("bfloat16 provides MODERATELY better accuracy (20-50% error reduction)")
        print("RECOMMENDATION: Use bfloat16 unless storage is critical")
    else:
        print("uint8 and bfloat16 have similar accuracy on this image")
        print("RECOMMENDATION: Either format acceptable, prefer bfloat16 for consistency")

if __name__ == "__main__":
    main()
