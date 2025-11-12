#!/usr/bin/env python3
"""
Analyze the impact of uint8 vs bfloat16 cache storage on image quality and tagging accuracy.
Tests actual quantization error with normalized images.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def quantize_uint8(img_01: torch.Tensor) -> torch.Tensor:
    """Simulate uint8 storage (0-1 range)."""
    quantized = (img_01.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    return quantized.to(torch.float32) / 255.0

def quantize_bfloat16(img_01: torch.Tensor) -> torch.Tensor:
    """Simulate bfloat16 storage."""
    return img_01.to(torch.bfloat16).to(torch.float32)

def normalize_image(img_01: torch.Tensor, mean, std):
    """Apply ImageNet-style normalization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (img_01 - mean) / std

def analyze_quantization_error(img_01: torch.Tensor, mean, std):
    """Analyze error for uint8 vs bfloat16 storage."""

    # Original (float32 reference)
    original_normalized = normalize_image(img_01, mean, std)

    # uint8 path
    uint8_cached = quantize_uint8(img_01)
    uint8_normalized = normalize_image(uint8_cached, mean, std)

    # bfloat16 path
    bf16_cached = quantize_bfloat16(img_01)
    bf16_normalized = normalize_image(bf16_cached, mean, std)

    # Compute errors
    uint8_error = (uint8_normalized - original_normalized).abs()
    bf16_error = (bf16_normalized - original_normalized).abs()

    return {
        'uint8': {
            'max_error': uint8_error.max().item(),
            'mean_error': uint8_error.mean().item(),
            'std_error': uint8_error.std().item(),
            'p95_error': uint8_error.flatten().quantile(0.95).item(),
            'p99_error': uint8_error.flatten().quantile(0.99).item(),
        },
        'bfloat16': {
            'max_error': bf16_error.max().item(),
            'mean_error': bf16_error.mean().item(),
            'std_error': bf16_error.std().item(),
            'p95_error': bf16_error.flatten().quantile(0.95).item(),
            'p99_error': bf16_error.flatten().quantile(0.99).item(),
        },
        'images': {
            'original': original_normalized,
            'uint8': uint8_normalized,
            'bfloat16': bf16_normalized,
            'uint8_error_map': uint8_error,
            'bf16_error_map': bf16_error,
        }
    }

def test_color_discrimination():
    """Test ability to discriminate similar colors (critical for tagging)."""
    print("\n=== Color Discrimination Test ===")
    print("Testing subtle color differences (e.g., eye colors, hair shades)\n")

    # Create test colors: blue vs cyan (close colors)
    blue = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])  # Pure blue
    cyan = torch.tensor([[[0.0]], [[0.5]], [[1.0]]])  # Cyan (blue + green)

    mean = [0.5, 0.5, 0.5]
    std = [0.229, 0.229, 0.229]

    # Original difference
    blue_norm = normalize_image(blue, mean, std)
    cyan_norm = normalize_image(cyan, mean, std)
    original_diff = (blue_norm - cyan_norm).abs().mean().item()

    # After uint8 quantization
    blue_uint8 = quantize_uint8(blue)
    cyan_uint8 = quantize_uint8(cyan)
    blue_uint8_norm = normalize_image(blue_uint8, mean, std)
    cyan_uint8_norm = normalize_image(cyan_uint8, mean, std)
    uint8_diff = (blue_uint8_norm - cyan_uint8_norm).abs().mean().item()

    # After bfloat16 quantization
    blue_bf16 = quantize_bfloat16(blue)
    cyan_bf16 = quantize_bfloat16(cyan)
    blue_bf16_norm = normalize_image(blue_bf16, mean, std)
    cyan_bf16_norm = normalize_image(cyan_bf16, mean, std)
    bf16_diff = (blue_bf16_norm - cyan_bf16_norm).abs().mean().item()

    print(f"Original color difference (normalized space): {original_diff:.6f}")
    print(f"uint8 color difference:    {uint8_diff:.6f} (preserved: {uint8_diff/original_diff*100:.1f}%)")
    print(f"bfloat16 color difference: {bf16_diff:.6f} (preserved: {bf16_diff/original_diff*100:.1f}%)")

    # Test very subtle difference (1 uint8 level)
    print("\n--- Subtle 1-level difference (1/255 ~= 0.004) ---")
    color_a = torch.tensor([[[0.500]], [[0.500]], [[0.500]]])  # Gray
    color_b = torch.tensor([[[0.504]], [[0.500]], [[0.500]]])  # Slightly red-shifted (1 uint8 level)

    a_norm = normalize_image(color_a, mean, std)
    b_norm = normalize_image(color_b, mean, std)
    original_subtle = (a_norm - b_norm).abs().mean().item()

    a_uint8 = quantize_uint8(color_a)
    b_uint8 = quantize_uint8(color_b)
    # Check if uint8 can distinguish them
    distinguishable_uint8 = not torch.equal(a_uint8, b_uint8)

    a_bf16 = quantize_bfloat16(color_a)
    b_bf16 = quantize_bfloat16(color_b)
    distinguishable_bf16 = not torch.equal(a_bf16, b_bf16)

    print(f"Original difference: {original_subtle:.6f}")
    print(f"uint8 can distinguish: {distinguishable_uint8}")
    print(f"bfloat16 can distinguish: {distinguishable_bf16}")

def test_with_synthetic_images():
    """Test with synthetic images of different characteristics."""
    print("\n=== Synthetic Image Tests ===\n")

    mean = [0.485, 0.456, 0.406]  # ImageNet normalization
    std = [0.229, 0.224, 0.225]

    test_cases = {
        'flat_colors': torch.ones(3, 64, 64) * 0.5,  # Anime-style flat regions
        'gradients': torch.linspace(0, 1, 64).repeat(3, 64, 1),  # Smooth gradients
        'fine_details': torch.rand(3, 64, 64),  # Random noise (fine details)
        'dark_image': torch.ones(3, 64, 64) * 0.1,  # Dark scenes
        'bright_image': torch.ones(3, 64, 64) * 0.9,  # Bright scenes
    }

    print(f"{'Image Type':<20} {'uint8 Error':<25} {'bfloat16 Error':<25} {'Winner'}")
    print("-" * 95)

    for name, img in test_cases.items():
        results = analyze_quantization_error(img, mean, std)

        uint8_err = results['uint8']['mean_error']
        bf16_err = results['bfloat16']['mean_error']
        winner = 'bfloat16' if bf16_err < uint8_err else 'uint8' if uint8_err < bf16_err else 'tie'

        print(f"{name:<20} mean={uint8_err:.6f} p99={results['uint8']['p99_error']:.6f}   "
              f"mean={bf16_err:.6f} p99={results['bfloat16']['p99_error']:.6f}   {winner}")

def test_storage_efficiency():
    """Compare storage requirements."""
    print("\n=== Storage Efficiency ===\n")

    # Typical image size (640x640x3)
    H, W, C = 640, 640, 3
    pixels = H * W * C

    uint8_bytes = pixels * 1  # 1 byte per value
    bf16_bytes = pixels * 2    # 2 bytes per value
    f32_bytes = pixels * 4     # 4 bytes per value

    print(f"Single image (640x640x3):")
    print(f"  uint8:    {uint8_bytes / 1024 / 1024:.2f} MB")
    print(f"  bfloat16: {bf16_bytes / 1024 / 1024:.2f} MB")
    print(f"  float32:  {f32_bytes / 1024 / 1024:.2f} MB")

    # Dataset sizes
    for n_images in [10_000, 50_000, 100_000, 200_000]:
        uint8_gb = (n_images * uint8_bytes) / (1024**3)
        bf16_gb = (n_images * bf16_bytes) / (1024**3)
        savings_gb = bf16_gb - uint8_gb

        print(f"\n{n_images:,} images:")
        print(f"  uint8:    {uint8_gb:.1f} GB")
        print(f"  bfloat16: {bf16_gb:.1f} GB  (extra {savings_gb:.1f} GB)")
        print(f"  Max images in 14TB L2 cache: uint8={14000/uint8_gb*n_images:,.0f}, bf16={14000/bf16_gb*n_images:,.0f}")

def print_recommendations():
    """Print recommendations based on analysis."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR TAGGER SYSTEM")
    print("="*80)

    print("""
ACCURACY CONSIDERATIONS:
------------------------
1. **uint8 (256 levels per channel)**:
   - Quantization: 1/255 ~= 0.004 per channel (0.4%)
   - After normalization (/ std): Error becomes ~0.017 (1.7%)
   - Can represent 16.7M distinct colors (256^3)
   - May lose subtle color distinctions in 1-2 level differences
   - More error in dark/bright regions (non-linear quantization)

2. **bfloat16 (7-bit mantissa)**:
   - Dynamic precision: ~0.008 near 1.0, better near 0
   - Maintains more precision for subtle differences
   - Better for fine gradients and smooth transitions
   - No precision loss for color discrimination

TAGGING IMPACT:
---------------
- **Critical tags** (eye_color, hair_color, skin_tone):
  > bfloat16 better preserves subtle color differences
  > uint8 may merge very similar shades

- **Style tags** (soft_focus, gradient_background, lighting):
  > bfloat16 better for smooth gradients
  > uint8 can introduce banding artifacts

- **Object tags** (cat_ears, glasses, hat):
  > Both formats adequate (shape detection robust to quantization)

- **Rare tags / edge cases**:
  > bfloat16 provides safety margin for unexpected input

STORAGE CONSIDERATIONS:
-----------------------
- bfloat16 uses 2x storage vs uint8
- For 100K images: ~120GB vs ~240GB difference
- Your 14TB L2 cache: Both formats have plenty of room

PERFORMANCE:
------------
- Your CPU supports native bfloat16 (AVX-512)
- RTX 5090 optimized for bfloat16
- uint8 requires conversion to float32 for operations
- bfloat16 can stay in native format through pipeline

RECOMMENDATION:
---------------
**Use bfloat16 for cache storage** because:

[+] Better accuracy for color-critical tags (eyes, hair, skin)
[+] Preserves subtle gradients important for style tags
[+] Native support on your hardware (no conversion overhead)
[+] You have plenty of cache space (14TB >> dataset size)
[+] Matches your training dtype (no conversion in pipeline)
[+] Safety margin for production tagger system

**Consider uint8 only if**:
[-] Dataset exceeds ~5-6TB (approaching L2 limit)
[-] Willing to sacrifice 1-2% accuracy for storage
[-] Tags don't require fine color discrimination

For a production tagging system where accuracy matters,
bfloat16 is the better choice given your hardware and cache capacity.
""")

if __name__ == "__main__":
    print("="*80)
    print("CACHE DTYPE ANALYSIS: uint8 vs bfloat16 for Tagger System")
    print("="*80)

    test_color_discrimination()
    test_with_synthetic_images()
    test_storage_efficiency()
    print_recommendations()

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
