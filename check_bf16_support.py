#!/usr/bin/env python3
"""Quick check for CPU and GPU bfloat16 support."""
import torch

print("=== PyTorch bfloat16 Support Check ===\n")

# Check CPU support
print("CPU bfloat16 support:")
try:
    test = torch.tensor([1.0, 2.0], dtype=torch.float32)
    bf16 = test.to(torch.bfloat16)
    result = bf16 + bf16
    print("  [OK] CPU supports bfloat16 operations")
    print(f"     Test result: {result.tolist()}")
except (RuntimeError, NotImplementedError) as e:
    print(f"  [NO] CPU does NOT support bfloat16: {e}")
    print("     Will fall back to float32 for cache operations")

# Check GPU support
print("\nGPU bfloat16 support:")
if torch.cuda.is_available():
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
    if torch.cuda.is_bf16_supported():
        print("  [OK] GPU supports bfloat16 (Ampere+ or newer)")
        try:
            test_gpu = torch.tensor([1.0, 2.0], device='cuda', dtype=torch.bfloat16)
            result_gpu = test_gpu + test_gpu
            print(f"     Test result: {result_gpu.cpu().tolist()}")
        except Exception as e:
            print(f"  [WARN] GPU claims support but operation failed: {e}")
    else:
        print("  [WARN] GPU does not support bfloat16 (pre-Ampere)")
        print("     Training will use float16 for AMP instead")
else:
    print("  [WARN] CUDA not available")

# Show PyTorch version
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
