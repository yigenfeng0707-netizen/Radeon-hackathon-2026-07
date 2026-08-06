# Upstream PR: Fix uint8 bilinear interpolate NotImplementedError on ROCm

## Summary

This PR fixes `NotImplementedError` raised by `F.interpolate(mode='bilinear')` when the input tensor has `torch.uint8` dtype and PyTorch is running on the **AMD ROCm** backend.

The fix adds an explicit `uint8 → float32` dtype cast at the entry of `resize_with_pad()` in `src/lerobot/policies/common/vla_utils.py`, ensuring the same code path works on both CUDA and ROCm.

## Problem

When a user manually constructs an inference batch with `uint8` image tensors (a natural pattern for closed-loop evaluation / deployment) and calls a SmolVLA policy's `forward()` or `select_action()` method, the image passes through `resize_with_pad()` which internally calls:

```python
F.interpolate(img, size=(...), mode="bilinear", align_corners=False)
```

On **CUDA**, this works because PyTorch ships a `uint8` bilinear-interpolation kernel. On **ROCm**, no such kernel exists, so the call raises:

```
NotImplementedError: Could not run 'aten::_empty_affine_quantized' with arguments from the 'Byte' backend.
```

This was reported in Issue #4205.

## Root Cause

`resize_with_pad()` (in `src/lerobot/policies/common/vla_utils.py`) performs the resize **before** any normalization. The downstream consumers (SigLIP, etc.) all expect `float32` input in `[0, 1]` or `[-1, 1]` range. The only reason `uint8` reaches this point is when a user bypasses `prepare_observation_for_inference()` (which already does the cast) and constructs a batch manually.

Note: `resize_with_pad_torch()` (the centered-padding variant used by pi0/pi0.5) already handles `uint8` explicitly — it rounds and clamps the output. The top-left-padding `resize_with_pad()` variant used by SmolVLA/xVLA does not, making it the sole gap.

## Fix

Add an explicit dtype guard at the entry of `resize_with_pad()`:

```python
# Cast uint8 to float32 before F.interpolate.
# PyTorch's ROCm backend lacks a uint8 bilinear kernel;
# CUDA has one. The cast is semantically correct: downstream
# consumers all expect float32 input.
original_dtype = img.dtype
if original_dtype == torch.uint8:
    img = img.to(dtype=torch.float32)
```

This is a minimal, surgical change:
- **No behavioral change on CUDA**: if the input is already `float32` (the normal case after `prepare_observation_for_inference`), the guard is a no-op.
- **Fixes ROCm**: if the input is `uint8`, it gets cast to `float32` before the interpolate call.
- **Semantically correct**: all downstream code (padding, normalization, SigLIP) expects `float32`.

## Testing

### Reproduction

```python
import torch
import torch.nn.functional as F

# On ROCm (W7900, ROCm 7.2.1, PyTorch 2.9.1+rocm7.2.1):
img = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8, device="cuda")
F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
# => NotImplementedError

# After fix:
img_f32 = img.to(dtype=torch.float32)
F.interpolate(img_f32, size=(224, 224), mode="bilinear", align_corners=False)
# => Works correctly
```

### Verification

Tested in the context of a SmolVLA closed-loop evaluation on AMD Radeon W7900 (ROCm 7.2.1, PyTorch 2.9.1+rocm7.2.1):

1. Without patch: `NotImplementedError` on every inference step when manually constructing batches.
2. With patch: 10 episodes × 3 fruits (plum, banana, lemon) completed successfully, 100% success rate on the nominal layout.

## Related

- Issue #4205: [ROCm] NotImplementedError for bilinear interpolate on uint8 tensor in SmolVLA resize_with_pad during manual inference
- Issue #2218: Prior discussion of resize_with_pad / F.interpolate uint8 handling (focused on pad_value, not ROCm)
- This fix was discovered and validated during the AMD Radeon Hackathon 2026 (Track 3: Robotics)

## Checklist

- [x] Code follows the project's style guidelines
- [x] Self-review completed
- [x] Comments added for complex logic (the dtype guard is commented)
- [x] No breaking changes (the cast is a no-op when input is already float32)
- [x] Tested on ROCm 7.2.1 (W7900) and verified no regression on CUDA
