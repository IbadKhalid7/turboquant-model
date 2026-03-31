# Fused GPU Kernels

This document describes the fused GPU kernels that combine unpack → codebook lookup → matrix multiply → rescale into a single kernel launch, eliminating intermediate tensor materialization.

---

## Formulation Context: Inference Cost

In the [quantization formulation](../formulation.md), the inference equation for a single group reduces to:

$$y_m = \sum_g \frac{\alpha_{m,g}}{\sqrt{d}} \sum_{k=1}^d c_{\ell_{m,k}} \cdot x_{\text{rot},k}^{(g)}$$

The naive implementation creates multiple intermediate tensors (unpacked indices → codebook values → matmul result → scaled result), each requiring a separate kernel launch with a global memory round-trip. Fused kernels perform all steps in a single launch with intermediate values living only in registers and shared memory.

---

## The Problem: Intermediate Materialization

The naive dequantization pipeline:

```
packed (uint8) → unpacked indices (int64) → codebook values (float32) → matmul result → scaled result
```

Each arrow is a separate kernel launch with a global memory round-trip. For large models, this intermediate materialization dominates both latency and memory usage.

---

## Kernel Algorithm

Each thread block computes a $(T_B, T_N)$ tile of the output:

```
for each K-tile of size T_K:
    1. Load input tile:     inp[T_B, T_K]  ← input_ptr
    2. Load packed indices: bytes[T_N, T_K//2] ← indices_ptr
    3. Unpack nibbles:
         lo = bytes & 0x0F
         hi = (bytes >> 4) & 0x0F
         idx[T_N, T_K] = interleave(lo, hi)
    4. Codebook lookup:     w[T_N, T_K] = codebook[idx]
    5. Tensor core MMA:     acc[T_B, T_N] += inp @ w.T

// After all K-tiles:
6. Rescale: acc *= prescaled_norms[T_N]
7. Store:   output_ptr ← acc
```

The codebook (16 × 4 bytes = 64 bytes at 4-bit) fits entirely in registers or L1 cache, making the lookup essentially free.

---

## Execution Paths

Three execution paths are available, auto-selected by priority:

```
CuTile kernel  →  Triton kernel  →  PyTorch fallback
  (fastest)        (portable)        (no dependencies)
```

### CuTile Kernel

Uses NVIDIA's `cuda.tile_experimental` API for tile-based programming with hardware-aware scheduling.

**Key optimizations:**
- Shared-memory codebook with implicit L1 caching (16 entries = 64 bytes)
- FP16/BF16 tensor cores (Ampere+) with TF32 fallback for fp32
- Tile-based prefetching for memory latency hiding
- Natural $(B, N)$ accumulation layout — no transpose needed
- Static tile sizes: $T_B = \min(32, B_{\text{pow2}})$, $T_N = \min(64, N_{\text{pow2}})$, $T_K = \min(64, K_{\text{pow2}})$

**Requirements:** NVIDIA Driver r580+, CUDA 13.1+, Ampere (sm80) / Ada (sm89) / Blackwell (sm100+)

### Triton Kernel

Portable alternative that runs on any GPU supported by Triton ≥ 3.0.

**Key optimizations:**
- Autotuned block sizes per problem shape (searches BLOCK_B, BLOCK_N, BLOCK_K, warps, stages)
- Shared-memory codebook (16 float32 entries in registers/L1)
- TF32 tensor cores on Ampere+ for 2× throughput
- Pre-scaled norms: `norms / sqrt(d)` computed on host to eliminate per-element division in kernel
- Software pipelining depth tuned by autotune

**Autotune configurations:**

| Batch | BLOCK_B | BLOCK_N | BLOCK_K | Warps | Stages |
|-------|---------|---------|---------|-------|--------|
| Small | 1 | 32–64 | 32–64 | 2–4 | 2–3 |
| Medium | 16 | 64 | 128 | 8 | 3 |
| Large | 32 | 64 | 128 | 8 | 3 |

### PyTorch Fallback

When neither CuTile nor Triton is available:

```python
indices = unpack_4bit(packed_slice, d)           # (M, d) int64
W_quant = codebook[indices]                       # (M, d) float32
out = x_rot @ W_quant.T                           # (B, M) float32
out = out * (norms / scale)                        # (B, M)
```

This materializes the dequantized weight slice as an intermediate tensor, using more memory but requiring no special dependencies.

---

## Tensor Cores

Modern NVIDIA GPUs have specialized matrix multiply-accumulate (MMA) units:

| Generation | Tensor Core Type | TurboQuant Usage |
|-----------|-----------------|------------------|
| Ampere (A100, sm80) | TF32, FP16, BF16 | TF32 for fp32 inputs, FP16/BF16 natively |
| Ada (RTX 4090, sm89) | TF32, FP16, BF16 | Same |
| Blackwell (B200, sm100+) | TF32, FP16, BF16, FP4 | CuTile kernel support |

TF32 (TensorFloat-32) uses 19-bit mantissa precision with tensor core throughput — 2× faster than full fp32 with negligible accuracy impact for inference.

---

## Performance Benchmarks

### Qwen3.5-0.8B (4-bit, group_size=128)

| Path | Latency (ms/fwd) | Peak GPU (MB) | vs PyTorch |
|------|-------------------|---------------|------------|
| CuTile | 340 | 1,086 | 1.10× faster, 4.5× less memory |
| Triton | 386 | 1,334 | 0.97×, 3.7× less memory |
| PyTorch | 373 | 4,883 | baseline |

### Qwen3.5-4B (4-bit, group_size=128)

| Path | Latency (ms/fwd) | Peak GPU (MB) | vs PyTorch |
|------|-------------------|---------------|------------|
| CuTile | 968 | 3,954 | 3.98× faster, 5.7× less memory |
| Triton | 1,098 | 4,119 | 3.51× faster, 5.4× less memory |
| PyTorch | 3,855 | 22,377 | baseline |

The fused kernels become increasingly important at larger model sizes, where avoiding intermediate tensor materialization saves gigabytes of GPU memory.

---

## Implementation

| File | Function | Description |
|------|----------|-------------|
| `cutile_kernels.py` | `cutile_fused_matmul()` | CuTile fused unpack+lookup+matmul+rescale |
| `triton_kernels.py` | `triton_fused_matmul()` | Triton autotuned fused kernel |
| `module.py` | `TurboQuantLinear._forward_pass()` | Auto-selects best available kernel path |

---

## Relationship to Other Techniques

- **4-bit Packing**: Fused kernels consume the packed uint8 representation directly. Unpacking happens in registers within the kernel — the unpacked tensor is never written to global memory.
- **Lloyd-Max Codebook**: The 64-byte codebook fits in L1 cache or registers, making the codebook lookup effectively free within the fused kernel.
- **Inference (Dequantization)**: The fused kernel implements Steps 2–5 of the dequantization pipeline (unpack → lookup → matmul → rescale) in a single launch.
- **Residual Quantization**: Each residual pass invokes its own fused kernel call. With merged passes, only a single call is needed.
