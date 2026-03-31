# Inference-Time Dequantization Pipeline

This document describes how TurboQuant performs on-the-fly dequantization during inference — converting packed 4-bit indices back into weight values fused with the matrix multiplication, without ever materializing the full weight matrix.

## Key Insight: Rotate the Input, Not the Weight

Naively, dequantization would reconstruct the full weight matrix $\tilde{W} \in \mathbb{R}^{M \times N}$ and then compute $y = x \tilde{W}^T$. This requires an expensive inverse rotation ($O(MN^2)$ or $O(MN \log N)$) and materializes a large dense matrix, defeating the purpose of compression.

TurboQuant's insight is to **pre-rotate the activation** instead:

$$x_{\text{rot}} = x \cdot \Pi^T \qquad \text{(cheap: } B \times d \text{ per group)}$$

Then the matmul becomes a lookup + multiply in the rotated domain:

$$\text{output} = x_{\text{rot}} \cdot C[\mathbf{i}]^T \cdot \frac{\alpha}{\sqrt{d}}$$

where $C[\mathbf{i}]$ is the codebook lookup by index. The rotation is applied to $x$ once per group per layer — a $(B, d)$ matrix multiply vs the $(M, d)$ inverse rotation on the weight side.

## Forward Pass Algorithm

Given input $x \in \mathbb{R}^{B \times N}$ (batch × in_features) and quantized weight data:

```
output = zeros(B, M)

for each group g in [0, n_groups):
    # 1. Rotate input slice
    x_g = x[:, g*d : (g+1)*d]                         # (B, d)
    x_rot = x_g @ Pi_g.T                               # (B, d)

    # 2. Unpack 4-bit indices
    idx_g = unpack_4bit(packed[:, g*d//2:(g+1)*d//2])  # (M, d)

    # 3. Codebook lookup
    W_g = codebook[idx_g]                               # (M, d)

    # 4. Matrix multiply
    out_g = x_rot @ W_g.T                               # (B, M)

    # 5. Rescale by norms
    out_g = out_g * (norms_g / sqrt(d))                 # (B, M)

    # 6. Accumulate
    output += out_g

# Optional: add residual pass
if has_residual:
    output += forward_pass(x, pass2_indices, pass2_codebook, pass2_norms, pass2_seed)

output += bias  # if present
```

**Implementation:** `module.py → TurboQuantLinear._forward_pass()` and `TurboQuantLinear.forward()`

## Fused Kernel Implementations

Steps 2–5 above (unpack → lookup → matmul → rescale) are fused into a single GPU kernel to avoid intermediate tensor materialization. Three execution paths are available, auto-selected by priority:

### Priority Order

```
CuTile kernel  →  Triton kernel  →  PyTorch fallback
  (fastest)        (portable)        (no dependencies)
```

The module auto-detects availability at construction time and selects the best available path. You can manually override per layer:

```python
layer.use_cutile = False   # disable CuTile, fall back to Triton
layer.use_triton = False   # disable Triton, fall back to PyTorch
```

### CuTile Kernel

The CuTile kernel uses NVIDIA's `cuda.tile_experimental` API for tile-based programming with hardware-aware scheduling.

**Key optimizations:**
- Shared-memory codebook with implicit L1 caching (16 entries = 64 bytes)
- FP16/BF16 tensor cores (Ampere+) with TF32 fallback for fp32
- Tile-based prefetching for memory latency hiding
- Natural $(B, N)$ accumulation layout — no transpose needed
- Static tile sizes: $T_B = \min(32, B_{\text{pow2}})$, $T_N = \min(64, N_{\text{pow2}})$, $T_K = \min(64, K_{\text{pow2}})$

**Requirements:** NVIDIA Driver r580+, CUDA 13.1+, Ampere (sm80) / Ada (sm89) / Blackwell (sm100+)

**Implementation:** `cutile_kernels.py → cutile_fused_matmul()`

### Triton Kernel

The Triton kernel is a portable alternative that runs on any GPU supported by Triton ≥ 3.0.

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

**Implementation:** `triton_kernels.py → triton_fused_matmul()`

### Kernel Algorithm (Shared by CuTile and Triton)

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

### PyTorch Fallback

When neither CuTile nor Triton is available, the forward pass falls back to explicit PyTorch operations:

```python
indices = unpack_4bit(packed_slice, d)           # (M, d) int64
W_quant = codebook[indices]                       # (M, d) float32
out = x_rot @ W_quant.T                           # (B, M) float32
out = out * (norms / scale)                        # (B, M)
```

This materializes the dequantized weight slice as an intermediate tensor, using more memory but requiring no special dependencies.

**Implementation:** `module.py → TurboQuantLinear._forward_pass()` (the `else` branch)

## Residual Pass Handling

When a layer has residual quantization (pass 2), the forward method simply runs `_forward_pass` twice with different packed data and sums the results:

```python
def forward(self, x):
    output = self._forward_pass(x, ..., pass1_data)
    if self.has_residual:
        output += self._forward_pass(x, ..., pass2_data)
    if self.bias is not None:
        output += self.bias
    return output
```

Each pass uses its own indices, codebook, norms, and rotation seed. The cost is 2× rotations and 2× matmuls per group. To reduce this to 1×, use `merge_passes()` (see [Quantize Pipeline → Merging](quantize-pipeline.md#merging-multi-pass-to-single-pass)).

## Input Shape Handling

`TurboQuantLinear.forward()` handles both 2D and 3D inputs:

| Input shape | Interpretation | Processing |
|------------|---------------|------------|
| $(B, K)$ | Batch of vectors | Direct forward pass |
| $(B, S, K)$ | Batch of sequences | Reshape to $(B \cdot S, K)$, forward, reshape back |

## Caching

The module caches frequently reused data to avoid redundant computation:

- **Rotation cache** (`_rotation_cache`): Rotation matrices keyed by seed — generated once, reused across forward calls
- **Index cache** (`_cached_indices`, `_cached_pass2_indices`): Lazily unpacked indices — avoids repeated `unpack_4bit` calls when Triton/CuTile are not used

## Memory Profile

The dequantization pipeline never materializes the full $M \times N$ weight matrix. Peak additional memory during a forward pass:

| Component | Size | Notes |
|-----------|------|-------|
| Rotated input `x_rot` | $B \times d$ × 4 bytes | Per group, reused |
| Dequantized weight slice (PyTorch path only) | $M \times d$ × 4 bytes | Per group, reused |
| Output accumulator | $B \times M$ × 4 bytes | Persistent |
| Rotation matrix | $d \times d$ × 4 bytes | Cached |

With fused kernels (CuTile/Triton), the dequantized weight slice is never materialized — it exists only in registers/shared memory within the kernel.

## Performance

Benchmarks on Qwen3.5 models (4-bit, group_size=128):

### Qwen3.5-0.8B

| Path | Latency (ms/fwd) | Peak GPU (MB) | vs PyTorch |
|------|-------------------|---------------|------------|
| CuTile | 340 | 1,086 | 1.10× faster, 4.5× less memory |
| Triton | 386 | 1,334 | 0.97×, 3.7× less memory |
| PyTorch | 373 | 4,883 | baseline |

### Qwen3.5-4B

| Path | Latency (ms/fwd) | Peak GPU (MB) | vs PyTorch |
|------|-------------------|---------------|------------|
| CuTile | 968 | 3,954 | 3.98× faster, 5.7× less memory |
| Triton | 1,098 | 4,119 | 3.51× faster, 5.4× less memory |
| PyTorch | 3,855 | 22,377 | baseline |

The fused kernels become increasingly important at larger model sizes, where avoiding intermediate tensor materialization saves gigabytes of GPU memory.

## Full Dequantization (Debug)

For debugging or analysis, `TurboQuantLinear.dequantize()` reconstructs the full weight matrix by running the inverse pipeline:

```
for each group g:
    indices = unpack_4bit(packed_g)           # (M, d)
    Y_quant = codebook[indices] / sqrt(d)     # (M, d)  rotated-domain values
    W_g = Y_quant @ Pi_g                      # (M, d)  inverse rotation
    W_g = W_g * norms_g                        # (M, d)  rescale
```

This is expensive and intended only for offline analysis — not for inference.
