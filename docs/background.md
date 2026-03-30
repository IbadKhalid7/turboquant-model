# Background: Techniques Used in TurboQuant

This document provides background on the core mathematical and algorithmic techniques underlying TurboQuant.

## Table of Contents

- [Lloyd-Max Scalar Quantization](#lloyd-max-scalar-quantization)
- [Random Rotation for Decorrelation](#random-rotation-for-decorrelation)
- [Walsh-Hadamard Transform](#walsh-hadamard-transform)
- [Residual Quantization](#residual-quantization)
- [4-bit Packing](#4-bit-packing)
- [Fused GPU Kernels](#fused-gpu-kernels)
- [Why Not QJL?](#why-not-qjl)

---

## Lloyd-Max Scalar Quantization

### Problem

Given a continuous random variable $X \sim p(x)$ and a budget of $L = 2^b$ levels, find reconstruction values (centroids) $c_1, \ldots, c_L$ and decision boundaries $t_0 < t_1 < \cdots < t_L$ that minimize mean squared error:

$$\text{MSE} = \mathbb{E}\left[(X - Q(X))^2\right] = \int_{-\infty}^{\infty} (x - Q(x))^2 \, p(x)\, dx$$

where $Q(x) = c_i$ when $x \in [t_{i-1}, t_i)$.

### Lloyd-Max Algorithm

The Lloyd-Max algorithm (Lloyd 1982, Max 1960) iteratively refines centroids and boundaries:

1. **Initialize** centroids uniformly across the distribution's range
2. **Update boundaries** (nearest-neighbor rule):
   $$t_i = \frac{c_i + c_{i+1}}{2}$$
3. **Update centroids** (conditional expectation):
   $$c_i = \frac{\int_{t_{i-1}}^{t_i} x \, p(x) \, dx}{\int_{t_{i-1}}^{t_i} p(x) \, dx}$$
4. **Repeat** until convergence (typically ~200 iterations)

### Application in TurboQuant

After rotation, each weight coordinate is approximately $\mathcal{N}(0, 1)$. The Lloyd-Max codebook for $\mathcal{N}(0,1)$ is computed once and shared across all layers — at 4 bits, it's just 16 float32 values (64 bytes).

This is **optimal** for the Gaussian distribution: no other scalar quantizer with the same number of levels achieves lower MSE. The paper shows TurboQuant's overall distortion is within 2.7× of the information-theoretic lower bound (Shannon rate-distortion).

**Implementation:** `codebook.py → _compute_lloyd_max_gaussian()`

---

## Random Rotation for Decorrelation

### The Problem with Correlated Weights

Neural network weight matrices have structured correlations — some coordinates carry much more information than others. Scalar quantization applied directly to correlated weights wastes bits on low-variance coordinates and under-quantizes high-variance ones.

### Solution: Random Orthogonal Rotation

Multiplying by a random orthogonal matrix $\Pi \in \mathbb{R}^{d \times d}$ (where $\Pi^T \Pi = I$) spreads information uniformly across all coordinates:

$$Y = W_{\text{norm}} \cdot \Pi^T$$

**Properties of the rotation:**
- **Norm-preserving:** $\|Y\|_2 = \|W_{\text{norm}}\|_2 = 1$ (orthogonal matrices preserve norms)
- **Decorrelating:** Coordinates of $Y$ become approximately independent
- **Gaussianizing:** By the central limit theorem on high-dimensional unit vectors, each coordinate of $Y$ is approximately $\mathcal{N}(0, 1/d)$
- **Invertible:** $W_{\text{norm}} = Y \cdot \Pi$ (the inverse is just the transpose)

After scaling by $\sqrt{d}$, each coordinate is $\mathcal{N}(0, 1)$ — exactly matching the Lloyd-Max codebook.

### Haar-Distributed Random Orthogonal Matrix (QR Method)

The "gold standard" for random rotations. A matrix drawn from the Haar measure on $O(d)$ is maximally random — it is the unique distribution invariant under left/right multiplication by any orthogonal matrix.

**Algorithm:**
1. Draw $A \in \mathbb{R}^{d \times d}$ with i.i.d. $\mathcal{N}(0, 1)$ entries
2. Compute QR decomposition: $A = QR$
3. Adjust signs: $\Pi = Q \cdot \text{diag}(\text{sign}(\text{diag}(R)))$

This ensures $\Pi$ is exactly Haar-distributed.

**Trade-off:** $O(d^2)$ storage and compute. For $d = 128$ (default group size), the rotation matrix is $128 \times 128 \times 4$ bytes = 64 KB — manageable. For full-dimension rotation on large models ($d = 4096$), this becomes 64 MB per layer.

**Implementation:** `rotation.py → generate_rotation_matrix()`

### Fast Walsh-Hadamard Alternative

See the dedicated section below.

---

## Walsh-Hadamard Transform

### Definition

The Walsh-Hadamard matrix $H_n$ of order $n = 2^k$ is defined recursively:

$$H_1 = [1], \qquad H_{2n} = \begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}$$

The **normalized** Hadamard matrix $\bar{H} = H / \sqrt{d}$ is orthogonal: $\bar{H}^T \bar{H} = I$.

### Fast Walsh-Hadamard Transform (FWHT)

Analogous to FFT, the FWHT computes $y = H \cdot x$ in $O(d \log d)$ time without materializing the full matrix. The butterfly-style algorithm operates in-place:

```
for step s in 0, 1, ..., log2(d) - 1:
    half = 2^s
    for each pair (i, i + half) in groups of 2^(s+1):
        a, b = x[i], x[i + half]
        x[i]        = a + b
        x[i + half] = a - b
```

### Randomized Hadamard Rotation

A plain Hadamard matrix is deterministic and structured. To get a **random** rotation that decorrelates arbitrary weight distributions, TurboQuant uses:

$$\Pi = \frac{1}{\sqrt{d}} H \cdot D$$

where $D = \text{diag}(s_1, \ldots, s_d)$ with each $s_i \in \{-1, +1\}$ drawn from a seeded PRNG.

**Forward rotation:**
$$Y = X \cdot \Pi^T = \text{FWHT}(X \odot \mathbf{s}) / \sqrt{d}$$

**Inverse rotation:**
$$X = Y \cdot \Pi = \text{FWHT}(Y) / \sqrt{d} \odot \mathbf{s}$$

**Advantages over QR:**

| | QR (Haar) | Hadamard |
|--|-----------|----------|
| Storage | $O(d^2)$ — full matrix | $O(d)$ — just the sign vector |
| Compute | $O(d^2)$ — matrix multiply | $O(d \log d)$ — FWHT |
| Randomness quality | Exact Haar distribution | Approximate (excellent in practice) |
| Constraint | None | $d$ must be power of 2 |

For the default group size of 128 (a power of 2), Hadamard rotation is both faster and more memory-efficient.

**Implementation:** `rotation.py → hadamard_rotate()`, `hadamard_rotate_inverse()`, `_fwht()`

---

## Residual Quantization

### Motivation

Single-pass quantization at $b$ bits per weight has a fundamental error floor — the Lloyd-Max distortion for $\mathcal{N}(0,1)$ at 4 bits. For LLMs, this translates to measurable perplexity degradation (~2 PPL on Qwen3.5-0.8B).

### Idea

Instead of using all bits in one shot, apply quantization iteratively to the **residual error**:

$$R_0 = W$$
$$\hat{W}_k = \text{TQ}(R_{k-1}, b_k\text{ bits}, \text{seed}_k)$$
$$R_k = R_{k-1} - \hat{W}_k$$
$$W_{\text{approx}} = \sum_{k=1}^{n} \hat{W}_k$$

Each pass captures progressively finer detail in the weight matrix. The residual after pass $k$ has smaller magnitude, so even a coarse quantizer captures significant information.

### Why It Works Better Than Higher Bit-Width

At first glance, 4+4 residual (8 total bits) might seem equivalent to single-pass 8-bit quantization. It's actually **better** for a subtle reason:

- **Single-pass 8-bit:** Allocates 256 levels uniformly across the whole dynamic range. Most levels are wasted in low-density regions.
- **4+4 residual:** Pass 1 captures the coarse structure with 16 levels optimized for the original distribution. Pass 2 captures fine corrections with 16 levels optimized for the *residual* distribution (which is also approximately Gaussian after rotation). The two-stage allocation is more efficient.

### Shared vs Independent Rotation Seeds

| | Independent seeds ($s_1 \neq s_2$) | Shared seed ($s_1 = s_2$) |
|--|-------------------------------------|--------------------------|
| Quality | Slightly better (independent errors) | Marginally worse |
| Merge support | Slow path (requires inverse rotation) | Fast path (merge in rotated domain) |
| Recommended | When merging is not needed | When merging is needed |

**Implementation:** `residual.py → residual_quantize()` (independent), `multi_residual_quantize()` (shared)

---

## 4-bit Packing

### Layout

Two 4-bit indices are packed into each uint8 byte:

```
Byte layout:  [hi_nibble (bits 7-4)] [lo_nibble (bits 3-0)]
              |--- index[k+1] -----|--- index[k] --------|
```

**Pack:** `packed[m, k//2] = indices[m, k] | (indices[m, k+1] << 4)`

**Unpack:**
- `lo = packed & 0x0F` → `indices[m, 2*j]`
- `hi = (packed >> 4) & 0x0F` → `indices[m, 2*j+1]`

This halves the storage for the index tensor: an $(M, N)$ index matrix becomes $(M, N/2)$ uint8.

When $N$ is odd, the last column is zero-padded before packing and the original $N$ is stored in metadata for correct unpacking.

**Implementation:** `quantize.py → pack_4bit()`, `unpack_4bit()`

---

## Fused GPU Kernels

### Problem

The naive dequantization pipeline creates multiple intermediate tensors:

```
packed (uint8) → unpacked indices (int64) → codebook values (float32) → matmul result → scaled result
```

Each arrow is a separate kernel launch with a global memory round-trip. For large models, this intermediate materialization dominates both latency and memory.

### Solution: Kernel Fusion

A single GPU kernel performs all steps:

1. **Load** packed uint8 bytes from global memory
2. **Unpack** nibbles (bitwise ops in registers)
3. **Lookup** codebook values (codebook in shared memory / registers — 64 bytes at 4-bit)
4. **Multiply-accumulate** using tensor cores (TF32/FP16)
5. **Rescale** by pre-computed `norms / sqrt(d)`
6. **Store** final result to global memory

No intermediate tensors are ever written to global memory.

### Tensor Cores

Modern NVIDIA GPUs have specialized matrix multiply-accumulate (MMA) units:

| Generation | Tensor Core Type | TurboQuant Usage |
|-----------|-----------------|------------------|
| Ampere (A100, sm80) | TF32, FP16, BF16 | TF32 for fp32 inputs, FP16/BF16 natively |
| Ada (RTX 4090, sm89) | TF32, FP16, BF16 | Same |
| Blackwell (B200, sm100+) | TF32, FP16, BF16, FP4 | CuTile kernel support |

TF32 (TensorFloat-32) uses 19-bit mantissa precision with tensor core throughput — 2× faster than full fp32 with negligible accuracy impact for inference.

### Autotuning (Triton)

The Triton kernel searches across multiple tile configurations at runtime to find the fastest for each problem shape:

- **BLOCK_B:** How many batch rows per thread block (1, 16, 32)
- **BLOCK_N:** How many output columns per thread block (32, 64)
- **BLOCK_K:** How many input columns per K-tile (32, 64, 128)
- **num_warps:** Thread parallelism (2, 4, 8)
- **num_stages:** Software pipeline depth (2, 3)

Results are cached by problem shape for subsequent calls.

**Implementation:** `triton_kernels.py`, `cutile_kernels.py`

---

## Why Not QJL?

The original TurboQuant paper defines **TurboQuant_prod**, a variant that applies QJL (Quantized Johnson-Lindenstrauss) as a 1-bit correction on the residual to produce an **unbiased** inner product estimator. This project does **not** use QJL. Here's why:

### 1. QJL Solves a Different Problem

QJL is designed for **online inner product estimation** — e.g., quantize KV-cache keys once, then compute attention scores with many different query vectors. It needs the estimator $\hat{\langle q, k \rangle}$ to be **unbiased**.

Weight quantization is **offline**: we compress $W$ once and compute $y = xW^T$ repeatedly. The goal is minimum reconstruction error $\|W - \tilde{W}\|$, not an unbiased dot-product estimator.

### 2. Unbiasedness Is Unnecessary for Weights

A small deterministic bias from MSE-optimal quantization is absorbed by layer norms, residual connections, and softmax normalization. An unbiased but **high-variance** estimator (QJL at 1 bit) introduces stochastic noise that changes every forward pass — worse for stable inference.

### 3. Residual Quantization Strictly Dominates

QJL uses **1 bit** (random sign projection) for the residual correction. TurboQuant's residual pass uses $b_2$ bits with a full Lloyd-Max codebook + independent rotation, capturing far more residual information.

At 4+4 total bits, residual TurboQuant achieves KL divergence of only **0.002 nats** (practically lossless). A 1-bit QJL correction cannot compete with a 4-bit Lloyd-Max pass.

### 4. QJL Requires the Query at Runtime

The QJL correction term depends on the input activation $x$, making it incompatible with offline weight compression. You'd need to recompute corrections per forward pass — defeating the purpose of weight-only quantization.

**Summary:** QJL is elegant for streaming / KV-cache inner product preservation. For weight compression, multi-pass residual quantization with optimal scalar codebooks is the natural and superior choice.

---

## References

- **TurboQuant paper:** Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," arXiv:2504.19874, 2025.
- **Lloyd-Max quantization:** S. Lloyd, "Least squares quantization in PCM," IEEE Trans. on Information Theory, 1982. J. Max, "Quantizing for minimum distortion," IRE Trans. on Information Theory, 1960.
- **Walsh-Hadamard transform:** J. Hadamard, 1893. J.L. Walsh, 1923.
- **Johnson-Lindenstrauss lemma:** W. Johnson & J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space," Contemporary Mathematics, 1984.
- **QJL:** Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead," 2024.
