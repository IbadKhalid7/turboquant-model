# Residual Quantization

This document describes the residual (multi-pass) quantization strategy that iteratively quantizes the reconstruction error, achieving near-lossless quality by capturing progressively finer detail.

---

## Formulation Context: Error Reduction

In the [quantization formulation](../formulation.md), single-pass quantization at $b$ bits has a fundamental error floor — the Lloyd-Max distortion $D_b$. For 4-bit quantization of Qwen3.5-0.8B, this translates to ~2 PPL degradation. Residual quantization multiplicatively reduces the MSE:

$$\text{MSE}_P \approx \overline{\alpha^2} \cdot \prod_{k=0}^{P-1} D_{b_k}$$

At $b_1 = b_2 = 4$: $D_4^2 \approx 9.0 \times 10^{-5}$ — two orders of magnitude better than single-pass.

---

## The Idea

Instead of using all bits in one shot, apply quantization iteratively to the **residual error**:

$$R_0 = W$$
$$\hat{W}_k = \text{TQ}(R_{k-1}, b_k\text{ bits}, \text{seed}_k)$$
$$R_k = R_{k-1} - \hat{W}_k$$
$$W_{\text{approx}} = \sum_{k=1}^{n} \hat{W}_k$$

Each pass applies the full TurboQuant pipeline (normalize, rotate, scale, quantize, pack) to the residual from the previous pass. The residual after pass $k$ has smaller magnitude, so even a coarse quantizer captures significant information.

---

## Why 4+4 Residual Beats Single-Pass 8-bit

At first glance, 4+4 residual (8 total bits) might seem equivalent to single-pass 8-bit. It is actually **better** for a subtle reason:

- **Single-pass 8-bit:** Allocates 256 levels uniformly across the whole dynamic range. Most levels are wasted in low-density regions.
- **4+4 residual:** Pass 1 captures the coarse structure with 16 levels optimized for the original distribution. Pass 2 captures fine corrections with 16 levels optimized for the *residual* distribution (which is also approximately Gaussian after rotation). The two-stage allocation is more efficient.

---

## Rotation Strategies

The choice of rotation seed across passes affects both quality and whether the fast merge optimization is available:

### Independent Seeds (Default)

Each pass uses a unique rotation seed: $s_k \neq s_j$ for $k \neq j$.

- **Advantage:** Quantization errors are projected onto different random subspaces, making them approximately uncorrelated. This maximizes the error reduction per pass.
- **Disadvantage:** Cannot merge passes in the rotated domain.

### Shared Seed

All passes use the same seed: $s_0 = s_1 = \cdots = s_{P-1} = s$.

- **Advantage:** Enables the fast-path merge, reducing storage and inference cost to single-pass levels.
- **Disadvantage:** Errors are correlated across passes (quantized in the same basis), reducing the benefit of each additional pass.

### Alternating Seeds

Even-numbered passes use seed $s_a$, odd-numbered passes use seed $s_b$:

$$s_k = \begin{cases} s_a & \text{if } k \bmod 2 = 0 \\ s_b & \text{if } k \bmod 2 = 1 \end{cases}$$

- **Advantage:** Adjacent passes project errors onto different bases, achieving near-independent error reduction with only two rotation matrices.
- **Disadvantage:** Cannot use the fast-path merge (requires fully shared rotation).

### Benchmark (Qwen3.5-0.8B, 4 × 2-bit)

| Strategy | PPL | KLD |
|----------|-----|-----|
| different | 17.87 | 0.0034 |
| alternating | 18.09 | 0.0041 |
| shared | 22.24 | 0.0498 |

---

## Effective Bit-Rate Configurations

Total storage is $\sum_{k=0}^{P-1} b_k$ bits per weight element (plus norm overhead per pass).

| Config | Passes | Bits per weight | Expected MSE ratio |
|--------|--------|-----------------|--------------------|
| 4-bit single | 1 | 4 | $D_4 \approx 0.0095$ |
| 4+4 residual | 2 | 8 | $D_4^2 \approx 9.0 \times 10^{-5}$ |
| 4+2 residual | 2 | 6 | $D_4 \cdot D_2 \approx 1.1 \times 10^{-3}$ |
| 3+2 residual | 2 | 5 | $D_3 \cdot D_2 \approx 4.1 \times 10^{-3}$ |
| 2+2+2+2 | 4 | 8 | $D_2^4 \approx 1.9 \times 10^{-4}$ |

### Quality Results (Qwen3.5-0.8B)

| Config | Bits | PPL | KLD |
|--------|------|-----|-----|
| Baseline bf16 | 16 | 14.29 | — |
| 4+4 residual | 8 | 14.28 | 0.0020 |
| 4+2 residual | 6 | 14.46 | 0.0159 |
| 3+2 residual | 5 | 15.15 | 0.0545 |
| 4-bit single | 4 | 16.58 | 0.1403 |

A 4+4 residual configuration achieves **near-lossless** quality: PPL 14.28 vs baseline 14.29, KLD only 0.002 nats.

---

## Merge Optimization (Shared Rotation Only)

When all passes share the same rotation $\Pi_g$, the sum of reconstructions can be computed entirely in the **rotated domain** — no inverse rotation needed:

$$\tilde{Y}^{(g)}_m = \sum_{k=0}^{P-1} \frac{\alpha_{m,g}^{(k)}}{\sqrt{d}} \cdot \mathbf{c}^{(k)}[\boldsymbol{\ell}_m^{(k)}]$$

The merged result is re-normalized and re-quantized into a single-pass representation, collapsing $P$ passes into one at the cost of re-quantization noise.

| Property | Multi-pass (separate) | Merged (single-pass) |
|----------|----------------------|---------------------|
| Storage | $P \times$ single-pass | $1 \times$ single-pass |
| Inference cost | $P$ rotations + $P$ matmuls per group | 1 rotation + 1 matmul per group |
| Quality | Best (no re-quantization loss) | Slightly worse (re-quantization) |

---

## Implementation

| File | Function | Description |
|------|----------|-------------|
| `residual.py` | `residual_quantize_packed()` | Two-pass residual with independent seeds |
| `residual.py` | `multi_residual_quantize_packed()` | N-pass with shared seed |
| `residual.py` | `alternating_residual_quantize_packed()` | N-pass with two alternating seeds |
| `residual.py` | `merge_and_requantize()` | Merge shared-rotation passes into single-pass format |
| `module.py` | `TurboQuantLinear.merge_passes()` | Module-level merge |

---

## Relationship to Other Techniques

- **Lloyd-Max Quantization**: Each pass applies Lloyd-Max quantization independently. The residual after each pass is approximately Gaussian (after re-rotation), so the same codebook family remains near-optimal.
- **Random Rotation**: The rotation strategy (independent, shared, alternating) is a key design choice that trades quality for mergeability.
- **Norm Compression**: Each residual pass produces its own norm tensor. Residual norms are smaller and more structured, making [norm factorization](norm-compression.md) especially effective.
- **Entropy Coding**: Each pass produces an independent index tensor that can be [entropy coded](entropy-codec.md) separately.
- **4-bit Packing**: Each pass's indices are packed independently into uint8 bytes.
