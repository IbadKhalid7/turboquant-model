# Lloyd-Max Scalar Quantization

This document describes the Lloyd-Max scalar quantizer that maps each rotated, unit-variance weight coordinate to the MSE-optimal centroid from a fixed codebook of $L = 2^b$ levels.

---

## Formulation Context: The Quantizer

In the [quantization formulation](../formulation.md), after row normalization, rotation, and variance scaling, each weight coordinate satisfies $Z \sim \text{approx.}\;\mathcal{N}(0,1)$. The Lloyd-Max codebook is the **optimal** scalar quantizer for this distribution:

$$\{c_\ell^*, t_\ell^*\} = \arg\min_{\{c_\ell, t_\ell\}} \;\mathbb{E}_{Z \sim \mathcal{N}(0,1)}\!\left[(Z - Q_b(Z))^2\right]$$

At 4 bits, this yields 16 centroids and 15 decision boundaries — just 64 bytes of codebook data shared across all layers and all groups.

---

## The Problem

Given a continuous random variable $X \sim p(x)$ and a budget of $L = 2^b$ levels, find reconstruction values (centroids) $c_1, \ldots, c_L$ and decision boundaries $t_0 < t_1 < \cdots < t_L$ that minimize mean squared error:

$$\text{MSE} = \mathbb{E}\left[(X - Q(X))^2\right] = \int_{-\infty}^{\infty} (x - Q(x))^2 \, p(x)\, dx$$

where $Q(x) = c_i$ when $x \in [t_{i-1}, t_i)$.

---

## The Lloyd-Max Algorithm

The Lloyd-Max algorithm (Lloyd 1982, Max 1960) iteratively refines centroids and boundaries:

1. **Initialize** centroids uniformly across the distribution's range
2. **Update boundaries** (nearest-neighbor rule):
   $$t_i = \frac{c_i + c_{i+1}}{2}$$
3. **Update centroids** (conditional expectation):
   $$c_i = \frac{\int_{t_{i-1}}^{t_i} x \, p(x) \, dx}{\int_{t_{i-1}}^{t_i} p(x) \, dx}$$
4. **Repeat** until convergence (typically ~200 iterations)

where $p(x) = \phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ is the standard Gaussian density.

---

## Distortion Values

The per-element distortion $D_b = \mathbb{E}_{Z \sim \mathcal{N}(0,1)}[(Z - Q_b(Z))^2]$ for the Lloyd-Max quantizer:

| $b$ (bits) | $L$ (levels) | $D_b$ | SNR ($-10\log_{10} D_b$) |
|------------|-------------|-------|--------------------------|
| 1 | 2 | 0.3634 | 4.40 dB |
| 2 | 4 | 0.1175 | 9.30 dB |
| 3 | 8 | 0.03454 | 14.62 dB |
| 4 | 16 | 0.009497 | 20.22 dB |
| 5 | 32 | 0.002499 | 26.02 dB |

Each additional bit roughly halves the distortion (~6 dB improvement).

---

## Near-Optimality

The Shannon rate-distortion function for $\mathcal{N}(0,1)$ at rate $R$ bits is:

$$D^*(R) = 2^{-2R}$$

For $b = 4$ bits, $D^*(4) = 2^{-8} \approx 0.00391$. The Lloyd-Max quantizer achieves $D_4 = 0.00950$, giving:

$$\frac{D_4}{D^*(4)} \approx 2.43$$

The gap is only ~3.9 dB from the information-theoretic optimum. No other scalar quantizer with the same number of levels achieves lower MSE for the Gaussian distribution.

---

## Non-Uniform Bin Probabilities

Because the Gaussian density is concentrated near zero, the inner codebook levels (near the mean) receive more probability mass than the outer (tail) levels. At 4 bits:

| Level position | Probability | Character |
|----------------|-------------|-----------|
| 7, 8 (near 0) | ~0.102 each | Most probable |
| 0, 15 (tails) | ~0.013 each | Least probable |

This non-uniformity is exploited by [entropy coding](entropy-codec.md) to compress indices below their nominal $b$-bit width.

---

## Application in TurboQuant

After rotation, each weight coordinate is approximately $\mathcal{N}(0, 1)$. The Lloyd-Max codebook for $\mathcal{N}(0,1)$ is computed **once** and shared across all layers. At 4 bits it is just 16 float32 values (64 bytes).

The quantization step (Step 4 of the pipeline) applies:

$$\text{idx}_{m,k} = \text{searchsorted}(\text{boundaries}, Z_{m,k})$$

clamped to $[0, 2^b - 1]$, producing an integer index per coordinate that references the shared codebook.

---

## Implementation

| File | Function | Description |
|------|----------|-------------|
| `codebook.py` | `_compute_lloyd_max_gaussian()` | Iterative Lloyd-Max algorithm for $\mathcal{N}(0,1)$ |
| `codebook.py` | `get_codebook()` | Return cached codebook and boundaries for a given bit-width |

---

## Relationship to Other Techniques

- **Random Rotation**: Rotation transforms correlated, non-Gaussian weight coordinates into approximately i.i.d. $\mathcal{N}(0,1)$ — the distribution for which Lloyd-Max is optimal. Without rotation, scalar quantization is suboptimal.
- **Variance Normalization**: Scaling by $\sqrt{d}$ after rotation ensures unit variance, exactly matching the codebook's design distribution.
- **Entropy Coding**: The non-uniform bin probabilities from Lloyd-Max quantization enable entropy coding to compress indices below $b$ bits.
- **Residual Quantization**: Each residual pass applies Lloyd-Max independently. The residual after pass $k$ is approximately Gaussian (after re-rotation), so the same codebook family remains near-optimal.
