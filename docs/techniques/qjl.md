# Why Not QJL?

This document explains why this project uses multi-pass residual quantization instead of the QJL (Quantized Johnson-Lindenstrauss) correction described in the TurboQuant paper's $\text{TurboQuant}_{\text{prod}}$ variant.

---

## Background: The Johnson-Lindenstrauss Lemma

The JL lemma (1984) states that any set of $n$ points in high-dimensional space can be embedded into $O(\log n / \varepsilon^2)$ dimensions while preserving all pairwise distances within a factor of $1 \pm \varepsilon$:

$$(1 - \varepsilon)\|u - v\|^2 \;\leq\; \|f(u) - f(v)\|^2 \;\leq\; (1 + \varepsilon)\|u - v\|^2$$

The projection is a random linear map — a matrix with i.i.d. Gaussian or sub-Gaussian entries, scaled appropriately.

---

## How QJL Works

QJL (Zandieh et al., 2024) extends the JL idea: instead of storing the full projected coordinates, it keeps only the **sign** — just 1 bit per projection. Given $m$ random directions $r_1, \ldots, r_m$, the inner product estimator is:

$$\hat{\langle q, k \rangle} = \frac{\|q\| \cdot \|k\|}{m} \sum_{i=1}^{m} \text{sign}(\langle r_i, q \rangle) \cdot \text{sign}(\langle r_i, k \rangle)$$

**Key properties:**
- **Unbiased:** $\mathbb{E}[\hat{\langle q,k \rangle}] = \langle q,k \rangle$
- **1 bit per projection:** Store only $\text{sign}(\langle r_i, v \rangle)$
- **Zero decode overhead:** Sign comparisons via bitwise XOR + popcount

---

## QJL in the TurboQuant Paper

The paper defines **TurboQuant$_{\text{prod}}$**, which combines standard TurboQuant with a QJL correction for an unbiased inner product estimator:

1. Quantize the vector using TurboQuant (rotation + Lloyd-Max) → $\tilde{k}$
2. Compute residual $e = k - \tilde{k}$
3. Apply **1-bit QJL** to $e$ for an unbiased correction: $\hat{\langle q,k \rangle} = \langle q, \tilde{k} \rangle + \widehat{\langle q, e \rangle}_{\text{QJL}}$

This makes the overall estimator unbiased — critical for KV-cache attention where you quantize keys once and query with many different vectors over the sequence lifetime.

---

## Why This Project Doesn't Use QJL

### 1. QJL Solves a Different Problem

QJL is designed for **online inner product estimation** — e.g., quantize KV-cache keys once, then compute attention scores with many different query vectors. It needs the estimator $\hat{\langle q, k \rangle}$ to be **unbiased**.

Weight quantization is **offline**: we compress $W$ once and compute $y = xW^T$ repeatedly. The goal is minimum reconstruction error $\|W - \tilde{W}\|$, not an unbiased dot-product estimator.

### 2. Unbiasedness Is Unnecessary for Weights

A small deterministic bias from MSE-optimal quantization is absorbed by layer norms, residual connections, and softmax normalization. An unbiased but **high-variance** estimator (QJL at 1 bit) introduces stochastic noise that changes every forward pass — worse for stable inference.

### 3. Residual Quantization Strictly Dominates

QJL uses **1 bit** (random sign projection) for the residual correction. TurboQuant's residual pass uses $b_2$ bits with a full Lloyd-Max codebook + independent rotation, capturing far more residual information.

| | QJL correction | Residual TurboQuant |
|--|---------------|---------------------|
| Bits per weight | 1 | $b_2$ (typically 4) |
| Codebook | None (sign only) | Full Lloyd-Max |
| Rotation | Random projection | Full orthogonal rotation |
| Quality | High variance | Near-lossless |

At 4+4 total bits, residual TurboQuant achieves KL divergence of only **0.002 nats** (practically lossless). A 1-bit QJL correction cannot compete with a 4-bit Lloyd-Max pass.

### 4. QJL Requires the Query at Runtime

The QJL correction term depends on the input activation $x$, making it incompatible with offline weight compression. You'd need to recompute corrections per forward pass — defeating the purpose of weight-only quantization.

---

## Summary

QJL is an elegant technique rooted in the JL lemma — perfect for streaming / KV-cache inner product preservation with 1-bit signed projections. For **offline weight compression**, multi-pass residual quantization with optimal scalar codebooks is the natural and superior choice — achieving practically lossless results at 4+4 bits with no runtime overhead.

---

## References

- **QJL:** Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead," 2024.
- **Johnson-Lindenstrauss:** W. Johnson & J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space," Contemporary Mathematics, 1984.
- **TurboQuant:** Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," arXiv:2504.19874, 2025.
