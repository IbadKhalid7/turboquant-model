# Norm Tensor Factorization & Compression

This document describes the norm codec module that reduces the storage overhead of the per-row, per-group norm tensor $\alpha_{m,g}$ via rank-1 SVD factorization with int8 residual quantization.

---

## Formulation Context: Norm Storage

In the [quantization formulation](quantization-formulation.md), the norm tensor $\alpha_{m,g} \in \mathbb{R}^{M \times G}$ is the **second-largest** storage component after the index tensor:

$$\text{BPW} \approx b + \underbrace{\frac{32 \cdot n_{\text{norm}}}{d}}_{\text{norm overhead}} + \text{BPW}_{\text{non-quant}}$$

At 4-bit quantization with group size $d = 128$, the norm overhead is already $32/128 = 0.25$ BPW per pass. With two residual passes, this doubles to 0.50 BPW — a significant fraction of the total budget.

The norm codec targets this term by factorizing the norm tensor into a compact representation.

---

## Rank-1 SVD Factorization

### Key Observation

The norm tensor $\alpha_{m,g}$ has strong low-rank structure: rows of the same layer tend to have similar magnitude patterns across groups. This motivates a rank-1 approximation.

### Decomposition

$$\alpha_{m,g} \approx \beta_m \cdot \gamma_g \cdot (1 + \varepsilon_{m,g})$$

where:
- $\beta_m$ — **row scale** (float16): captures the per-row magnitude
- $\gamma_g$ — **group scale** (float16): captures the per-group magnitude pattern
- $\varepsilon_{m,g}$ — **fractional residual** (int8): captures the remaining deviation

### Computation

1. Compute the rank-1 SVD of the norm matrix: $\alpha \approx \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$
2. Set $\beta_m = \sigma_1 \cdot u_{1,m}$ and $\gamma_g = v_{1,g}$
3. Compute the fractional residual: $\varepsilon_{m,g} = \frac{\alpha_{m,g}}{\beta_m \gamma_g} - 1$
4. Quantize $\varepsilon$ to int8 with symmetric quantization: $\hat{\varepsilon} = \text{round}(\varepsilon / s)$ where $s = \max|\varepsilon| / 127$

---

## Storage Comparison

| Method | Storage | BPW overhead ($d = 128$) |
|--------|---------|--------------------------|
| float32 (baseline) | $M \cdot G \cdot 32$ bits | $32/d = 0.250$ |
| float16 | $M \cdot G \cdot 16$ bits | $16/d = 0.125$ |
| **Factored int8** | $M \cdot 16 + G \cdot 16 + M \cdot G \cdot 8 + 32$ bits | **$\approx 0.0625 + \epsilon$** |

The factored representation achieves roughly **4× compression** vs float32 norms and **2× compression** vs float16 norms.

### Breakdown of Factored Storage

| Component | Size | Description |
|-----------|------|-------------|
| $\beta_m$ | $M \times 16$ bits | Row scales (float16) |
| $\gamma_g$ | $G \times 16$ bits | Group scales (float16) |
| $\varepsilon_{m,g}$ | $M \times G \times 8$ bits | Fractional residual (int8) |
| $s$ | 32 bits | Residual quantization scale |

For a typical layer with $M = 1536$, $G = 12$ ($N = 1536$, $d = 128$):
- float32: $1536 \times 12 \times 32 = 589,824$ bits
- Factored: $1536 \times 16 + 12 \times 16 + 1536 \times 12 \times 8 + 32 = 172,064$ bits
- **Saving: 3.4×**

---

## Reconstruction

$$\hat{\alpha}_{m,g} = \beta_m \cdot \gamma_g \cdot (1 + s \cdot \hat{\varepsilon}_{m,g})$$

The reconstruction error is bounded by the int8 quantization granularity: $|\alpha - \hat{\alpha}| \leq \beta_m \gamma_g \cdot s$, which is typically $< 0.5\%$ of the norm value.

---

## BPW Computation

The `norm_bpw()` function computes the exact BPW overhead for any configuration:

```python
norm_bpw(M=1536, N=1536, group_size=128, method="fp32")       # → 0.250
norm_bpw(M=1536, N=1536, group_size=128, method="fp16")       # → 0.125
norm_bpw(M=1536, N=1536, group_size=128, method="factored_int8")  # → 0.063
```

---

## Implementation

| File | Function | Description |
|------|----------|-------------|
| `norm_codec.py` | `factorize_norms()` | Rank-1 SVD + int8 residual quantization |
| `norm_codec.py` | `reconstruct_norms()` | Reconstruct norms from factored form |
| `norm_codec.py` | `norm_bpw()` | Compute BPW overhead for different methods |

---

## Relationship to Other Techniques

- **Row Normalization (Pipeline Step 1)**: The norm tensor is produced during the normalization step of the quantization pipeline. This codec compresses that output.
- **Residual Quantization**: Each residual pass produces its own norm tensor. The factored representation is especially beneficial here since residual norms are highly structured (and smaller in magnitude).
- **Entropy Coding**: Entropy coding compresses the index tensor; norm factorization compresses the norm tensor. Together they address both major storage components beyond the raw quantized indices.
- **BPW Budget**: At $d = 128$, switching from float32 to factored int8 norms saves ~0.19 BPW per pass — directly reducing the norm overhead term in the formulation.
