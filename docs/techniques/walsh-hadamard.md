# Walsh-Hadamard Transform

This document describes the Walsh-Hadamard transform (WHT) used as a fast, memory-efficient alternative to QR-based random rotation for decorrelating weight coordinates.

---

## Formulation Context: Fast Rotation

In the [quantization formulation](../formulation.md), the rotation step requires applying an orthogonal matrix $\Pi_g$ to each row-group slice. The QR method produces an exact Haar-distributed matrix but costs $O(d^2)$ in storage and compute. The Walsh-Hadamard transform reduces both to $O(d \log d)$ compute and $O(d)$ storage — critical for scaling to larger group sizes.

---

## Definition

The Walsh-Hadamard matrix $H_n$ of order $n = 2^k$ is defined recursively:

$$H_1 = [1], \qquad H_{2n} = \begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}$$

The **normalized** Hadamard matrix $\bar{H} = H / \sqrt{d}$ is orthogonal: $\bar{H}^T \bar{H} = I$.

---

## Fast Walsh-Hadamard Transform (FWHT)

Analogous to FFT, the FWHT computes $y = H \cdot x$ in $O(d \log d)$ time without materializing the full $d \times d$ matrix. The butterfly-style algorithm operates in-place:

```
for step s in 0, 1, ..., log2(d) - 1:
    half = 2^s
    for each pair (i, i + half) in groups of 2^(s+1):
        a, b = x[i], x[i + half]
        x[i]        = a + b
        x[i + half] = a - b
```

At each of $\log_2 d$ stages, every element participates in exactly one butterfly (add/subtract) pair. The total work is $d \log_2 d$ additions — no multiplications needed.

---

## Randomized Hadamard Rotation

A plain Hadamard matrix is deterministic and structured — it does not decorrelate arbitrary weight distributions. To get a **random** rotation, TurboQuant uses:

$$\Pi = \frac{1}{\sqrt{d}} H \cdot D$$

where $D = \text{diag}(s_1, \ldots, s_d)$ with each $s_i \in \{-1, +1\}$ drawn from a seeded PRNG.

### Forward Rotation

$$Y = X \cdot \Pi^T = \text{FWHT}(X \odot \mathbf{s}) / \sqrt{d}$$

Element-wise multiply by the random sign vector, then apply the FWHT, then scale.

### Inverse Rotation

$$X = Y \cdot \Pi = \text{FWHT}(Y) / \sqrt{d} \odot \mathbf{s}$$

Apply the FWHT (self-inverse up to scaling), scale, then element-wise multiply by the sign vector.

---

## Comparison with QR

| Property | QR (Haar) | Hadamard |
|----------|-----------|----------|
| Storage | $O(d^2)$ — full matrix | $O(d)$ — just the sign vector |
| Compute | $O(d^2)$ — matrix multiply | $O(d \log d)$ — FWHT |
| Randomness quality | Exact Haar distribution | Approximate (excellent in practice) |
| Constraint | None | $d$ must be power of 2 |

For the default group size of 128 (a power of 2), Hadamard rotation is both faster and more memory-efficient.

### Quality Benchmark

On Qwen3.5-0.8B at 4+4 residual:

| Rotation | PPL | KLD |
|----------|-----|-----|
| QR (Haar) | 14.28 | 0.0020 |
| Hadamard | 14.30 | 0.0020 |

The quality difference is negligible. Use `--rotation hadamard` to enable.

---

## Why Randomized Hadamard Works

The random sign-flip vector $\mathbf{s}$ ensures that the combined transform $HD/\sqrt{d}$ is an approximate random orthogonal matrix. While not exactly Haar-distributed, the randomized Hadamard transform satisfies the Johnson-Lindenstrauss property and provides excellent decorrelation for practical weight distributions. The key insight is that for high-dimensional vectors, the specific choice of orthogonal matrix matters less than the fact that it is random — concentration of measure on $\mathbb{S}^{d-1}$ ensures approximate Gaussianity regardless.

---

## Implementation

| File | Function | Description |
|------|----------|-------------|
| `rotation.py` | `hadamard_rotate()` | Forward: $\text{FWHT}(X \odot \mathbf{s}) / \sqrt{d}$ |
| `rotation.py` | `hadamard_rotate_inverse()` | Inverse: $\text{FWHT}(Y) / \sqrt{d} \odot \mathbf{s}$ |
| `rotation.py` | `_fwht()` | In-place butterfly FWHT kernel |

---

## Relationship to Other Techniques

- **Random Rotation (QR)**: The Walsh-Hadamard transform is the fast alternative to QR rotation. Both achieve the same goal (decorrelation + Gaussianization) but Hadamard trades exact Haar randomness for $O(d \log d)$ speed and $O(d)$ storage.
- **Lloyd-Max Quantization**: Like QR rotation, Hadamard rotation ensures the input to the Lloyd-Max quantizer is approximately $\mathcal{N}(0,1)$, making the codebook near-optimal.
- **Inference (Dequantization)**: At inference, the FWHT is used to pre-rotate the input activation, benefiting from the same $O(d \log d)$ speedup.
- **Group Size Constraint**: Requires $d$ to be a power of 2. The default group size of 128 satisfies this. For non-power-of-2 dimensions, the QR method must be used instead.
