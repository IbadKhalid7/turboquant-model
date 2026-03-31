# Random Rotation for Decorrelation

This document describes the random orthogonal rotation step that transforms correlated, non-Gaussian weight coordinates into approximately i.i.d. Gaussian scalars — the prerequisite for optimal scalar quantization.

---

## Formulation Context: Why Rotation Is Needed

In the [quantization formulation](../formulation.md), the Lloyd-Max codebook is optimal for $\mathcal{N}(0,1)$ inputs. But trained neural network weight matrices have structured correlations — some coordinates carry much more information than others. Scalar quantization applied directly to correlated weights wastes bits on low-variance coordinates and under-quantizes high-variance ones.

Random orthogonal rotation (Step 2 of the pipeline) is what bridges this gap: it projects the weights into a basis where scalar quantization is near-optimal.

---

## The Rotation

Multiplying by a random orthogonal matrix $\Pi \in \mathbb{R}^{d \times d}$ (where $\Pi^T \Pi = I$) spreads information uniformly across all coordinates:

$$Y = W_{\text{norm}} \cdot \Pi^T$$

### Properties

| Property | Statement | Why it matters |
|----------|-----------|----------------|
| **Norm-preserving** | $\|Y\|_2 = \|W_{\text{norm}}\|_2 = 1$ | Orthogonal matrices preserve Frobenius norm — no information lost |
| **Decorrelating** | Coordinates of $Y$ become approximately independent | Enables per-coordinate scalar quantization without cross-coordinate waste |
| **Gaussianizing** | Each coordinate $Y_k \sim \text{approx.}\;\mathcal{N}(0, 1/d)$ | Matches the Lloyd-Max codebook distribution (after $\times\sqrt{d}$ scaling) |
| **Invertible** | $W_{\text{norm}} = Y \cdot \Pi$ | The inverse is just the transpose — exact reconstruction from rotated domain |

The Gaussianization property follows from concentration-of-measure on the sphere $\mathbb{S}^{d-1}$: a random projection of a unit vector in $\mathbb{R}^d$ concentrates around $\mathcal{N}(0, 1/d)$ per coordinate for large $d$.

---

## Haar-Distributed Random Orthogonal Matrix (QR Method)

The "gold standard" for random rotations. A matrix drawn from the Haar measure on $O(d)$ is maximally random — it is the unique distribution invariant under left/right multiplication by any orthogonal matrix.

### Algorithm

1. Draw $A \in \mathbb{R}^{d \times d}$ with i.i.d. $\mathcal{N}(0, 1)$ entries
2. Compute QR decomposition: $A = QR$
3. Adjust signs: $\Pi = Q \cdot \text{diag}(\text{sign}(\text{diag}(R)))$

This ensures $\Pi$ is exactly Haar-distributed.

### Trade-offs

- **Storage:** $O(d^2)$ — the full $d \times d$ matrix must be stored or regenerated from the seed
- **Compute:** $O(d^2)$ per row — matrix-vector multiply
- For $d = 128$ (default group size): 128 × 128 × 4 bytes = 64 KB — manageable
- For full-dimension rotation ($d = 4096$): 64 MB per layer — expensive

The rotation matrix is deterministic given the seed, so only the seed (a single integer) needs to be stored. The matrix is regenerated at inference time and cached.

---

## Rotation in the Pipeline

In the quantization pipeline, rotation is applied per-group:

$$Y^{(g)}_m = \bar{W}^{(g)}_m \cdot \Pi_g^T$$

where $\bar{W}^{(g)}_m$ is the normalized row-group slice and $\Pi_g$ is the rotation for group $g$. The rotation seed determines $\Pi_g$ for all groups (groups share the same seed but use group-indexed slicing in practice).

At inference, the **input** is rotated instead of inverse-rotating the weight:

$$x_{\text{rot}}^{(g)} = \Pi_g \cdot x^{(g)}$$

This is cheaper: a $(B, d)$ operation vs the $(M, d)$ inverse rotation on the weight side.

---

## Fast Alternative: Walsh-Hadamard

For group sizes that are powers of 2 (the common case with $d = 128$), the [Walsh-Hadamard transform](walsh-hadamard.md) provides a faster alternative with $O(d \log d)$ compute and $O(d)$ storage, at the cost of approximate (rather than exact) Haar randomness.

| | QR (Haar) | Hadamard |
|--|-----------|----------|
| Storage | $O(d^2)$ — full matrix | $O(d)$ — just the sign vector |
| Compute | $O(d^2)$ — matrix multiply | $O(d \log d)$ — FWHT |
| Randomness quality | Exact Haar distribution | Approximate (excellent in practice) |
| Constraint | None | $d$ must be power of 2 |

---

## Implementation

| File | Function | Description |
|------|----------|-------------|
| `rotation.py` | `generate_rotation_matrix()` | Generate Haar-distributed random orthogonal matrix via QR |
| `rotation.py` | `hadamard_rotate()` | Fast Walsh-Hadamard forward rotation |
| `rotation.py` | `hadamard_rotate_inverse()` | Fast Walsh-Hadamard inverse rotation |

---

## Relationship to Other Techniques

- **Lloyd-Max Quantization**: Rotation is what makes Lloyd-Max near-optimal — by transforming arbitrary weight distributions into the $\mathcal{N}(0,1)$ distribution the codebook is designed for.
- **Walsh-Hadamard Transform**: The fast alternative to QR rotation. See [walsh-hadamard.md](walsh-hadamard.md).
- **Row Normalization (Step 1)**: Normalization ensures $\|W\|_2 = 1$ before rotation, so the rotated coordinates have the correct variance ($1/d$).
- **Residual Quantization**: Each residual pass can use its own rotation seed (independent strategy) or share seeds (shared/alternating strategies). The seed choice affects error correlation and merge capability.
- **Inference (Dequantization)**: At inference, the input is pre-rotated instead of inverse-rotating the weights — a key computational saving.
