# Quantization Pipeline

This document describes how TurboQuant compresses model weights from full-precision (bf16/fp32) down to 4-bit indices, covering the single-pass pipeline, residual (multi-pass) quantization, and the merge optimization.

## Overview

The quantization pipeline transforms each `nn.Linear` weight matrix $W \in \mathbb{R}^{M \times N}$ (where $M$ = out_features, $N$ = in_features) into a compact representation:

| Component | Shape | Dtype | Purpose |
|-----------|-------|-------|---------|
| `indices_packed` | $(M, N/2)$ | uint8 | Two 4-bit codebook indices per byte |
| `weight_norms` | $(M,)$ or $(M, G)$ | float32 | Row or group norms for rescaling |
| `codebook` | $(2^b,)$ | float32 | Lloyd-Max centroids (shared globally) |
| `seed` | scalar | int | Rotation seed for reproducibility |

At 4-bit, this achieves roughly **4× compression** vs bf16.

## Single-Pass Quantization

The core algorithm processes each weight matrix group-by-group (group size $d$, default 128). For each group $g$:

### Step 1: Row Normalization

$$W_{\text{norm}}^{(g)} = \frac{W^{(g)}}{\|W^{(g)}\|_2}, \qquad \alpha^{(g)} = \|W^{(g)}\|_2$$

Each row of the group slice is divided by its $\ell_2$-norm. The norm $\alpha$ is stored separately and applied during inference. This ensures the rotated coordinates have a known variance ($1/d$), which is critical for the Lloyd-Max codebook to be optimal.

**Implementation:** `model.py → _quantize_weight()`

### Step 2: Random Rotation

$$Y^{(g)} = W_{\text{norm}}^{(g)} \cdot \Pi_g^T$$

A random orthogonal matrix $\Pi_g$ decorrelates the weight coordinates. After rotation, each coordinate is approximately i.i.d. $\mathcal{N}(0, 1/d)$ — the ideal input distribution for scalar Lloyd-Max quantization.

Two rotation methods are supported:

| Method | Storage | Compute | Constraint |
|--------|---------|---------|------------|
| **QR** (Haar random) | $O(d^2)$ | $O(d^2)$ | None |
| **Hadamard** (Walsh-Hadamard + random signs) | $O(d)$ | $O(d \log d)$ | $d$ must be power of 2 |

The rotation matrix is deterministic given the seed, so only the seed needs to be stored.

**Implementation:** `rotation.py → generate_rotation_matrix()` (QR) or `hadamard_rotate()` (Hadamard)

### Step 3: Scaling

$$Y_{\text{scaled}}^{(g)} = Y^{(g)} \cdot \sqrt{d}$$

Since rotation preserves norms and each coordinate has variance $1/d$, multiplying by $\sqrt{d}$ brings the coordinates to unit variance: $\mathcal{N}(0, 1)$. This is exactly the distribution for which the Lloyd-Max codebook is computed.

### Step 4: Scalar Quantization

$$\text{idx}_{m,k} = \text{searchsorted}(\text{boundaries}, Y_{\text{scaled},m,k})$$

Each scalar coordinate is independently quantized using the Lloyd-Max optimal boundaries for $\mathcal{N}(0,1)$. At 4 bits, there are 16 centroids and 15 decision boundaries.

**Implementation:** `codebook.py → get_codebook()`

### Step 5: 4-bit Packing

$$\text{packed}_{m, k/2} = \text{lo}_k \;|\; (\text{hi}_{k+1} \ll 4)$$

Consecutive pairs of 4-bit indices are packed into a single uint8 byte, halving the storage for the index tensor.

**Implementation:** `quantize.py → pack_4bit()`

### Complete Single-Pass Pseudocode

```
for each group g in [0, n_groups):
    W_g = W[:, g*d : (g+1)*d]                        # (M, d)
    norms = ||W_g||₂  (per row)                        # (M,)
    W_norm = W_g / norms                               # (M, d)
    Y = W_norm @ Pi_g.T                                # (M, d)  rotate
    Y_scaled = Y * sqrt(d)                             # (M, d)  to N(0,1)
    indices = searchsorted(boundaries, Y_scaled)       # (M, d)  quantize
    indices = clamp(indices, 0, 2^b - 1)
    packed[:, g*d//2 : (g+1)*d//2] = pack_4bit(indices)

store: packed (uint8), norms (float32), codebook (float32), seed (int)
```

**Entry point:** `quantize.py → turboquant_quantize_packed()` for standalone use, or `model.py → quantize_model()` for full-model quantization.

## Residual Quantization (Multi-Pass)

Single-pass 4-bit quantization introduces noticeable error. Residual quantization dramatically reduces this by running multiple passes, each quantizing the error left by the previous pass.

### Two-Pass Pipeline

```
Pass 1:  W_hat1 = TQ(W,         bit_width=b₁, seed=s₁)
         R      = W - W_hat1
Pass 2:  R_hat  = TQ(R,          bit_width=b₂, seed=s₂)
         W_approx = W_hat1 + R_hat
```

Total storage is $b_1 + b_2$ bits per weight. A **4+4 residual** (8 total bits) achieves near-lossless quality: PPL 14.28 vs baseline 14.29 on Qwen3.5-0.8B, KLD only 0.002 nats.

**Implementation:** `residual.py → residual_quantize_packed()`

### N-Pass Pipeline (Shared Rotation)

For more than two passes, a shared rotation seed enables an important optimization (see Merging below):

```
for k in 1..n_passes:
    W_hat_k = TQ(R_{k-1}, bit_width=b, seed=s)   # same seed s for all passes
    R_k = R_{k-1} - W_hat_k
    W_approx += W_hat_k
```

MSE decreases monotonically with each additional pass.

**Implementation:** `residual.py → multi_residual_quantize_packed()`

## Merging Multi-Pass to Single-Pass

When all passes share the same rotation seed, the rotation factors out of the sum, enabling a merge that avoids the expensive inverse rotation:

### Fast-Path Merge (Shared Rotation)

Since $\hat{W}_k^{(g)} = \frac{\alpha_k}{\sqrt{d}} \cdot C_k[\mathbf{i}_k] \cdot \Pi_g$ for all passes, the sum becomes:

$$\tilde{Y}^{(g)} = \sum_k \frac{\alpha_k}{\sqrt{d}} \cdot C_k[\mathbf{i}_k]$$

This sum is computed entirely in the **rotated domain** — no inverse rotation needed. The merged result is then re-normalized and re-quantized into a single-pass representation:

```
for each group g:
    Y_sum = Σ_k (norms_k * codebook_k[indices_k] / sqrt(d))   # sum in rotated domain
    norms_merged = ||Y_sum||                                     # new norms
    Y_norm = Y_sum / norms_merged                                # re-normalize
    indices_merged = searchsorted(boundaries, Y_norm * sqrt(d))  # re-quantize
```

**Result:** A single packed representation with the same format as single-pass, but approximating the multi-pass quality.

**Implementation:** `residual.py → merge_and_requantize()`

### Why Merge?

| Property | Multi-pass (separate) | Merged (single-pass) |
|----------|----------------------|---------------------|
| Storage | $n \times$ single-pass | $1 \times$ single-pass |
| Inference cost | $n$ rotations + $n$ matmuls per group | 1 rotation + 1 matmul per group |
| Quality | Best (no re-quantization loss) | Slightly worse (re-quantization) |

Merging trades a small accuracy loss for halved inference latency and storage.

**Module-level merge:** `module.py → TurboQuantLinear.merge_passes()`

## Model-Level Quantization

`quantize_model()` orchestrates the full pipeline over an `nn.Module`:

1. Iterate all `nn.Linear` layers (optionally skip embeddings / lm_head)
2. For each layer:
   - Run single-pass quantization → create `TurboQuantLinear`
   - If residual configured: dequantize pass 1, compute residual, quantize pass 2, call `set_pass2()`
   - Replace the original `nn.Linear` in the model
3. Log compression statistics (original size → compressed size, ratio)

**Implementation:** `model.py → quantize_model()`

## Saved Format

`save_quantized()` writes the following directory structure:

```
output_dir/
├── turboquant_config.json         # TurboQuantConfig (JSON)
├── codebook.pt                    # Shared codebook tensor
├── layers/
│   ├── {name}.indices.pt          # Packed 4-bit indices
│   ├── {name}.norms.pt            # Per-row/group norms
│   ├── {name}.bias.pt             # (optional) bias vector
│   ├── {name}.pass2_indices.pt    # (optional) residual indices
│   ├── {name}.pass2_norms.pt      # (optional) residual norms
│   └── {name}.pass2_codebook.pt   # (optional) residual codebook
├── non_quantized.pt               # Non-linear params (LayerNorm, embeddings, etc.)
└── config.json                    # (optional) HuggingFace model config
```

**Implementation:** `model.py → save_quantized()` / `load_quantized()`

## CLI Usage

```bash
# Single-pass 4-bit
turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized --bit-width 4

# 4+4 residual (near-lossless)
turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized \
    --bit-width 4 --residual-bit-width 4

# With Hadamard rotation (faster for power-of-2 dimensions)
turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized \
    --bit-width 4 --rotation hadamard
```
