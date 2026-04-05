# Block-wise Norm Calibration

This document describes the block-wise norm calibration technique that fine-tunes the per-row norm vectors $\alpha_m$ to minimize end-to-end reconstruction error through each transformer block.

---

## Motivation

After quantization, the per-row norms $\alpha_m = \|W_m\|_2$ are computed analytically from the original weight matrix. While this is optimal in a per-layer MSE sense, it does not account for **error propagation through the network** — the output error of block $\ell$ becomes the input error for block $\ell + 1$, compounding through all $L$ blocks.

Per-layer calibration (optimizing each layer's norms independently) was found to **degrade** end-to-end quality: per-layer MSE improves, but PPL and KLD regress because locally optimal norms can amplify errors when composed.

---

## Formulation

### Block Structure

A transformer block $\mathcal{B}_\ell$ contains multiple quantized linear layers (attention projections, MLP layers). Its output is:

$$h_\ell = \mathcal{B}_\ell(h_{\ell-1}; \{\alpha^{(i)}_\ell\})$$

where $\{\alpha^{(i)}_\ell\}$ are the norm vectors of all TurboQuant linears in block $\ell$.

### Calibration Objective

For each block $\ell$, given a fixed input $h_{\ell-1}$ (from the quantized model's output of block $\ell-1$), optimize:

$$\min_{\{\alpha^{(i)}_\ell\}} \; \underbrace{\|h_\ell^{\text{fp}} - h_\ell^{\text{tq}}\|^2}_{\text{MSE}} + \lambda \left( \underbrace{1 - \cos(h_\ell^{\text{fp}}, h_\ell^{\text{tq}})}_{\text{angular}} + \underbrace{D_{\text{KL}}(\text{softmax}(h_\ell^{\text{fp}}) \| \text{softmax}(h_\ell^{\text{tq}}))}_{\text{distributional}} \right)$$

where:
- $h_\ell^{\text{fp}}$ — full-precision block output (pre-captured in one forward pass)
- $h_\ell^{\text{tq}}$ — TurboQuant block output with current norms
- The softmax and KLD terms treat hidden states as distributions over the feature dimension

### Parameterization

Instead of directly optimizing $\alpha_{m,g}$, we parameterize the correction as a multiplicative scale:

$$\hat{\alpha}_{m,g} = \alpha_{m,g} \cdot \exp(\beta_{m,g})$$

where $\beta_{m,g} \in \mathbb{R}$ is initialized to zero. This ensures:
- The initial solution is the analytical norm (identity transform)
- Positive-definiteness: $\hat{\alpha}_{m,g} > 0$ always
- Smooth optimization landscape near the identity

By default, $\beta$ is **per-group** (shape $M \times G$), allowing each group's norm to be independently adjusted. This is critical: per-group correction recovers 2× more quality than per-row because different groups contribute differently to the output error.

After optimization, the correction is folded: $\alpha_{m,g} \leftarrow \alpha_{m,g} \cdot \exp(\beta_{m,g}^*)$.

---

## Algorithm

```
Input: FP model, TQ model, calibration data X
Output: Calibrated TQ model

1. Pre-capture all L FP block outputs {h_1^fp, ..., h_L^fp} in one forward pass
2. Offload FP model to CPU
3. Compute initial TQ backbone forward to get block-0 inputs

4. For ℓ = 1, ..., L:
   a. Disable fused kernels for current block's TQ linears
   b. Create learnable α parameters (one per TQ linear, per-row)
   c. Patch forwards: y = (y * exp(α)[None, :]).to(x.dtype)
   d. Optimize with AdamW for T iterations:
      - Forward: h_ℓ^tq = B_ℓ(input; patched norms)
      - Loss = MSE(h_ℓ^fp, h_ℓ^tq) + λ * (angular + KLD)
      - Backprop through α only (all other params frozen)
   e. Fold optimal α into weight_norms
   f. Restore fused kernels and original forwards
   g. Forward TQ backbone to capture next block's input

5. Calibrate lm_head separately (per-layer fallback)
```

### Key Design Decisions

- **Sequential block processing**: Block $\ell$'s input comes from the *calibrated* model's output at block $\ell-1$, so each block sees the correct error landscape
- **Pre-captured FP targets**: One full FP forward pass captures all targets, then FP model is offloaded to CPU to free GPU memory
- **Sub-batch gradient**: Uses $\min(4, B)$ samples for gradient computation to manage memory, but evaluates metrics on all samples

---

## Results (Qwen3.5-0.8B-Base)

### 4-bit Quantization

| Method | PPL | $\Delta$PPL | KLD | $\Delta$KLD | Time |
|--------|-----|-------------|-----|-------------|------|
| Analytical norms (baseline) | 13.9564 | — | 0.130127 | — | — |
| Per-layer calibration | 14.0220 | +0.0656 | 0.135156 | +0.005 | ~35 min |
| Blockwise per-row (4s / 50i) | 13.6971 | −0.2592 | 0.117041 | −0.0131 | 12.9 min |
| **Blockwise per-group (4s / 50i)** | **13.4427** | **−0.5137** | **0.095947** | **−0.0342** | **14.0 min** |

### 4+4 Residual Quantization

| Method | PPL | $\Delta$PPL | KLD | $\Delta$KLD |
|--------|-----|-------------|-----|-------------|
| Analytical norms | 12.1540 | — | 0.001887 | — |
| Blockwise per-row (16s / 200i) | 12.1540 | +0.0000 | 0.001912 | +0.000025 |

**Key findings:**
1. **Per-group** blockwise calibration improves 4-bit PPL by **3.68%** and KLD by **26.3%**
2. Per-group is **2× better than per-row** at the same cost (different groups contribute differently to output error)
3. Per-layer calibration is **harmful** — it reduces per-layer MSE but increases end-to-end PPL
4. 4+4 residual is already near-perfect ($\cos \approx 1.000$) and does not benefit from calibration
5. Recovers **28.1%** of the quantization gap (bf16 12.13 → 4-bit 13.96 → calibrated 13.44)

### Default Configuration

Based on the above analysis, the default configuration is:
- `n_samples = 4` (calibration sequences from WikiText-103)
- `n_iters = 50` (AdamW optimization steps per block)
- `per_group = True` (per-group alpha correction)
- `lr = 1e-3`, `lambda = 1.0`

This adds approximately **14 minutes** to the quantization pipeline for a 0.8B model and recovers ~28% of the quantization gap.

---

## Integration

Blockwise calibration is integrated into the main pipeline:

```bash
# During quantization (adds ~13 min for 0.8B model)
turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized \
    --bit-width 4 --calibrate

# Post-hoc calibration of a saved model
turboquant calibrate --model Qwen/Qwen3.5-0.8B-Base --quantized ./quantized

# Python API
from turboquant_model import calibrate_norms_blockwise, CalibrationConfig

config = CalibrationConfig(n_samples=4, n_iters=50)
stats = calibrate_norms_blockwise(tq_model, fp_model, tokenizer, device="cuda", config=config)
```

---

## When to Use

| Scenario | Recommendation |
|----------|---------------|
| 4-bit single-pass | **Yes** — significant PPL and KLD improvement |
| 4+4 residual | **No** — already near-perfect, calibration has no effect |
| Latency-constrained pipeline | Use default 4s/50i (13 min overhead) |
| Maximum quality | Try 16s/200i (51 min, marginal gain over default) |
