# Quantization Problem Formulation

This document formulates the TurboQuant weight quantization problem in general mathematical terms, under the assumptions of random orthogonal rotation and Lloyd-Max scalar codebook.

---

## 1. Problem Statement

Given a pre-trained weight matrix $W \in \mathbb{R}^{M \times N}$, find a compressed representation $\hat{W}$ that minimizes the mean squared reconstruction error:

$$\min_{\hat{W}} \;\; \frac{1}{MN} \|W - \hat{W}\|_F^2$$

subject to $\hat{W}$ being stored using only $b$ bits per weight element, plus a small side-information budget (norms, codebook, seed).

---

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| $W \in \mathbb{R}^{M \times N}$ | Full-precision weight matrix ($M$ = out_features, $N$ = in_features) |
| $b$ | Bit-width of the quantizer ($L = 2^b$ levels) |
| $d$ | Group size (columns processed together; $d \mid N$ for simplicity) |
| $G = N / d$ | Number of groups |
| $\Pi_g \in \mathbb{R}^{d \times d}$ | Random orthogonal rotation matrix for group $g$ |
| $\alpha_{m,g} \in \mathbb{R}_{>0}$ | Row norm of group $g$ of row $m$ |
| $Q_b : \mathbb{R} \to \{c_1, \ldots, c_L\}$ | Scalar quantizer mapping to $L$ centroids |
| $\{c_\ell\}_{\ell=1}^L$ | Lloyd-Max codebook (centroids) |
| $\{t_\ell\}_{\ell=0}^L$ | Decision boundaries ($t_0 = -\infty$, $t_L = +\infty$) |

---

## 3. Decomposition into Groups

Partition columns into $G$ groups of size $d$:

$$W = \bigl[W^{(0)} \;\big|\; W^{(1)} \;\big|\; \cdots \;\big|\; W^{(G-1)}\bigr], \quad W^{(g)} \in \mathbb{R}^{M \times d}$$

Each group is quantized independently, so the overall MSE decomposes:

$$\frac{1}{MN}\|W - \hat{W}\|_F^2 = \frac{1}{G}\sum_{g=0}^{G-1} \frac{1}{Md}\|W^{(g)} - \hat{W}^{(g)}\|_F^2$$

---

## 4. Single-Pass Quantization Pipeline

For each group $g$ and each row $m$, the pipeline proceeds in five steps.

### 4.1 Row Normalization

Extract the row norm and normalize:

$$\alpha_{m,g} = \|W^{(g)}_m\|_2, \qquad \bar{W}^{(g)}_m = \frac{W^{(g)}_m}{\alpha_{m,g}}$$

After normalization, $\|\bar{W}^{(g)}_m\|_2 = 1$ and each component has expected magnitude $1/\sqrt{d}$.

### 4.2 Random Rotation

Apply a random orthogonal transformation:

$$Y^{(g)}_m = \bar{W}^{(g)}_m \cdot \Pi_g^T \in \mathbb{R}^d$$

where $\Pi_g$ is drawn from the Haar measure on $\mathcal{O}(d)$ (or approximated via the Walsh-Hadamard transform with random sign flips).

**Key property:** Since $\Pi_g$ is orthogonal, $\|Y^{(g)}_m\|_2 = \|\bar{W}^{(g)}_m\|_2 = 1$, so each component satisfies:

$$Y^{(g)}_{m,k} \;\sim\; \text{approx.} \;\; \mathcal{N}\!\left(0, \frac{1}{d}\right), \quad k = 1, \ldots, d$$

and the components are approximately independent for large $d$. This is guaranteed by concentration-of-measure on the sphere $\mathbb{S}^{d-1}$.

### 4.3 Variance Normalization

Scale to unit variance:

$$Z^{(g)}_{m,k} = \sqrt{d} \;\cdot\; Y^{(g)}_{m,k}$$

Now each scalar satisfies $Z^{(g)}_{m,k} \sim \text{approx.} \;\mathcal{N}(0, 1)$.

### 4.4 Scalar Quantization (Lloyd-Max)

Apply the optimal scalar quantizer for $\mathcal{N}(0,1)$ to each component independently:

$$\hat{Z}^{(g)}_{m,k} = Q_b(Z^{(g)}_{m,k}) = c_\ell \quad \text{where } \ell = \arg\min_{j} |Z^{(g)}_{m,k} - c_j|$$

Equivalently, using decision boundaries:

$$\ell = \ell(z) = \sum_{j=1}^{L-1} \mathbf{1}[z > t_j]$$

The Lloyd-Max codebook $\{c_\ell, t_\ell\}$ is the unique minimizer of the scalar MSE for $\mathcal{N}(0,1)$:

$$\{c_\ell^*, t_\ell^*\} = \arg\min_{\{c_\ell, t_\ell\}} \; \mathbb{E}_{Z \sim \mathcal{N}(0,1)}\!\left[(Z - Q_b(Z))^2\right]$$

satisfying the necessary conditions:

$$c_\ell = \frac{\int_{t_{\ell-1}}^{t_\ell} z \,\phi(z)\,dz}{\int_{t_{\ell-1}}^{t_\ell} \phi(z)\,dz}, \qquad t_\ell = \frac{c_\ell + c_{\ell+1}}{2}$$

where $\phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ is the standard Gaussian density.

### 4.5 Reconstruction

The quantized approximation of the original weight is:

$$\hat{W}^{(g)}_m = \frac{\alpha_{m,g}}{\sqrt{d}} \cdot \hat{Z}^{(g)}_m \cdot \Pi_g = \frac{\alpha_{m,g}}{\sqrt{d}} \cdot \mathbf{c}[\boldsymbol{\ell}_m] \cdot \Pi_g$$

where $\mathbf{c}[\boldsymbol{\ell}_m] = (c_{\ell_{m,1}}, \ldots, c_{\ell_{m,d}})$ is the vector of quantized centroids.

---

## 5. Storage Representation

The compressed form stores:

| Component | Per group | Total | Bits/weight |
|-----------|-----------|-------|-------------|
| Indices $\boldsymbol{\ell}_{m,k}$ | $M \times d$ at $b$ bits each | $MN \cdot b$ bits | $b$ |
| Norms $\alpha_{m,g}$ | $M$ floats | $M \cdot G \cdot 32$ bits | $32/d$ |
| Codebook $\{c_\ell\}$ | $L$ floats (shared globally) | $L \cdot 32$ bits | $\approx 0$ |
| Seed $s$ | 1 integer | 32 bits | $\approx 0$ |

**Effective bit-rate:**

$$r = b + \frac{32}{d} \;\approx\; b \quad \text{for } d \gg 32$$

---

## 6. Quantization MSE Analysis

### 6.1 Per-Element MSE

The reconstruction error for a single group element is:

$$\mathbb{E}\!\left[(W^{(g)}_{m,k} - \hat{W}^{(g)}_{m,k})^2\right] = \frac{\alpha_{m,g}^2}{d} \cdot D_b$$

where $D_b = \mathbb{E}_{Z \sim \mathcal{N}(0,1)}[(Z - Q_b(Z))^2]$ is the distortion of the $b$-bit Lloyd-Max quantizer on $\mathcal{N}(0,1)$.

**Proof sketch:** The orthogonal rotation preserves the Frobenius norm, so:

$$\|W^{(g)}_m - \hat{W}^{(g)}_m\|_2^2 = \frac{\alpha_{m,g}^2}{d} \cdot \|\hat{Z}^{(g)}_m - Z^{(g)}_m\|_2^2 = \frac{\alpha_{m,g}^2}{d} \cdot \sum_{k=1}^d (Z^{(g)}_{m,k} - \hat{Z}^{(g)}_{m,k})^2$$

Taking expectations and using approximate independence of the $Z_{m,k}$ terms gives $d \cdot D_b$ for the sum.

### 6.2 Overall MSE

$$\text{MSE} = \frac{1}{MN}\|W - \hat{W}\|_F^2 = \frac{D_b}{MN} \sum_{m=1}^M \sum_{g=0}^{G-1} \alpha_{m,g}^2 = D_b \cdot \overline{\alpha^2}$$

where $\overline{\alpha^2} = \frac{1}{MN}\sum_{m,g} \alpha_{m,g}^2$ is the average squared norm per weight element.

### 6.3 Lloyd-Max Distortion Values

| $b$ (bits) | $L$ (levels) | $D_b$ | SNR ($-10\log_{10} D_b$) |
|------------|-------------|-------|--------------------------|
| 1 | 2 | 0.3634 | 4.40 dB |
| 2 | 4 | 0.1175 | 9.30 dB |
| 3 | 8 | 0.03454 | 14.62 dB |
| 4 | 16 | 0.009497 | 20.22 dB |
| 5 | 32 | 0.002499 | 26.02 dB |

Each additional bit roughly halves the distortion (~6 dB improvement).

---

## 7. Near-Optimality Argument

**Claim:** Under the rotation + Lloyd-Max scheme, the per-element MSE is near the rate-distortion lower bound for Gaussian sources at rate $b$ bits per sample.

The Shannon rate-distortion function for $\mathcal{N}(0,1)$ at rate $R$ bits is:

$$D^*(R) = 2^{-2R}$$

For $b = 4$ bits, $D^*(4) = 2^{-8} \approx 0.00391$. The Lloyd-Max quantizer achieves $D_4 = 0.00950$, giving:

$$\frac{D_4}{D^*(4)} \approx 2.43$$

The gap is only $\sim 3.9$ dB from the theoretical optimum. Moreover, this gap decreases for higher $b$.

**Why rotation makes this possible:** Without rotation, the weight coordinates of a trained neural network are correlated and non-Gaussian. Correlation means that scalar quantization (operating per-coordinate) leaves inter-coordinate redundancy unexploited. The random rotation decorrelates the coordinates and projects them onto i.i.d. approximate Gaussians, reducing the problem to the case where scalar Lloyd-Max is near-optimal.

---

## 8. Residual Quantization

The single-pass error can be reduced by iteratively quantizing the reconstruction residual.

### 8.1 Multi-Pass Formulation

Define the residual sequence:

$$R^{(0)} = W, \qquad R^{(k)} = R^{(k-1)} - \hat{R}^{(k-1)}, \quad k = 1, 2, \ldots, P$$

where $\hat{R}^{(k-1)} = \text{TQ}(R^{(k-1)}, b_k, s_k)$ applies the full single-pass pipeline (normalize, rotate with seed $s_k$, scale, quantize at $b_k$ bits, reconstruct).

The final approximation is:

$$\hat{W} = \sum_{k=0}^{P-1} \hat{R}^{(k)}$$

### 8.2 Error Reduction

At each pass, the residual magnitude decreases:

$$\|R^{(k)}\|_F^2 = \|R^{(k-1)} - \hat{R}^{(k-1)}\|_F^2 = \|R^{(k-1)}\|_F^2 \cdot D_{b_k} \cdot \rho_k$$

where $\rho_k \leq 1$ is a correction factor accounting for the residual's deviation from the ideal Gaussian model. In practice, $\rho_k \approx 1$ for small $b_k$ where the residuals remain approximately Gaussian.

After $P$ passes, the total MSE is approximately:

$$\text{MSE}_P \approx \overline{\alpha^2} \cdot \prod_{k=0}^{P-1} D_{b_k}$$

### 8.3 Effective Bit-Rate

Total storage is $\sum_{k=0}^{P-1} b_k$ bits per weight element. For example:

| Config | Passes | Bits per weight | Expected MSE ratio |
|--------|--------|-----------------|--------------------|
| 4-bit single | 1 | 4 | $D_4 \approx 0.0095$ |
| 4+4 residual | 2 | 8 | $D_4^2 \approx 9.0 \times 10^{-5}$ |
| 4+2 residual | 2 | 6 | $D_4 \cdot D_2 \approx 1.1 \times 10^{-3}$ |
| 2+2+2+2 | 4 | 8 | $D_2^4 \approx 1.9 \times 10^{-4}$ |

---

## 9. Rotation Strategy

### 9.1 Independent Rotations (Different Seeds)

Each pass uses a distinct rotation seed $s_k \neq s_j$ for $k \neq j$.

**Advantage:** The quantization errors across passes are projected onto different random subspaces, making them approximately uncorrelated. This maximizes the error reduction per pass.

**Disadvantage:** Cannot merge passes in the rotated domain (see §10).

### 9.2 Shared Rotation (Same Seed)

All passes use the same seed: $s_0 = s_1 = \cdots = s_{P-1} = s$.

**Advantage:** Enables the fast-path merge (§10), reducing storage and inference cost to single-pass levels.

**Disadvantage:** Errors are correlated across passes (quantized in the same basis), reducing the benefit of each additional pass.

### 9.3 Alternating Rotation (Two Seeds)

Even-numbered passes use seed $s_a$, odd-numbered passes use seed $s_b$:

$$s_k = \begin{cases} s_a & \text{if } k \bmod 2 = 0 \\ s_b & \text{if } k \bmod 2 = 1 \end{cases}$$

**Advantage:** Adjacent passes project errors onto different bases, achieving near-independent error reduction while using only two rotation matrices.

**Disadvantage:** Cannot use the fast-path merge (requires fully shared rotation).

---

## 10. Merge Optimization (Shared Rotation Only)

When all passes share the same rotation $\Pi_g$, the sum of reconstructions telescopes in the rotated domain.

### 10.1 Derivation

For pass $k$, the reconstructed group is:

$$\hat{R}^{(k,g)}_m = \frac{\alpha_{m,g}^{(k)}}{\sqrt{d}} \cdot \mathbf{c}^{(k)}[\boldsymbol{\ell}_m^{(k)}] \cdot \Pi_g$$

Summing over passes:

$$\hat{W}^{(g)}_m = \sum_{k=0}^{P-1} \hat{R}^{(k,g)}_m = \left(\sum_{k=0}^{P-1} \frac{\alpha_{m,g}^{(k)}}{\sqrt{d}} \cdot \mathbf{c}^{(k)}[\boldsymbol{\ell}_m^{(k)}]\right) \cdot \Pi_g$$

Define the **merged rotated vector:**

$$\tilde{Y}^{(g)}_m = \sum_{k=0}^{P-1} \frac{\alpha_{m,g}^{(k)}}{\sqrt{d}} \cdot \mathbf{c}^{(k)}[\boldsymbol{\ell}_m^{(k)}] \in \mathbb{R}^d$$

### 10.2 Re-Quantization

To collapse the multi-pass representation into a single-pass format, re-quantize $\tilde{Y}^{(g)}_m$:

$$\tilde{\alpha}_{m,g} = \|\tilde{Y}^{(g)}_m\|_2$$

$$\tilde{Z}^{(g)}_m = \frac{\sqrt{d} \cdot \tilde{Y}^{(g)}_m}{\tilde{\alpha}_{m,g}}$$

$$\tilde{\boldsymbol{\ell}}_m = Q_{b'}(\tilde{Z}^{(g)}_m)$$

This converts $P$ passes at $b$ bits each into a single pass at $b'$ bits, at the cost of re-quantization noise.

### 10.3 Merge MSE

The merge introduces an additional quantization error:

$$\text{MSE}_{\text{merge}} \approx \text{MSE}_P + D_{b'} \cdot \overline{\tilde{\alpha}^2}$$

where $\overline{\tilde{\alpha}^2}$ is the average squared norm of the merged vectors. For $b' \geq b$, the merge overhead is small.

---

## 11. Inference (On-the-Fly Dequantization)

Instead of materializing the full $M \times N$ weight matrix, the forward pass operates in the rotated domain.

### 11.1 Standard Forward Pass

For input $\mathbf{x} \in \mathbb{R}^N$ and output $\mathbf{y} = W\mathbf{x} \in \mathbb{R}^M$:

$$y_m = \mathbf{W}_m \cdot \mathbf{x} = \sum_{g=0}^{G-1} \hat{W}^{(g)}_m \cdot \mathbf{x}^{(g)}$$

### 11.2 Pre-Rotate Input

Substituting the reconstruction:

$$y_m = \sum_g \frac{\alpha_{m,g}}{\sqrt{d}} \left(\mathbf{c}[\boldsymbol{\ell}_m] \cdot \Pi_g\right) \cdot \mathbf{x}^{(g)} = \sum_g \frac{\alpha_{m,g}}{\sqrt{d}} \cdot \mathbf{c}[\boldsymbol{\ell}_m] \cdot \underbrace{(\Pi_g \mathbf{x}^{(g)})}_{\mathbf{x}_{\text{rot}}^{(g)}}$$

By pre-rotating the input: $\mathbf{x}_{\text{rot}}^{(g)} = \Pi_g \mathbf{x}^{(g)}$, the inner product reduces to a lookup + dot product:

$$y_m = \sum_g \frac{\alpha_{m,g}}{\sqrt{d}} \sum_{k=1}^d c_{\ell_{m,k}} \cdot x_{\text{rot},k}^{(g)}$$

This avoids ever materializing the $M \times d$ dequantized weight block, requiring only $O(b)$ memory per element (the packed indices).

### 11.3 Computational Cost

| Operation | Cost | Comment |
|-----------|------|---------|
| Input rotation $\Pi_g \mathbf{x}^{(g)}$ | $O(d^2)$ or $O(d \log d)$ | QR vs Hadamard |
| Codebook lookup | $O(Md)$ | Index → centroid |
| Fused dot product | $O(Md)$ | Sum of products |
| Norm rescaling | $O(M)$ | Multiply by $\alpha/\sqrt{d}$ |

Total: $O(MN)$ per forward pass — same asymptotic cost as dense matmul, but with much smaller memory footprint.

---

## 12. Summary

The TurboQuant quantization objective can be written compactly as:

$$\hat{W} = \sum_{k=0}^{P-1} \text{TQ}(R^{(k)}, b_k, s_k)$$

where $R^{(0)} = W$, $R^{(k)} = R^{(k-1)} - \text{TQ}(R^{(k-1)}, b_{k-1}, s_{k-1})$, and:

$$\text{TQ}(X, b, s)^{(g)}_m = \frac{\|X^{(g)}_m\|_2}{\sqrt{d}} \cdot Q_b\!\left(\sqrt{d} \cdot \frac{X^{(g)}_m}{\|X^{(g)}_m\|_2} \cdot \Pi_g(s)^T\right) \cdot \Pi_g(s)$$

The scheme is **near-optimal** because:

1. **Rotation** decorrelates weight coordinates → i.i.d. approximate $\mathcal{N}(0, 1/d)$
2. **Normalization** matches the codebook's design distribution → $\mathcal{N}(0, 1)$
3. **Lloyd-Max** is the optimal scalar quantizer for known distributions → minimal $D_b$
4. **Residual passes** exploit the remaining structure → multiplicative MSE reduction
5. **On-the-fly dequantization** preserves the $b$-bit memory advantage at inference

---

## 13. Bits-Per-Weight (BPW) Analysis

Beyond minimizing MSE at a fixed bit-width, a second objective is **minimizing the total bits per weight** (BPW) for a given quality target. BPW determines model size on disk and memory footprint at inference.

### 13.1 BPW Decomposition

The total storage per weight element is:

$$\text{BPW} = \underbrace{b_{\text{idx}}}_{\text{index bits}} + \underbrace{\frac{32}{d} \cdot n_{\text{norm}}}_{\text{norm overhead}} + \underbrace{\frac{L \cdot 32}{M \cdot N}}_{\text{codebook}} + \underbrace{\frac{32}{M \cdot N}}_{\text{seed}} + \underbrace{\frac{S_{\text{non-quant}}}{M_{\text{total}} \cdot N_{\text{total}}}}_{\text{non-quantized params}}$$

For practical models the codebook and seed terms are negligible. The dominant terms are:

$$\text{BPW} \approx b_{\text{idx}} + \frac{32 \cdot n_{\text{norm}}}{d} + \text{BPW}_{\text{non-quant}}$$

### 13.2 Variables That Affect BPW

| Variable | Symbol | How it affects BPW | Current default |
|----------|--------|--------------------|-----------------|
| **Index bit-width** | $b$ | Directly: BPW ∝ $b$ | 4 |
| **Number of residual passes** | $P$ | Total index bits = $\sum_k b_k$ | 1 or 2 |
| **Per-pass bit-width** | $b_k$ | Each pass adds $b_k$ to BPW | 4 (or 2 for residual) |
| **Group size** | $d$ | Norm overhead = $32/d$ per norm set | 128 |
| **Norm precision** | $p_{\text{norm}}$ | Overhead = $p_{\text{norm}}/d$ (currently 32-bit) | float32 |
| **Number of norm sets** | $n_{\text{norm}}$ | Each pass stores one set of norms | $P$ |
| **Codebook size** | $L = 2^b$ | Tiny: $32L / (MN)$ | 16 |
| **Non-quantized layers** | — | Embeddings, LayerNorm, lm_head at full precision | model-dependent |
| **Packing efficiency** | — | Sub-byte packing (4-bit → 2 per byte) | $\lfloor 8/b \rfloor$ per byte |

### 13.3 BPW Budget Example

For Qwen3.5-0.8B (752M params) with 4+2 residual quantization, $d = 128$:

| Component | Bits per weight | % of total |
|-----------|:-:|:-:|
| Pass 1 indices (4-bit) | 4.000 | 57.0% |
| Pass 2 indices (2-bit) | 2.000 | 28.5% |
| Pass 1 norms (float32, per-group) | 0.250 | 3.6% |
| Pass 2 norms (float32, per-group) | 0.250 | 3.6% |
| Non-quantized params (embeddings, LN) | ~0.52 | 7.3% |
| **Total** | **~7.02** | **100%** |

---

## 14. Ideas to Lower BPW

### 14.1 Reduce Norm Precision

**Current:** Norms stored at float32 (32 bits), contributing $32/d$ BPW per pass.

**Idea:** Store norms in float16 or bfloat16, halving the overhead to $16/d$ BPW.

$$\Delta\text{BPW} = -\frac{16 \cdot P}{d}$$

At $d = 128$, $P = 2$: saves $0.25$ BPW.

**Risk:** Norms multiply the entire reconstructed row. Reduced precision amplifies rounding error by $\alpha$. Likely safe for $\alpha < 10$ (typical for LLM weights), but sensitive layers (lm_head, attention projections) should be validated.

**Variant:** Adaptive norm precision — use fp16 for small norms, fp32 only for outlier rows.

### 14.2 Shared / Delta Norms Across Passes

**Current:** Each residual pass stores independent norms.

**Idea:** The residual norms are much smaller than pass-1 norms (the residual has lower energy). Store pass-2+ norms as a ratio or delta:

$$\alpha^{(k)}_{m,g} = \alpha^{(0)}_{m,g} \cdot r^{(k)}_{m,g}$$

If the ratios $r^{(k)}$ cluster tightly, they can be quantized to 8-bit or shared across groups.

**Potential saving:** Replace $P-1$ norm sets (each $32/d$ BPW) with 8-bit ratios ($8/d$ BPW each).

### 14.3 Larger Group Size

**Current:** $d = 128$.

**Idea:** Increase to $d = 256$ or $d = 512$:

$$\frac{32}{d}: \quad 128 \to 0.25, \quad 256 \to 0.125, \quad 512 \to 0.0625$$

**Saving per pass:** 0.125 BPW (at $d = 256$) or 0.1875 BPW (at $d = 512$).

**Risk:** The Gaussian approximation $Y_{m,k} \sim \mathcal{N}(0, 1/d)$ improves with larger $d$ (better concentration of measure), but:
- Requires $N$ divisible by $d$ (or padding)
- Larger rotation matrices: $O(d^2)$ storage for QR, $O(d \log d)$ for Hadamard
- Inference: each group rotation operates on $d$-dim input; larger $d$ may reduce GEMM parallelism

### 14.4 Lower Index Bit-Width (Sub-4-bit)

**Current:** 4-bit (16 levels).

**Idea:** Use 3-bit (8 levels) or 2-bit (4 levels) for the primary pass, relying on residual passes for quality recovery.

| Config | Total BPW (indices only) | Expected $\prod D_{b_k}$ |
|--------|:-:|:-:|
| 4-bit single | 4 | $9.5 \times 10^{-3}$ |
| 3+3 residual | 6 | $1.2 \times 10^{-3}$ |
| 2+2+2 residual | 6 | $1.6 \times 10^{-3}$ |
| 3+2 residual | 5 | $4.1 \times 10^{-3}$ |
| 2+2 residual | 4 | $1.4 \times 10^{-2}$ |

A 3+2 config achieves 5 BPW with distortion comparable to 4-bit single-pass.

**Risk:** The Gaussian assumption weakens for very low bit-widths — the quantization error (residual) after a 2-bit pass is large and less Gaussian, reducing the effectiveness of subsequent passes.

### 14.5 Non-Uniform Bit Allocation Across Layers

**Current:** All layers use the same $b$ and $b_{\text{residual}}$.

**Idea:** Assign higher bit-width to sensitive layers (e.g., attention Q/K projections, first/last layers) and lower bit-width to less sensitive layers (MLP intermediate projections).

**Method:** Use a sensitivity metric (Fisher information, Hessian diagonal, or calibration MSE) to rank layers, then solve:

$$\min_{\{b_\ell\}} \sum_\ell w_\ell \cdot D_{b_\ell} \cdot \|\alpha_\ell\|^2 \quad \text{s.t.} \quad \frac{1}{|\mathcal{L}|}\sum_\ell b_\ell \leq B_{\text{target}}$$

This is a knapsack-like problem solvable via dynamic programming or Lagrangian relaxation.

**Potential saving:** 0.5–1.0 BPW at iso-quality, based on mixed-precision quantization literature.

### 14.6 Skip Quantizing Non-Critical Layers

**Current:** All `nn.Linear` layers are quantized; embeddings and LN kept at full precision.

**Idea:** Identify and skip layers where quantization introduces disproportionate error relative to their parameter count:
- **Embedding layer** (already skippable via `--skip-embeddings`): typically 10–30% of total params
- **lm_head** (already skippable via `--skip-lm-head`): often tied to embeddings
- **Gate projections** in MoE models: small per-expert, high sensitivity

For the non-quantized portion, consider fp16 instead of fp32 to halve their BPW contribution.

### 14.7 Entropy Coding the Indices

**Current:** Each $b$-bit index takes exactly $b$ bits — no compression of the index stream.

**Idea:** The quantized indices are not uniformly distributed (inner levels near 0 are more probable for $\mathcal{N}(0,1)$). The Shannon entropy of the index distribution is:

$$H = -\sum_{\ell=0}^{L-1} p_\ell \log_2 p_\ell$$

For 4-bit Lloyd-Max on $\mathcal{N}(0,1)$:

| Level | Probability | Contribution to $H$ |
|-------|:-:|:-:|
| 7, 8 (near 0) | 0.102 each | 0.669 |
| 0, 15 (tails) | 0.013 each | 0.167 |
| Others | varies | ... |
| **Total $H$** | | **~3.76 bits** |

The entropy is $\sim 3.76$ bits vs the 4 bits allocated — a potential saving of **0.24 BPW**.

**Implementation options:**
- Asymmetric Numeral Systems (ANS): near-optimal compression with fast decode
- Huffman coding: simpler, slightly less efficient
- Range coding: good for streaming decode

**Risk:** Adds decoding latency to the inference critical path. Only beneficial if index loading (not compute) is the bottleneck.

### 14.8 Codebook Sharing Across Bit-Widths

**Current:** Primary and residual passes may use different codebooks (e.g., 4-bit + 2-bit).

**Idea:** If both passes use the same bit-width, they share the codebook — saving $L \cdot 32$ bits. When bit-widths differ, a hierarchical codebook structure (fine codebook as a subset of coarse) could allow partial sharing.

**Saving:** Negligible per-weight ($< 0.001$ BPW), but simplifies the format.

### 14.9 Norm-Free Quantization

**Current:** Norms are stored explicitly per row per group.

**Idea:** Absorb the norm into the codebook by using a learnable or per-group scaled codebook:

$$c_\ell^{(g)} = \sigma_g \cdot c_\ell$$

where $\sigma_g$ is a single per-group scale factor (1 float per group instead of $M$ floats per group).

**Saving:** Reduces norm overhead from $M \cdot 32 / d$ to $32 / d$ bits (per group, amortized across $M$ rows).

**Risk:** Assumes all rows in a group have similar norms — violated when weight row magnitudes vary widely. A hybrid approach could use per-group scale + low-bit per-row residual scale.

### 14.10 Summary: BPW Reduction Roadmap

| Idea | Estimated saving | Difficulty | Quality risk |
|------|:-:|:-:|:-:|
| Norm precision fp16 | 0.25 BPW | Low | Low |
| Larger group size ($d=256$) | 0.13 BPW per pass | Low | Low |
| Sub-4-bit primary (3-bit) | 1.0 BPW | Medium | Medium |
| Non-uniform bit allocation | 0.5–1.0 BPW | Medium | Low |
| Entropy coding | 0.24 BPW | Medium | None |
| Delta norms | 0.19 BPW | Medium | Low |
| Norm-free (per-group scale) | variable | High | Medium |
| Skip non-critical layers at fp16 | model-dependent | Low | Low |

**Combined potential:** With fp16 norms, $d = 256$, 3+2 config, and entropy coding, the effective BPW could reach $\sim 4.5$ at quality comparable to current 4+2 (7 BPW).
