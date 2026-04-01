# Recent Quantization Papers vs. TurboQuant's Rotation + Lloyd-Max

Survey of recent LLM post-training quantization papers, extracting core ideas and evaluating compatibility with TurboQuant's rotation + Lloyd-Max scalar quantization pipeline.

## TurboQuant's Core Pipeline (Reference)

Per group of size $d$:
1. **Normalize** — extract row norms $\alpha_{m,g}$, get unit-norm rows
2. **Rotate** — $Y = \bar{W} \Pi^T$ via random orthogonal or Hadamard matrix
3. **Scale** — $Z = \sqrt{d} \cdot Y$ → entries $\sim \mathcal{N}(0,1)$
4. **Lloyd-Max scalar quantize** — optimal $b$-bit quantizer for $\mathcal{N}(0,1)$
5. **Pack** — 4-bit indices into uint8, optional rANS entropy coding

Key properties:
- Rotation makes entries approximately i.i.d. $\mathcal{N}(0,1)$ — enables optimal scalar quantization
- Lloyd-Max is the minimum-MSE scalar quantizer for known distributions
- rANS entropy coding exploits non-uniform bin probabilities (saves ~0.24 BPW at 4-bit)
- Per-row norms stored separately (fp32 or factored int8)
- Residual passes quantize $R^{(k)} = W - \hat{W}^{(k-1)}$ with same pipeline

---

## 1. QTIP — Quantization with Trellises and Incoherence Processing

**Paper:** Tseng et al., NeurIPS 2024 Spotlight ([arXiv:2406.11235](https://arxiv.org/abs/2406.11235))

### Core Idea

Replace scalar quantization with **trellis-coded quantization (TCQ)** to achieve ultra-high-dimensional vector quantization without exponential codebook cost.

A trellis is a directed graph with $2^L$ nodes, each having $2^k$ incoming/outgoing edges and an associated scalar value. To quantize a length-$T$ sequence:
- Each element is assigned to a node, constrained to form a **walk** on the graph
- The Viterbi algorithm finds the minimum-distortion walk in $O(2^L T)$ time — **linear in sequence length**
- Storage: only $k$ bits per element (the edge choice), plus $L - kV$ bits for initial state

**Key innovations:**
1. **Bitshift trellis** — node $i$ connects to node $j$ if top $L-kV$ bits of $j$ equal bottom $L-kV$ bits of $i$. This means each weight depends only on a **contiguous window of $L$ bits**, enabling **parallel decoding**.
2. **Lookup-free computed codes** — instead of storing a $2^L$-entry codebook, compute pseudo-random approximate Gaussians from the $L$-bit state using ≤4 GPU instructions (LCG + byte-sum for "1MAD", or LCG + XOR + FP16 tricks for "3INST").
3. **Hybrid codes** — compute a hash into a small 2D lookup table ($2^Q \times 2$ entries, fits in L1 cache). The LUT is fine-tunable.
4. **Tail-biting** — approximate tail-biting via 2 Viterbi calls to eliminate initial-state overhead.

**Why it works better than scalar quantization:**
- TCQ approaches the **distortion-rate bound** $D_R$ as $L$ increases, which is unreachable by scalar quantizers
- For $\mathcal{N}(0,1)$ at 2 bits: Lloyd-Max MSE = 0.118, 8D VQ (QuIP#) = 0.089, TCQ ($L$=16) = **0.069**, $D_R$ = 0.063
- The trellis introduces **controlled dependencies** between quantized values, effectively doing high-dimensional VQ with linear cost

**Results:** At 2-bit, QTIP outperforms QuIP# by 0.3–0.5 perplexity on Llama-2. At 4-bit, roughly halves the perplexity gap to fp16. Matches QuIP# inference speed (>80% peak memory BW).

### Compatibility with TurboQuant

| Aspect | Analysis |
|---|---|
| **Incoherence processing** | QTIP uses the **same Hadamard rotation** as TurboQuant to produce i.i.d. Gaussian entries. Fully compatible — TurboQuant's rotation step IS QTIP's incoherence processing. |
| **Replaces what** | TCQ replaces both Lloyd-Max quantization AND rANS entropy coding. The trellis structure itself is the source code — no separate entropy coding step needed. |
| **Encoding (quantization time)** | Viterbi algorithm: $O(2^L \cdot d)$ per row per group. With $L=16$, this is ~65K states × $d$ — feasible but ~1000× slower than `searchsorted`. Acceptable for offline quantization. |
| **Decoding (inference)** | Bitshift trellis + computed codes: ≤4 instructions per weight, parallel across positions. **Comparable speed to current 4-bit unpack.** However, requires custom kernel (LCG + byte-sum or XOR tricks) instead of simple bitwise unpack. |
| **Norm handling** | QTIP doesn't change how norms are handled — still needs per-row/group norms. Fully orthogonal to TurboQuant's norm compression. |
| **Residual quantization** | QTIP quantizes each row as a single long sequence. This naturally extends to residuals — the residual could be another trellis-coded sequence. However, the merge optimization (summing in rotated domain) would be lost since TCQ doesn't produce simple index + codebook form. |

### What TurboQuant Could Adopt

**Option A: TCQ as a storage-only codec (minimal change)**
- Keep Lloyd-Max at quantization time
- Apply TCQ as a lossless/near-lossless compression of the index sequence (replacing rANS)
- Decode to plain 4-bit indices at load time
- Benefit: better compression than rANS (~0.4 BPW savings vs ~0.24)
- Cost: slower model loading, no inference change

**Option B: Full TCQ replacement (major change)**
- Replace Lloyd-Max + rANS with Viterbi encoder + bitshift trellis decoder
- Need new GPU kernels for trellis-based matmul (LCG + lookup tricks)
- Benefit: better distortion at same bit rate, especially at 2-3 bits
- Cost: custom kernels for every backend (CUDA, Metal, Triton), loss of merge optimization

**Option C: Hybrid — TCQ for low-bit, Lloyd-Max for 4-bit**
- TCQ shows biggest advantage at 2-bit (0.118 → 0.069 MSE, 42% reduction)
- At 4-bit, advantage is smaller (0.0095 → ~0.007 MSE, ~26% reduction)
- Keep Lloyd-Max for 4-bit (simpler, fast), use TCQ for 2-bit and 3-bit modes

### Verdict

**High value at 2-3 bits, moderate at 4-bit.** The distortion improvement is real and well-demonstrated. The main barrier is kernel complexity — QTIP's inference kernels are specialized CUDA only. For TurboQuant's multi-backend strategy (cuTile, Triton, Metal), implementing trellis decoding on all backends is significant work. **Option A (storage codec) is the lowest-risk adoption path.**

---

## 2. QuaRot — Outlier-Free 4-Bit Inference in Rotated LLMs

**Paper:** Ashkboos et al., 2024 ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456))

### Core Idea

Use **computational invariance** of RMSNorm to fuse Hadamard rotations into weight matrices permanently, making both weights AND activations outlier-free for joint quantization (W4A4KV4).

Key insight: Since $\text{RMSNorm}(\mathbf{X}\mathbf{Q}^T) \mathbf{Q} = \text{RMSNorm}(\mathbf{X})$ for orthogonal $\mathbf{Q}$, we can:
1. Absorb RMSNorm scaling ($\text{diag}(\alpha)$) into adjacent weight matrices
2. Multiply input-side weights by $\mathbf{Q}^T$ and output-side weights by $\mathbf{Q}$ — cancels out
3. Now the hidden state is permanently rotated: $\mathbf{X} \leftarrow \mathbf{X}\mathbf{Q}$
4. Add small online Hadamard ops for FFN down-projection and attention values/keys

Result: **all activations, weights, and KV cache** quantized to INT4 with minimal accuracy loss. No outlier channels needed.

### Compatibility with TurboQuant

| Aspect | Analysis |
|---|---|
| **Rotation technique** | Same Hadamard/random orthogonal rotation as TurboQuant. QuaRot fuses it into the model graph; TurboQuant applies it per-group at inference. |
| **Scope** | QuaRot targets **activation + KV cache quantization** (joint W4A4KV4). TurboQuant is weight-only. These are complementary — different problems. |
| **Weight quantization** | QuaRot uses GPTQ for weight quantization after rotation. TurboQuant uses Lloyd-Max. The rotation preprocessing is shared. |
| **Relevance** | QuaRot validates that **rotation-based incoherence processing works for activations too**. If TurboQuant ever extends to activation quantization, this is the path. |

### What TurboQuant Could Adopt

**Computational invariance fusion:** Instead of applying rotation online at inference, fuse Hadamard matrices into the stored weights. This would:
- Eliminate the per-group rotation during inference (currently $O(d \log d)$ per group)
- But restrict to a single global rotation per layer (not per-group)
- Would break the current group-level rotation flexibility

**Verdict:** QuaRot solves a different problem (activation quantization). Its **computational invariance** technique is interesting but conflicts with TurboQuant's per-group rotation design. Not directly applicable without architectural changes.

---

## 3. QuIP# — Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

**Paper:** Tseng et al., 2024 ([arXiv:2402.04396](https://arxiv.org/abs/2402.04396))

### Core Idea

Combine **incoherence processing** (random Hadamard transform) with the **$E_8$ lattice codebook** for 8D vector quantization (VQ) at 2 bits.

Key components:
1. **Random Hadamard Transform (RHT):** $\tilde{W} = V_m S_m W S_n V_n^T$ — makes weights approximately i.i.d. Gaussian
2. **$E_8$ lattice codebook:** The densest 8D lattice packing. 256 codewords per 8D vector (1 byte = 8 weights at 1 bit each, but with VQ shaping gain). Highly symmetric — compressible by 256× to fit in L1 cache.
3. **BlockLDLQ:** Hessian-aware adaptive rounding within blocks, minimizing proxy loss $\ell(\hat{W}) = \text{tr}((\hat{W}-W)H(\hat{W}-W)^T)$

**Why lattice VQ beats scalar:**
- $E_8$ lattice is the densest sphere packing in 8D
- 8D VQ has ~24% lower MSE than scalar for Gaussian sources at 2-bit
- The lattice structure enables fast nearest-neighbor search (no brute-force over exponential codebook)

### Compatibility with TurboQuant

| Aspect | Analysis |
|---|---|
| **Rotation step** | Identical concept — RHT produces i.i.d. Gaussian weights. TurboQuant uses the same approach. |
| **Quantization step** | QuIP# uses 8D lattice VQ where TurboQuant uses 1D Lloyd-Max. Lattice VQ gives better distortion but requires 8-element vector grouping. |
| **Hessian-aware rounding** | BlockLDLQ uses the proxy Hessian $H = \mathbb{E}[xx^T]$ for adaptive rounding. TurboQuant doesn't use Hessian information — purely MSE-optimal for $\mathcal{N}(0,1)$. |

### What TurboQuant Could Adopt

**Lattice-based VQ for 2-bit mode:** At 2-bit, the shaping gain of 8D VQ is substantial (~24% MSE reduction). Could replace Lloyd-Max for 2-bit quantization while keeping it for 4-bit where the gap is smaller.

**Hessian-aware calibration (BlockLDLQ):** Even keeping Lloyd-Max scalar quantization, applying Hessian-weighted rounding decisions could improve task-specific quality. The modification: instead of rounding to minimize $\|W - \hat{W}\|_F^2$, minimize $\text{tr}((\hat{W}-W)H(\hat{W}-W)^T)$. This biases rounding decisions for columns that have larger Hessian eigenvalues.

**Verdict:** Hessian-aware rounding (BlockLDLQ) is the most transferable idea. It's orthogonal to the quantizer choice and could improve TurboQuant's quality at any bit width. Lattice VQ is interesting but high-dimensional — similar concerns as QTIP's TCQ for kernel complexity.

---

## 4. SpinQuant — LLM Quantization with Learned Rotations

**Paper:** Liu et al., 2024 ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406))

### Core Idea

Instead of random Hadamard rotations, **learn the rotation matrices** via Cayley optimization to minimize quantization error on calibration data.

Key insight: Random Hadamard is good but not optimal. The rotation that minimizes quantization distortion depends on the actual weight distribution, which isn't perfectly i.i.d. Gaussian.

**Method:**
1. Parameterize rotation as $R = \text{Cayley}(A) = (I - A)(I + A)^{-1}$ where $A$ is skew-symmetric
2. Optimize $A$ with gradient descent on calibration loss
3. Apply optimized rotation in the same positions as QuaRot (computational invariance)

**Result:** Consistently outperforms random Hadamard by 0.1-0.5 perplexity at 4-bit W4A4.

### Compatibility with TurboQuant

| Aspect | Analysis |
|---|---|
| **Core idea** | Directly applicable — TurboQuant could learn rotation matrices instead of using random Haar/Hadamard matrices. |
| **Constraint** | TurboQuant rotates per-group, not per-layer. Learning per-group rotations would be prohibitively expensive (thousands of small matrices). |
| **Benefit** | Marginal at 4-bit (already close to optimal with random rotation). More significant at 2-3 bits. |
| **Fast Walsh-Hadamard** | Learned rotations lose the $O(d \log d)$ fast transform — would need full $O(d^2)$ matmul. At $d=128$, this is 16K vs 896 FLOPs. |

### What TurboQuant Could Adopt

**Per-layer learned rotation (if group_size = hidden_dim):** If using a single rotation per layer (entire hidden dimension), learned rotation could replace random Hadamard. But TurboQuant's group-level design makes this impractical.

**Cayley-parameterized per-group rotations:** Learn a small number of rotation templates and select per-group. But calibration cost would be high and benefit is marginal at 4-bit.

**Verdict:** Interesting theoretically but impractical for TurboQuant's per-group rotation design. The fast Hadamard transform is too valuable to give up for marginal quality gains. **Not recommended.**

---

## 5. AQLM — Extreme Compression via Additive Quantization

**Paper:** Egiazarian et al., 2024 ([arXiv:2401.06118](https://arxiv.org/abs/2401.06118))

### Core Idea

Use **additive (multi-codebook) vector quantization**: each weight vector is the sum of codewords from multiple independent codebooks.

For a $d$-dimensional weight vector $w$:
$$\hat{w} = \sum_{m=1}^{M} C_m[i_m]$$
where $C_m \in \mathbb{R}^{K \times d}$ are learned codebooks and $i_m$ are indices.

Cost: $M \cdot \lceil \log_2 K \rceil$ bits per $d$-dimensional vector.

**Beam search encoding** with Hessian-weighted objective, plus end-to-end codebook fine-tuning.

### Compatibility with TurboQuant

| Aspect | Analysis |
|---|---|
| **Two-codebook additive structure** | TurboQuant's residual quantization IS additive quantization with $M=2$: $\hat{W} = \hat{R}^{(0)} + \hat{R}^{(1)}$, each pass contributing a reconstruction. |
| **Key difference** | AQLM jointly optimizes all codebooks end-to-end. TurboQuant quantizes greedily (pass 1, then residual). Joint optimization could yield better results. |
| **Codebook size** | AQLM's codebooks are per-layer and large (1MiB), don't fit in L1 cache → slow inference. TurboQuant's Lloyd-Max codebook is 16 entries × fp32 = 64 bytes — tiny. |

### What TurboQuant Could Adopt

**Joint optimization of residual passes:** Instead of greedy residual quantization, jointly optimize pass-1 and pass-2 indices/norms to minimize total reconstruction error. This is the AQLM insight applied to TurboQuant's existing structure.

**Verdict:** The joint optimization idea is valuable, but AQLM's actual VQ approach is too slow for inference. TurboQuant's residual quantization is already a form of additive quantization with much faster inference.

---

## 6. Bonsai — End-to-End 1-Bit Language Models

**Project:** Prism ML, March 2026 ([HuggingFace](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit), [Whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf))

### Core Idea

Produce a **commercially viable 1-bit weight** LLM (Bonsai-8B) by training/distilling into a format where every weight is a single sign bit: `0 → −scale`, `1 → +scale`, with one fp16 scale per 128-weight group. This gives **1.125 BPW** (GGUF Q1_0_g128) or 1.25 BPW (MLX format with scale+bias).

Key properties:
- **1-bit coverage:** embeddings, attention projections, MLP projections, and LM head — all quantized to 1-bit
- **Architecture:** Qwen3-8B base (8.19B params, 36 layers, GQA 32/8, SwiGLU, RoPE, RMSNorm)
- **Inference:** custom fused 1-bit dequant kernels for MLX (Apple Silicon), llama.cpp (Metal + CUDA), and mlx-swift (iOS). No fp16 weight materialization at runtime — bitwise ops + scale multiplication only
- **Size:** 1.28 GB MLX / 1.15 GB GGUF (down from 16.38 GB fp16 = **12.8–14.2× compression**)

**Results:**
| Model | Size | Avg Score (6 benchmarks) |
|---|---|---|
| Qwen 3 8B (fp16) | 16 GB | 79.3 |
| **1-bit Bonsai 8B** | **1.15 GB** | **70.5** |
| Llama 3.1 8B (fp16) | 16 GB | 67.1 |

Throughput: 131 tok/s on M4 Pro (8.4× faster than fp16), 44 tok/s on iPhone 17 Pro Max. Energy efficiency: 5.6× better energy-per-token than fp16.

### Method (Inferred from Public Info)

The whitepaper details are not fully public, but the approach appears to be:
1. **Quantization-aware training (QAT) or knowledge distillation** from a full-precision Qwen3-8B teacher — this is NOT post-training quantization
2. **1-bit sign quantization per group:** $w_i \in \{-s_g, +s_g\}$ where $s_g$ is a learned fp16 scale per group of 128
3. **Custom kernels** that exploit the binary structure: each weight is a single bit, so the "dequantize + matmul" reduces to sign-flip + accumulate, which is extremely fast and parallelizable

The "intelligence density" metric $\alpha = -\ln(1 - \text{score}/100) / \text{size\_GB}$ gives Bonsai 10.8× higher density than fp16 Qwen3-8B.

### Compatibility with TurboQuant

| Aspect | Analysis |
|---|---|
| **Quantization approach** | Fundamentally different — Bonsai uses **QAT/distillation** (training the model TO be 1-bit), while TurboQuant is **post-training quantization** (compressing an existing model). |
| **Rotation** | No random rotation needed. At 1-bit (sign quantization), rotation cannot help — there's zero freedom to optimize within a 2-level quantizer per dimension. The distribution of the rotated weight doesn't matter when you only have 2 levels. |
| **Lloyd-Max** | Lloyd-Max for a symmetric distribution at 1-bit is trivially $\{-\mu, +\mu\}$ where $\mu = \mathbb{E}[|x|]$. For $\mathcal{N}(0,1)$: $\mu = \sqrt{2/\pi} \approx 0.798$. MSE = $1 - 2/\pi \approx 0.363$ per weight. This is extremely lossy — 36.3% relative MSE. That's why PTQ at 1-bit doesn't work; Bonsai needs QAT. |
| **Residual quantization** | Not applicable — Bonsai trains the model to produce 1-bit weights directly, no residuals needed. |
| **Entropy coding** | At 1-bit, the maximum entropy is 1 bit per weight. If the sign distribution is balanced (50/50), rANS would save nothing. If biased, there's some room, but likely minimal since the model is trained to use both signs. |

### What TurboQuant Could Adopt

**Nothing directly.** Bonsai operates in a completely different paradigm:
- TurboQuant: take a trained fp16 model → compress it losslessly/near-losslessly
- Bonsai: train a model from scratch (or distill) to work with 1-bit weights

The key insight from Bonsai is about the **limits of PTQ**: at 1-bit, no PTQ method can preserve quality — the MSE of optimal scalar quantization ($1 - 2/\pi$) is too high. QAT is required to let the model adapt its weight structure to extreme quantization.

**Indirect takeaway:** Bonsai demonstrates that 1-bit weights CAN work if the model is trained for it. This validates the theoretical floor — if TurboQuant needs to go below ~2 bits, QAT or fine-tuning becomes necessary regardless of how good the quantizer is.

### Verdict

**Not applicable to TurboQuant's PTQ pipeline.** Bonsai and TurboQuant solve the same deployment problem (efficient inference) through fundamentally different means. Bonsai is relevant as a competitive benchmark: at 1.15 GB, it achieves 70.5 avg score vs fp16's 79.3 — an 11% quality drop for 14× size reduction. TurboQuant's 4+4bit residual achieves near-lossless quality at ~8.5 BPW (~2× compression). Different points on the Pareto frontier.

---

## Summary: Actionable Ideas Ranked by Impact

| Rank | Idea | Source | Impact | Effort | Bit Regime |
|---|---|---|---|---|---|
| 1 | **Trellis-coded storage compression** | QTIP | Better compression than rANS (~0.4 vs ~0.24 BPW savings) | Medium | 2-4 bit |
| 2 | **Hessian-aware adaptive rounding** | QuIP#/AQLM | Better task-specific quality by considering which weights matter more | Medium | All |
| 3 | **TCQ for 2-bit quantizer** | QTIP | 42% lower MSE than Lloyd-Max at 2-bit. Huge quality uplift. | High | 2-bit |
| 4 | **Joint residual optimization** | AQLM | Better residual capture by jointly optimizing both passes | Medium | Multi-pass |
| 5 | **Activation quantization via computational invariance** | QuaRot | Enable W4A4KV4 deployment | High | 4-bit |
| 6 | **Lattice VQ for 2-bit** | QuIP# | ~24% lower MSE at 2-bit via $E_8$ lattice | High | 2-bit |
| 7 | **Learned rotations** | SpinQuant | Marginal quality gain, loses fast Hadamard | Low-Med | All |

### Recommended Priority

1. **Item 2 (Hessian-aware rounding)** — broadest impact, works with existing pipeline, moderate implementation effort. Add an optional GPTQ/LDLQ-style calibration step after the current rotation + Lloyd-Max quantization.

2. **Item 1 (Trellis storage codec)** — drop-in replacement for rANS in `entropy_codec.py`. Better compression, no inference change. Pure storage-side optimization.

3. **Item 3 (TCQ for 2-bit)** — only if 2-bit mode is a priority. The quality difference is transformative at 2-bit but requires per-backend kernel work.

4. **Item 4 (Joint residual)** — improves multi-pass mode without changing the quantizer. Could be implemented as an optimization pass after the current greedy residual quantization.

---

## Theoretical Analysis: Why TCQ > Scalar for Gaussian Sources

For an i.i.d. $\mathcal{N}(0,1)$ source at rate $R$ bits/sample, the distortion-rate function is:
$$D(R) = 2^{-2R}$$

| Rate $R$ | $D(R)$ (bound) | Lloyd-Max $D_b$ | TCQ $D$ ($L$=16) | Gap to bound |
|---|---|---|---|---|
| 1 bit | 0.250 | 0.363 | ~0.28 | ~12% |
| 2 bit | 0.063 | 0.118 | 0.069 | ~10% |
| 3 bit | 0.016 | 0.035 | ~0.019 | ~19% |
| 4 bit | 0.004 | 0.0095 | ~0.005 | ~25% |

TCQ's advantage comes from the **shaping gain**: the trellis constrains quantized sequences to lie on a "good" lattice-like structure, achieving better sphere-packing density than independent scalar quantization. As $L \to \infty$, TCQ approaches $D(R)$ for i.i.d. sources.

At 4-bit, Lloyd-Max distortion is 0.0095 and the bound is 0.004 — there's room for a ~2.4× improvement. TCQ captures roughly half this gap. At 2-bit, the gap is much larger (0.118 vs 0.063 = 1.87×), and TCQ captures most of it.

**Implication for TurboQuant:** The current pipeline is near-optimal for scalar quantization. To push further, the quantizer must become multi-dimensional — whether via trellis (QTIP), lattice (QuIP#), or additive VQ (AQLM). The rotation step remains unchanged in all approaches.
