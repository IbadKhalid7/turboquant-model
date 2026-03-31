# Entropy Coding (rANS)

This document describes the entropy coding module that compresses quantized indices below their nominal bit-width by exploiting the non-uniform Gaussian bin probability distribution.

---

## Formulation Context: Index Storage

In the [quantization formulation](quantization-formulation.md), the **dominant** term in the BPW budget is the **index tensor** $\boldsymbol{\ell}_{m,k}$, stored at $b$ bits per weight:

$$\text{BPW} \approx \underbrace{b}_{\text{indices}} + \frac{32}{d} + \text{BPW}_{\text{non-quant}}$$

The entropy codec targets the **index term** $b$. Because Lloyd-Max quantization of $\mathcal{N}(0,1)$ produces non-uniform bin probabilities (inner levels are more probable than outer), the Shannon entropy $H$ is strictly less than $b$ bits:

| $b$ | $L = 2^b$ | $H$ (bits/symbol) | Saving ($b - H$) |
|-----|-----------|-------------------|-------------------|
| 2   | 4         | 1.911             | 0.089             |
| 3   | 8         | 2.832             | 0.168             |
| 4   | 16        | 3.764             | 0.236             |
| 5   | 32        | 4.755             | 0.245             |

At 4 bits, entropy coding can save **~0.24 BPW** — bringing the index cost from 4.0 to ~3.76 bits per weight.

---

## How It Works: rANS (Asymmetric Numeral Systems)

### Why rANS?

- **GPU-friendly**: Block-based encoding means each block of $B$ symbols can be decoded independently — perfect for GPU parallel decoding.
- **Near-entropy-optimal**: Achieves compression within fractions of a bit of the Shannon limit.
- **Tiny decode tables**: At 4-bit, the frequency + cumulative tables fit in ~128 bytes of GPU shared memory.

### Encoding

Symbols (quantization indices) are split into blocks of $B = 4096$ symbols. Within each block, rANS processes symbols in **reverse order** (the ANS property), maintaining a state integer that is periodically renormalized by emitting bytes.

For symbol $s$ with frequency $f_s$ and cumulative frequency $c_s$:

$$\text{state}' = \left\lfloor \frac{\text{state}}{f_s} \right\rfloor \cdot 2^{P} + (\text{state} \bmod f_s) + c_s$$

where $P = 14$ is the probability precision (frequencies quantized to sum to $2^{14}$).

### Decoding (GPU-parallel)

Each block starts from a known 4-byte state. The decode loop per symbol is:

1. **Slot lookup**: $\text{slot} = \text{state} \;\&\; (2^P - 1)$
2. **Symbol find**: table lookup $\text{slot} \to s$ (O(1) with precomputed LUT)
3. **State update**: $\text{state} = f_s \cdot (\text{state} \gg P) + \text{slot} - c_s$
4. **Renormalize**: read bytes while $\text{state} < 2^{16}$

This is 1 table lookup + 1 multiply + 1 shift + renormalize per symbol — well-suited for GPU shared memory execution.

### Frequency Tables

The `ANSTable` stores:

| Field | Size | Description |
|-------|------|-------------|
| `freqs` | $L \times 2$ bytes | Quantized frequencies (uint16) |
| `cumuls` | $(L+1) \times 4$ bytes | Cumulative frequencies (uint32) |
| **Total** | ~128 bytes (4-bit) | Fits in GPU shared memory |

Frequencies are derived from the known Gaussian bin probabilities of the Lloyd-Max codebook — no training data needed.

---

## Compression Ratios

For a weight matrix with $M \times N$ elements at $b$-bit quantization:

- **Uncompressed index storage**: $M \cdot N \cdot b$ bits (or $M \cdot N \cdot b/8$ bytes with packing)
- **Entropy-coded storage**: $\approx M \cdot N \cdot H$ bits
- **Compression ratio**: $b / H$

At 4 bits: $4 / 3.764 \approx 1.063\times$ compression on the index tensor alone.

The saving is modest per-element but significant at scale: for a 7B parameter model at 4-bit, entropy coding saves ~200 MB.

---

## Implementation

| File | Function | Description |
|------|----------|-------------|
| `entropy_codec.py` | `gaussian_bin_probs()` | Compute bin probabilities from Lloyd-Max boundaries |
| `entropy_codec.py` | `compute_entropy()` | Theoretical entropy for a given bit-width |
| `entropy_codec.py` | `build_ans_table()` | Build rANS frequency/cumulative tables |
| `entropy_codec.py` | `rANSCodec.encode()` | Block-based rANS encoding |
| `entropy_codec.py` | `rANSCodec.decode()` | Block-based rANS decoding |

---

## Relationship to Other Techniques

- **Lloyd-Max Quantization**: Entropy coding exploits the non-uniform bin probabilities that arise from optimal Gaussian quantization. Uniform quantizers would have $H = b$ (no saving).
- **4-bit Packing**: Packing reduces storage by fitting two indices per byte; entropy coding goes further by exploiting statistical redundancy.
- **Residual Quantization**: Each residual pass produces its own index tensor — entropy coding applies independently to each pass's indices.
