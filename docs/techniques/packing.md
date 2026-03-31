# 4-bit Index Packing

This document describes the bit-packing scheme that stores two 4-bit quantization indices per uint8 byte, halving the storage for the index tensor.

---

## Formulation Context: Index Storage

In the [quantization formulation](../formulation.md), the **dominant** storage term is the index tensor $\boldsymbol{\ell}_{m,k}$ at $b$ bits per weight. At 4 bits, each index naturally fits in a nibble (half-byte). Packing two indices per byte converts an $(M, N)$ index matrix into $(M, N/2)$ uint8 — an immediate 2× reduction in index storage.

---

## Byte Layout

Two 4-bit indices are packed into each uint8 byte:

```
Byte layout:  [hi_nibble (bits 7-4)] [lo_nibble (bits 3-0)]
              |--- index[k+1] -----|--- index[k] --------|
```

### Pack

$$\text{packed}[m, k/2] = \text{indices}[m, k] \;|\; (\text{indices}[m, k+1] \ll 4)$$

The low nibble (bits 3-0) holds the even-indexed value; the high nibble (bits 7-4) holds the odd-indexed value.

### Unpack

- **Low nibble:** $\text{lo} = \text{packed} \;\&\; \texttt{0x0F} \;\to\; \text{indices}[m, 2j]$
- **High nibble:** $\text{hi} = (\text{packed} \gg 4) \;\&\; \texttt{0x0F} \;\to\; \text{indices}[m, 2j+1]$

Both operations are single-cycle bitwise instructions on any modern CPU or GPU.

---

## Odd-Length Handling

When $N$ is odd, the last column is zero-padded before packing. The original $N$ is stored in metadata (`turboquant_config.json`) so that unpacking correctly discards the padding.

---

## Storage Savings

| Representation | Shape | Bytes per weight |
|----------------|-------|-----------------|
| int64 (unpacked) | $(M, N)$ | 8.0 |
| uint8 (unpacked) | $(M, N)$ | 1.0 |
| **uint8 (packed)** | $(M, N/2)$ | **0.5** |

For a typical 4096 × 4096 layer: unpacked uint8 = 16 MB; packed = 8 MB.

---

## Interaction with Fused Kernels

The fused GPU kernels (CuTile and Triton) operate directly on the packed uint8 representation. During a tile computation, the unpack step is performed in registers via bitwise ops — the unpacked index tensor is never materialized in global memory:

```
bytes = load(packed_ptr)         # uint8 from global memory
lo = bytes & 0x0F                # even indices
hi = (bytes >> 4) & 0x0F         # odd indices
idx = interleave(lo, hi)         # reconstruct index order
w = codebook[idx]                # lookup (codebook in shared memory)
```

This avoids the 2× memory footprint that explicit unpacking would require.

---

## Implementation

| File | Function | Description |
|------|----------|-------------|
| `quantize.py` | `pack_4bit()` | Pack pairs of 4-bit indices into uint8 |
| `quantize.py` | `unpack_4bit()` | Unpack uint8 back to individual indices |

---

## Relationship to Other Techniques

- **Scalar Quantization (Lloyd-Max)**: Packing is applied to the index tensor produced by Lloyd-Max quantization. The 4-bit index range [0, 15] fits exactly in a nibble.
- **Entropy Coding**: Packing achieves exactly $b$ bits per index (4 bits). [Entropy coding](entropy-codec.md) goes further by exploiting the non-uniform distribution of indices to compress below $b$ bits.
- **Fused GPU Kernels**: The [fused kernels](fused-kernels.md) consume packed uint8 directly, performing unpack + lookup + matmul in a single kernel without intermediate buffers.
- **Residual Quantization**: Each residual pass's indices are packed independently.
