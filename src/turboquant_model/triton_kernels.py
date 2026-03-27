"""Triton fused dequant + matmul kernels for on-the-fly inference.

These kernels avoid materializing the full dequantized weight by fusing
4-bit unpack → codebook lookup → matmul → norm rescale in one kernel launch.

Main kernel: _turboquant_fused_matmul_kernel
  - Input: x_rot (pre-rotated activations), packed indices, codebook, norms
  - Output: x_rot @ codebook[indices].T * (norms / scale)

Supports group-wise calls: pass a packed index slice (N, g_dim//2) with K=g_dim.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _turboquant_fused_matmul_kernel(
    # Input
    input_ptr,        # (B, K) pre-rotated activations
    # Quantized weight
    indices_ptr,      # (N, K//2) packed uint8
    codebook_ptr,     # (n_levels,) float32
    norms_ptr,        # (N,) float32
    # Output
    output_ptr,       # (B, N)
    # Dims
    B, N, K,
    PACKED_K,         # K // 2 (stride for packed index rows)
    SCALE: tl.constexpr,
    N_LEVELS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused dequant-matmul: output[b,n] = norm[n]/scale * Σ_k x_rot[b,k] * codebook[indices[n,k]]"""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K

        # Load input tile: (BLOCK_B, BLOCK_K)
        inp_off = rb[:, None] * K + rk[None, :]
        inp_mask = mask_b[:, None] & mask_k[None, :]
        inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

        # Load + unpack weight indices: (BLOCK_N, BLOCK_K)
        byte_col = rk // 2
        is_high = (rk % 2) == 1
        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        idx = tl.where(is_high[None, :], hi, lo)

        # Codebook lookup
        w_quant = tl.load(codebook_ptr + idx.to(tl.int32), mask=w_mask, other=0.0)

        # Accumulate: (BLOCK_B, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc += tl.dot(inp_tile.to(tl.float32), tl.trans(w_quant.to(tl.float32)))

    # Scale by norms / scale
    norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
    acc = acc * (norm_vals[None, :] / SCALE)

    # Store
    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_fused_matmul(
    x_rot: torch.Tensor,           # (B, K) pre-rotated input
    indices_packed: torch.Tensor,   # (N, K//2) packed uint8
    codebook: torch.Tensor,         # (n_levels,) float32
    norms: torch.Tensor,            # (N,) float32
    K: int,                         # in_features (or group_size for per-group calls)
    scale: float | None = None,     # override sqrt(K) if needed
) -> torch.Tensor:
    """Fused dequant + matmul via Triton.

    Expects pre-rotated input: x_rot = x @ Pi.T

    Supports per-group calls: pass a slice of packed indices (N, g_dim//2)
    with K=g_dim. The kernel handles unpack + codebook lookup + matmul + norm
    rescale in one launch, avoiding materialization of the (N, K) float weight.

    Args:
        x_rot: (B, K) pre-rotated activations
        indices_packed: (N, K//2) packed 4-bit weight indices
        codebook: centroids
        norms: per-row weight norms (N,)
        K: dimension of this group (in_features or group_size)
        scale: norm divisor (default: sqrt(K))

    Returns:
        output: (B, N)
    """
    B = x_rot.shape[0]
    N = indices_packed.shape[0]
    PACKED_K = indices_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    # Block sizes — must be powers of 2 for tl.dot
    BLOCK_B = 1
    for p in [1, 2, 4, 8, 16, 32]:
        if p <= B:
            BLOCK_B = p
    BLOCK_N = min(64, N)
    # Round BLOCK_N up to power of 2
    BLOCK_N = 1
    for p in [1, 2, 4, 8, 16, 32, 64]:
        if p <= N:
            BLOCK_N = p
    BLOCK_K = 1
    for p in [1, 2, 4, 8, 16, 32, 64, 128]:
        if p <= K:
            BLOCK_K = p

    grid = (
        (B + BLOCK_B - 1) // BLOCK_B,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    _turboquant_fused_matmul_kernel[grid](
        x_rot, indices_packed, codebook, norms, output,
        B, N, K, PACKED_K,
        SCALE=scale,
        N_LEVELS=codebook.shape[0],
        BLOCK_B=BLOCK_B,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output
