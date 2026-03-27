"""Core TurboQuant quantization: single-pass rotation + Lloyd-Max scalar quantization.

Provides both simulation (returns fp32/bf16 approximation) and packed storage
(returns 4-bit packed indices + norms + codebook).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from turboquant_model.codebook import get_codebook
from turboquant_model.rotation import generate_rotation_matrix


# ---------------------------------------------------------------------------
# 4-bit packing / unpacking
# ---------------------------------------------------------------------------


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices (0-15) into uint8, 2 per byte.

    Layout: byte = lo_nibble | (hi_nibble << 4)
    where lo = indices[..., 2i], hi = indices[..., 2i+1]

    Args:
        indices: int tensor (..., N) with values in [0, 15], N must be even

    Returns:
        packed: uint8 tensor (..., N//2)
    """
    assert indices.shape[-1] % 2 == 0, "Last dim must be even for 4-bit packing"
    lo = indices[..., 0::2].to(torch.uint8)
    hi = indices[..., 1::2].to(torch.uint8)
    return lo | (hi << 4)


def unpack_4bit(packed: torch.Tensor, N: int) -> torch.Tensor:
    """Unpack uint8 -> 4-bit indices.

    Args:
        packed: uint8 tensor (..., N//2)
        N: original last dimension

    Returns:
        indices: int32 tensor (..., N)
    """
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    result = torch.stack([lo, hi], dim=-1)
    return result.reshape(*packed.shape[:-1], N)


# ---------------------------------------------------------------------------
# Single-pass quantization (simulation)
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_quantize(
    W: torch.Tensor,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed: int = 42,
) -> torch.Tensor:
    """Apply TurboQuant quantization and return the dequantized approximation.

    Steps:
      1. Row-normalize
      2. Rotate by random orthogonal Pi
      3. Scalar quantize with Lloyd-Max codebook
      4. Dequantize (centroids), inverse-rotate, rescale

    Args:
        W: (out_features, in_features) weight matrix
        bit_width: bits per coordinate
        group_size: group size along in_features (None = full row)
        seed: rotation seed

    Returns:
        W_approx: same shape/dtype as W
    """
    orig_dtype = W.dtype
    W = W.float()
    out_features, in_features = W.shape

    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    if group_size is None:
        group_size = in_features

    W_approx = torch.zeros_like(W)

    for g_start in range(0, in_features, group_size):
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
        Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1)
        Y_quant = centroids[indices].reshape(Y_scaled.shape)

        Y_unscaled = Y_quant / scale
        W_g_approx = Y_unscaled @ Pi
        W_approx[:, g_start:g_end] = W_g_approx * norms

    return W_approx.to(orig_dtype)


# ---------------------------------------------------------------------------
# Single-pass quantization (packed storage)
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_quantize_packed(
    W: torch.Tensor,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Quantize and return packed representation for storage/inference.

    Args:
        W: (M, N) weight matrix
        bit_width: bits per element (currently 4-bit only for packing)
        group_size: group size (None = full row)
        seed: rotation seed

    Returns:
        dict with:
            indices_packed: (M, N//2) uint8
            codebook: (2^b,) float32
            norms: (M,) or (M, n_groups) float32
            seed: int
            group_size: int
            shape: (M, N)
            bit_width: int
    """
    assert bit_width == 4, "Packed format supports 4-bit only"

    M, N = W.shape
    if group_size is None:
        group_size = N

    W = W.float()
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    all_norms = []
    all_indices = []

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
        Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

    # Pad to even for packing
    if N % 2 != 0:
        full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

    packed = pack_4bit(full_indices)

    return {
        "indices_packed": packed,
        "codebook": centroids.cpu(),
        "norms": norms_out.cpu(),
        "seed": seed,
        "group_size": group_size,
        "shape": (M, N),
        "bit_width": bit_width,
    }
