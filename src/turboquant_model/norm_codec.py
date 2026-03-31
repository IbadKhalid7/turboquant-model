"""Norm tensor factorization and compression.

Decomposes the norm tensor α_{m,g} ∈ R^{M×G} via rank-1 SVD:
    α_{m,g} = β_m · γ_g · (1 + ε_{m,g})

where β_m is the row scale, γ_g is the group scale,
and ε_{m,g} is a small fractional residual quantized to int8.

Storage: M·16 + G·16 + M·G·8 + 32 bits
vs full: M·G·32 bits
Saving (d=128): 0.1875 BPW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FactoredNorms:
    """Factored norm representation."""
    row_scale: torch.Tensor       # (M,) float16
    group_scale: torch.Tensor     # (G,) float16
    residual_int8: torch.Tensor   # (M, G) int8
    residual_scale: float         # scalar for dequantizing residual


def factorize_norms(norms: torch.Tensor) -> FactoredNorms:
    """Factorize (M, G) norm tensor into rank-1 + int8 residual.

    Args:
        norms: (M, G) or (M,) norm tensor

    Returns:
        FactoredNorms with row/group scales and int8 residual
    """
    if norms.dim() == 1:
        return FactoredNorms(
            row_scale=norms.half(),
            group_scale=torch.ones(1, dtype=torch.float16, device=norms.device),
            residual_int8=torch.zeros(norms.shape[0], 1, dtype=torch.int8, device=norms.device),
            residual_scale=1.0,
        )

    M, G = norms.shape
    norms_f = norms.float()

    # SVD for rank-1 approximation
    U, S, Vh = torch.linalg.svd(norms_f, full_matrices=False)
    row_scale = U[:, 0] * S[0]  # (M,)
    group_scale = Vh[0, :]       # (G,)

    # Ensure positive (norms are always positive, but SVD can flip signs)
    if (row_scale < 0).sum() > M // 2:
        row_scale = -row_scale
        group_scale = -group_scale

    # Handle any remaining negative signs by flipping per-element
    row_sign = row_scale.sign().clamp(min=0) * 2 - 1  # default to +1 for zeros
    row_scale = row_scale.abs().clamp(min=1e-8)
    group_scale = group_scale * row_sign[0]  # correct for first-row sign
    # If group_scale has negatives, fallback to mean-based factorization
    if (group_scale <= 0).any():
        row_scale = norms_f.mean(dim=1).clamp(min=1e-8)
        group_scale = norms_f.mean(dim=0).clamp(min=1e-8)
        overall_mean = norms_f.mean().clamp(min=1e-8)
        group_scale = group_scale / overall_mean

    # Compute fractional residual
    rank1 = row_scale.unsqueeze(1) * group_scale.unsqueeze(0)  # (M, G)
    residual = norms_f / rank1.clamp(min=1e-8) - 1.0

    # Quantize residual to int8 (symmetric)
    res_amax = residual.abs().amax()
    if res_amax > 0:
        res_scale = res_amax.item() / 127.0
        residual_int8 = (residual / res_scale).round().clamp(-127, 127).to(torch.int8)
    else:
        res_scale = 1.0
        residual_int8 = torch.zeros(M, G, dtype=torch.int8, device=norms.device)

    return FactoredNorms(
        row_scale=row_scale.half(),
        group_scale=group_scale.half(),
        residual_int8=residual_int8,
        residual_scale=res_scale,
    )


def reconstruct_norms(fn: FactoredNorms) -> torch.Tensor:
    """Reconstruct norms from factored representation.

    Returns:
        norms: (M, G) or (M,) float32 tensor
    """
    row_scale = fn.row_scale.float()
    group_scale = fn.group_scale.float()

    if fn.residual_int8.shape[1] == 1 and group_scale.shape[0] == 1:
        return row_scale

    rank1 = row_scale.unsqueeze(1) * group_scale.unsqueeze(0)
    residual = fn.residual_int8.float() * fn.residual_scale
    return rank1 * (1.0 + residual)


def norm_bpw(M: int, N: int, group_size: int, method: str = "fp32") -> float:
    """Compute norm BPW overhead.

    Args:
        M: out_features
        N: in_features
        group_size: columns per group
        method: "fp32", "fp16", "factored_int8"

    Returns:
        BPW overhead (bits per weight element)
    """
    G = (N + group_size - 1) // group_size
    total_elements = M * N

    if method == "fp32":
        norm_bits = M * G * 32
    elif method == "fp16":
        norm_bits = M * G * 16
    elif method == "factored_int8":
        norm_bits = M * 16 + G * 16 + M * G * 8 + 32  # +32 for residual_scale
    else:
        raise ValueError(f"Unknown norm method: {method}")

    return norm_bits / total_elements
