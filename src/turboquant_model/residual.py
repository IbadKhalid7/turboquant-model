"""Residual (multi-pass) TurboQuant quantization.

Residual quantization applies TurboQuant multiple times:
  Pass 1: Quantize W at bit_width b1 → W_hat1, residual R1 = W - W_hat1
  Pass 2: Quantize R1 at bit_width b2 → R_hat1
  Final:  W_approx = W_hat1 + R_hat1  (total bits = b1 + b2)

This achieves better quality than single-pass at the same total bit budget
because the residual has different structure from the original weight.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from turboquant_model.codebook import get_codebook
from turboquant_model.rotation import generate_rotation_matrix
from turboquant_model.quantize import (
    turboquant_quantize,
    turboquant_quantize_packed,
    pack_4bit,
    unpack_4bit,
)


@torch.no_grad()
def residual_quantize(
    W: torch.Tensor,
    bit_width_1: int = 4,
    bit_width_2: int = 4,
    group_size: Optional[int] = None,
    seed_1: int = 42,
    seed_2: int = 1042,
) -> torch.Tensor:
    """Two-pass residual TurboQuant: returns the dequantized approximation.

    Args:
        W: (M, N) weight matrix
        bit_width_1: bits for first pass
        bit_width_2: bits for second pass (residual)
        group_size: group size (None = full row)
        seed_1: rotation seed for pass 1
        seed_2: rotation seed for pass 2

    Returns:
        W_approx: same shape/dtype as W
    """
    # Pass 1
    W_hat1 = turboquant_quantize(W, bit_width=bit_width_1, group_size=group_size, seed=seed_1)

    # Residual
    residual = W.float() - W_hat1.float()

    # Pass 2
    R_hat = turboquant_quantize(residual, bit_width=bit_width_2, group_size=group_size, seed=seed_2)

    return (W_hat1.float() + R_hat.float()).to(W.dtype)


@torch.no_grad()
def residual_quantize_packed(
    W: torch.Tensor,
    bit_width_1: int = 4,
    bit_width_2: int = 4,
    group_size: Optional[int] = None,
    seed_1: int = 42,
    seed_2: int = 1042,
) -> dict:
    """Two-pass residual TurboQuant: returns packed representations for both passes.

    Args:
        W: (M, N) weight matrix
        bit_width_1: bits for first pass
        bit_width_2: bits for second pass
        group_size: group size (None = full row)
        seed_1: rotation seed for pass 1
        seed_2: rotation seed for pass 2

    Returns:
        dict with:
            pass1: dict (same format as turboquant_quantize_packed output)
            pass2: dict (same format, for the residual)
            total_bits: bit_width_1 + bit_width_2
    """
    M, N = W.shape
    if group_size is None:
        group_size = N

    # Pass 1: quantize and pack
    pass1 = turboquant_quantize_packed(W, bit_width=bit_width_1, group_size=group_size, seed=seed_1)

    # Reconstruct pass 1 to compute residual
    W_hat1 = _dequantize_from_packed(pass1, device=W.device)
    residual = W.float() - W_hat1

    # Pass 2: quantize residual
    pass2 = turboquant_quantize_packed(residual, bit_width=bit_width_2, group_size=group_size, seed=seed_2)

    return {
        "pass1": pass1,
        "pass2": pass2,
        "total_bits": bit_width_1 + bit_width_2,
    }


def _dequantize_from_packed(packed_data: dict, device: torch.device) -> torch.Tensor:
    """Reconstruct weight from packed representation.

    Args:
        packed_data: output from turboquant_quantize_packed
        device: target device

    Returns:
        W_approx: (M, N) float32 tensor
    """
    M, N = packed_data["shape"]
    group_size = packed_data["group_size"]
    seed = packed_data["seed"]

    indices_packed = packed_data["indices_packed"].to(device)
    codebook = packed_data["codebook"].to(device)
    norms = packed_data["norms"].to(device)

    indices = unpack_4bit(indices_packed, N if N % 2 == 0 else N + 1)
    indices = indices[:, :N]  # trim padding

    n_groups = math.ceil(N / group_size)
    W_approx = torch.zeros(M, N, dtype=torch.float32, device=device)

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
        scale = math.sqrt(g_dim)

        Y_g = codebook[indices[:, g_start:g_end].long()] / scale
        W_g = Y_g @ Pi

        if norms.dim() == 1:
            W_g = W_g * norms.unsqueeze(1)
        else:
            W_g = W_g * norms[:, g].unsqueeze(1)

        W_approx[:, g_start:g_end] = W_g

    return W_approx
