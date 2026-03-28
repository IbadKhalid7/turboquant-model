"""Sensitivity analysis for per-layer adaptive group size selection.

Provides utilities to measure quantization error per layer and select the
largest group_size (fewest norms) that stays within an acceptable SNR threshold.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

from turboquant_model.quantize import turboquant_quantize

logger = logging.getLogger(__name__)

# Small constant to guard against log(0) in SNR computation.
_SNR_EPSILON = 1e-30

# Candidate group sizes ordered from most-compressed (fewest norms) to
# least-compressed (most norms).  None means full-row (one norm per row).
DEFAULT_CANDIDATE_SIZES: list[Optional[int]] = [None, 256, 128, 64]


def compute_layer_snr(
    W: torch.Tensor,
    bit_width: int,
    group_size: Optional[int],
    seed: int = 42,
) -> float:
    """Compute the Signal-to-Noise Ratio (SNR) for quantizing a weight matrix.

    SNR = 10 * log10(||W||_F^2 / ||W - W_hat||_F^2)

    A higher SNR means less quantization error.

    Args:
        W: (out_features, in_features) weight matrix (float32 or bfloat16)
        bit_width: bits per coordinate
        group_size: group size along in_features (None = full row)
        seed: rotation seed

    Returns:
        SNR in dB (float; +inf if quantization is exact)
    """
    W_f = W.float()
    W_hat = turboquant_quantize(W_f, bit_width=bit_width, group_size=group_size, seed=seed)
    noise = W_f - W_hat
    signal_power = W_f.pow(2).sum().item()
    noise_power = noise.pow(2).sum().item()
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * math.log10(signal_power / (noise_power + _SNR_EPSILON))


def compute_layer_mse(
    W: torch.Tensor,
    bit_width: int,
    group_size: Optional[int],
    seed: int = 42,
) -> float:
    """Compute the Mean Squared Error (MSE) for quantizing a weight matrix.

    Args:
        W: (out_features, in_features) weight matrix (float32 or bfloat16)
        bit_width: bits per coordinate
        group_size: group size along in_features (None = full row)
        seed: rotation seed

    Returns:
        MSE (float)
    """
    W_f = W.float()
    W_hat = turboquant_quantize(W_f, bit_width=bit_width, group_size=group_size, seed=seed)
    return (W_f - W_hat).pow(2).mean().item()


def select_group_size(
    W: torch.Tensor,
    bit_width: int,
    seed: int = 42,
    candidate_sizes: list[Optional[int]] | None = None,
    target_snr: float = 20.0,
) -> Optional[int]:
    """Select the largest group_size (fewest norms) that meets the SNR threshold.

    Iterates through *candidate_sizes* from most-compressed to least-compressed
    and returns the first candidate whose SNR >= *target_snr*.  Falls back to
    the last (most-precise) candidate if none meet the threshold.

    Args:
        W: (out_features, in_features) weight matrix
        bit_width: bits per coordinate
        seed: rotation seed
        candidate_sizes: list of group sizes to try, ordered from most- to
            least-compressed (None means full-row).  Defaults to
            [None, 256, 128, 64].
        target_snr: minimum acceptable SNR in dB

    Returns:
        Selected group_size (None = full row, i.e. most compressed)
    """
    if candidate_sizes is None:
        candidate_sizes = DEFAULT_CANDIDATE_SIZES

    fallback: Optional[int] = candidate_sizes[-1]  # most precise option

    for gs in candidate_sizes:
        snr = compute_layer_snr(W, bit_width=bit_width, group_size=gs, seed=seed)
        logger.debug(
            "  group_size=%s → SNR=%.2f dB (target=%.2f dB)",
            gs,
            snr,
            target_snr,
        )
        if snr >= target_snr:
            return gs  # First (most-compressed) that meets the threshold

    return fallback
