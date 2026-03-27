"""Random rotation matrix generation for TurboQuant."""

from __future__ import annotations

import torch


def generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor:
    """Generate a Haar-distributed random orthogonal matrix via QR decomposition.

    The resulting matrix maps any unit vector to a nearly uniform point on
    the unit hypersphere, making coordinates approximately independent
    with N(0, 1/d) distribution.

    Args:
        d: dimension
        seed: random seed for reproducibility

    Returns:
        Q: orthogonal matrix of shape (d, d), float32
    """
    gen = torch.Generator().manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity to get proper Haar distribution
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q
