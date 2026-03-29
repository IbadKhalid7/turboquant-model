"""TurboQuantLinear — Drop-in nn.Linear replacement with on-the-fly 4-bit dequantization.

Stores weights as packed 4-bit indices + per-row norms + shared codebook.
On-the-fly forward (Approach C: pre-rotate input):
  1. x_rot = x @ Pi.T           (rotate input, not weight)
  2. out = x_rot @ codebook[indices].T  (fused lookup + matmul)
  3. out = out * (norms / scale)  (rescale per output row)

Supports both single-pass and residual (two-pass) quantization.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from turboquant_model.rotation import (
    generate_rotation_matrix,
    hadamard_rotate,
    hadamard_rotate_inverse,
)
from turboquant_model.quantize import unpack_4bit, pack_4bit
from turboquant_model.codebook import get_codebook

# Try to import fused kernels — prefers cuTile > Triton > PyTorch fallback
_HAS_CUTILE = False
try:
    from turboquant_model.cutile_kernels import cutile_fused_matmul
    _HAS_CUTILE = True
except ImportError:
    pass

_HAS_TRITON = False
try:
    from turboquant_model.triton_kernels import triton_fused_matmul
    _HAS_TRITON = True
except ImportError:
    pass


class TurboQuantLinear(nn.Module):
    """Linear layer with TurboQuant-compressed weights and on-the-fly dequantization.

    Storage per layer:
      - indices_packed: (M, N//2) uint8 — packed 4-bit quantization indices
      - weight_norms: (M,) or (M, n_groups) float32 — per-row norms
      - codebook: (16,) float32 — Lloyd-Max centroids (shared)
      - [optional] pass2_*: same buffers for residual pass

    Forward pass:
      x_rot = x @ Pi.T
      output = x_rot @ codebook[indices].T * (norms / sqrt(group_size))
      [+ residual pass if present]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bit_width: int = 4,
        group_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        rotation: str = "qr",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        self.group_size = group_size or in_features
        self.n_levels = 2**bit_width
        self.rotation = rotation

        pack_factor = 8 // bit_width
        packed_dim = math.ceil(in_features / pack_factor)
        n_groups = math.ceil(in_features / self.group_size)

        # Pass 1 buffers
        self.register_buffer(
            "indices_packed",
            torch.zeros(out_features, packed_dim, dtype=torch.uint8, device=device),
        )
        if n_groups == 1:
            self.register_buffer(
                "weight_norms",
                torch.ones(out_features, dtype=torch.float32, device=device),
            )
        else:
            self.register_buffer(
                "weight_norms",
                torch.ones(out_features, n_groups, dtype=torch.float32, device=device),
            )
        self.register_buffer(
            "codebook",
            torch.zeros(self.n_levels, dtype=torch.float32, device=device),
        )

        # Pass 2 (residual) buffers — None until set
        self.register_buffer("pass2_indices_packed", None)
        self.register_buffer("pass2_weight_norms", None)
        self.register_buffer("pass2_codebook", None)
        self._pass2_seed: Optional[int] = None

        # Bias
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float32, device=device),
            )
        else:
            self.bias = None

        # Rotation cache: dict[seed_offset → Pi tensor]
        self._rotation_cache: dict[int, torch.Tensor] = {}
        self._rotation_seed: int = 42
        self._scale: float = math.sqrt(self.group_size)

        # Cached unpacked indices (lazy, freed on device change)
        self._cached_indices: Optional[torch.Tensor] = None
        self._cached_pass2_indices: Optional[torch.Tensor] = None
        self._n_groups: int = math.ceil(in_features / self.group_size)

        # Fused kernel priority: cuTile > Triton > PyTorch fallback
        self.use_cutile: bool = _HAS_CUTILE
        self.use_triton: bool = _HAS_TRITON

    def set_rotation(self, seed: int):
        self._rotation_seed = seed
        self._rotation_cache.clear()

    def set_pass2(
        self,
        indices_packed: torch.Tensor,
        weight_norms: torch.Tensor,
        codebook: torch.Tensor,
        seed: int,
    ):
        """Set residual (pass 2) quantization data."""
        self.register_buffer("pass2_indices_packed", indices_packed)
        self.register_buffer("pass2_weight_norms", weight_norms)
        self.register_buffer("pass2_codebook", codebook)
        self._pass2_seed = seed
        self._cached_pass2_indices = None

    @property
    def has_residual(self) -> bool:
        return self.pass2_indices_packed is not None

    def _get_rotation(self, seed: int, g_start: int = 0) -> torch.Tensor:
        """Get cached rotation matrix for a specific group.

        Args:
            seed: base rotation seed
            g_start: group start column (each group uses seed + g_start)
        """
        key = seed + g_start
        if key not in self._rotation_cache:
            self._rotation_cache[key] = generate_rotation_matrix(
                self.group_size, seed=key
            ).to(self.indices_packed.device)
        return self._rotation_cache[key]

    def _get_indices(self) -> torch.Tensor:
        """Get unpacked indices (cached)."""
        if self._cached_indices is None:
            self._cached_indices = unpack_4bit(self.indices_packed, self.in_features)
        return self._cached_indices

    def _get_pass2_indices(self) -> torch.Tensor:
        if self._cached_pass2_indices is None and self.pass2_indices_packed is not None:
            self._cached_pass2_indices = unpack_4bit(self.pass2_indices_packed, self.in_features)
        return self._cached_pass2_indices

    def _forward_pass(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None,
        indices_packed: torch.Tensor,
        codebook: torch.Tensor,
        weight_norms: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """Single-pass on-the-fly dequant matmul with group-wise rotation.

        For each group g in [0, n_groups):
          x_rot_g = x[:, g_start:g_end] @ Pi_g.T     (rotate input slice)
          output += x_rot_g @ codebook[indices[:, g_start:g_end]].T * (norms_g / scale)

        Uses Triton fused kernel when available (operates on packed indices directly,
        avoiding unpack + codebook[idx] intermediate materialization).

        Args:
            x: (B, K) input (float32)
            indices: (N, K) unpacked int32 (None if using Triton path)
            indices_packed: (N, K//2) packed uint8 (for Triton path)
            codebook: (n_levels,) float32
            weight_norms: (N,) or (N, n_groups) float32
            seed: base rotation seed

        Returns:
            output: (B, N) float32
        """
        B = x.shape[0]
        N = indices_packed.shape[0]
        K = self.in_features
        device = x.device
        scale = self._scale

        output = torch.zeros(B, N, dtype=torch.float32, device=device)

        for g in range(self._n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, K)
            g_dim = g_end - g_start

            # Rotate this group's input slice
            if self.rotation == "hadamard":
                x_rot_g = hadamard_rotate(x[:, g_start:g_end], seed + g_start)
            else:
                Pi_g = self._get_rotation(seed, g_start).to(device)
                x_rot_g = x[:, g_start:g_end] @ Pi_g.T  # (B, g_dim)

            # Per-group norms
            if weight_norms.dim() == 1:
                norms_g = weight_norms  # (N,) — same norm for all groups
            else:
                norms_g = weight_norms[:, g]  # (N,) — per-group norms

            if self.use_cutile and g_dim == self.group_size:
                # cuTile fused path: unpack + codebook lookup + matmul in one kernel
                packed_g = indices_packed[:, g_start // 2 : g_end // 2]
                out_g = cutile_fused_matmul(
                    x_rot_g.contiguous(), packed_g.contiguous(),
                    codebook, norms_g.contiguous(), g_dim, scale,
                )
            elif self.use_triton and g_dim == self.group_size:
                # Triton fused path: unpack + codebook lookup + matmul in one kernel
                packed_g = indices_packed[:, g_start // 2 : g_end // 2]  # (N, g_dim//2)
                out_g = triton_fused_matmul(
                    x_rot_g.contiguous(), packed_g.contiguous(),
                    codebook, norms_g.contiguous(), g_dim, scale,
                )
            else:
                # PyTorch fallback: explicit unpack + lookup + matmul
                if indices is None:
                    indices = unpack_4bit(indices_packed, K)
                idx_g = indices[:, g_start:g_end]  # (N, g_dim)
                W_g = codebook[idx_g.long()]       # (N, g_dim)
                out_g = x_rot_g @ W_g.T
                out_g = out_g * (norms_g[None, :] / scale)

            output += out_g

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """On-the-fly dequant forward pass with group-wise rotation.

        Handles 2D (B, K) and 3D (B, S, K) inputs.
        """
        device = x.device
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        x_f = x.float()

        # Pass 1
        _use_fused = self.use_cutile or self.use_triton
        indices = None if _use_fused else self._get_indices()
        output = self._forward_pass(
            x_f, indices, self.indices_packed, self.codebook,
            self.weight_norms, self._rotation_seed,
        )

        # Pass 2 (residual) if present
        if self.has_residual:
            indices2 = None if _use_fused else self._get_pass2_indices()
            output += self._forward_pass(
                x_f, indices2, self.pass2_indices_packed, self.pass2_codebook,
                self.pass2_weight_norms, self._pass2_seed,
            )

        # Restore shape
        if len(orig_shape) == 3:
            output = output.reshape(orig_shape[0], orig_shape[1], self.out_features)

        out = output.to(x.dtype)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def dequantize(self) -> torch.Tensor:
        """Full dequantization: returns (M, N) bf16 weight (for debugging)."""
        indices = self._get_indices()
        scale = self._scale

        W = torch.zeros(
            self.out_features, self.in_features,
            dtype=torch.float32, device=self.indices_packed.device,
        )

        for g in range(self._n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, self.in_features)

            if self.rotation == "hadamard":
                Y_g = self.codebook[indices[:, g_start:g_end].long()] / scale
                W_g = hadamard_rotate_inverse(Y_g, self._rotation_seed + g_start)
            else:
                Pi_g = self._get_rotation(self._rotation_seed, g_start)
                Y_g = self.codebook[indices[:, g_start:g_end].long()] / scale
                W_g = Y_g @ Pi_g[:g_end - g_start, :g_end - g_start]

            if self.weight_norms.dim() == 1:
                W_g = W_g * self.weight_norms.unsqueeze(1)
            else:
                W_g = W_g * self.weight_norms[:, g].unsqueeze(1)

            W[:, g_start:g_end] = W_g

        if self.has_residual:
            indices2 = self._get_pass2_indices()
            for g in range(self._n_groups):
                g_start = g * self.group_size
                g_end = min(g_start + self.group_size, self.in_features)
                if self.rotation == "hadamard":
                    Y_g = self.pass2_codebook[indices2[:, g_start:g_end].long()] / scale
                    W_g = hadamard_rotate_inverse(Y_g, self._pass2_seed + g_start)
                else:
                    Pi2_g = self._get_rotation(self._pass2_seed, g_start)
                    Y_g = self.pass2_codebook[indices2[:, g_start:g_end].long()] / scale
                    W_g = Y_g @ Pi2_g[:g_end - g_start, :g_end - g_start]
                if self.pass2_weight_norms.dim() == 1:
                    W_g = W_g * self.pass2_weight_norms.unsqueeze(1)
                else:
                    W_g = W_g * self.pass2_weight_norms[:, g].unsqueeze(1)
                W[:, g_start:g_end] += W_g

        return W.to(torch.bfloat16)

    @torch.no_grad()
    def merge_passes(self) -> None:
        """Merge residual pass into the primary pass via rotated-domain addition.

        When both passes share the **same rotation seed**, the rotation matrix
        factors out of the sum and the merge happens entirely in the rotated
        domain — no inverse rotation is ever computed.  The merged rotated-domain
        values are re-normalized and re-quantized into a single set of packed
        4-bit indices.

        After calling this method ``has_residual`` becomes False and forward
        uses a single pass, cutting inference cost in half while retaining
        multi-pass quality (up to re-quantisation).

        When the rotation seeds differ, a dense dequantize-sum-requantize
        fallback is used (still correct, but involves inverse rotations).
        """
        if not self.has_residual:
            return

        device = self.indices_packed.device
        K = self.in_features
        N = self.out_features
        scale = self._scale

        centroids, boundaries = get_codebook(self.bit_width)
        centroids = centroids.to(device)
        boundaries = boundaries.to(device)

        same_rotation = self._pass2_seed == self._rotation_seed

        if same_rotation:
            # ---- Fast path: merge in the rotated domain ----
            indices1 = self._get_indices()
            indices2 = self._get_pass2_indices()

            all_merged_indices: list[torch.Tensor] = []
            all_merged_norms: list[torch.Tensor] = []

            for g in range(self._n_groups):
                g_start = g * self.group_size
                g_end = min(g_start + self.group_size, K)
                g_dim = g_end - g_start

                # Pass-1 contribution in rotated domain
                Y1 = self.codebook[indices1[:, g_start:g_end].long()] / scale
                n1 = (
                    self.weight_norms
                    if self.weight_norms.dim() == 1
                    else self.weight_norms[:, g]
                )
                Y1 = Y1 * n1.unsqueeze(1)

                # Pass-2 contribution in rotated domain
                Y2 = self.pass2_codebook[indices2[:, g_start:g_end].long()] / scale
                n2 = (
                    self.pass2_weight_norms
                    if self.pass2_weight_norms.dim() == 1
                    else self.pass2_weight_norms[:, g]
                )
                Y2 = Y2 * n2.unsqueeze(1)

                Y_merged = Y1 + Y2

                # Re-normalize
                merged_norms = Y_merged.norm(dim=1, keepdim=True).clamp(min=1e-8)
                Y_norm = Y_merged / merged_norms

                # Re-quantize (already rotated — no Pi needed)
                Y_scaled = Y_norm * scale
                idx = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
                idx = idx.clamp(0, len(centroids) - 1).reshape(N, g_dim)

                all_merged_indices.append(idx)
                all_merged_norms.append(merged_norms.squeeze(1))

            full_indices = torch.cat(all_merged_indices, dim=1)
            norms_out = (
                torch.stack(all_merged_norms, dim=1)
                if len(all_merged_norms) > 1
                else all_merged_norms[0]
            )
        else:
            # ---- Fallback: dequantize both, sum, re-quantize ----
            W_merged = self.dequantize().float()
            all_indices: list[torch.Tensor] = []
            all_norms: list[torch.Tensor] = []

            for g_start in range(0, K, self.group_size):
                g_end = min(g_start + self.group_size, K)
                g_dim = g_end - g_start
                W_g = W_merged[:, g_start:g_end]

                norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
                W_norm = W_g / norms
                all_norms.append(norms.squeeze(1))

                Pi = generate_rotation_matrix(
                    g_dim, seed=self._rotation_seed + g_start
                ).to(device)
                Y = W_norm @ Pi.T
                Y_scaled = Y * scale

                idx = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
                idx = idx.clamp(0, len(centroids) - 1).reshape(N, g_dim)
                all_indices.append(idx)

            full_indices = torch.cat(all_indices, dim=1)
            norms_out = (
                torch.stack(all_norms, dim=1)
                if len(all_norms) > 1
                else all_norms[0]
            )

        # Pack and replace buffers
        if K % 2 != 0:
            full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

        packed = pack_4bit(full_indices)
        self.indices_packed.copy_(packed)
        self.weight_norms.copy_(norms_out)
        self.codebook.copy_(centroids)

        # Clear residual buffers
        self.register_buffer("pass2_indices_packed", None)
        self.register_buffer("pass2_weight_norms", None)
        self.register_buffer("pass2_codebook", None)
        self._pass2_seed = None
        self._cached_indices = None
        self._cached_pass2_indices = None

    def memory_bytes(self) -> int:
        """Compressed storage in bytes."""
        total = self.indices_packed.numel()  # uint8
        total += self.weight_norms.numel() * 4
        total += self.codebook.numel() * 4
        if self.bias is not None:
            total += self.bias.numel() * 4
        if self.pass2_indices_packed is not None:
            total += self.pass2_indices_packed.numel()
            total += self.pass2_weight_norms.numel() * 4
            total += self.pass2_codebook.numel() * 4
        return total

    def extra_repr(self) -> str:
        residual = ", residual=True" if self.has_residual else ""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bit_width={self.bit_width}, group_size={self.group_size}{residual}, "
            f"compressed={self.memory_bytes() / 1024:.1f} KB"
        )
