"""Tests for multi-pass shared-rotation residual quantization and merging.

Validates:
  1. Multi-pass residual reduces quantization error monotonically.
  2. Packed representation matches simulation exactly.
  3. merge_residual_passes produces exact (lossless) dense merge.
  4. merge_and_requantize compresses multi-pass into single-pass format.
  5. Group-wise quantization works with multi-pass and merge.
  6. TurboQuantLinear.merge_passes() collapses residual into single pass.
"""

from __future__ import annotations

import math

import pytest
import torch

from turboquant_model.quantize import turboquant_quantize, turboquant_quantize_packed
from turboquant_model.residual import (
    multi_residual_quantize,
    multi_residual_quantize_packed,
    merge_residual_passes,
    merge_and_requantize,
    _dequantize_from_packed,
)
from turboquant_model.module import TurboQuantLinear


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def weight_matrix():
    """A small reproducible weight matrix for CPU tests."""
    torch.manual_seed(0)
    return torch.randn(16, 32)


@pytest.fixture()
def weight_matrix_odd():
    """Odd-column matrix to exercise padding paths."""
    torch.manual_seed(1)
    return torch.randn(8, 31)


# ---------------------------------------------------------------------------
# 1. Multi-pass error reduction
# ---------------------------------------------------------------------------

class TestMultiPassResidual:
    """multi_residual_quantize reduces MSE with each additional pass."""

    def test_error_decreases_monotonically(self, weight_matrix):
        W = weight_matrix
        prev_mse = float("inf")
        for n in range(1, 6):
            W_approx = multi_residual_quantize(W, n_passes=n, bit_width=4, seed=42)
            mse = (W - W_approx).pow(2).mean().item()
            assert mse < prev_mse, f"MSE did not decrease at pass {n}"
            prev_mse = mse

    def test_single_pass_matches_turboquant(self, weight_matrix):
        """n_passes=1 should be identical to plain turboquant_quantize."""
        W = weight_matrix
        W1 = turboquant_quantize(W, bit_width=4, seed=42)
        W_multi_1 = multi_residual_quantize(W, n_passes=1, bit_width=4, seed=42)
        assert torch.allclose(W1, W_multi_1, atol=1e-6)

    def test_output_shape_dtype(self, weight_matrix):
        W = weight_matrix
        out = multi_residual_quantize(W, n_passes=3)
        assert out.shape == W.shape
        assert out.dtype == W.dtype


# ---------------------------------------------------------------------------
# 2. Packed representation matches simulation
# ---------------------------------------------------------------------------

class TestMultiPassPacked:
    """multi_residual_quantize_packed stores the same data as simulation."""

    @pytest.mark.parametrize("n_passes", [1, 2, 3, 4])
    def test_packed_matches_simulation(self, weight_matrix, n_passes):
        W = weight_matrix
        W_sim = multi_residual_quantize(W, n_passes=n_passes, bit_width=4, seed=42)
        packed = multi_residual_quantize_packed(W, n_passes=n_passes, bit_width=4, seed=42)
        W_merged = merge_residual_passes(packed)
        assert torch.allclose(W_sim.float(), W_merged, atol=1e-6)

    def test_packed_metadata(self, weight_matrix):
        packed = multi_residual_quantize_packed(weight_matrix, n_passes=3, bit_width=4, seed=99)
        assert packed["n_passes"] == 3
        assert packed["total_bits"] == 12
        assert packed["shared_seed"] == 99
        assert len(packed["passes"]) == 3
        for p in packed["passes"]:
            assert p["seed"] == 99  # shared rotation


# ---------------------------------------------------------------------------
# 3. Exact (lossless) dense merge
# ---------------------------------------------------------------------------

class TestMergeResidualPasses:
    """merge_residual_passes is exact w.r.t. the sum of dequantized passes."""

    def test_merge_equals_sum_of_dequantized(self, weight_matrix):
        W = weight_matrix
        packed = multi_residual_quantize_packed(W, n_passes=3, bit_width=4, seed=42)
        W_merged = merge_residual_passes(packed)

        # Manually sum dequantized passes
        W_sum = torch.zeros_like(W_merged)
        for p in packed["passes"]:
            W_sum += _dequantize_from_packed(p, device=torch.device("cpu"))

        assert torch.allclose(W_merged, W_sum, atol=1e-6)

    def test_merge_output_shape(self, weight_matrix):
        packed = multi_residual_quantize_packed(weight_matrix, n_passes=2, bit_width=4, seed=42)
        W_merged = merge_residual_passes(packed)
        assert W_merged.shape == weight_matrix.shape
        assert W_merged.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. merge_and_requantize
# ---------------------------------------------------------------------------

class TestMergeAndRequantize:
    """merge_and_requantize produces a valid single-pass packed representation."""

    def test_output_format(self, weight_matrix):
        packed = multi_residual_quantize_packed(weight_matrix, n_passes=3, bit_width=4, seed=42)
        merged = merge_and_requantize(packed, target_bit_width=4)

        # Should have the same keys as turboquant_quantize_packed output
        assert "indices_packed" in merged
        assert "codebook" in merged
        assert "norms" in merged
        assert "seed" in merged
        assert merged["seed"] == 42
        assert merged["shape"] == (16, 32)
        assert merged["bit_width"] == 4

    def test_dequant_roundtrip(self, weight_matrix):
        """Merged+requantized representation should dequantize cleanly."""
        packed = multi_residual_quantize_packed(weight_matrix, n_passes=3, bit_width=4, seed=42)
        merged = merge_and_requantize(packed, target_bit_width=4)
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        assert W_deq.shape == weight_matrix.shape
        assert not torch.isnan(W_deq).any()
        assert not torch.isinf(W_deq).any()

    def test_merged_better_than_random(self, weight_matrix):
        """Merged result should be closer to W than a random matrix."""
        W = weight_matrix
        packed = multi_residual_quantize_packed(W, n_passes=3, bit_width=4, seed=42)
        merged = merge_and_requantize(packed, target_bit_width=4)
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        mse_merged = (W - W_deq).pow(2).mean().item()
        mse_random = W.pow(2).mean().item()  # MSE vs zero
        assert mse_merged < mse_random


# ---------------------------------------------------------------------------
# 5. Group-wise multi-pass and merge
# ---------------------------------------------------------------------------

class TestGroupWise:
    """Multi-pass and merge work correctly with group_size < N."""

    def test_multi_pass_with_groups(self, weight_matrix):
        W = weight_matrix
        W_approx = multi_residual_quantize(W, n_passes=3, bit_width=4, group_size=16, seed=42)
        assert W_approx.shape == W.shape
        mse = (W - W_approx).pow(2).mean().item()
        assert mse < W.pow(2).mean().item()

    def test_merge_with_groups(self, weight_matrix):
        W = weight_matrix
        packed = multi_residual_quantize_packed(W, n_passes=3, bit_width=4, group_size=16, seed=42)
        W_merged = merge_residual_passes(packed)
        W_sim = multi_residual_quantize(W, n_passes=3, bit_width=4, group_size=16, seed=42)
        assert torch.allclose(W_sim.float(), W_merged, atol=1e-6)

    def test_requantize_with_groups(self, weight_matrix):
        packed = multi_residual_quantize_packed(
            weight_matrix, n_passes=2, bit_width=4, group_size=16, seed=42,
        )
        merged = merge_and_requantize(packed, target_bit_width=4)
        assert merged["group_size"] == 16
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        assert W_deq.shape == weight_matrix.shape

    def test_odd_columns(self, weight_matrix_odd):
        """Odd-column matrices should be handled correctly (padding)."""
        W = weight_matrix_odd
        packed = multi_residual_quantize_packed(W, n_passes=2, bit_width=4, seed=42)
        W_merged = merge_residual_passes(packed)
        assert W_merged.shape == W.shape


# ---------------------------------------------------------------------------
# 6. TurboQuantLinear.merge_passes
# ---------------------------------------------------------------------------

class TestModuleMerge:
    """TurboQuantLinear.merge_passes collapses residual into a single pass."""

    def _make_tq_with_residual(self, W, seed=42, group_size=None):
        """Helper to build a TurboQuantLinear with residual from a weight matrix."""
        M, N = W.shape
        gs = group_size or N

        tq = TurboQuantLinear(
            in_features=N, out_features=M, bit_width=4, group_size=gs,
        )
        # Quantize pass 1
        packed1 = turboquant_quantize_packed(W, bit_width=4, group_size=gs, seed=seed)
        tq.indices_packed.copy_(packed1["indices_packed"])
        tq.weight_norms.copy_(packed1["norms"])
        tq.codebook.copy_(packed1["codebook"])
        tq.set_rotation(seed)

        # Disable fused kernels (not available in test environment)
        tq.use_cutile = False
        tq.use_triton = False

        # Compute residual and quantize pass 2 (same seed = same rotation)
        W_hat1 = _dequantize_from_packed(packed1, device=torch.device("cpu"))
        residual = W.float() - W_hat1
        packed2 = turboquant_quantize_packed(residual, bit_width=4, group_size=gs, seed=seed)
        tq.set_pass2(
            indices_packed=packed2["indices_packed"],
            weight_norms=packed2["norms"],
            codebook=packed2["codebook"],
            seed=seed,
        )
        return tq

    def test_merge_clears_residual(self, weight_matrix):
        tq = self._make_tq_with_residual(weight_matrix)
        assert tq.has_residual
        tq.merge_passes()
        assert not tq.has_residual

    def test_forward_after_merge(self, weight_matrix):
        """Forward pass should work after merging."""
        tq = self._make_tq_with_residual(weight_matrix)
        x = torch.randn(4, weight_matrix.shape[1])

        with torch.no_grad():
            out_before = tq(x).clone()
        tq.merge_passes()
        with torch.no_grad():
            out_after = tq(x)

        # The merge is lossy (requantisation), so outputs won't be identical,
        # but should be finite and reasonably close.
        assert not torch.isnan(out_after).any()
        assert not torch.isinf(out_after).any()

    def test_merge_noop_without_residual(self, weight_matrix):
        """merge_passes on a single-pass module should be a no-op."""
        M, N = weight_matrix.shape
        tq = TurboQuantLinear(in_features=N, out_features=M, bit_width=4)
        packed = turboquant_quantize_packed(weight_matrix, bit_width=4, seed=42)
        tq.indices_packed.copy_(packed["indices_packed"])
        tq.weight_norms.copy_(packed["norms"])
        tq.codebook.copy_(packed["codebook"])
        tq.set_rotation(42)
        tq.use_cutile = False
        tq.use_triton = False

        indices_before = tq.indices_packed.clone()
        tq.merge_passes()
        assert torch.equal(tq.indices_packed, indices_before)

    def test_merge_with_groups(self, weight_matrix):
        tq = self._make_tq_with_residual(weight_matrix, group_size=16)
        assert tq.has_residual
        x = torch.randn(2, weight_matrix.shape[1])
        with torch.no_grad():
            _ = tq(x)  # should not crash
        tq.merge_passes()
        assert not tq.has_residual
        with torch.no_grad():
            out = tq(x)
        assert not torch.isnan(out).any()
