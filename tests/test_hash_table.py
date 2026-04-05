"""Tests for TurboQuant-Model hash-based weight compression.

Validates:
  1. Pre-rotation statistics computation and quantization.
  2. Hash key construction is deterministic.
  3. Multi-head lookup produces correct shapes and is deterministic.
  4. Table training reduces MSE (convergence).
  5. hash_compress / hash_decompress round-trip.
  6. BPW calculation matches expected values.
  7. Reconstruction quality beats random baseline.
"""

from __future__ import annotations

import math

import pytest
import torch

from turboquant_model.hash_table import (
    compute_group_stats,
    quantize_stats,
    build_hash_keys,
    multi_head_lookup,
    HashTableConfig,
    HashWeightTable,
    train_hash_table,
    hash_compress,
    hash_decompress,
    compute_bpw,
    _compute_rotated_groups,
    DEFAULT_TABLE_SIZE,
    DEFAULT_GROUP_SIZE,
    DEFAULT_NUM_HEADS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def weight_matrix():
    """A small reproducible weight matrix for CPU tests."""
    torch.manual_seed(0)
    return torch.randn(16, 128)


@pytest.fixture()
def weight_matrix_multi_group():
    """Weight matrix requiring multiple groups (N > group_size)."""
    torch.manual_seed(1)
    return torch.randn(8, 256)


@pytest.fixture()
def small_config():
    """Small hash table config for fast tests."""
    return HashTableConfig(
        table_size=1024,
        group_size=128,
        num_heads=4,
        lr=1e-2,
        n_steps=50,
        rotation="qr",
    )


# ---------------------------------------------------------------------------
# 1. Pre-rotation statistics
# ---------------------------------------------------------------------------

class TestGroupStats:
    """compute_group_stats and quantize_stats."""

    def test_shapes(self, weight_matrix):
        W = weight_matrix
        M, N = W.shape
        means, stds = compute_group_stats(W, group_size=128)
        n_groups = math.ceil(N / 128)
        assert means.shape == (M, n_groups)
        assert stds.shape == (M, n_groups)

    def test_stds_positive(self, weight_matrix):
        _, stds = compute_group_stats(weight_matrix, group_size=128)
        assert (stds > 0).all()

    def test_multi_group_shapes(self, weight_matrix_multi_group):
        W = weight_matrix_multi_group
        M, N = W.shape
        means, stds = compute_group_stats(W, group_size=128)
        n_groups = math.ceil(N / 128)
        assert means.shape == (M, n_groups)
        assert n_groups == 2

    def test_quantize_stats_ranges(self, weight_matrix):
        means, stds = compute_group_stats(weight_matrix, group_size=128)
        mu_q, sigma_q = quantize_stats(means, stds)
        assert mu_q.dtype == torch.int8
        assert sigma_q.dtype == torch.uint8
        assert mu_q.min() >= -127
        assert mu_q.max() <= 127
        assert sigma_q.min() >= 0
        assert sigma_q.max() <= 255


# ---------------------------------------------------------------------------
# 2. Hash key determinism
# ---------------------------------------------------------------------------

class TestHashKeys:
    """build_hash_keys is deterministic and produces correct shapes."""

    def test_shape(self, weight_matrix):
        M, N = weight_matrix.shape
        means, stds = compute_group_stats(weight_matrix, group_size=128)
        mu_q, sigma_q = quantize_stats(means, stds)
        n_groups = mu_q.shape[1]
        keys = build_hash_keys(0, M, n_groups, mu_q, sigma_q)
        assert keys.shape == (M, n_groups)

    def test_deterministic(self, weight_matrix):
        means, stds = compute_group_stats(weight_matrix, group_size=128)
        mu_q, sigma_q = quantize_stats(means, stds)
        n_groups = mu_q.shape[1]
        keys1 = build_hash_keys(0, weight_matrix.shape[0], n_groups, mu_q, sigma_q)
        keys2 = build_hash_keys(0, weight_matrix.shape[0], n_groups, mu_q, sigma_q)
        assert torch.equal(keys1, keys2)

    def test_different_layers_differ(self, weight_matrix):
        means, stds = compute_group_stats(weight_matrix, group_size=128)
        mu_q, sigma_q = quantize_stats(means, stds)
        M = weight_matrix.shape[0]
        n_groups = mu_q.shape[1]
        keys_l0 = build_hash_keys(0, M, n_groups, mu_q, sigma_q)
        keys_l1 = build_hash_keys(1, M, n_groups, mu_q, sigma_q)
        assert not torch.equal(keys_l0, keys_l1)


# ---------------------------------------------------------------------------
# 3. Multi-head lookup
# ---------------------------------------------------------------------------

class TestMultiHeadLookup:
    """multi_head_lookup shape, determinism, and averaging."""

    def test_output_shape(self):
        table = torch.randn(1024, 128)
        keys = torch.randint(0, 10000, (4, 2), dtype=torch.int64)
        result = multi_head_lookup(keys, table, num_heads=4)
        assert result.shape == (4, 2, 128)

    def test_deterministic(self):
        table = torch.randn(1024, 128)
        keys = torch.randint(0, 10000, (4, 2), dtype=torch.int64)
        r1 = multi_head_lookup(keys, table, num_heads=4)
        r2 = multi_head_lookup(keys, table, num_heads=4)
        assert torch.equal(r1, r2)

    def test_single_head_equals_direct(self):
        """With 1 head, lookup should be equivalent to direct table indexing."""
        table = torch.randn(1024, 128)
        keys = torch.tensor([[500]], dtype=torch.int64)
        result = multi_head_lookup(keys, table, num_heads=1)
        # h_0(key) = (key * prime_0) % 1024
        prime = 6_700_417
        idx = (500 * prime) % 1024
        expected = table[idx].unsqueeze(0).unsqueeze(0)
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# 4. Table training convergence
# ---------------------------------------------------------------------------

class TestTableTraining:
    """train_hash_table reduces reconstruction MSE."""

    def test_training_reduces_loss(self, weight_matrix, small_config):
        W = weight_matrix
        # Train with 0 steps = random table
        config_0 = HashTableConfig(
            table_size=small_config.table_size,
            group_size=small_config.group_size,
            num_heads=small_config.num_heads,
            lr=small_config.lr,
            n_steps=0,
        )
        table_0, keys, mu_q, sigma_q, norms = train_hash_table(
            W, layer_idx=0, config=config_0, seed=42,
        )

        # Compute initial MSE
        rotated, _ = _compute_rotated_groups(W, group_size=128, seed=42)
        recon_0 = multi_head_lookup(keys.to(W.device), table_0.table.data, num_heads=4)
        mse_0 = (rotated - recon_0).pow(2).mean().item()

        # Train with 50 steps
        table_50, _, _, _, _ = train_hash_table(
            W, layer_idx=0, config=small_config, seed=42,
        )
        recon_50 = multi_head_lookup(keys.to(W.device), table_50.table.data, num_heads=4)
        mse_50 = (rotated - recon_50).pow(2).mean().item()

        assert mse_50 < mse_0, f"Training did not reduce MSE: {mse_50} >= {mse_0}"

    def test_returns_correct_types(self, weight_matrix, small_config):
        table, keys, mu_q, sigma_q, norms = train_hash_table(
            weight_matrix, layer_idx=0, config=small_config, seed=42,
        )
        assert isinstance(table, HashWeightTable)
        assert keys.dtype == torch.int64
        assert mu_q.dtype == torch.int8
        assert sigma_q.dtype == torch.uint8
        assert norms.dtype == torch.float32


# ---------------------------------------------------------------------------
# 5. Compress / decompress round-trip
# ---------------------------------------------------------------------------

class TestCompressDecompress:
    """hash_compress / hash_decompress round-trip."""

    def test_round_trip_shape(self, weight_matrix, small_config):
        packed = hash_compress(weight_matrix, layer_idx=0, config=small_config, seed=42)
        W_recon = hash_decompress(packed)
        assert W_recon.shape == weight_matrix.shape

    def test_round_trip_finite(self, weight_matrix, small_config):
        packed = hash_compress(weight_matrix, layer_idx=0, config=small_config, seed=42)
        W_recon = hash_decompress(packed)
        assert not torch.isnan(W_recon).any()
        assert not torch.isinf(W_recon).any()

    def test_reconstruction_better_than_random(self, weight_matrix, small_config):
        """Reconstructed weights should be closer to W than zero matrix."""
        W = weight_matrix
        packed = hash_compress(W, layer_idx=0, config=small_config, seed=42)
        W_recon = hash_decompress(packed)
        mse_recon = (W - W_recon).pow(2).mean().item()
        mse_zero = W.pow(2).mean().item()
        assert mse_recon < mse_zero, (
            f"Reconstruction MSE ({mse_recon:.4f}) >= zero-baseline ({mse_zero:.4f})"
        )

    def test_multi_group_round_trip(self, weight_matrix_multi_group, small_config):
        W = weight_matrix_multi_group
        packed = hash_compress(W, layer_idx=0, config=small_config, seed=42)
        W_recon = hash_decompress(packed)
        assert W_recon.shape == W.shape
        assert not torch.isnan(W_recon).any()

    def test_packed_metadata(self, weight_matrix, small_config):
        packed = hash_compress(weight_matrix, layer_idx=0, config=small_config, seed=42)
        assert packed["shape"] == (16, 128)
        assert packed["layer_idx"] == 0
        assert packed["seed"] == 42
        assert packed["group_size"] == 128
        assert packed["num_heads"] == 4
        assert "table" in packed
        assert "keys" in packed
        assert "mu_q" in packed
        assert "sigma_q" in packed
        assert "norms" in packed


# ---------------------------------------------------------------------------
# 6. BPW calculation
# ---------------------------------------------------------------------------

class TestBPW:
    """compute_bpw returns expected values."""

    def test_0_8B_model(self):
        """For 0.8B params, the paper claims ~0.83 bpw."""
        total_weights = 800_000_000
        bpw = compute_bpw(
            total_weights=total_weights,
            table_size=262_144,
            group_size=128,
        )
        # Table: 262144 * 128 * 16 = ~537M bits = 0.67 bpw
        # Stats: (800M / 128) * 16 = ~100M bits = 0.125 bpw
        # Norms: (800M / 128) * 32 = ~200M bits = 0.25 bpw
        # Total ≈ 1.05 bpw (actual will vary based on exact arithmetic)
        assert 0.5 < bpw < 2.0, f"BPW out of expected range: {bpw}"

    def test_scales_inversely_with_model_size(self):
        """Larger models have lower amortised table cost."""
        bpw_small = compute_bpw(100_000_000, table_size=262_144, group_size=128)
        bpw_large = compute_bpw(10_000_000_000, table_size=262_144, group_size=128)
        assert bpw_large < bpw_small

    def test_stats_overhead(self):
        """Stats overhead alone should be 0.125 bpw (16 bits / 128 weights)."""
        # Isolate stats contribution: large model so table cost is negligible
        total_weights = 10_000_000_000
        bpw = compute_bpw(total_weights, table_size=1, group_size=128)
        # With table_size=1: table cost ≈ 0, stats = 0.125, norms = 0.25
        # Total ≈ 0.375
        expected_stats_overhead = 16.0 / 128.0  # 0.125
        expected_norm_overhead = 32.0 / 128.0    # 0.25
        assert abs(bpw - (expected_stats_overhead + expected_norm_overhead + 16 / total_weights)) < 0.001


# ---------------------------------------------------------------------------
# 7. HashWeightTable module
# ---------------------------------------------------------------------------

class TestHashWeightTable:
    """HashWeightTable nn.Module."""

    def test_memory_bytes(self):
        t = HashWeightTable(table_size=1024, group_size=128)
        # 1024 * 128 * 2 bytes (fp16) = 262144
        assert t.memory_bytes() == 1024 * 128 * 2

    def test_table_is_parameter(self):
        t = HashWeightTable(table_size=1024, group_size=128)
        params = list(t.parameters())
        assert len(params) == 1
        assert params[0].shape == (1024, 128)

    def test_lookup_shape(self):
        t = HashWeightTable(table_size=1024, group_size=128, num_heads=4)
        keys = torch.randint(0, 10000, (4, 2), dtype=torch.int64)
        result = t.lookup(keys)
        assert result.shape == (4, 2, 128)


# ---------------------------------------------------------------------------
# 8. Rotated groups computation
# ---------------------------------------------------------------------------

class TestRotatedGroups:
    """_compute_rotated_groups matches TurboQuant pre-processing."""

    def test_output_shapes(self, weight_matrix):
        W = weight_matrix
        M, N = W.shape
        rotated, norms = _compute_rotated_groups(W, group_size=128, seed=42)
        n_groups = math.ceil(N / 128)
        assert rotated.shape == (M, n_groups, 128)
        assert norms.shape == (M, n_groups)

    def test_norms_positive(self, weight_matrix):
        _, norms = _compute_rotated_groups(weight_matrix, group_size=128, seed=42)
        assert (norms > 0).all()

    def test_deterministic(self, weight_matrix):
        r1, n1 = _compute_rotated_groups(weight_matrix, group_size=128, seed=42)
        r2, n2 = _compute_rotated_groups(weight_matrix, group_size=128, seed=42)
        assert torch.equal(r1, r2)
        assert torch.equal(n1, n2)


# ---------------------------------------------------------------------------
# 9. Shared table across layers
# ---------------------------------------------------------------------------

class TestSharedTable:
    """A single table can be shared across multiple layers."""

    def test_shared_table_training(self, small_config):
        torch.manual_seed(42)
        W1 = torch.randn(8, 128)
        W2 = torch.randn(8, 128)

        # Train on layer 0
        table, _, _, _, _ = train_hash_table(W1, layer_idx=0, config=small_config, seed=42)

        # Fine-tune on layer 1 (reuse table)
        table, _, _, _, _ = train_hash_table(W2, layer_idx=1, config=small_config, seed=42, table=table)

        assert isinstance(table, HashWeightTable)
        # Table should still be the correct size
        assert table.table.shape == (small_config.table_size, small_config.group_size)
