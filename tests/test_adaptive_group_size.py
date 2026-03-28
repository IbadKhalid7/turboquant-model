"""Unit tests for per-layer adaptive group size selection.

Tests:
  1. compute_layer_snr returns a finite positive value
  2. compute_layer_mse returns a non-negative value
  3. select_group_size returns a valid group size from candidates
  4. select_group_size chooses full-row when SNR is easily achievable
  5. select_group_size falls back to smallest group_size when target is very high
  6. TurboQuantConfig per_layer_group_size serializes/deserializes correctly
  7. quantize_model with per_layer_group_size uses layer-specific group sizes
  8. quantize_model with target_snr populates per_layer_group_size
  9. save/load round-trip preserves per-layer group sizes
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from turboquant_model.sensitivity import (
    DEFAULT_CANDIDATE_SIZES,
    compute_layer_mse,
    compute_layer_snr,
    select_group_size,
)
from turboquant_model.model import TurboQuantConfig, quantize_model, save_quantized, load_quantized
from turboquant_model.module import TurboQuantLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weight(out_features: int = 32, in_features: int = 64, seed: int = 0) -> torch.Tensor:
    """Create a reproducible random weight matrix."""
    torch.manual_seed(seed)
    return torch.randn(out_features, in_features)


def _make_small_model(seed: int = 0) -> nn.Module:
    """Create a tiny 2-layer MLP for model-level tests."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 32, bias=False),
        nn.Linear(32, 16, bias=False),
    )


# ---------------------------------------------------------------------------
# 1. compute_layer_snr
# ---------------------------------------------------------------------------

class TestComputeLayerSNR:
    def test_returns_finite_positive(self):
        W = _make_weight()
        snr = compute_layer_snr(W, bit_width=4, group_size=None)
        assert isinstance(snr, float)
        assert snr > 0

    def test_smaller_group_size_gives_higher_snr(self):
        """Smaller group_size (more norms) should yield higher SNR for non-uniform rows.

        We create a weight matrix where the second half of each row contains
        values 100x larger than the first half.  With full-row normalisation
        the small values are crushed; smaller group_size protects them.
        """
        torch.manual_seed(0)
        W = torch.randn(16, 256)
        # Inject large-magnitude cluster in the second half of each row
        W[:, 128:] = W[:, 128:] * 100.0

        snr_full = compute_layer_snr(W, bit_width=4, group_size=None, seed=42)
        snr_128 = compute_layer_snr(W, bit_width=4, group_size=128, seed=42)
        # With outlier clusters, smaller group size should be strictly better
        assert snr_128 > snr_full, (
            f"Expected group_size=128 ({snr_128:.2f} dB) > full-row ({snr_full:.2f} dB)"
        )

    def test_accepts_bfloat16(self):
        W = _make_weight().to(torch.bfloat16)
        snr = compute_layer_snr(W, bit_width=4, group_size=128)
        assert snr > 0

    def test_different_seeds_give_different_snr(self):
        W = _make_weight(out_features=16, in_features=128)
        snr1 = compute_layer_snr(W, bit_width=4, group_size=None, seed=42)
        snr2 = compute_layer_snr(W, bit_width=4, group_size=None, seed=99)
        # Different rotation seeds produce different quantization errors
        assert snr1 != snr2


# ---------------------------------------------------------------------------
# 2. compute_layer_mse
# ---------------------------------------------------------------------------

class TestComputeLayerMSE:
    def test_returns_non_negative(self):
        W = _make_weight()
        mse = compute_layer_mse(W, bit_width=4, group_size=None)
        assert mse >= 0.0

    def test_smaller_group_size_gives_lower_mse(self):
        """Smaller group_size gives lower MSE for non-uniform rows (outlier cluster)."""
        torch.manual_seed(0)
        W = torch.randn(16, 256)
        W[:, 128:] = W[:, 128:] * 100.0  # Large cluster in second half

        mse_full = compute_layer_mse(W, bit_width=4, group_size=None, seed=42)
        mse_128 = compute_layer_mse(W, bit_width=4, group_size=128, seed=42)
        assert mse_128 < mse_full, (
            f"Expected group_size=128 MSE ({mse_128:.6f}) < full-row MSE ({mse_full:.6f})"
        )


# ---------------------------------------------------------------------------
# 3-5. select_group_size
# ---------------------------------------------------------------------------

class TestSelectGroupSize:
    def test_returns_candidate(self):
        W = _make_weight(out_features=16, in_features=128)
        gs = select_group_size(W, bit_width=4, seed=42, target_snr=0.0)
        assert gs in DEFAULT_CANDIDATE_SIZES

    def test_very_low_target_returns_full_row(self):
        """A very low target SNR should be met by full-row (None)."""
        W = _make_weight(out_features=16, in_features=128)
        gs = select_group_size(W, bit_width=4, seed=42, target_snr=0.0)
        assert gs is None

    def test_very_high_target_returns_smallest_group(self):
        """Unreachably high target should return the last (most precise) candidate."""
        W = _make_weight(out_features=16, in_features=128)
        candidates = [None, 64]
        gs = select_group_size(
            W, bit_width=4, seed=42,
            candidate_sizes=candidates,
            target_snr=1e9,
        )
        assert gs == 64

    def test_custom_candidates(self):
        W = _make_weight(out_features=16, in_features=128)
        candidates = [128, 64]
        gs = select_group_size(
            W, bit_width=4, seed=42,
            candidate_sizes=candidates,
            target_snr=0.0,
        )
        assert gs in candidates


# ---------------------------------------------------------------------------
# 6. TurboQuantConfig serialization
# ---------------------------------------------------------------------------

class TestTurboQuantConfigSerialization:
    def test_per_layer_group_size_roundtrip(self, tmp_path):
        cfg = TurboQuantConfig(
            bit_width=4,
            group_size=128,
            per_layer_group_size={"model.layer1": 64, "model.layer2": None},
            target_snr=25.0,
        )
        path = tmp_path / "config.json"
        cfg.save(path)
        loaded = TurboQuantConfig.load(path)
        assert loaded.per_layer_group_size == cfg.per_layer_group_size
        assert loaded.target_snr == cfg.target_snr

    def test_empty_per_layer_group_size_roundtrip(self, tmp_path):
        cfg = TurboQuantConfig(bit_width=4, group_size=128)
        path = tmp_path / "config.json"
        cfg.save(path)
        loaded = TurboQuantConfig.load(path)
        assert loaded.per_layer_group_size == {}
        assert loaded.target_snr is None

    def test_json_keys_with_dots(self, tmp_path):
        """Layer names with dots serialize as JSON string keys."""
        cfg = TurboQuantConfig(
            per_layer_group_size={"model.layers.0.mlp.down_proj": 64},
        )
        path = tmp_path / "config.json"
        cfg.save(path)
        data = json.loads(path.read_text())
        assert "model.layers.0.mlp.down_proj" in data["per_layer_group_size"]


# ---------------------------------------------------------------------------
# 7. quantize_model with explicit per_layer_group_size
# ---------------------------------------------------------------------------

class TestQuantizeModelPerLayerGroupSize:
    def test_explicit_per_layer_group_size(self):
        """Explicit per_layer_group_size overrides the global group_size."""
        model = _make_small_model()
        # Use None (full-row) for the first layer, 32 for the second
        config = TurboQuantConfig(
            bit_width=4,
            group_size=128,
            per_layer_group_size={"0": None, "1": 32},
        )
        q_model = quantize_model(model, config)

        tq_layers = {
            name: m
            for name, m in q_model.named_modules()
            if isinstance(m, TurboQuantLinear)
        }
        assert len(tq_layers) == 2

        # Layer "0": group_size should be full-row (in_features=64)
        assert tq_layers["0"].group_size == 64  # in_features
        assert tq_layers["0"].weight_norms.dim() == 1  # single norm per row

        # Layer "1": group_size should be 32
        assert tq_layers["1"].group_size == 32

    def test_global_group_size_used_when_no_per_layer(self):
        model = _make_small_model()
        config = TurboQuantConfig(bit_width=4, group_size=32)
        q_model = quantize_model(model, config)

        for m in q_model.modules():
            if isinstance(m, TurboQuantLinear):
                assert m.group_size == 32


# ---------------------------------------------------------------------------
# 8. quantize_model with target_snr populates per_layer_group_size
# ---------------------------------------------------------------------------

class TestQuantizeModelTargetSNR:
    def test_target_snr_populates_per_layer_group_size(self):
        """quantize_model with target_snr should fill per_layer_group_size."""
        model = _make_small_model()
        config = TurboQuantConfig(
            bit_width=4,
            group_size=128,
            target_snr=0.0,  # Very low → should select full-row for all
        )
        quantize_model(model, config)

        assert len(config.per_layer_group_size) == 2
        for name, gs in config.per_layer_group_size.items():
            assert gs is None  # Full-row expected for target_snr=0

    def test_target_snr_selected_group_sizes_are_valid(self):
        model = _make_small_model()
        config = TurboQuantConfig(
            bit_width=4,
            group_size=128,
            target_snr=5.0,
        )
        quantize_model(model, config)

        for name, gs in config.per_layer_group_size.items():
            assert gs in DEFAULT_CANDIDATE_SIZES


# ---------------------------------------------------------------------------
# 9. save/load round-trip preserves per-layer group sizes
# ---------------------------------------------------------------------------

class TestSaveLoadPerLayerGroupSize:
    def test_roundtrip_with_per_layer_group_size(self, tmp_path):
        """Saved per_layer_group_size is restored on load."""
        model = _make_small_model()
        config = TurboQuantConfig(
            bit_width=4,
            group_size=128,
            per_layer_group_size={"0": None, "1": 32},
        )
        q_model = quantize_model(model, config)
        save_quantized(q_model, config, tmp_path)

        # Verify the config file contains per_layer_group_size
        saved_cfg = TurboQuantConfig.load(tmp_path / "turboquant_config.json")
        assert saved_cfg.per_layer_group_size == config.per_layer_group_size

    def test_roundtrip_with_target_snr(self, tmp_path):
        """After adaptive selection, saved config captures per-layer decisions."""
        model = _make_small_model()
        config = TurboQuantConfig(
            bit_width=4,
            group_size=128,
            target_snr=0.0,
        )
        q_model = quantize_model(model, config)
        save_quantized(q_model, config, tmp_path)

        saved_cfg = TurboQuantConfig.load(tmp_path / "turboquant_config.json")
        assert len(saved_cfg.per_layer_group_size) == 2
        assert saved_cfg.target_snr == 0.0
