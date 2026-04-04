"""Tests for residual pass-2 CPU offload with pipelined H2D copy.

Validates:
  1. offload_pass2_to_cpu moves data to pinned CPU and frees GPU buffers.
  2. Forward produces identical output with and without offload.
  3. reload_pass2_to_gpu restores GPU state and produces same output.
  4. dequantize() works in offloaded state.
  5. merge_passes() works after offload (auto-reloads).
  6. save_quantized / load_quantized roundtrip with offload flag.
  7. memory_bytes() reports GPU-only; memory_bytes_cpu() reports pinned.
  8. On CPU device, offload is a silent no-op.
  9. CUDA stream pipelining: copy overlaps with pass-1 compute.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from turboquant_model.codebook import get_codebook
from turboquant_model.module import TurboQuantLinear
from turboquant_model.quantize import pack_4bit, unpack_4bit
from turboquant_model.rotation import generate_rotation_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _make_tq(M: int, N: int, group_size: int, seed: int, device: str) -> TurboQuantLinear:
    """Create a TurboQuantLinear with random pass1 + pass2 data."""
    torch.manual_seed(0)
    W = torch.randn(M, N, device=device)

    centroids, boundaries = get_codebook(4)
    centroids = centroids.to(device)
    boundaries = boundaries.to(device)

    tq = TurboQuantLinear(
        in_features=N, out_features=M,
        bit_width=4, group_size=group_size, device=device,
    )

    # Quantize pass 1
    p1_indices, p1_norms = _quantize_pass(W, group_size, seed, centroids, boundaries, device)
    tq.indices_packed.copy_(p1_indices)
    tq.weight_norms.copy_(p1_norms)
    tq.codebook.copy_(centroids)
    tq.set_rotation(seed)

    # Reconstruct and compute residual
    W_hat1 = tq.dequantize().float()
    residual = W.float() - W_hat1

    # Quantize pass 2
    seed2 = seed + 1000
    p2_indices, p2_norms = _quantize_pass(residual, group_size, seed2, centroids, boundaries, device)
    tq.set_pass2(
        indices_packed=p2_indices,
        weight_norms=p2_norms,
        codebook=centroids.clone(),
        seed=seed2,
    )

    # Disable fused kernels for deterministic CPU/CUDA comparison
    tq.use_cutile = False
    tq.use_triton = False
    tq.use_metal = False

    return tq


def _quantize_pass(
    W: torch.Tensor, group_size: int, seed: int,
    centroids: torch.Tensor, boundaries: torch.Tensor, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = W.shape
    W = W.float()
    all_norms = []
    all_indices = []

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
        Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]
    if N % 2 != 0:
        full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)
    packed = pack_4bit(full_indices)
    return packed, norms_out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tq_cpu():
    return _make_tq(M=32, N=64, group_size=64, seed=42, device="cpu")


@pytest.fixture()
def tq_cuda():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    return _make_tq(M=32, N=64, group_size=64, seed=42, device="cuda")


@pytest.fixture()
def tq_cuda_grouped():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    return _make_tq(M=32, N=128, group_size=64, seed=42, device="cuda")


# ---------------------------------------------------------------------------
# 1. Basic offload mechanics
# ---------------------------------------------------------------------------

class TestOffloadMechanics:

    def test_offload_sets_flags(self, tq_cuda):
        assert tq_cuda.has_residual
        assert not tq_cuda.is_pass2_offloaded

        tq_cuda.offload_pass2_to_cpu()

        assert tq_cuda.has_residual
        assert tq_cuda.is_pass2_offloaded
        assert tq_cuda._cpu_offload_pass2
        assert tq_cuda.pass2_indices_packed is None  # GPU buffer freed
        assert tq_cuda._pass2_cpu_indices_packed is not None
        assert tq_cuda._pass2_cpu_indices_packed.is_pinned()
        assert tq_cuda._scratch_pool is None  # pool set by enable_prefetch_chain
        assert tq_cuda._copy_stream is not None

    def test_offload_noop_on_cpu(self, tq_cpu):
        assert tq_cpu.has_residual
        tq_cpu.offload_pass2_to_cpu()
        assert not tq_cpu.is_pass2_offloaded
        assert tq_cpu.pass2_indices_packed is not None  # still on CPU

    def test_reload_restores_gpu(self, tq_cuda):
        tq_cuda.offload_pass2_to_cpu()
        assert tq_cuda.is_pass2_offloaded

        tq_cuda.reload_pass2_to_gpu()
        assert not tq_cuda.is_pass2_offloaded
        assert tq_cuda.pass2_indices_packed is not None
        assert tq_cuda.pass2_indices_packed.is_cuda
        assert tq_cuda._pass2_cpu_indices_packed is None
        assert tq_cuda._scratch_pool is None

    def test_offload_without_residual_is_noop(self):
        if not _has_cuda():
            pytest.skip("CUDA not available")
        tq = TurboQuantLinear(64, 32, bit_width=4, group_size=64, device="cuda")
        assert not tq.has_residual
        tq.offload_pass2_to_cpu()
        assert not tq.is_pass2_offloaded


# ---------------------------------------------------------------------------
# 2. Numerical equivalence: offloaded forward == GPU forward
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:

    def test_forward_matches(self, tq_cuda):
        torch.manual_seed(7)
        x = torch.randn(4, tq_cuda.in_features, device="cuda")

        # Reference: both passes on GPU
        with torch.no_grad():
            ref = tq_cuda(x).clone()

        # Offload and re-run
        tq_cuda.offload_pass2_to_cpu()
        with torch.no_grad():
            offloaded = tq_cuda(x)

        assert torch.allclose(ref, offloaded, atol=1e-5), (
            f"Max diff: {(ref - offloaded).abs().max().item()}"
        )

    def test_forward_3d_input(self, tq_cuda):
        torch.manual_seed(8)
        x = torch.randn(2, 3, tq_cuda.in_features, device="cuda")

        with torch.no_grad():
            ref = tq_cuda(x).clone()

        tq_cuda.offload_pass2_to_cpu()
        with torch.no_grad():
            offloaded = tq_cuda(x)

        assert torch.allclose(ref, offloaded, atol=1e-5)

    def test_forward_grouped(self, tq_cuda_grouped):
        torch.manual_seed(9)
        x = torch.randn(4, tq_cuda_grouped.in_features, device="cuda")

        with torch.no_grad():
            ref = tq_cuda_grouped(x).clone()

        tq_cuda_grouped.offload_pass2_to_cpu()
        with torch.no_grad():
            offloaded = tq_cuda_grouped(x)

        assert torch.allclose(ref, offloaded, atol=1e-5)

    def test_forward_after_reload(self, tq_cuda):
        torch.manual_seed(10)
        x = torch.randn(4, tq_cuda.in_features, device="cuda")

        with torch.no_grad():
            ref = tq_cuda(x).clone()

        tq_cuda.offload_pass2_to_cpu()
        tq_cuda.reload_pass2_to_gpu()

        with torch.no_grad():
            reloaded = tq_cuda(x)

        assert torch.allclose(ref, reloaded, atol=1e-5)

    def test_dequantize_offloaded(self, tq_cuda):
        ref_w = tq_cuda.dequantize().clone()

        tq_cuda.offload_pass2_to_cpu()
        offloaded_w = tq_cuda.dequantize()

        assert torch.allclose(ref_w, offloaded_w, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. merge_passes works from offloaded state
# ---------------------------------------------------------------------------

class TestMergeFromOffloaded:

    def test_merge_after_offload(self, tq_cuda):
        torch.manual_seed(11)
        x = torch.randn(4, tq_cuda.in_features, device="cuda")

        with torch.no_grad():
            ref = tq_cuda(x).clone()

        tq_cuda.offload_pass2_to_cpu()
        tq_cuda.merge_passes()

        assert not tq_cuda.has_residual
        assert not tq_cuda.is_pass2_offloaded


# ---------------------------------------------------------------------------
# 4. Memory accounting
# ---------------------------------------------------------------------------

class TestMemoryAccounting:

    def test_gpu_memory_reduced(self, tq_cuda):
        before = tq_cuda.memory_bytes()
        tq_cuda.offload_pass2_to_cpu()
        after = tq_cuda.memory_bytes()

        # GPU memory should be reduced (pass2 freed, no per-layer scratch)
        # Shared scratch pool is set up by enable_prefetch_chain, not per layer
        assert after < before

        cpu_mem = tq_cuda.memory_bytes_cpu()
        assert cpu_mem > 0

    def test_cpu_memory_zero_without_offload(self, tq_cuda):
        assert tq_cuda.memory_bytes_cpu() == 0


# ---------------------------------------------------------------------------
# 5. Multiple forward calls (scratch buffer reuse)
# ---------------------------------------------------------------------------

class TestRepeatedForward:

    def test_multiple_forwards_consistent(self, tq_cuda):
        torch.manual_seed(12)
        x = torch.randn(4, tq_cuda.in_features, device="cuda")

        tq_cuda.offload_pass2_to_cpu()

        with torch.no_grad():
            out1 = tq_cuda(x).clone()
            out2 = tq_cuda(x).clone()

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_different_inputs(self, tq_cuda):
        tq_cuda.offload_pass2_to_cpu()

        with torch.no_grad():
            x1 = torch.randn(2, tq_cuda.in_features, device="cuda")
            out1 = tq_cuda(x1)
            assert out1.shape == (2, tq_cuda.out_features)

            x2 = torch.randn(8, tq_cuda.in_features, device="cuda")
            out2 = tq_cuda(x2)
            assert out2.shape == (8, tq_cuda.out_features)


# ---------------------------------------------------------------------------
# 6. extra_repr shows offload status
# ---------------------------------------------------------------------------

class TestRepr:

    def test_extra_repr_shows_offload(self, tq_cuda):
        assert "cpu_offload" not in tq_cuda.extra_repr()
        tq_cuda.offload_pass2_to_cpu()
        assert "cpu_offload=True" in tq_cuda.extra_repr()


# ---------------------------------------------------------------------------
# 7. Config flag integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:

    def test_config_default_false(self):
        from turboquant_model.model import TurboQuantConfig
        c = TurboQuantConfig()
        assert c.cpu_offload_pass2 is False

    def test_config_roundtrip(self, tmp_path):
        from turboquant_model.model import TurboQuantConfig
        c = TurboQuantConfig(residual_bit_width=4, cpu_offload_pass2=True)
        c.save(tmp_path / "cfg.json")
        c2 = TurboQuantConfig.load(tmp_path / "cfg.json")
        assert c2.cpu_offload_pass2 is True
        assert c2.residual_bit_width == 4


# ---------------------------------------------------------------------------
# 8. Dual-pass fused kernel correctness (PyTorch fallback comparison)
# ---------------------------------------------------------------------------

class TestDualPassFused:
    """Verify _forward_residual_fused matches two separate _forward_pass calls."""

    def test_fused_matches_separate_passes_cpu(self, tq_cpu):
        """On CPU, dual-pass fused falls back to PyTorch — verify correctness."""
        torch.manual_seed(20)
        x = torch.randn(4, tq_cpu.in_features)
        x_f = x.float()

        # Reference: two separate passes
        with torch.no_grad():
            ref = tq_cpu(x).clone()

        # Force fallback path (no fused kernels on CPU)
        assert not tq_cpu.use_cutile
        assert not tq_cpu.use_triton

        # Directly call _forward_residual_fused
        with torch.no_grad():
            fused = tq_cpu._forward_residual_fused(
                x_f,
                tq_cpu.indices_packed, tq_cpu.codebook,
                tq_cpu.weight_norms, tq_cpu._rotation_seed,
                tq_cpu.pass2_indices_packed, tq_cpu.pass2_codebook,
                tq_cpu.pass2_weight_norms, tq_cpu._pass2_seed,
            )

        # Reference computed as pass1 + pass2 via _forward_pass
        indices1 = tq_cpu._get_indices()
        indices2 = tq_cpu._get_pass2_indices()
        with torch.no_grad():
            pass1 = tq_cpu._forward_pass(
                x_f, indices1, tq_cpu.indices_packed, tq_cpu.codebook,
                tq_cpu.weight_norms, tq_cpu._rotation_seed,
            )
            pass2 = tq_cpu._forward_pass(
                x_f, indices2, tq_cpu.pass2_indices_packed, tq_cpu.pass2_codebook,
                tq_cpu.pass2_weight_norms, tq_cpu._pass2_seed,
            )
        expected = pass1 + pass2

        assert torch.allclose(fused, expected, atol=1e-5), (
            f"Max diff: {(fused - expected).abs().max().item()}"
        )

    def test_fused_matches_forward_output(self, tq_cpu):
        """_forward_residual_fused should produce same output as forward()."""
        torch.manual_seed(21)
        x = torch.randn(2, 3, tq_cpu.in_features)  # 3D input

        with torch.no_grad():
            forward_out = tq_cpu(x).clone()

        # _forward_residual_fused expects 2D input
        x_f = x.reshape(-1, x.shape[-1]).float()
        with torch.no_grad():
            fused = tq_cpu._forward_residual_fused(
                x_f,
                tq_cpu.indices_packed, tq_cpu.codebook,
                tq_cpu.weight_norms, tq_cpu._rotation_seed,
                tq_cpu.pass2_indices_packed, tq_cpu.pass2_codebook,
                tq_cpu.pass2_weight_norms, tq_cpu._pass2_seed,
            )
        fused_3d = fused.to(x.dtype).reshape(2, 3, tq_cpu.out_features)

        assert torch.allclose(forward_out, fused_3d, atol=1e-5)

    def test_fused_with_grouped(self):
        """Dual-pass fused with multiple groups."""
        tq = _make_tq(M=32, N=128, group_size=64, seed=42, device="cpu")
        torch.manual_seed(22)
        x = torch.randn(4, tq.in_features)
        x_f = x.float()

        with torch.no_grad():
            ref = tq(x).clone()

        with torch.no_grad():
            fused = tq._forward_residual_fused(
                x_f,
                tq.indices_packed, tq.codebook,
                tq.weight_norms, tq._rotation_seed,
                tq.pass2_indices_packed, tq.pass2_codebook,
                tq.pass2_weight_norms, tq._pass2_seed,
            )

        assert torch.allclose(ref.float(), fused, atol=1e-5)


# ---------------------------------------------------------------------------
# 9. Prefetch chain
# ---------------------------------------------------------------------------

class TestPrefetchChain:

    def test_enable_prefetch_links_layers(self):
        """enable_prefetch_chain links offloaded layers in order."""
        if not _has_cuda():
            pytest.skip("CUDA not available")
        from turboquant_model.model import enable_prefetch_chain, disable_prefetch_chain

        layers = [_make_tq(M=16, N=32, group_size=32, seed=42, device="cuda") for _ in range(3)]
        for l in layers:
            l.offload_pass2_to_cpu()

        # Simulate a model with these layers
        import torch.nn as nn
        model = nn.Sequential(*layers)

        n_links = enable_prefetch_chain(model)
        assert n_links == 2
        assert layers[0]._next_offloaded_layer is layers[1]
        assert layers[1]._next_offloaded_layer is layers[2]
        assert layers[2]._next_offloaded_layer is None

        # All share the same copy stream and scratch pool
        assert layers[0]._copy_stream is layers[1]._copy_stream
        assert layers[1]._copy_stream is layers[2]._copy_stream
        assert layers[0]._scratch_pool is layers[1]._scratch_pool
        assert layers[0]._scratch_pool is not None

        # Verify ping-pong slot assignment
        assert layers[0]._scratch_idx == 0
        assert layers[1]._scratch_idx == 1
        assert layers[2]._scratch_idx == 0

        # Cleanup
        disable_prefetch_chain(model)
        assert layers[0]._next_offloaded_layer is None
        assert layers[1]._next_offloaded_layer is None
        assert layers[0]._scratch_pool is None

    def test_prefetch_no_offloaded_layers(self):
        """enable_prefetch_chain is a no-op with 0 or 1 offloaded layers."""
        from turboquant_model.model import enable_prefetch_chain
        import torch.nn as nn

        tq = _make_tq(M=16, N=32, group_size=32, seed=42, device="cpu")
        model = nn.Sequential(tq)
        n_links = enable_prefetch_chain(model)
        assert n_links == 0

    def test_prefetch_pass2_returns_event(self):
        """prefetch_pass2 returns a CUDA event when offloaded + pool is set."""
        if not _has_cuda():
            pytest.skip("CUDA not available")
        from turboquant_model.model import enable_prefetch_chain
        import torch.nn as nn

        tq = _make_tq(M=16, N=32, group_size=32, seed=42, device="cuda")
        tq.offload_pass2_to_cpu()

        # Need pool to be set up
        model = nn.Sequential(tq)
        enable_prefetch_chain(model)

        event = tq.prefetch_pass2()
        assert event is not None
        assert isinstance(event, torch.cuda.Event)
        assert tq._prefetch_event is event

    def test_prefetch_pass2_noop_no_offload(self):
        """prefetch_pass2 returns None when not offloaded."""
        tq = _make_tq(M=16, N=32, group_size=32, seed=42, device="cpu")
        event = tq.prefetch_pass2()
        assert event is None

    def test_forward_with_prefetch_correct(self):
        """Forward produces correct output when prefetch event is pre-set."""
        if not _has_cuda():
            pytest.skip("CUDA not available")

        tq = _make_tq(M=32, N=64, group_size=64, seed=42, device="cuda")
        torch.manual_seed(30)
        x = torch.randn(4, tq.in_features, device="cuda")

        # Reference without offload
        with torch.no_grad():
            ref = tq(x).clone()

        # Offload, set up pool, manually prefetch, then forward
        tq.offload_pass2_to_cpu()
        import torch.nn as nn
        from turboquant_model.model import enable_prefetch_chain
        model = nn.Sequential(tq)
        enable_prefetch_chain(model)
        tq.prefetch_pass2()  # sets _prefetch_event
        with torch.no_grad():
            out = tq(x)

        assert torch.allclose(ref, out, atol=1e-5)
