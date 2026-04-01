"""Test entropy coding integration in save/load pipeline.

Validates:
  1. save_quantized with entropy_coding=True produces _ec keys
  2. Round-trip: save(EC) → load tensors → indices match originals
  3. File size reduction vs non-EC save
  4. Residual pass also compresses correctly

Run: python tests/test_entropy_save_load.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "src")

from turboquant_model.model import (
    TurboQuantConfig,
    save_quantized,
    _quantize_weight,
    _replace_module,
    _entropy_compress_indices,
    _entropy_decompress_indices,
)
from turboquant_model.module import TurboQuantLinear
from turboquant_model.codebook import get_codebook
from turboquant_model.quantize import pack_4bit, unpack_4bit


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(128)
        self.linear1 = nn.Linear(128, 256, bias=False)
        self.linear2 = nn.Linear(256, 128, bias=True)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        return self.linear2(x)


def _quantize_small(config):
    torch.manual_seed(42)
    model = SmallModel()
    centroids, boundaries = get_codebook(config.bit_width)

    for name in ["linear1", "linear2"]:
        module = getattr(model, name)
        W = module.weight.data
        M, N = W.shape
        group_size = config.group_size or N

        packed, norms, _ = _quantize_weight(
            W, config.bit_width, group_size, config.seed,
            centroids, boundaries, W.device, rotation=config.rotation,
        )
        tq = TurboQuantLinear(
            in_features=N, out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=W.device,
            rotation=config.rotation,
        )
        tq.indices_packed.copy_(packed)
        tq.weight_norms.copy_(norms)
        tq.codebook.copy_(centroids.to(W.device))
        tq.set_rotation(config.seed)
        if module.bias is not None:
            tq.bias.copy_(module.bias.data)
        _replace_module(model, name, tq)

    return model


# -------------------------------------------------------
# Test 1: Low-level compress/decompress round-trip
# -------------------------------------------------------

def test_compress_decompress_roundtrip():
    print("TEST 1: Low-level compress/decompress round-trip")
    torch.manual_seed(0)
    centroids, boundaries = get_codebook(4)

    W = torch.randn(64, 128) * 0.02
    from turboquant_model.rotation import hadamard_rotate
    import math
    Y = hadamard_rotate(W / W.norm(dim=1, keepdim=True).clamp(min=1e-8), seed=42)
    Y_scaled = Y * math.sqrt(128)
    idx = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
    idx = idx.clamp(0, 15).reshape(64, 128)
    packed = pack_4bit(idx)

    # Compress
    ec = _entropy_compress_indices(packed, 4, 128)
    # Decompress
    packed_back = _entropy_decompress_indices(ec, 4, 64, 128)

    assert (packed == packed_back).all(), "FAIL: packed indices mismatch"
    print(f"  ✓ 64x128 packed indices match")
    print(f"    Original: {packed.numel()} bytes, Compressed: {ec.numel()} bytes "
          f"({ec.numel() / packed.numel() * 100:.1f}%)")


# -------------------------------------------------------
# Test 2: save_quantized produces _ec keys
# -------------------------------------------------------

def test_save_ec_keys():
    print("\nTEST 2: save_quantized with entropy_coding=True produces _ec keys")
    config = TurboQuantConfig(
        bit_width=4, group_size=128, seed=42,
        rotation="hadamard", entropy_coding=True,
    )
    model = _quantize_small(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_quantized(model, config, tmpdir)
        from safetensors.torch import load_file
        tensors = load_file(str(Path(tmpdir) / "model.safetensors"))

        # Should have _ec keys, not plain indices
        assert "linear1.indices_ec" in tensors, "Missing linear1.indices_ec"
        assert "linear1.indices_ec_shape" in tensors, "Missing linear1.indices_ec_shape"
        assert "linear1.indices" not in tensors, "Should NOT have uncompressed indices"
        assert "linear2.indices_ec" in tensors, "Missing linear2.indices_ec"
        assert "codebook" in tensors, "Missing codebook"
        assert "linear2.bias" in tensors, "Missing bias"

        print(f"  ✓ Found expected _ec keys in safetensors")

        # Check shape metadata
        shape = tensors["linear1.indices_ec_shape"]
        assert shape[0] == 256 and shape[1] == 128, f"Bad shape: {shape}"
        print(f"  ✓ Shape metadata correct: {shape.tolist()}")


# -------------------------------------------------------
# Test 3: File size reduction
# -------------------------------------------------------

def test_file_size_reduction():
    print("\nTEST 3: File size savings with entropy_coding=True")
    config_plain = TurboQuantConfig(
        bit_width=4, group_size=128, seed=42,
        rotation="hadamard", entropy_coding=False,
    )
    config_ec = TurboQuantConfig(
        bit_width=4, group_size=128, seed=42,
        rotation="hadamard", entropy_coding=True,
    )
    model_plain = _quantize_small(config_plain)
    model_ec = _quantize_small(config_ec)

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_plain = Path(tmpdir) / "plain"
        dir_ec = Path(tmpdir) / "ec"
        save_quantized(model_plain, config_plain, dir_plain)
        save_quantized(model_ec, config_ec, dir_ec)

        size_plain = (dir_plain / "model.safetensors").stat().st_size
        size_ec = (dir_ec / "model.safetensors").stat().st_size

        savings_pct = (1 - size_ec / size_plain) * 100
        print(f"  Plain:  {size_plain:>8,d} bytes")
        print(f"  EC:     {size_ec:>8,d} bytes")
        print(f"  Savings: {savings_pct:.1f}%")
        assert size_ec < size_plain, "EC should be smaller"
        print(f"  ✓ EC file is smaller")


# -------------------------------------------------------
# Test 4: Full round-trip (save EC → load tensors → decompress → match)
# -------------------------------------------------------

def test_full_roundtrip():
    print("\nTEST 4: Full save/load round-trip with EC")
    config = TurboQuantConfig(
        bit_width=4, group_size=128, seed=42,
        rotation="hadamard", entropy_coding=True,
    )
    model = _quantize_small(config)

    # Grab original packed indices before saving
    orig_packed = {}
    for name, mod in model.named_modules():
        if isinstance(mod, TurboQuantLinear):
            orig_packed[name] = mod.indices_packed.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_quantized(model, config, tmpdir)

        # Manually load and decompress to verify
        from safetensors.torch import load_file
        tensors = load_file(str(Path(tmpdir) / "model.safetensors"))

        for name in orig_packed:
            safe = name.replace(".", "_")
            ec_key = f"{safe}.indices_ec"
            shape_key = f"{safe}.indices_ec_shape"

            shape_t = tensors[shape_key]
            M, N = int(shape_t[0]), int(shape_t[1])
            recovered = _entropy_decompress_indices(
                tensors[ec_key], config.bit_width, M, N,
            )

            assert (orig_packed[name] == recovered).all(), (
                f"Round-trip mismatch for {name}"
            )
            print(f"  ✓ {name}: indices match after EC round-trip")


# -------------------------------------------------------
# Test 5: Dequantized weights match
# -------------------------------------------------------

def test_dequantized_weights_match():
    print("\nTEST 5: Dequantized weights identical before/after EC save")
    config = TurboQuantConfig(
        bit_width=4, group_size=128, seed=42,
        rotation="hadamard", entropy_coding=True,
    )
    model = _quantize_small(config)

    # Get dequantized weights before save
    orig_deq = {}
    for name, mod in model.named_modules():
        if isinstance(mod, TurboQuantLinear):
            orig_deq[name] = mod.dequantize().clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_quantized(model, config, tmpdir)

        # Load tensors and manually reconstruct
        from safetensors.torch import load_file
        tensors = load_file(str(Path(tmpdir) / "model.safetensors"))

        for name in orig_deq:
            safe = name.replace(".", "_")
            shape_t = tensors[f"{safe}.indices_ec_shape"]
            M, N = int(shape_t[0]), int(shape_t[1])
            packed = _entropy_decompress_indices(
                tensors[f"{safe}.indices_ec"], config.bit_width, M, N,
            )

            # Reconstruct TQ module
            tq = TurboQuantLinear(
                in_features=N, out_features=M, bias=False,
                bit_width=config.bit_width, group_size=config.group_size,
                rotation=config.rotation,
            )
            tq.indices_packed = packed
            tq.weight_norms = tensors[f"{safe}.norms"]
            tq.codebook = tensors["codebook"]
            tq.set_rotation(config.seed)

            w_recon = tq.dequantize()
            diff = (orig_deq[name].float() - w_recon.float()).abs().max().item()
            print(f"  ✓ {name}: max dequantize diff = {diff:.2e}")
            assert diff < 1e-5, f"Dequantized mismatch for {name}: {diff}"


def main():
    print("=" * 60)
    print("  ENTROPY CODING SAVE/LOAD INTEGRATION TEST")
    print("=" * 60 + "\n")

    test_compress_decompress_roundtrip()
    test_save_ec_keys()
    test_file_size_reduction()
    test_full_roundtrip()
    test_dequantized_weights_match()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
