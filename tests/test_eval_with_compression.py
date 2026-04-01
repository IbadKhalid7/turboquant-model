"""E2E eval test: save with norm_compression + entropy_coding → load → forward match.

Verifies the full round-trip:
  1. Quantize model with norm_compression + entropy_coding
  2. Capture pre-save forward outputs
  3. Save to disk
  4. Load tensors from disk (simulating load_quantized)
  5. Reconstruct TQ modules from loaded tensors
  6. Verify forward outputs match exactly

Run: python tests/test_eval_with_compression.py
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
    _entropy_decompress_indices,
)
from turboquant_model.module import TurboQuantLinear
from turboquant_model.codebook import get_codebook
from turboquant_model.norm_compression import FactoredNorms, reconstruct_norms


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(256)
        self.up = nn.Linear(256, 512, bias=False)
        self.gate = nn.Linear(256, 512, bias=False)
        self.down = nn.Linear(512, 256, bias=True)
        self.norm2 = nn.LayerNorm(256)

    def forward(self, x):
        h = self.norm1(x)
        h = self.up(h) * torch.sigmoid(self.gate(h))
        h = self.down(h)
        return self.norm2(x + h)


def quantize_model_tiny(config):
    torch.manual_seed(42)
    model = TinyModel()
    centroids, boundaries = get_codebook(config.bit_width)

    for name in ["up", "gate", "down"]:
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
        tq.use_cutile = False
        tq.use_triton = False

        if module.bias is not None:
            tq.bias.copy_(module.bias.data)

        if config.norm_codec in ("factored_int8", "factored_int4") and norms.dim() == 2:
            tq.apply_norm_codec(config.norm_codec)

        _replace_module(model, name, tq)

    return model


def load_from_tensors(tensors: dict, config: TurboQuantConfig, ref_model: nn.Module):
    """Reconstruct a quantized model from loaded safetensors (simulates load_quantized)."""
    codebook = tensors["codebook"]

    for name in ["up", "gate", "down"]:
        module = getattr(ref_model, name)
        if not isinstance(module, nn.Linear):
            continue

        safe = name
        M, N = module.weight.shape
        group_size = config.group_size or N

        tq = TurboQuantLinear(
            in_features=N, out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            rotation=config.rotation,
        )
        tq.use_cutile = False
        tq.use_triton = False

        # Load indices (EC or plain)
        ec_key = f"{safe}.indices_ec"
        if ec_key in tensors:
            shape_t = tensors[f"{safe}.indices_ec_shape"]
            ec_M, ec_N = int(shape_t[0]), int(shape_t[1])
            tq.indices_packed = _entropy_decompress_indices(
                tensors[ec_key], config.bit_width, ec_M, ec_N,
            )
        else:
            tq.indices_packed = tensors[f"{safe}.indices"]

        # Load norms (factored or full)
        norms_row_key = f"{safe}.norms.row_scale"
        norms_full_key = f"{safe}.norms"
        if norms_row_key in tensors:
            res_bits_key = f"{safe}.norms.residual_bits"
            res_bits = int(tensors[res_bits_key][0]) if res_bits_key in tensors else 8
            fn = FactoredNorms(
                row_scale=tensors[f"{safe}.norms.row_scale"],
                group_scale=tensors[f"{safe}.norms.group_scale"],
                residual_int8=tensors[f"{safe}.norms.residual"],
                residual_scale=float(tensors[f"{safe}.norms.residual_scale"][0]),
                residual_bits=res_bits,
            )
            tq.weight_norms = reconstruct_norms(fn)
        elif norms_full_key in tensors:
            tq.weight_norms = tensors[norms_full_key].float()

        tq.codebook = codebook
        tq.set_rotation(config.seed)

        if module.bias is not None:
            bias_key = f"{safe}.bias"
            if bias_key in tensors:
                tq.bias = tensors[bias_key]

        _replace_module(ref_model, name, tq)

    return ref_model


def main():
    print("=" * 70)
    print("  EVAL TEST: save with compression → load → forward match")
    print("=" * 70 + "\n")

    configs = [
        ("Baseline (fp32 norms, no EC)", TurboQuantConfig(
            bit_width=4, group_size=128, seed=42, rotation="hadamard",
            norm_codec="fp32", entropy_coding=False,
        )),
        ("Norm int8 only", TurboQuantConfig(
            bit_width=4, group_size=128, seed=42, rotation="hadamard",
            norm_codec="factored_int8", entropy_coding=False,
        )),
        ("Norm int4 only", TurboQuantConfig(
            bit_width=4, group_size=128, seed=42, rotation="hadamard",
            norm_codec="factored_int4", entropy_coding=False,
        )),
        ("Entropy coding only", TurboQuantConfig(
            bit_width=4, group_size=128, seed=42, rotation="hadamard",
            norm_codec="fp32", entropy_coding=True,
        )),
        ("Both int8 + EC", TurboQuantConfig(
            bit_width=4, group_size=128, seed=42, rotation="hadamard",
            norm_codec="factored_int8", entropy_coding=True,
        )),
        ("Both int4 + EC", TurboQuantConfig(
            bit_width=4, group_size=128, seed=42, rotation="hadamard",
            norm_codec="factored_int4", entropy_coding=True,
        )),
    ]

    x = torch.randn(2, 8, 256)  # test input

    for label, config in configs:
        print(f"--- {label} ---")

        # 1. Quantize
        model = quantize_model_tiny(config)

        # 2. Pre-save forward
        with torch.no_grad():
            out_before = model(x).clone()

        # 3. Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "model"
            save_quantized(model, config, save_dir)

            # 4. Load tensors
            from safetensors.torch import load_file
            tensors = load_file(str(save_dir / "model.safetensors"))
            non_q = load_file(str(save_dir / "non_quantized.safetensors"))

            # 5. Reconstruct model from scratch
            torch.manual_seed(42)
            loaded_model = TinyModel()

            # Load non-quantized params
            for pname, tensor in non_q.items():
                parts = pname.split(".")
                parent = loaded_model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                target = getattr(parent, parts[-1], None)
                if target is not None:
                    if isinstance(target, nn.Parameter):
                        target.data.copy_(tensor)
                    elif isinstance(target, torch.Tensor):
                        target.copy_(tensor)

            loaded_model = load_from_tensors(tensors, config, loaded_model)

            # 6. Post-load forward
            with torch.no_grad():
                out_after = loaded_model(x)

            # Compare
            max_diff = (out_before - out_after).abs().max().item()
            match = max_diff < 1e-4

            # File size
            model_bytes = (save_dir / "model.safetensors").stat().st_size
            print(f"  Save size:   {model_bytes:>10,d} bytes")
            print(f"  Max diff:    {max_diff:.2e}")
            print(f"  Match:       {'✓' if match else '✗ FAIL'}")

            if not match:
                # Debug: check individual layers
                for name in ["up", "gate", "down"]:
                    orig_mod = getattr(model, name)
                    loaded_mod = getattr(loaded_model, name)
                    if isinstance(orig_mod, TurboQuantLinear) and isinstance(loaded_mod, TurboQuantLinear):
                        w_orig = orig_mod.dequantize()
                        w_loaded = loaded_mod.dequantize()
                        layer_diff = (w_orig - w_loaded).abs().max().item()
                        idx_match = (orig_mod.indices_packed == loaded_mod.indices_packed).all()
                        print(f"    {name}: weight diff={layer_diff:.2e}, "
                              f"indices match={idx_match}")

            assert match, f"Forward mismatch for {label}: max diff = {max_diff}"
            print()

    print("=" * 70)
    print("  ALL EVAL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
