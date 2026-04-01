"""E2E test: 4-bit baseline vs 4-bit + norm_compression + entropy_coding.

Compares file sizes, dequantized weight fidelity, and forward pass
correctness across three configurations:
  A) Baseline: 4-bit, fp32 norms, no entropy coding
  B) + norm compression: factored_int8 norms
  C) + norm compression + entropy coding

Uses a small synthetic model (no HF download needed).

Run: python tests/test_e2e_compression.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "src")

from turboquant_model.model import (
    TurboQuantConfig,
    quantize_model_advanced,
    save_quantized,
    _quantize_weight,
    _replace_module,
)
from turboquant_model.module import TurboQuantLinear
from turboquant_model.codebook import get_codebook


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Synthetic model with multiple linear layers for realistic testing."""
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


def quantize_tiny(config: TurboQuantConfig) -> nn.Module:
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

        if module.bias is not None:
            tq.bias.copy_(module.bias.data)

        # Disable fused kernels (CPU only)
        tq.use_cutile = False
        tq.use_triton = False

        # Apply norm compression if configured
        if config.norm_codec != "fp32" and norms.dim() == 2:
            tq.apply_norm_codec(config.norm_codec)

        _replace_module(model, name, tq)

    return model


def sqnr(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    signal = W.float().pow(2).mean()
    noise = (W.float() - W_hat.float()).pow(2).mean()
    if noise < 1e-30:
        return 999.0
    return 10 * math.log10(signal.item() / noise.item())


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  E2E COMPARISON: baseline vs norm_compression vs +entropy_coding")
    print("=" * 70 + "\n")

    configs = {
        "A) Baseline (4-bit, fp32 norms)": TurboQuantConfig(
            bit_width=4, group_size=128, seed=42,
            rotation="hadamard",
            norm_codec="fp32", entropy_coding=False,
        ),
        "B) + norm_compression (factored_int8)": TurboQuantConfig(
            bit_width=4, group_size=128, seed=42,
            rotation="hadamard",
            norm_codec="factored_int8", entropy_coding=False,
        ),
        "C) + norm_compression + entropy_coding": TurboQuantConfig(
            bit_width=4, group_size=128, seed=42,
            rotation="hadamard",
            norm_codec="factored_int8", entropy_coding=True,
        ),
    }

    # Reference weights (before quantization)
    torch.manual_seed(42)
    ref_model = TinyModel()
    ref_weights = {
        name: getattr(ref_model, name).weight.data.clone()
        for name in ["up", "gate", "down"]
    }

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for label, config in configs.items():
            print(f"--- {label} ---")
            model = quantize_tiny(config)

            # Save
            save_dir = Path(tmpdir) / label.split(")")[0].strip()
            save_quantized(model, config, save_dir)

            # Measure file sizes
            model_st = save_dir / "model.safetensors"
            non_q_st = save_dir / "non_quantized.safetensors"
            model_bytes = model_st.stat().st_size if model_st.exists() else 0
            non_q_bytes = non_q_st.stat().st_size if non_q_st.exists() else 0
            total_bytes = model_bytes + non_q_bytes

            # Dequantized weight SQNR
            sqnrs = {}
            for name in ["up", "gate", "down"]:
                mod = getattr(model, name)
                if isinstance(mod, TurboQuantLinear):
                    w_deq = mod.dequantize()
                    sqnrs[name] = sqnr(ref_weights[name], w_deq)

            avg_sqnr = sum(sqnrs.values()) / len(sqnrs)

            # Forward pass
            x = torch.randn(2, 8, 256)
            with torch.no_grad():
                out = model(x)
            out_ok = out.shape == (2, 8, 256) and not out.isnan().any()

            print(f"  model.safetensors:       {model_bytes:>10,d} bytes")
            print(f"  non_quantized.safetensors: {non_q_bytes:>10,d} bytes")
            print(f"  Total:                   {total_bytes:>10,d} bytes")
            print(f"  Avg SQNR:                {avg_sqnr:>10.2f} dB")
            print(f"  Forward pass:            {'✓ OK' if out_ok else '✗ FAIL'}")
            print()

            results[label] = {
                "model_bytes": model_bytes,
                "non_q_bytes": non_q_bytes,
                "total_bytes": total_bytes,
                "avg_sqnr": round(avg_sqnr, 2),
                "forward_ok": out_ok,
            }

        # Summary
        print("=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"\n  {'Config':<45s} {'Size':>10s} {'Savings':>10s} {'SQNR':>10s}")
        print(f"  {'-' * 77}")

        baseline_bytes = results[list(results.keys())[0]]["total_bytes"]
        for label, r in results.items():
            short = label.split(")")[1].strip() or "Baseline"
            savings = (1 - r["total_bytes"] / baseline_bytes) * 100
            print(f"  {short:<45s} {r['total_bytes']:>10,d} "
                  f"{savings:>9.1f}% {r['avg_sqnr']:>10.2f} dB")

        # Assertions
        print()
        a = results[list(results.keys())[0]]
        b = results[list(results.keys())[1]]
        c = results[list(results.keys())[2]]

        assert b["total_bytes"] < a["total_bytes"], "B should be smaller than A"
        assert c["total_bytes"] < b["total_bytes"], "C should be smaller than B"
        assert c["total_bytes"] < a["total_bytes"], "C should be smaller than A"
        assert a["forward_ok"] and b["forward_ok"] and c["forward_ok"], "All forwards must pass"

        # SQNR: baseline and entropy coding should be identical (lossless)
        assert abs(a["avg_sqnr"] - c["avg_sqnr"]) < 0.5, \
            f"Entropy coding changed SQNR: {a['avg_sqnr']} vs {c['avg_sqnr']}"

        print("  ✓ All assertions passed.")
        print(f"  ✓ Total size reduction A→C: "
              f"{(1 - c['total_bytes']/a['total_bytes'])*100:.1f}%")


if __name__ == "__main__":
    main()
