"""Experiment: Entropy-coded packing for TurboQuant indices.

Measures the BPW savings from rANS entropy coding of the quantized
index stream vs. naive fixed-width packing.

Tests:
  1. Theoretical entropy vs. fixed bit-widths (2, 3, 4 bit)
  2. Actual rANS compression on Gaussian synthetic weights
  3. Round-trip correctness (compress → decompress = identity)
  4. Impact on effective BPW budget (freed bits → quality reinvestment)
  5. Scaling with matrix dimensions and group sizes
  6. Per-layer empirical entropy distribution on real-ish weights

Run: python tests/exp_entropy_coding.py
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")

from turboquant_model.codebook import get_codebook
from turboquant_model.entropy_codec import (
    compute_entropy,
    compress_indices,
    decompress_indices,
    measure_compressed_bpw,
    gaussian_bin_probs,
    get_ans_table,
    get_codec,
    BLOCK_SIZE,
)
from turboquant_model.rotation import hadamard_rotate, hadamard_rotate_inverse
from turboquant_model.quantize import turboquant_quantize


def sqnr(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    signal = W.float().pow(2).mean()
    noise = (W.float() - W_hat.float()).pow(2).mean()
    if noise < 1e-30:
        return 999.0
    return 10 * math.log10(signal.item() / noise.item())


def quantize_to_indices(
    W: torch.Tensor,
    bit_width: int,
    group_size: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize and return raw indices + norms + dequantized approximation.

    Returns:
        indices: (M, N) int32 — quantization indices
        norms: (M, n_groups) float32 — per-group norms
        W_approx: (M, N) float32 — dequantized weights
    """
    W = W.float()
    M, N = W.shape
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    all_indices = torch.zeros(M, N, dtype=torch.int32, device=W.device)
    all_norms = []
    W_approx = torch.zeros_like(W)

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(-1))

        Y = hadamard_rotate(W_norm, seed=seed + g_start)
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        idx = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        idx = idx.clamp(0, 2**bit_width - 1)
        all_indices[:, g_start:g_end] = idx.reshape(M, g_dim)

        Y_quant = centroids[idx].reshape(Y_scaled.shape) / scale
        W_g_approx = hadamard_rotate_inverse(Y_quant, seed=seed + g_start) * norms
        W_approx[:, g_start:g_end] = W_g_approx

    norms_tensor = torch.stack(all_norms, dim=1)
    return all_indices, norms_tensor, W_approx


# ---------------------------------------------------------------------------
# Test 1: Theoretical entropy analysis
# ---------------------------------------------------------------------------

def test_theoretical_entropy():
    print("=" * 70)
    print("TEST 1: Theoretical entropy of Lloyd-Max quantized N(0,1)")
    print("=" * 70)

    results = []
    print(f"\n  {'Bits':<6s} {'Levels':<8s} {'Entropy':>10s} {'Fixed BPW':>10s} "
          f"{'Savings':>10s} {'Savings%':>10s}")
    print(f"  {'-' * 56}")

    for bw in [2, 3, 4, 5, 6]:
        H = compute_entropy(bw)
        savings = bw - H
        pct = savings / bw * 100
        print(f"  {bw:<6d} {2**bw:<8d} {H:>10.4f} {bw:>10.4f} "
              f"{savings:>10.4f} {pct:>9.1f}%")
        results.append({
            "bit_width": bw, "n_levels": 2**bw,
            "entropy": round(H, 4), "fixed_bpw": bw,
            "savings_bits": round(savings, 4),
            "savings_pct": round(pct, 1),
        })

    # Show bin probabilities for 4-bit
    print(f"\n  4-bit bin probabilities (N(0,1) Lloyd-Max):")
    probs = gaussian_bin_probs(4)
    for i, p in enumerate(probs):
        bar = "█" * int(p * 200)
        print(f"    bin {i:2d}: {p:.4f}  {bar}")

    return results


# ---------------------------------------------------------------------------
# Test 2: Actual rANS compression on synthetic weights
# ---------------------------------------------------------------------------

def test_actual_compression():
    print("\n" + "=" * 70)
    print("TEST 2: Actual rANS compression vs. fixed packing")
    print("=" * 70)

    results = []
    torch.manual_seed(42)

    print(f"\n  {'Config':<25s} {'Fixed BPW':>10s} {'rANS BPW':>12s} "
          f"{'Entropy':>10s} {'Overhead':>10s} {'Savings':>10s}")
    print(f"  {'-' * 80}")

    for bw in [2, 3, 4]:
        for shape in [(256, 512), (1024, 1024), (4096, 4096)]:
            M, N = shape
            W = torch.randn(M, N) * 0.02
            group_size = min(128, N)

            indices, norms, _ = quantize_to_indices(W, bw, group_size)

            # Measure compression
            compressed_bpw, empirical_H = measure_compressed_bpw(indices, bw)
            theoretical_H = compute_entropy(bw)

            # Overhead = rANS BPW - entropy (codec inefficiency + block headers)
            overhead = compressed_bpw - empirical_H
            savings = bw - compressed_bpw

            label = f"{bw}b {M}x{N} g={group_size}"
            print(f"  {label:<25s} {bw:>10.3f} {compressed_bpw:>12.4f} "
                  f"{empirical_H:>10.4f} {overhead:>10.4f} {savings:>10.4f}")

            results.append({
                "bit_width": bw, "shape": list(shape), "group_size": group_size,
                "fixed_bpw": bw,
                "rans_bpw": round(compressed_bpw, 4),
                "empirical_entropy": round(empirical_H, 4),
                "theoretical_entropy": round(theoretical_H, 4),
                "overhead": round(overhead, 4),
                "savings_bits": round(savings, 4),
            })

    return results


# ---------------------------------------------------------------------------
# Test 3: Round-trip correctness
# ---------------------------------------------------------------------------

def test_roundtrip():
    print("\n" + "=" * 70)
    print("TEST 3: rANS round-trip correctness")
    print("=" * 70)

    torch.manual_seed(0)

    for bw in [2, 3, 4, 5]:
        for n in [128, 4096, BLOCK_SIZE, BLOCK_SIZE * 3 + 128]:
            W = torch.randn(1, n) * 0.02
            indices, _, _ = quantize_to_indices(W, bw, min(128, n))
            flat = indices.reshape(-1)

            compressed, _ = compress_indices(flat, bw)
            recovered = decompress_indices(compressed, bw, flat.shape)

            match = (flat.cpu() == recovered).all().item()
            status = "✓" if match else "✗ MISMATCH"
            print(f"  {bw}-bit, n={n:>6d}: {status}")
            assert match, f"Round-trip failed for {bw}-bit, n={n}"

    print("  ✓ All round-trip tests passed.\n")


# ---------------------------------------------------------------------------
# Test 4: BPW budget analysis — freed bits → quality reinvestment
# ---------------------------------------------------------------------------

def test_bpw_reinvestment():
    print("=" * 70)
    print("TEST 4: Freed-bit reinvestment — what can we do with entropy savings?")
    print("=" * 70)

    d = 128
    torch.manual_seed(0)
    W = torch.randn(512, d * 8) * 0.02  # 512x1024 matrix, 8 groups

    results = []

    print(f"\n  {'Scenario':<40s} {'Total BPW':>10s} {'SQNR':>10s}")
    print(f"  {'-' * 62}")

    # Baseline: naive 4-bit packing
    W_hat_4 = turboquant_quantize(W, 4, d, 42, "hadamard")
    s_4 = sqnr(W, W_hat_4)
    bpw_4_naive = 4 + 32 / d  # indices + norms
    print(f"  {'4-bit naive (4.25 bpw stored)':<40s} {bpw_4_naive:>10.2f} {s_4:>10.2f}")

    # Entropy-coded 4-bit: measure actual compressed BPW
    indices_4, norms_4, _ = quantize_to_indices(W, 4, d)
    rans_bpw_4, _ = measure_compressed_bpw(indices_4, 4)
    total_bpw_4_rans = rans_bpw_4 + 32 / d
    print(f"  {'4-bit + rANS (same quality)':<40s} "
          f"{total_bpw_4_rans:>10.2f} {s_4:>10.2f}")

    results.append({
        "scenario": "4-bit naive", "total_bpw": round(bpw_4_naive, 3),
        "sqnr": round(s_4, 2),
    })
    results.append({
        "scenario": "4-bit + rANS", "total_bpw": round(total_bpw_4_rans, 3),
        "sqnr": round(s_4, 2),
    })

    # What if we use the freed bits for 5-bit quantization?
    for bw in [3, 5]:
        W_hat = turboquant_quantize(W, bw, d, 42, "hadamard")
        s = sqnr(W, W_hat)
        indices, _, _ = quantize_to_indices(W, bw, d)
        rans_bpw, _ = measure_compressed_bpw(indices, bw)
        total = rans_bpw + 32 / d
        label = f"{bw}-bit + rANS"
        print(f"  {label:<40s} {total:>10.2f} {s:>10.2f}")
        results.append({
            "scenario": label, "total_bpw": round(total, 3), "sqnr": round(s, 2),
        })

    # Compare: 3-bit naive vs 4-bit+rANS (do they land at similar BPW?)
    W_hat_3 = turboquant_quantize(W, 3, d, 42, "hadamard")
    s_3 = sqnr(W, W_hat_3)
    bpw_3_naive = 3 + 32 / d
    print(f"\n  Key comparison at ~3.3 bpw:")
    print(f"    3-bit naive:  {bpw_3_naive:.2f} bpw → {s_3:.2f} dB SQNR")
    print(f"    4-bit + rANS: {total_bpw_4_rans:.2f} bpw → {s_4:.2f} dB SQNR")
    if total_bpw_4_rans < bpw_3_naive:
        delta = s_4 - s_3
        print(f"    → 4-bit+rANS is SMALLER and {delta:.1f} dB better!")
    else:
        print(f"    → 4-bit+rANS costs {total_bpw_4_rans - bpw_3_naive:.2f} more bpw")

    results.append({
        "scenario": "3-bit naive", "total_bpw": round(bpw_3_naive, 3),
        "sqnr": round(s_3, 2),
    })

    return results


# ---------------------------------------------------------------------------
# Test 5: Scaling with matrix dimensions
# ---------------------------------------------------------------------------

def test_scaling():
    print("\n" + "=" * 70)
    print("TEST 5: Compression ratio scaling with matrix size")
    print("=" * 70)

    torch.manual_seed(42)
    results = []

    print(f"\n  {'Shape':<15s} {'Fixed(4b)':>10s} {'rANS':>10s} "
          f"{'Ratio':>8s} {'Enc MB/s':>10s} {'Dec MB/s':>10s}")
    print(f"  {'-' * 66}")

    for M, N in [(64, 128), (256, 512), (1024, 1024), (2048, 2048), (4096, 4096)]:
        W = torch.randn(M, N) * 0.02
        group_size = min(128, N)

        indices, _, _ = quantize_to_indices(W, 4, group_size)
        flat = indices.reshape(-1)
        n_elements = flat.numel()
        fixed_bytes = n_elements * 4 / 8  # 4-bit packing

        # Measure compression speed
        t0 = time.perf_counter()
        compressed, rans_bpw = compress_indices(flat, 4)
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        recovered = decompress_indices(compressed, 4, flat.shape)
        t_dec = time.perf_counter() - t0

        ratio = fixed_bytes / len(compressed)
        enc_mbs = (n_elements / 1e6) / max(t_enc, 1e-9)
        dec_mbs = (n_elements / 1e6) / max(t_dec, 1e-9)

        label = f"{M}x{N}"
        print(f"  {label:<15s} {fixed_bytes:>10,.0f} {len(compressed):>10,d} "
              f"{ratio:>7.2f}x {enc_mbs:>10.1f} {dec_mbs:>10.1f}")

        results.append({
            "shape": [M, N], "n_elements": n_elements,
            "fixed_bytes": int(fixed_bytes),
            "rans_bytes": len(compressed),
            "compression_ratio": round(ratio, 3),
            "rans_bpw": round(rans_bpw, 4),
            "encode_msyms_per_sec": round(enc_mbs, 1),
            "decode_msyms_per_sec": round(dec_mbs, 1),
        })

    return results


# ---------------------------------------------------------------------------
# Test 6: Per-group empirical entropy variation
# ---------------------------------------------------------------------------

def test_per_group_entropy():
    print("\n" + "=" * 70)
    print("TEST 6: Per-group empirical entropy (4-bit, d=128)")
    print("=" * 70)

    d = 128
    torch.manual_seed(42)
    W = torch.randn(1024, d * 4) * 0.02  # 1024x512, 4 groups

    indices, _, _ = quantize_to_indices(W, 4, d)

    n_groups = indices.shape[1] // d
    group_entropies = []

    for g in range(n_groups):
        g_idx = indices[:, g * d:(g + 1) * d].reshape(-1).cpu().numpy()
        counts = np.bincount(g_idx, minlength=16)
        probs = counts / counts.sum()
        H = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
        group_entropies.append(H)

    theoretical = compute_entropy(4)
    mean_H = np.mean(group_entropies)
    std_H = np.std(group_entropies)

    print(f"\n  Theoretical entropy (Gaussian): {theoretical:.4f} bits/symbol")
    print(f"  Empirical mean:                 {mean_H:.4f} ± {std_H:.4f}")
    print(f"  Min/Max:                        {min(group_entropies):.4f} / "
          f"{max(group_entropies):.4f}")
    print(f"  Potential savings vs 4-bit:      {4 - mean_H:.4f} bits/symbol "
          f"({(4 - mean_H)/4*100:.1f}%)")

    # Non-Gaussian weight distributions
    print(f"\n  Non-Gaussian distributions:")
    from torch.distributions import StudentT

    for name, W_alt in [
        ("Heavy-tailed (t3)", StudentT(df=3.0).sample((1024, d * 4)).float() * 0.01),
        ("Sparse (50%)", torch.where(torch.rand(1024, d * 4) > 0.5,
                                     torch.randn(1024, d * 4) * 0.02,
                                     torch.zeros(1024, d * 4))),
    ]:
        idx_alt, _, _ = quantize_to_indices(W_alt, 4, d)
        alt_H_list = []
        for g in range(idx_alt.shape[1] // d):
            g_idx = idx_alt[:, g * d:(g + 1) * d].reshape(-1).cpu().numpy()
            counts = np.bincount(g_idx, minlength=16)
            probs = counts / counts.sum()
            H = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
            alt_H_list.append(H)
        mean_alt = np.mean(alt_H_list)
        print(f"    {name}: empirical entropy = {mean_alt:.4f}, "
              f"savings = {4 - mean_alt:.4f} bits ({(4 - mean_alt)/4*100:.1f}%)")

    return {
        "theoretical_entropy": round(theoretical, 4),
        "empirical_mean": round(mean_H, 4),
        "empirical_std": round(std_H, 4),
        "group_entropies": [round(h, 4) for h in group_entropies],
    }


# ---------------------------------------------------------------------------
# Test 7: End-to-end comparison table
# ---------------------------------------------------------------------------

def test_end_to_end_comparison():
    print("\n" + "=" * 70)
    print("TEST 7: End-to-end comparison — SQNR at matched storage BPW")
    print("=" * 70)

    d = 128
    torch.manual_seed(0)
    W = torch.randn(512, d * 8) * 0.02

    results = []

    print(f"\n  {'Method':<35s} {'Index BPW':>10s} {'Total BPW':>10s} "
          f"{'SQNR':>10s} {'Quality level':>15s}")
    print(f"  {'-' * 82}")

    for bw in [2, 3, 4, 5]:
        W_hat = turboquant_quantize(W, bw, d, 42, "hadamard")
        s = sqnr(W, W_hat)
        indices, _, _ = quantize_to_indices(W, bw, d)
        rans_bpw, emp_H = measure_compressed_bpw(indices, bw)
        norm_bpw = 32 / d

        total_fixed = bw + norm_bpw
        total_rans = rans_bpw + norm_bpw

        label_fixed = f"TQ {bw}-bit (naive pack)"
        label_rans = f"TQ {bw}-bit + rANS"
        savings = total_fixed - total_rans

        print(f"  {label_fixed:<35s} {bw:>10.3f} {total_fixed:>10.3f} "
              f"{s:>10.2f} {'':>15s}")
        print(f"  {label_rans:<35s} {rans_bpw:>10.3f} {total_rans:>10.3f} "
              f"{s:>10.2f}   saves {savings:.3f} bpw")

        results.append({
            "bit_width": bw,
            "sqnr_db": round(s, 2),
            "index_bpw_fixed": bw,
            "index_bpw_rans": round(rans_bpw, 4),
            "total_bpw_fixed": round(total_fixed, 3),
            "total_bpw_rans": round(total_rans, 3),
            "savings_bpw": round(savings, 4),
            "empirical_entropy": round(emp_H, 4),
        })

    # Key insight: rANS TQ-4 vs naive TQ-3
    print(f"\n  === KEY INSIGHT ===")
    r4 = next(r for r in results if r["bit_width"] == 4)
    r3 = next(r for r in results if r["bit_width"] == 3)
    print(f"  TQ 4-bit + rANS: {r4['total_bpw_rans']:.3f} bpw, "
          f"{r4['sqnr_db']:.2f} dB SQNR")
    print(f"  TQ 3-bit naive:  {r3['total_bpw_fixed']:.3f} bpw, "
          f"{r3['sqnr_db']:.2f} dB SQNR")
    delta_bpw = r4["total_bpw_rans"] - r3["total_bpw_fixed"]
    delta_sqnr = r4["sqnr_db"] - r3["sqnr_db"]
    if delta_bpw < 0:
        print(f"  → rANS 4-bit is {-delta_bpw:.3f} bpw SMALLER and "
              f"{delta_sqnr:.1f} dB better!")
    else:
        print(f"  → rANS 4-bit costs {delta_bpw:.3f} more bpw but "
              f"{delta_sqnr:.1f} dB better")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  ENTROPY-CODED PACKING INTEGRATION EXPERIMENT")
    print("  rANS compression of TurboQuant quantized indices")
    print("=" * 70 + "\n")

    t0 = time.time()

    # Tests
    results_entropy = test_theoretical_entropy()
    results_compression = test_actual_compression()
    test_roundtrip()
    results_reinvest = test_bpw_reinvestment()
    results_scaling = test_scaling()
    results_pergroup = test_per_group_entropy()
    results_e2e = test_end_to_end_comparison()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total experiment time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    output = {
        "experiment": "entropy_coded_packing",
        "description": "rANS entropy coding integration for TurboQuant indices",
        "elapsed_s": round(elapsed, 1),
        "theoretical_entropy": results_entropy,
        "actual_compression": results_compression,
        "bpw_reinvestment": results_reinvest,
        "scaling": results_scaling,
        "per_group_entropy": results_pergroup,
        "end_to_end": results_e2e,
    }
    with open("logs/exp_entropy_coding.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to logs/exp_entropy_coding.json")


if __name__ == "__main__":
    main()
