"""Experiment: Polar decomposition quantization vs. standard TurboQuant.

Tests the PolarQuant-style recursive polar decomposition pipeline:
  1. Numerical validation (lossless round-trip)
  2. Angle PDF verification
  3. Bit allocation optimization
  4. SQNR/MSE comparison vs standard TurboQuant at matching bpw
  5. Per-level angle quantization error analysis
  6. Scaling with group size (d=32, 64, 128, 256)

Run: python tests/exp_polar_decomposition.py
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")

from turboquant_model.polar import (
    recursive_polar_decompose,
    recursive_polar_reconstruct,
    quantize_angles,
    compute_bpw,
    polar_quantize,
    optimize_bit_allocation,
    angle_distortion_at_bits,
    get_angle_codebook,
    angle_pdf_level1,
    angle_pdf_higher,
)
from turboquant_model.quantize import turboquant_quantize
from turboquant_model.rotation import hadamard_rotate


def sqnr(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB."""
    signal = W.float().pow(2).mean()
    noise = (W.float() - W_hat.float()).pow(2).mean()
    if noise < 1e-30:
        return 999.0
    return 10 * math.log10(signal.item() / noise.item())


def mse(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    return (W.float() - W_hat.float()).pow(2).mean().item()


# ---------------------------------------------------------------------------
# Test 1: Lossless round-trip (no quantization)
# ---------------------------------------------------------------------------


def test_lossless_roundtrip():
    print("=" * 70)
    print("TEST 1: Lossless polar decomposition round-trip")
    print("=" * 70)

    for d in [4, 8, 16, 32, 64, 128, 256]:
        torch.manual_seed(42)
        y = torch.randn(100, d)  # 100 random vectors
        r, angles = recursive_polar_decompose(y)
        y_recon = recursive_polar_reconstruct(r, angles)
        err = (y - y_recon).abs().max().item()
        # Check that radius equals L2 norm
        norm_err = (r - y.norm(dim=1)).abs().max().item()
        print(f"  d={d:4d}: max reconstruction error = {err:.2e}, "
              f"radius-norm error = {norm_err:.2e}")
        assert err < 1e-5, f"Round-trip error too large: {err}"
        assert norm_err < 1e-5, f"Norm error too large: {norm_err}"

    print("  ✓ All round-trip tests passed.\n")


# ---------------------------------------------------------------------------
# Test 2: Angle distribution verification
# ---------------------------------------------------------------------------


def test_angle_distributions():
    print("=" * 70)
    print("TEST 2: Angle distribution verification")
    print("=" * 70)

    d = 128
    L = int(math.log2(d))
    torch.manual_seed(42)
    y = torch.randn(50000, d)
    _, angles = recursive_polar_decompose(y)

    for level in range(min(L, 5)):
        psi = angles[level].reshape(-1).numpy()
        n_angles = len(psi)

        if level == 0:
            # Should be approximately uniform on [-π, π]
            # KS test against uniform
            expected_mean = 0.0
            expected_std = math.pi / math.sqrt(3)
            actual_mean = np.mean(psi)
            actual_std = np.std(psi)
            lo, hi = psi.min(), psi.max()
            print(f"  Level {level} (d/2={d // 2} angles per vector):")
            print(f"    Range: [{lo:.4f}, {hi:.4f}] (expected [-π, π])")
            print(f"    Mean:  {actual_mean:.4f} (expected {expected_mean:.4f})")
            print(f"    Std:   {actual_std:.4f} (expected {expected_std:.4f})")
        else:
            # Should be concentrated around π/4
            k = 2**level
            actual_mean = np.mean(psi)
            actual_std = np.std(psi)
            lo, hi = psi.min(), psi.max()
            print(f"  Level {level} (d/{2**(level+1)}={d // 2**(level+1)} angles per vector):")
            print(f"    Range: [{lo:.4f}, {hi:.4f}] (expected [0, π/2])")
            print(f"    Mean:  {actual_mean:.4f} (expected ~{math.pi/4:.4f})")
            print(f"    Std:   {actual_std:.4f} (shrinks with level)")

    print()


# ---------------------------------------------------------------------------
# Test 3: Bit allocation optimization
# ---------------------------------------------------------------------------


def test_bit_allocation():
    print("=" * 70)
    print("TEST 3: Optimal bit allocation for various bpw targets")
    print("=" * 70)

    for d in [64, 128, 256]:
        L = int(math.log2(d))
        print(f"\n  d = {d} (L = {L} levels):")

        for target_bpw in [2.0, 2.5, 3.0, 3.5, 4.0]:
            alloc = optimize_bit_allocation(d, target_bpw)
            actual_bpw = compute_bpw(alloc, d)
            # Compute total estimated distortion
            total_dist = sum(
                angle_distortion_at_bits(lv, alloc[lv], d) * (d // 2 ** (lv + 1))
                for lv in range(L)
            )
            alloc_str = ",".join(str(b) for b in alloc)
            print(f"    target={target_bpw:.1f} → actual={actual_bpw:.3f} bpw, "
                  f"alloc=[{alloc_str}], est_distortion={total_dist:.6f}")

    print()


# ---------------------------------------------------------------------------
# Test 4: Polar vs Standard TurboQuant comparison
# ---------------------------------------------------------------------------


def compare_polar_vs_standard():
    print("=" * 70)
    print("TEST 4: Polar vs Standard TurboQuant — SQNR/MSE comparison")
    print("=" * 70)

    results = []

    for d in [64, 128, 256]:
        L = int(math.log2(d))
        torch.manual_seed(0)
        W = torch.randn(256, d) * 0.02  # simulate small weight matrix

        print(f"\n  d = {d}, W shape = {W.shape}")
        print(f"  {'Method':<25s} {'bpw':>6s} {'SQNR(dB)':>10s} {'MSE':>12s}")
        print(f"  {'-'*55}")

        # Standard TurboQuant at various bit widths
        for bw in [2, 3, 4]:
            W_hat = turboquant_quantize(W, bit_width=bw, group_size=d, seed=42,
                                        rotation="hadamard")
            s = sqnr(W, W_hat)
            m = mse(W, W_hat)
            bpw_tq = bw + 16.0 / d  # bits + norm cost
            label = f"TurboQuant {bw}-bit"
            print(f"  {label:<25s} {bpw_tq:>6.2f} {s:>10.2f} {m:>12.2e}")
            results.append({
                "d": d, "method": label, "bpw": round(bpw_tq, 3),
                "sqnr_db": round(s, 2), "mse": m,
            })

        # Polar quantization at various bpw targets
        for target_bpw in [2.0, 2.5, 3.0, 3.5, 4.0]:
            alloc = optimize_bit_allocation(d, target_bpw)
            actual_bpw = compute_bpw(alloc, d)

            W_hat, info = polar_quantize(
                W, bit_alloc=alloc, group_size=d, seed=42,
                rotation="hadamard",
            )
            s = sqnr(W, W_hat)
            m = mse(W, W_hat)
            alloc_str = ",".join(str(b) for b in alloc)
            label = f"Polar [{alloc_str}]"
            print(f"  {label:<25s} {actual_bpw:>6.2f} {s:>10.2f} {m:>12.2e}")
            results.append({
                "d": d, "method": label, "bpw": round(actual_bpw, 3),
                "sqnr_db": round(s, 2), "mse": m,
                "bit_alloc": alloc,
                "level_angle_mse": info["level_angle_mse"],
            })

    return results


# ---------------------------------------------------------------------------
# Test 5: Per-level angle quantization analysis
# ---------------------------------------------------------------------------


def analyze_per_level_errors():
    print("\n" + "=" * 70)
    print("TEST 5: Per-level angle quantization error analysis (d=128)")
    print("=" * 70)

    d = 128
    L = int(math.log2(d))
    torch.manual_seed(0)
    y = torch.randn(10000, d)  # rotated-like vectors
    _, angles = recursive_polar_decompose(y)

    print(f"\n  {'Level':<8s} {'#Angles':<10s} {'Bits':>5s} "
          f"{'Angle MSE':>12s} {'Angle SQNR':>12s} {'Angle Range':>15s}")

    for bits in [1, 2, 3, 4]:
        print(f"\n  --- {bits}-bit per level ---")
        bit_alloc = [bits] * L
        _, q_angles = quantize_angles(angles, bit_alloc, d)
        bpw = compute_bpw(bit_alloc, d)
        print(f"  (bpw = {bpw:.3f})")

        for lv in range(L):
            n_ang = d // 2 ** (lv + 1)
            err = (angles[lv] - q_angles[lv]).pow(2).mean().item()
            sig = angles[lv].pow(2).mean().item()
            ang_sqnr = 10 * math.log10(sig / max(err, 1e-30))
            lo = angles[lv].min().item()
            hi = angles[lv].max().item()
            print(f"  {lv:<8d} {n_ang:<10d} {bits:>5d} "
                  f"{err:>12.6f} {ang_sqnr:>12.2f} [{lo:.3f}, {hi:.3f}]")


# ---------------------------------------------------------------------------
# Test 6: Codebook visualization (print centroids per level)
# ---------------------------------------------------------------------------


def show_codebooks():
    print("\n" + "=" * 70)
    print("TEST 6: Angle codebook centroids (d=128)")
    print("=" * 70)

    d = 128
    L = int(math.log2(d))

    for level in range(min(L, 5)):
        for bits in [2, 3, 4]:
            centroids, boundaries = get_angle_codebook(level, bits, d)
            c_str = ", ".join(f"{c:.4f}" for c in centroids.numpy())
            if level == 0:
                label = f"Uniform[-π,π]"
            else:
                k = 2**level
                label = f"chi({k}) polar"
            print(f"  Level {level} ({label}), {bits}-bit: [{c_str}]")
        print()


# ---------------------------------------------------------------------------
# Test 7: Scaling behavior with d
# ---------------------------------------------------------------------------


def test_scaling():
    print("=" * 70)
    print("TEST 7: Scaling — polar SQNR vs d at fixed ~2.5 bpw")
    print("=" * 70)

    results = []
    print(f"\n  {'d':<6s} {'bpw':>6s} {'SQNR_polar':>12s} {'SQNR_tq2b':>12s} "
          f"{'SQNR_tq3b':>12s} {'SQNR_tq4b':>12s}")

    for d in [16, 32, 64, 128, 256]:
        torch.manual_seed(0)
        W = torch.randn(512, d) * 0.02

        # Polar at ~2.5 bpw
        alloc = optimize_bit_allocation(d, 2.5)
        actual_bpw = compute_bpw(alloc, d)
        W_hat_p, _ = polar_quantize(W, bit_alloc=alloc, group_size=d, seed=42,
                                     rotation="hadamard")
        s_polar = sqnr(W, W_hat_p)

        # TurboQuant baselines
        sqnrs_tq = {}
        for bw in [2, 3, 4]:
            W_hat_tq = turboquant_quantize(W, bit_width=bw, group_size=d, seed=42,
                                           rotation="hadamard")
            sqnrs_tq[bw] = sqnr(W, W_hat_tq)

        print(f"  {d:<6d} {actual_bpw:>6.2f} {s_polar:>12.2f} "
              f"{sqnrs_tq[2]:>12.2f} {sqnrs_tq[3]:>12.2f} {sqnrs_tq[4]:>12.2f}")

        results.append({
            "d": d, "bpw_polar": round(actual_bpw, 3),
            "sqnr_polar": round(s_polar, 2),
            "sqnr_tq_2bit": round(sqnrs_tq[2], 2),
            "sqnr_tq_3bit": round(sqnrs_tq[3], 2),
            "sqnr_tq_4bit": round(sqnrs_tq[4], 2),
            "bit_alloc": alloc,
        })

    return results


# ---------------------------------------------------------------------------
# Test 8: Impact on real-ish weight distribution (non-Gaussian)
# ---------------------------------------------------------------------------


def test_weight_like_distribution():
    print("\n" + "=" * 70)
    print("TEST 8: Non-Gaussian weights (heavy-tailed, sparse)")
    print("=" * 70)

    d = 128
    torch.manual_seed(42)

    # Heavy-tailed: Student-t with 3 dof
    from torch.distributions import StudentT
    W_heavy = StudentT(df=3.0).sample((256, d)).float() * 0.01

    # Sparse-ish: 50% zeros + Gaussian
    W_sparse = torch.randn(256, d) * 0.02
    mask = torch.rand(256, d) > 0.5
    W_sparse[mask] = 0.0

    for name, W in [("Heavy-tailed (t3)", W_heavy), ("Sparse (50%)", W_sparse)]:
        print(f"\n  {name}, shape {W.shape}:")
        print(f"    {'Method':<25s} {'bpw':>6s} {'SQNR':>10s} {'MSE':>12s}")

        for target in [2.5, 3.0, 4.0]:
            alloc = optimize_bit_allocation(d, target)
            bpw = compute_bpw(alloc, d)
            W_hat, _ = polar_quantize(W, bit_alloc=alloc, group_size=d,
                                       seed=42, rotation="hadamard")
            s = sqnr(W, W_hat)
            m = mse(W, W_hat)
            print(f"    Polar@{target:.1f}bpw         {bpw:>6.2f} {s:>10.2f} {m:>12.2e}")

        for bw in [2, 3, 4]:
            W_hat = turboquant_quantize(W, bit_width=bw, group_size=d,
                                        seed=42, rotation="hadamard")
            s = sqnr(W, W_hat)
            m = mse(W, W_hat)
            bpw = bw + 16.0 / d
            print(f"    TQ {bw}-bit              {bpw:>6.2f} {s:>10.2f} {m:>12.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("\n" + "=" * 70)
    print("  POLAR DECOMPOSITION QUANTIZATION EXPERIMENT")
    print("  TurboQuant + PolarQuant-style recursive polar transform")
    print("=" * 70 + "\n")

    t0 = time.time()

    # Validation tests
    test_lossless_roundtrip()
    test_angle_distributions()

    # Codebook analysis
    show_codebooks()

    # Bit allocation
    test_bit_allocation()

    # Per-level error analysis
    analyze_per_level_errors()

    # Core comparison
    results_comparison = compare_polar_vs_standard()

    # Scaling
    results_scaling = test_scaling()

    # Non-Gaussian
    test_weight_like_distribution()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total experiment time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Save results
    output = {
        "experiment": "polar_decomposition_quantization",
        "description": "PolarQuant-style recursive polar decomposition vs standard TurboQuant",
        "elapsed_s": round(elapsed, 1),
        "comparison": results_comparison,
        "scaling": results_scaling,
    }
    with open("logs/exp_polar_decomposition.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to logs/exp_polar_decomposition.json")


if __name__ == "__main__":
    main()
