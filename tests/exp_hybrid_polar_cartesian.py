"""Experiment: Hybrid polar + Cartesian quantization.

Tests partial polar decomposition (n_polar_levels < L) combined with
Cartesian scalar quantization for the intermediate radii.

Hypothesis: Stopping polar decomposition early avoids error amplification
at coarse levels, while fine-level angles exploit geometric structure.

Run: python tests/exp_hybrid_polar_cartesian.py
"""

from __future__ import annotations

import json
import math
import sys
import time

import torch

sys.path.insert(0, "src")

from turboquant_model.polar import (
    partial_polar_decompose,
    partial_polar_reconstruct,
    polar_quantize,
    hybrid_polar_cartesian_quantize,
    optimize_bit_allocation,
    compute_bpw,
)
from turboquant_model.quantize import turboquant_quantize


def sqnr(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    signal = W.float().pow(2).mean()
    noise = (W.float() - W_hat.float()).pow(2).mean()
    if noise < 1e-30:
        return 999.0
    return 10 * math.log10(signal.item() / noise.item())


def mse(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    return (W.float() - W_hat.float()).pow(2).mean().item()


# ---------------------------------------------------------------------------
# Test 1: Lossless round-trip for partial polar
# ---------------------------------------------------------------------------

def test_partial_roundtrip():
    print("=" * 70)
    print("TEST 1: Partial polar decomposition — lossless round-trip")
    print("=" * 70)

    d = 128
    L = int(math.log2(d))
    torch.manual_seed(42)
    y = torch.randn(200, d)

    for n_levels in range(0, L + 1):
        radii, angles = partial_polar_decompose(y, n_levels)
        y_recon = partial_polar_reconstruct(radii, angles)
        err = (y - y_recon).abs().max().item()
        d_r = d // (2 ** n_levels)
        print(f"  n_polar={n_levels}: radii dim={d_r:>4d}, "
              f"max recon error = {err:.2e}")
        assert err < 1e-5, f"Round-trip error too large at n_levels={n_levels}"

    print("  ✓ All partial round-trip tests passed.\n")


# ---------------------------------------------------------------------------
# Test 2: Sweep n_polar_levels at fixed ~3 bpw
# ---------------------------------------------------------------------------

def sweep_polar_depth(W: torch.Tensor, group_size: int = 128):
    print("=" * 70)
    print("TEST 2: Sweep n_polar_levels at ~3 bpw (d=128)")
    print("=" * 70)

    L = int(math.log2(group_size))
    results = []

    hdr = (f"  {'n_polar':<8s} {'angle_bits':<20s} {'cart_bits':>9s} "
           f"{'bpw':>6s} {'SQNR':>10s} {'MSE':>12s} "
           f"{'ang_mse':>12s} {'cart_mse':>12s}")
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    # n_polar = 0 → pure Cartesian (same as TQ)
    for n_polar in range(0, L + 1):
        d_radii = group_size // (2 ** n_polar)

        # Pick angle + cartesian bits to land near 3 bpw
        # total bits per group = sum(n_ang_lv * angle_bits_lv) + d_radii * cart_bits + 16
        # target total = 3 * group_size
        target_total = 3.0 * group_size - 16  # subtract radius norm cost

        # Strategy: try uniform angle bits, then allocate remaining to cartesian
        best = None
        for ab in range(1, 7):
            angle_bits_list = [ab] * n_polar
            angle_cost = sum(
                (group_size // (2 ** (lv + 1))) * ab for lv in range(n_polar)
            )
            remaining = target_total - angle_cost
            if remaining <= 0:
                continue
            cart_b = remaining / max(d_radii, 1)
            cart_b_int = max(0, min(6, round(cart_b)))
            if cart_b_int == 0 and n_polar < L:
                continue
            actual_total = angle_cost + d_radii * cart_b_int + 16
            actual_bpw = actual_total / group_size
            if best is None or abs(actual_bpw - 3.0) < abs(best[2] - 3.0):
                best = (angle_bits_list, cart_b_int, actual_bpw)

        if best is None:
            continue
        angle_bits_list, cart_b_int, actual_bpw = best

        if n_polar == 0:
            # Pure Cartesian: just use TQ directly
            W_hat = turboquant_quantize(
                W, bit_width=cart_b_int, group_size=group_size,
                seed=42, rotation="hadamard"
            )
            s = sqnr(W, W_hat)
            m = mse(W, W_hat)
            row = {
                "n_polar": 0, "angle_bits": [], "cartesian_bits": cart_b_int,
                "bpw": round(cart_b_int + 16 / group_size, 3),
                "sqnr_db": round(s, 2), "mse": m,
                "angle_mses": [], "cartesian_mse": 0.0,
            }
            bpw_disp = cart_b_int + 16 / group_size
            print(f"  {0:<8d} {'[]':<20s} {cart_b_int:>9d} "
                  f"{bpw_disp:>6.2f} {s:>10.2f} {m:>12.2e} "
                  f"{'N/A':>12s} {'N/A':>12s}")
        elif n_polar == L:
            # Full polar: use polar_quantize
            alloc = optimize_bit_allocation(group_size, actual_bpw)
            actual_bpw2 = compute_bpw(alloc, group_size)
            W_hat, info = polar_quantize(
                W, bit_alloc=alloc, group_size=group_size,
                seed=42, rotation="hadamard"
            )
            s = sqnr(W, W_hat)
            m = mse(W, W_hat)
            alloc_str = ",".join(str(b) for b in alloc)
            row = {
                "n_polar": L, "angle_bits": alloc,
                "cartesian_bits": 0,
                "bpw": round(actual_bpw2, 3),
                "sqnr_db": round(s, 2), "mse": m,
                "angle_mses": info["level_angle_mse"],
                "cartesian_mse": 0.0,
            }
            print(f"  {L:<8d} {'[' + alloc_str + ']':<20s} {'0':>9s} "
                  f"{actual_bpw2:>6.2f} {s:>10.2f} {m:>12.2e} "
                  f"{'(full polar)':>12s} {'N/A':>12s}")
        else:
            # Hybrid
            W_hat, info = hybrid_polar_cartesian_quantize(
                W, n_polar_levels=n_polar,
                angle_bits=angle_bits_list,
                cartesian_bits=cart_b_int,
                group_size=group_size, seed=42,
            )
            s = sqnr(W, W_hat)
            m = mse(W, W_hat)
            ab_str = ",".join(str(b) for b in angle_bits_list)
            row = {
                "n_polar": n_polar, "angle_bits": angle_bits_list,
                "cartesian_bits": cart_b_int,
                "bpw": round(info["bpw"], 3),
                "sqnr_db": round(s, 2), "mse": m,
                "angle_mses": info["angle_mses"],
                "cartesian_mse": info["cartesian_mse"],
            }
            ang_mse_str = f"{sum(info['angle_mses']):.4e}"
            cart_mse_str = f"{info['cartesian_mse']:.4e}"
            print(f"  {n_polar:<8d} {'[' + ab_str + ']':<20s} {cart_b_int:>9d} "
                  f"{info['bpw']:>6.2f} {s:>10.2f} {m:>12.2e} "
                  f"{ang_mse_str:>12s} {cart_mse_str:>12s}")

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Test 3: Sweep bpw for best hybrid depth
# ---------------------------------------------------------------------------

def sweep_bpw(W: torch.Tensor, group_size: int = 128):
    print("\n" + "=" * 70)
    print("TEST 3: Hybrid SQNR across bpw targets (d=128)")
    print("=" * 70)

    L = int(math.log2(group_size))
    results = []

    print(f"\n  {'bpw_target':<12s} {'n_polar':<8s} {'config':<25s} "
          f"{'actual_bpw':>10s} {'SQNR':>10s}")
    print(f"  {'-' * 70}")

    for target in [2.0, 2.5, 3.0, 3.5, 4.0]:
        best_sqnr = -999.0
        best_row = None

        for n_polar in range(0, L + 1):
            d_radii = group_size // (2 ** n_polar)
            target_total = target * group_size - 16

            # Try multiple angle bit combinations
            max_ab = min(6, n_polar + 3) if n_polar > 0 else 0
            ab_range = range(1, max_ab + 1) if n_polar > 0 else [0]

            for ab in ab_range:
                angle_bits_list = [ab] * n_polar if n_polar > 0 else []
                angle_cost = sum(
                    (group_size // (2 ** (lv + 1))) * ab for lv in range(n_polar)
                ) if n_polar > 0 else 0
                remaining = target_total - angle_cost
                if remaining <= 0:
                    continue

                if n_polar == L:
                    # Full polar — no cartesian
                    cart_b = 0
                    actual_bpw = (angle_cost + 16) / group_size
                else:
                    cart_b = max(1, min(6, round(remaining / max(d_radii, 1))))
                    actual_bpw = (angle_cost + d_radii * cart_b + 16) / group_size

                if abs(actual_bpw - target) > 0.6:
                    continue

                try:
                    if n_polar == 0:
                        W_hat = turboquant_quantize(
                            W, bit_width=cart_b, group_size=group_size,
                            seed=42, rotation="hadamard"
                        )
                    elif n_polar == L:
                        alloc = [ab] * L
                        W_hat, _ = polar_quantize(
                            W, bit_alloc=alloc, group_size=group_size,
                            seed=42, rotation="hadamard"
                        )
                    else:
                        W_hat, _ = hybrid_polar_cartesian_quantize(
                            W, n_polar_levels=n_polar,
                            angle_bits=angle_bits_list,
                            cartesian_bits=cart_b,
                            group_size=group_size, seed=42,
                        )
                    s = sqnr(W, W_hat)
                    if s > best_sqnr:
                        best_sqnr = s
                        best_row = {
                            "bpw_target": target,
                            "n_polar": n_polar,
                            "angle_bits": angle_bits_list,
                            "cartesian_bits": cart_b,
                            "actual_bpw": round(actual_bpw, 3),
                            "sqnr_db": round(s, 2),
                        }
                except Exception:
                    pass

        if best_row:
            ab_str = ",".join(str(b) for b in best_row["angle_bits"])
            config = f"[{ab_str}]+{best_row['cartesian_bits']}b"
            print(f"  {target:<12.1f} {best_row['n_polar']:<8d} "
                  f"{config:<25s} {best_row['actual_bpw']:>10.3f} "
                  f"{best_row['sqnr_db']:>10.2f}")
            results.append(best_row)

    return results


# ---------------------------------------------------------------------------
# Test 4: Dimension scaling
# ---------------------------------------------------------------------------

def sweep_dimensions(target_bpw: float = 3.0):
    print("\n" + "=" * 70)
    print(f"TEST 4: Dimension scaling at ~{target_bpw} bpw")
    print("=" * 70)

    results = []

    print(f"\n  {'d':<6s} {'Method':<30s} {'bpw':>6s} {'SQNR':>10s}")
    print(f"  {'-' * 55}")

    for d in [32, 64, 128, 256]:
        torch.manual_seed(0)
        W = torch.randn(512, d) * 0.02
        L = int(math.log2(d))

        # Pure Cartesian (TQ 3-bit)
        W_hat_tq = turboquant_quantize(
            W, bit_width=3, group_size=d, seed=42, rotation="hadamard"
        )
        s_tq = sqnr(W, W_hat_tq)
        bpw_tq = 3 + 16 / d

        # Full polar
        alloc = optimize_bit_allocation(d, target_bpw)
        bpw_polar = compute_bpw(alloc, d)
        W_hat_polar, _ = polar_quantize(
            W, bit_alloc=alloc, group_size=d, seed=42, rotation="hadamard"
        )
        s_polar = sqnr(W, W_hat_polar)

        print(f"  {d:<6d} {'TQ 3-bit':<30s} {bpw_tq:>6.2f} {s_tq:>10.2f}")
        print(f"  {'':<6s} {'Full polar (optimized)':<30s} "
              f"{bpw_polar:>6.2f} {s_polar:>10.2f}")

        # Hybrid at various depths
        best_hybrid_sqnr = -999.0
        best_hybrid_desc = ""
        best_hybrid_bpw = 0.0
        best_hybrid_n = 0

        for n_polar in range(1, L):
            d_radii = d // (2 ** n_polar)
            for ab in range(1, 5):
                angle_bits_list = [ab] * n_polar
                angle_cost = sum(
                    (d // (2 ** (lv + 1))) * ab for lv in range(n_polar)
                )
                remaining = target_bpw * d - 16 - angle_cost
                if remaining <= 0:
                    continue
                cart_b = max(1, min(6, round(remaining / d_radii)))
                actual_bpw = (angle_cost + d_radii * cart_b + 16) / d
                if abs(actual_bpw - target_bpw) > 0.5:
                    continue

                try:
                    W_hat, info = hybrid_polar_cartesian_quantize(
                        W, n_polar_levels=n_polar,
                        angle_bits=angle_bits_list,
                        cartesian_bits=cart_b,
                        group_size=d, seed=42,
                    )
                    s = sqnr(W, W_hat)
                    if s > best_hybrid_sqnr:
                        best_hybrid_sqnr = s
                        best_hybrid_bpw = info["bpw"]
                        best_hybrid_n = n_polar
                        ab_s = ",".join(str(b) for b in angle_bits_list)
                        best_hybrid_desc = f"Hybrid n={n_polar} [{ab_s}]+{cart_b}b"
                except Exception:
                    pass

        if best_hybrid_sqnr > -999:
            print(f"  {'':<6s} {best_hybrid_desc:<30s} "
                  f"{best_hybrid_bpw:>6.2f} {best_hybrid_sqnr:>10.2f}")

        results.append({
            "d": d,
            "tq_3bit": {"bpw": round(bpw_tq, 3), "sqnr": round(s_tq, 2)},
            "full_polar": {"bpw": round(bpw_polar, 3), "sqnr": round(s_polar, 2),
                           "alloc": alloc},
            "best_hybrid": {
                "bpw": round(best_hybrid_bpw, 3),
                "sqnr": round(best_hybrid_sqnr, 2),
                "n_polar": best_hybrid_n,
                "desc": best_hybrid_desc,
            },
        })
        print()

    return results


# ---------------------------------------------------------------------------
# Test 5: Non-Gaussian weights
# ---------------------------------------------------------------------------

def test_nongaussian():
    print("=" * 70)
    print("TEST 5: Non-Gaussian weights — hybrid vs baselines")
    print("=" * 70)

    d = 128
    L = int(math.log2(d))
    torch.manual_seed(42)

    from torch.distributions import StudentT
    W_heavy = StudentT(df=3.0).sample((256, d)).float() * 0.01
    W_sparse = torch.randn(256, d) * 0.02
    W_sparse[torch.rand(256, d) > 0.5] = 0.0

    results = []
    for name, W in [("Heavy-tailed (t3)", W_heavy), ("Sparse (50%)", W_sparse)]:
        print(f"\n  {name}:")
        print(f"    {'Method':<30s} {'bpw':>6s} {'SQNR':>10s}")

        # TQ 3-bit
        W_hat = turboquant_quantize(W, bit_width=3, group_size=d, seed=42,
                                    rotation="hadamard")
        s_tq = sqnr(W, W_hat)
        print(f"    {'TQ 3-bit':<30s} {3 + 16/d:>6.2f} {s_tq:>10.2f}")

        # Full polar
        alloc = optimize_bit_allocation(d, 3.0)
        W_hat, _ = polar_quantize(W, bit_alloc=alloc, group_size=d, seed=42,
                                   rotation="hadamard")
        s_polar = sqnr(W, W_hat)
        bpw_p = compute_bpw(alloc, d)
        print(f"    {'Full polar (opt)':<30s} {bpw_p:>6.2f} {s_polar:>10.2f}")

        # Hybrid sweep
        best_s = -999.0
        best_desc = ""
        best_bpw = 0.0
        for n_polar in range(1, min(4, L)):
            for ab in [2, 3]:
                d_radii = d // (2 ** n_polar)
                angle_cost = sum(
                    (d // (2 ** (lv + 1))) * ab for lv in range(n_polar)
                )
                remaining = 3.0 * d - 16 - angle_cost
                if remaining <= 0:
                    continue
                cart_b = max(1, min(6, round(remaining / d_radii)))
                try:
                    W_hat, info = hybrid_polar_cartesian_quantize(
                        W, n_polar_levels=n_polar,
                        angle_bits=[ab] * n_polar,
                        cartesian_bits=cart_b,
                        group_size=d, seed=42,
                    )
                    s = sqnr(W, W_hat)
                    if s > best_s:
                        best_s = s
                        best_bpw = info["bpw"]
                        best_desc = f"Hybrid n={n_polar} [{ab}]*{n_polar}+{cart_b}b"
                except Exception:
                    pass

        if best_s > -999:
            print(f"    {best_desc:<30s} {best_bpw:>6.2f} {best_s:>10.2f}")

        results.append({
            "dist": name,
            "tq_sqnr": round(s_tq, 2),
            "polar_sqnr": round(s_polar, 2),
            "hybrid_sqnr": round(best_s, 2),
            "hybrid_desc": best_desc,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  HYBRID POLAR + CARTESIAN QUANTIZATION EXPERIMENT")
    print("=" * 70 + "\n")

    t0 = time.time()

    # Validation
    test_partial_roundtrip()

    # Main test: sweep depth at ~3 bpw
    torch.manual_seed(0)
    W = torch.randn(256, 128) * 0.02
    results_depth = sweep_polar_depth(W)

    # BPW sweep
    results_bpw = sweep_bpw(W)

    # Dimension scaling
    results_dim = sweep_dimensions()

    # Non-Gaussian
    results_nongauss = test_nongaussian()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total experiment time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    output = {
        "experiment": "hybrid_polar_cartesian",
        "description": "Partial polar decomposition + Cartesian scalar quantization",
        "elapsed_s": round(elapsed, 1),
        "depth_sweep": results_depth,
        "bpw_sweep": results_bpw,
        "dimension_scaling": results_dim,
        "nongaussian": results_nongauss,
    }
    with open("logs/exp_hybrid_polar_cartesian.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to logs/exp_hybrid_polar_cartesian.json")


if __name__ == "__main__":
    main()
