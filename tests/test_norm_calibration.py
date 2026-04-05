"""Benchmark norm calibration across bit widths: 8-bit, 4-bit, 4+4-bit.

Measures PPL and KLD before and after calibration for each configuration.

Usage:
    python tests/test_norm_calibration.py [--device cuda]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate_ppl(model, tokenizer, device, seq_length=2048, n_chunks=20):
    """Perplexity on WikiText-103 validation."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * seq_length
            end = start + seq_length + 1
            if end > len(input_ids):
                break
            chunk = input_ids[start:end].unsqueeze(0).to(device)
            outputs = model(chunk[:, :-1])
            logits = outputs.logits
            targets = chunk[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    return math.exp(total_loss / total_tokens)


def evaluate_kld(model, ref_model, tokenizer, device, seq_length=2048, n_chunks=10):
    """Average KL divergence between model and ref_model."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    total_kld = 0.0
    total_tokens = 0
    model.eval()
    ref_model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * seq_length
            end = start + seq_length + 1
            if end > len(input_ids):
                break
            chunk = input_ids[start:end].unsqueeze(0).to(device)
            logits_q = model(chunk[:, :-1]).logits
            logits_r = ref_model(chunk[:, :-1]).logits
            log_p = F.log_softmax(logits_q, dim=-1)
            log_q = F.log_softmax(logits_r, dim=-1)
            kl = F.kl_div(
                log_p.reshape(-1, logits_q.shape[-1]),
                log_q.reshape(-1, logits_q.shape[-1]),
                log_target=True,
                reduction="sum",
            )
            total_kld += kl.item()
            total_tokens += (chunk.shape[1] - 1)

    return total_kld / total_tokens


def run_config(
    fp_model, tokenizer, device, dtype,
    bit_width, residual_bit_width,
    n_cal_samples, n_chunks, seq_length,
    n_iters=200,
    blockwise=False,
    per_group=False,
):
    """Quantize, measure baseline, calibrate, measure again."""
    from turboquant_model.model import TurboQuantConfig, quantize_model
    from turboquant_model.module import TurboQuantLinear
    from turboquant_model.norm_calibration import (
        calibrate_norms, calibrate_norms_blockwise, CalibrationConfig,
    )

    label = f"{bit_width}-bit" + (f"+{residual_bit_width}" if residual_bit_width else "")
    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"{'='*60}")

    config = TurboQuantConfig(
        bit_width=bit_width,
        residual_bit_width=residual_bit_width or None,
        seed=42,
        residual_seed=1042,
    )

    # Quantize a fresh copy
    tq_model = copy.deepcopy(fp_model)
    tq_model = quantize_model(tq_model, config)

    # Baseline eval (fused kernels stay enabled for speed)
    print("\n  Baseline (analytical norms):")
    t0 = time.time()
    ppl_before = evaluate_ppl(tq_model, tokenizer, device, seq_length=seq_length, n_chunks=n_chunks)
    kld_before = evaluate_kld(tq_model, fp_model, tokenizer, device, seq_length=seq_length, n_chunks=min(n_chunks, 10))
    print(f"    PPL: {ppl_before:.4f}")
    print(f"    KLD: {kld_before:.6f}")
    print(f"    Time: {time.time() - t0:.1f}s")

    # Calibrate
    print("\n  Calibrating norms...")
    cal_config = CalibrationConfig(
        n_samples=n_cal_samples,
        seq_length=seq_length,
        lam=1.0,
        lr=1e-3,
        n_iters=n_iters,
        batch_size=64,
        per_group=per_group,
    )
    t0 = time.time()
    if blockwise:
        layer_stats = calibrate_norms_blockwise(
            tq_model, fp_model, tokenizer, device=device, config=cal_config,
        )
    else:
        layer_stats = calibrate_norms(
            tq_model, fp_model, tokenizer, device=device, config=cal_config,
        )
    cal_time = time.time() - t0
    unit = "blocks" if blockwise else "layers"
    print(f"    Calibrated {len(layer_stats)} {unit} in {cal_time:.1f}s")

    # Post-calibration eval
    print("\n  After calibration:")
    ppl_after = evaluate_ppl(tq_model, tokenizer, device, seq_length=seq_length, n_chunks=n_chunks)
    kld_after = evaluate_kld(tq_model, fp_model, tokenizer, device, seq_length=seq_length, n_chunks=min(n_chunks, 10))
    print(f"    PPL: {ppl_after:.4f} (delta: {ppl_after - ppl_before:+.4f})")
    print(f"    KLD: {kld_after:.6f} (delta: {kld_after - kld_before:+.6f})")

    # Layer/block stats summary
    if layer_stats and "before_mse" in layer_stats[0]:
        avg_mse_before = sum(s["before_mse"] for s in layer_stats) / len(layer_stats)
        avg_mse_after = sum(s["after_mse"] for s in layer_stats) / len(layer_stats)
        avg_cos_before = sum(s["before_cos"] for s in layer_stats) / len(layer_stats)
        avg_cos_after = sum(s["after_cos"] for s in layer_stats) / len(layer_stats)
    else:
        avg_mse_before = avg_mse_after = 0.0
        avg_cos_before = avg_cos_after = 0.0
    print(f"\n    Layer avg MSE:     {avg_mse_before:.6f} -> {avg_mse_after:.6f}")
    print(f"    Layer avg cos_sim: {avg_cos_before:.6f} -> {avg_cos_after:.6f}")

    del tq_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "config": label,
        "bit_width": bit_width,
        "residual_bit_width": residual_bit_width,
        "baseline": {"ppl": ppl_before, "kld": kld_before},
        "calibrated": {"ppl": ppl_after, "kld": kld_after},
        "ppl_delta": ppl_after - ppl_before,
        "kld_delta": kld_after - kld_before,
        "calibration_time_s": cal_time,
        "n_layers_calibrated": len(layer_stats),
        "avg_mse_before": avg_mse_before,
        "avg_mse_after": avg_mse_after,
        "avg_cos_before": avg_cos_before,
        "avg_cos_after": avg_cos_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Norm calibration benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-cal-samples", type=int, default=64,
                        help="Calibration sequences (lower = faster)")
    parser.add_argument("--n-iters", type=int, default=200,
                        help="Optimization iterations per layer/block")
    parser.add_argument("--n-chunks", type=int, default=20,
                        help="Evaluation chunks for PPL/KLD")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--output", type=str, default="tests/report_norm_calibration.json")
    parser.add_argument("--blockwise", action="store_true",
                        help="Use block-wise end-to-end calibration instead of per-layer")
    parser.add_argument("--per-group", action="store_true",
                        help="Per-group alpha (M,G) instead of per-row alpha (M,)")
    parser.add_argument("--only-4bit", action="store_true",
                        help="Only run 4-bit config (skip 4+4)")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Calibration samples: {args.n_cal_samples}")
    print(f"Eval chunks: {args.n_chunks}")
    mode = 'blockwise' if args.blockwise else 'per-layer'
    if args.per_group:
        mode += '+per-group'
    print(f"Calibration mode: {mode}")

    # Load reference model once
    print("\nLoading reference model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    fp_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    # bf16 baseline
    print("\n" + "=" * 60)
    print("  bf16 Baseline")
    print("=" * 60)
    ppl_bf16 = evaluate_ppl(fp_model, tokenizer, device, seq_length=args.seq_length, n_chunks=args.n_chunks)
    print(f"  PPL: {ppl_bf16:.4f}")

    configs = [
        (4, None),     # 4-bit single pass
    ]
    if not args.only_4bit:
        configs.append((4, 4))  # 4+4 residual

    results = {"model": args.model, "bf16_ppl": ppl_bf16, "configs": []}

    for bw, rbw in configs:
        r = run_config(
            fp_model, tokenizer, device, dtype,
            bit_width=bw,
            residual_bit_width=rbw,
            n_cal_samples=args.n_cal_samples,
            n_chunks=args.n_chunks,
            seq_length=args.seq_length,
            n_iters=args.n_iters,
            blockwise=args.blockwise,
            per_group=args.per_group,
        )
        results["configs"].append(r)

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"bf16 PPL: {ppl_bf16:.4f}")
    print()
    print(f"{'Config':<12} {'PPL before':>12} {'PPL after':>12} {'Δ PPL':>10} "
          f"{'KLD before':>12} {'KLD after':>12} {'Δ KLD':>10}")
    print("-" * 80)
    for r in results["configs"]:
        print(f"{r['config']:<12} "
              f"{r['baseline']['ppl']:>12.4f} "
              f"{r['calibrated']['ppl']:>12.4f} "
              f"{r['ppl_delta']:>+10.4f} "
              f"{r['baseline']['kld']:>12.6f} "
              f"{r['calibrated']['kld']:>12.6f} "
              f"{r['kld_delta']:>+10.6f}")

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
