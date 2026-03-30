"""Compare three residual quantization rotation strategies on PPL and KLD.

Strategies:
  1. Shared rotation:      all passes use the same seed (current default)
  2. Alternating rotation:  even passes → seed_a, odd passes → seed_b
  3. Independent rotation:  each pass uses a unique seed

Usage:
    python tests/benchmark_rotation_strategies.py \
        --model Qwen/Qwen3.5-0.8B-Base \
        --n-passes 3 \
        --bit-width 4

Requires a CUDA GPU and the `transformers`, `datasets` packages.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
import time

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantize model with a given rotation strategy
# ---------------------------------------------------------------------------

def quantize_with_strategy(
    model: nn.Module,
    strategy: str,
    n_passes: int,
    bit_width: int,
    group_size: int | None,
    seed_a: int = 42,
    seed_b: int = 1042,
) -> nn.Module:
    """Quantize all nn.Linear layers using the specified residual strategy.

    Args:
        model: the model (will be modified in-place)
        strategy: "shared" | "alternating" | "independent"
        n_passes: number of residual passes
        bit_width: bits per pass
        group_size: group size (None = full row)
        seed_a: primary rotation seed
        seed_b: secondary rotation seed (for alternating)

    Returns:
        quantized model (same object, modified in-place)
    """
    from turboquant_model.quantize import turboquant_quantize

    def _get_seed(strategy: str, pass_idx: int, seed_a: int, seed_b: int) -> int:
        if strategy == "shared":
            return seed_a
        elif strategy == "alternating":
            return seed_a if pass_idx % 2 == 0 else seed_b
        elif strategy == "independent":
            return seed_a + pass_idx * 1000
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if "embed" in name.lower() or "lm_head" in name.lower():
            continue

        W = module.weight.data.float()
        current = W.clone()
        W_approx = torch.zeros_like(W)

        for i in range(n_passes):
            seed = _get_seed(strategy, i, seed_a, seed_b)
            W_hat = turboquant_quantize(
                current,
                bit_width=bit_width,
                group_size=group_size,
                seed=seed,
            )
            W_approx += W_hat.float()
            current = current - W_hat.float()

        module.weight.data = W_approx.to(module.weight.dtype)
        replaced += 1

    logger.info(f"  [{strategy}] Replaced {replaced} layers, {n_passes} passes × {bit_width}-bit")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ppl_kld(
    model: nn.Module,
    ref_model: nn.Module | None,
    input_ids: torch.Tensor,
    seq_len: int,
    n_chunks: int,
    device: str = "cuda",
) -> dict:
    """Compute PPL and optionally KLD."""
    n_chunks = min(n_chunks, (len(input_ids) - 1) // seq_len)

    total_loss = 0.0
    total_tokens = 0
    total_kld = 0.0

    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * seq_len
            chunk = input_ids[start : start + seq_len + 1].unsqueeze(0).to(device)

            logits = model(chunk[:, :-1]).logits
            targets = chunk[:, 1:]

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

            if ref_model is not None:
                ref_logits = ref_model(chunk[:, :-1]).logits
                log_p = torch.nn.functional.log_softmax(logits, dim=-1)
                log_q = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                kld = torch.nn.functional.kl_div(
                    log_p.reshape(-1, logits.shape[-1]),
                    log_q.reshape(-1, logits.shape[-1]),
                    log_target=True,
                    reduction="sum",
                )
                total_kld += kld.item()

    ppl = math.exp(total_loss / total_tokens)
    avg_kld = total_kld / total_tokens if ref_model is not None else None

    return {"ppl": ppl, "kld": avg_kld}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare rotation strategies for residual quantization")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--n-passes", type=int, default=2)
    parser.add_argument("--bit-width", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--n-chunks", type=int, default=10)
    parser.add_argument("--seed-a", type=int, default=42)
    parser.add_argument("--seed-b", type=int, default=1042)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', or auto-detect (default)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    logger.info("Loading WikiText-103 validation set")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    logger.info(f"Loading reference (bf16) model: {args.model}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    strategies = ["shared", "alternating", "independent"]
    results = {}

    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"{'='*60}")

        # Load a fresh copy of the model for each strategy
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()

        t0 = time.time()
        quantize_with_strategy(
            model,
            strategy=strategy,
            n_passes=args.n_passes,
            bit_width=args.bit_width,
            group_size=args.group_size,
            seed_a=args.seed_a,
            seed_b=args.seed_b,
        )
        quant_time = time.time() - t0

        t0 = time.time()
        result = evaluate_ppl_kld(
            model, ref_model, input_ids,
            seq_len=args.seq_length,
            n_chunks=args.n_chunks,
            device=device,
        )
        eval_time = time.time() - t0

        result["quant_time"] = quant_time
        result["eval_time"] = eval_time
        results[strategy] = result

        logger.info(f"  PPL:  {result['ppl']:.4f}")
        logger.info(f"  KLD:  {result['kld']:.6f}")
        logger.info(f"  Quantize: {quant_time:.1f}s, Eval: {eval_time:.1f}s")

        # Free memory for next strategy
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"Rotation Strategy Comparison: {args.model}")
    print(f"  {args.n_passes} passes × {args.bit_width}-bit, group_size={args.group_size}")
    print(f"  {args.n_chunks} chunks × seq_len={args.seq_length}")
    print(f"{'='*70}")
    print(f"{'Strategy':<16} {'PPL':>10} {'KLD':>12} {'Quant(s)':>10} {'Eval(s)':>10}")
    print(f"{'-'*16} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    for strategy, r in results.items():
        print(
            f"{strategy:<16} {r['ppl']:>10.4f} {r['kld']:>12.6f} "
            f"{r['quant_time']:>10.1f} {r['eval_time']:>10.1f}"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
