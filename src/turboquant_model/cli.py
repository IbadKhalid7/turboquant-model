"""CLI for TurboQuant model quantization and evaluation.

Usage:
    # Single-pass 4-bit quantization
    turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized --bit-width 4

    # Residual quantization (4+4 = 8 total bits)
    turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized \
        --bit-width 4 --residual-bit-width 4

    # Evaluate PPL
    turboquant eval --model Qwen/Qwen3.5-0.8B-Base --quantized ./quantized

    # Evaluate PPL + KL divergence vs. reference model
    turboquant eval --model Qwen/Qwen3.5-0.8B-Base --quantized ./quantized --kld

    # Run inference
    turboquant generate --model Qwen/Qwen3.5-0.8B-Base --quantized ./quantized \
        --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import torch

logger = logging.getLogger("turboquant_model")


def cmd_quantize(args: argparse.Namespace):
    """Quantize a model and save to disk."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from turboquant_model.model import TurboQuantConfig, quantize_model, save_quantized

    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    config = TurboQuantConfig(
        bit_width=args.bit_width,
        group_size=args.group_size,
        seed=args.seed,
        skip_embeddings=args.skip_embeddings,
        skip_lm_head=args.skip_lm_head,
        residual_bit_width=args.residual_bit_width,
        residual_seed=args.residual_seed,
    )

    logger.info(f"Quantizing: {config.bit_width}-bit"
                + (f"+{config.residual_bit_width}-bit residual" if config.residual_bit_width else ""))
    model = quantize_model(model, config)

    logger.info(f"Saving to: {args.output}")
    save_quantized(model, config, args.output)

    # Quick sanity check
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = tokenizer.encode("Hello world", return_tensors="pt").cuda()
    with torch.no_grad():
        logits = model(input_ids).logits
    logger.info(f"Sanity check passed: logits shape {logits.shape}")


def cmd_eval(args: argparse.Namespace):
    """Evaluate PPL (and optionally KL divergence) on WikiText-103."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    if args.quantized:
        from turboquant_model.model import load_quantized
        logger.info(f"Loading quantized model from: {args.quantized}")
        model = load_quantized(args.model, args.quantized, device="cuda")
    else:
        from turboquant_model.model import TurboQuantConfig, quantize_model
        logger.info(f"Loading and quantizing model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()
        config = TurboQuantConfig(
            bit_width=args.bit_width,
            group_size=args.group_size,
            seed=args.seed,
            residual_bit_width=args.residual_bit_width,
            residual_seed=args.residual_seed,
        )
        model = quantize_model(model, config)

    # Load reference model for KLD computation
    ref_model = None
    if args.kld:
        logger.info(f"Loading reference model for KLD: {args.model}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load WikiText-103 validation
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    seq_len = args.seq_length
    n_chunks = min(args.n_chunks, (len(input_ids) - 1) // seq_len)

    total_loss = 0.0
    total_tokens = 0
    total_kld = 0.0

    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * seq_len
            chunk = input_ids[start : start + seq_len + 1].unsqueeze(0).cuda()
            outputs = model(chunk[:, :-1])
            logits = outputs.logits
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

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    logger.info(f"PPL ({n_chunks} chunks, seq_len={seq_len}): {ppl:.4f}")
    print(f"PPL: {ppl:.4f}")

    if ref_model is not None:
        avg_kld = total_kld / total_tokens
        logger.info(f"KLD ({n_chunks} chunks, seq_len={seq_len}): {avg_kld:.6f}")
        print(f"KLD: {avg_kld:.6f}")


def cmd_generate(args: argparse.Namespace):
    """Generate text with a quantized model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.quantized:
        from turboquant_model.model import load_quantized
        model = load_quantized(args.model, args.quantized, device="cuda")
    else:
        from turboquant_model.model import TurboQuantConfig, quantize_model
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()
        config = TurboQuantConfig(
            bit_width=args.bit_width,
            residual_bit_width=args.residual_bit_width,
        )
        model = quantize_model(model, config)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").cuda()

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-6),
        )
    elapsed = time.time() - t0

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    n_tokens = output.shape[1] - input_ids.shape[1]

    print(text)
    logger.info(f"Generated {n_tokens} tokens in {elapsed:.1f}s ({n_tokens/elapsed:.1f} tok/s)")


def cmd_benchmark(args: argparse.Namespace):
    """Benchmark memory and latency."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from turboquant_model.model import TurboQuantConfig, quantize_model

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    config = TurboQuantConfig(
        bit_width=args.bit_width,
        residual_bit_width=args.residual_bit_width,
    )
    model = quantize_model(model, config)

    torch.cuda.reset_peak_memory_stats()
    input_ids = tokenizer.encode("The quick brown fox", return_tensors="pt").cuda()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(input_ids)

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.n_iters):
            model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    mem_base = torch.cuda.memory_allocated() / 1024**2
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Memory: {mem_base:.0f} MB base, {mem_peak:.0f} MB peak")
    print(f"Latency: {elapsed / args.n_iters * 1000:.1f} ms/forward")


def main():
    parser = argparse.ArgumentParser(
        prog="turboquant",
        description="TurboQuant: Near-optimal weight quantization with on-the-fly dequantization",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- quantize ---
    p_quant = subparsers.add_parser("quantize", help="Quantize a model and save")
    p_quant.add_argument("--model", required=True, help="HF model name or path")
    p_quant.add_argument("--output", required=True, help="Output directory")
    p_quant.add_argument("--bit-width", type=int, default=4)
    p_quant.add_argument("--group-size", type=int, default=None)
    p_quant.add_argument("--seed", type=int, default=42)
    p_quant.add_argument("--skip-embeddings", action="store_true")
    p_quant.add_argument("--skip-lm-head", action="store_true")
    p_quant.add_argument("--residual-bit-width", type=int, default=None, help="Bits for residual pass")
    p_quant.add_argument("--residual-seed", type=int, default=1042)

    # --- eval ---
    p_eval = subparsers.add_parser("eval", help="Evaluate PPL on WikiText-103")
    p_eval.add_argument("--model", required=True, help="HF model name or path")
    p_eval.add_argument("--quantized", default=None, help="Quantized model dir (skip live quant)")
    p_eval.add_argument("--bit-width", type=int, default=4)
    p_eval.add_argument("--group-size", type=int, default=None)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--residual-bit-width", type=int, default=None)
    p_eval.add_argument("--residual-seed", type=int, default=1042)
    p_eval.add_argument("--seq-length", type=int, default=512)
    p_eval.add_argument("--n-chunks", type=int, default=50)
    p_eval.add_argument("--kld", action="store_true", help="Compute KL divergence vs. reference model")

    # --- generate ---
    p_gen = subparsers.add_parser("generate", help="Generate text")
    p_gen.add_argument("--model", required=True, help="HF model name or path")
    p_gen.add_argument("--quantized", default=None, help="Quantized model dir")
    p_gen.add_argument("--bit-width", type=int, default=4)
    p_gen.add_argument("--residual-bit-width", type=int, default=None)
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--max-tokens", type=int, default=64)
    p_gen.add_argument("--temperature", type=float, default=0.0)

    # --- benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="Benchmark memory and latency")
    p_bench.add_argument("--model", required=True, help="HF model name or path")
    p_bench.add_argument("--bit-width", type=int, default=4)
    p_bench.add_argument("--residual-bit-width", type=int, default=None)
    p_bench.add_argument("--n-iters", type=int, default=10)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    commands = {
        "quantize": cmd_quantize,
        "eval": cmd_eval,
        "generate": cmd_generate,
        "benchmark": cmd_benchmark,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
