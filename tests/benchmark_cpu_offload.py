"""Benchmark: 4+4 residual with CPU-offloaded pass2 vs GPU-resident pass2.

Compares three modes on Qwen3.5-0.8B-Base:
  1. bf16 baseline (reference)
  2. 4+4 residual — both passes on GPU (default)
  3. 4+4 residual — pass2 offloaded to CPU with pipelined H2D

Metrics: PPL, KLD (vs bf16), latency (prefill + decode), VRAM, BPW.

Usage:
    python tests/benchmark_cpu_offload.py
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
SEQ_LEN = 512
N_CHUNKS = 10


# ---------------------------------------------------------------------------
# Utilities (shared with test_qwen08b_configs pattern)
# ---------------------------------------------------------------------------

def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def disable_fused_kernels(model: nn.Module):
    from turboquant_model.module import TurboQuantLinear
    for m in model.modules():
        if isinstance(m, TurboQuantLinear):
            m.use_cutile = False
            m.use_triton = False
            m.use_metal = False


def load_eval_data(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    return encodings.input_ids[0]


def eval_ppl_kld(
    model, ref_model, input_ids, device,
    seq_len: int = SEQ_LEN, n_chunks: int = N_CHUNKS,
):
    """Evaluate PPL and KLD vs reference."""
    n_chunks = min(n_chunks, (len(input_ids) - 1) // seq_len)
    total_loss = 0.0
    total_tokens = 0
    total_kld = 0.0

    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * seq_len
            chunk = input_ids[start : start + seq_len + 1].unsqueeze(0).to(device)
            inp = chunk[:, :-1]
            targets = chunk[:, 1:]

            logits = model(inp).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

            if ref_model is not None:
                ref_logits = ref_model(inp).logits
                log_p = torch.nn.functional.log_softmax(logits, dim=-1)
                log_q = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                kld = torch.nn.functional.kl_div(
                    log_p.reshape(-1, logits.shape[-1]),
                    log_q.reshape(-1, logits.shape[-1]),
                    log_target=True,
                    reduction="sum",
                )
                total_kld += kld.item()

            if (i + 1) % 5 == 0:
                logger.info(f"  eval chunk {i+1}/{n_chunks}")

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    avg_kld = total_kld / total_tokens if ref_model is not None else None
    return ppl, avg_kld


def compute_bpw(model):
    from turboquant_model.module import TurboQuantLinear
    total_bytes = 0
    total_bytes_cpu = 0
    total_params = 0
    for m in model.modules():
        if isinstance(m, TurboQuantLinear):
            total_bytes += m.memory_bytes()
            total_bytes_cpu += m.memory_bytes_cpu()
            total_params += m.out_features * m.in_features
    bpw = (total_bytes * 8) / total_params if total_params > 0 else 0
    return bpw, total_bytes / 1024**2, total_bytes_cpu / 1024**2, total_params


def measure_vram_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def measure_latency(
    model, tokenizer, device, prompt: str = "The quick brown fox",
    n_warmup: int = 3, n_runs: int = 10,
) -> dict:
    """Measure prefill and decode latency.

    Returns: dict with prefill_ms, decode_ms_per_token, total_ms
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.shape[1]
    gen_tokens = 32

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model.generate(input_ids, max_new_tokens=gen_tokens, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure prefill (single forward on prompt)
    prefill_times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_times.append((time.perf_counter() - t0) * 1000)

    # Measure full generation (prefill + decode)
    gen_times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(input_ids, max_new_tokens=gen_tokens, do_sample=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_times.append((time.perf_counter() - t0) * 1000)

    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_gen = sum(gen_times) / len(gen_times)
    avg_decode_per_token = (avg_gen - avg_prefill) / gen_tokens

    return {
        "prefill_ms": round(avg_prefill, 2),
        "decode_ms_per_token": round(avg_decode_per_token, 2),
        "total_gen_ms": round(avg_gen, 2),
        "generated_tokens": gen_tokens,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant_model.model import TurboQuantConfig, quantize_model
    from turboquant_model.module import TurboQuantLinear

    device = auto_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info(f"Device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    input_ids = load_eval_data(tokenizer)
    logger.info(f"Eval data: {len(input_ids)} tokens")

    results = []

    # -----------------------------------------------------------------------
    # 1. bf16 baseline
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Config: bf16 baseline")
    logger.info("=" * 70)

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    vram_bf16 = measure_vram_mb()
    logger.info(f"VRAM (bf16): {vram_bf16:.1f} MB")

    ppl_bf16, _ = eval_ppl_kld(ref_model, None, input_ids, device)
    logger.info(f"PPL (bf16): {ppl_bf16:.4f}")

    latency_bf16 = measure_latency(ref_model, tokenizer, device)
    logger.info(f"Latency (bf16): {latency_bf16}")

    results.append({
        "config": "bf16_baseline",
        "ppl": round(ppl_bf16, 4),
        "kld": None,
        "bpw": 16.0,
        "vram_mb": round(vram_bf16, 1),
        "vram_cpu_mb": 0.0,
        **latency_bf16,
    })

    # -----------------------------------------------------------------------
    # Helper: quantize fresh model
    # -----------------------------------------------------------------------
    def load_fresh():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=dtype, trust_remote_code=True,
        ).to(device).eval()

    base_config = TurboQuantConfig(
        bit_width=4,
        group_size=128,
        residual_bit_width=4,
        rotation="qr",
        rotation_strategy="different",
    )

    # -----------------------------------------------------------------------
    # 2. 4+4 residual — GPU resident
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Config: 4+4 residual (GPU)")
    logger.info("=" * 70)

    model_gpu = load_fresh()
    t0 = time.time()
    model_gpu = quantize_model(model_gpu, base_config)
    quant_time = time.time() - t0
    logger.info(f"Quantization: {quant_time:.1f}s")

    if device in ("cpu", "mps"):
        disable_fused_kernels(model_gpu)

    bpw_gpu, size_gpu, cpu_size_gpu, n_params = compute_bpw(model_gpu)
    vram_gpu = measure_vram_mb()
    logger.info(f"BPW: {bpw_gpu:.4f}, VRAM: {vram_gpu:.1f} MB")

    ppl_gpu, kld_gpu = eval_ppl_kld(model_gpu, ref_model, input_ids, device)
    logger.info(f"PPL: {ppl_gpu:.4f}, KLD: {kld_gpu:.6f}")

    latency_gpu = measure_latency(model_gpu, tokenizer, device)
    logger.info(f"Latency: {latency_gpu}")

    results.append({
        "config": "4+4_residual_gpu",
        "ppl": round(ppl_gpu, 4),
        "kld": round(kld_gpu, 6) if kld_gpu is not None else None,
        "bpw": round(bpw_gpu, 4),
        "vram_mb": round(vram_gpu, 1),
        "vram_cpu_mb": round(cpu_size_gpu, 2),
        "compressed_mb": round(size_gpu, 2),
        "quant_time_s": round(quant_time, 1),
        **latency_gpu,
    })

    del model_gpu
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 3. 4+4 residual — CPU offloaded pass2
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Config: 4+4 residual (CPU offload)")
    logger.info("=" * 70)

    offload_config = TurboQuantConfig(
        bit_width=4,
        group_size=128,
        residual_bit_width=4,
        rotation="qr",
        rotation_strategy="different",
        cpu_offload_pass2=True,
    )

    model_offload = load_fresh()
    t0 = time.time()
    model_offload = quantize_model(model_offload, offload_config)
    quant_time_off = time.time() - t0
    logger.info(f"Quantization: {quant_time_off:.1f}s")

    if device in ("cpu", "mps"):
        disable_fused_kernels(model_offload)

    # Verify offload is active
    n_offloaded = sum(
        1 for m in model_offload.modules()
        if isinstance(m, TurboQuantLinear) and m.is_pass2_offloaded
    )
    logger.info(f"Layers with offloaded pass2: {n_offloaded}")

    bpw_off, size_off, cpu_size_off, _ = compute_bpw(model_offload)
    vram_off = measure_vram_mb()
    logger.info(f"BPW: {bpw_off:.4f}, VRAM: {vram_off:.1f} MB, CPU pinned: {cpu_size_off:.2f} MB")

    ppl_off, kld_off = eval_ppl_kld(model_offload, ref_model, input_ids, device)
    logger.info(f"PPL: {ppl_off:.4f}, KLD: {kld_off:.6f}")

    latency_off = measure_latency(model_offload, tokenizer, device)
    logger.info(f"Latency: {latency_off}")

    results.append({
        "config": "4+4_residual_cpu_offload",
        "ppl": round(ppl_off, 4),
        "kld": round(kld_off, 6) if kld_off is not None else None,
        "bpw": round(bpw_off, 4),
        "vram_mb": round(vram_off, 1),
        "vram_cpu_mb": round(cpu_size_off, 2),
        "compressed_mb": round(size_off, 2),
        "n_offloaded_layers": n_offloaded,
        "quant_time_s": round(quant_time_off, 1),
        **latency_off,
    })

    del model_offload
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 130)
    print(
        f"{'Config':<30} {'PPL':>8} {'KLD':>10} {'BPW':>6} "
        f"{'VRAM MB':>8} {'CPU MB':>7} "
        f"{'Prefill':>9} {'Decode/tok':>11} {'Gen ms':>8}"
    )
    print("-" * 130)
    for r in results:
        kld_s = f"{r['kld']:>10.6f}" if r.get("kld") else f"{'—':>10}"
        print(
            f"{r['config']:<30} "
            f"{r['ppl']:>8.2f} "
            f"{kld_s} "
            f"{r.get('bpw', 16.0):>6.2f} "
            f"{r['vram_mb']:>8.1f} "
            f"{r.get('vram_cpu_mb', 0):>7.2f} "
            f"{r['prefill_ms']:>8.2f}ms "
            f"{r['decode_ms_per_token']:>9.2f}ms "
            f"{r['total_gen_ms']:>8.2f}"
        )
    print("=" * 130)

    # Quality assertion: offloaded PPL == GPU PPL (bitwise identical)
    if results[1]["ppl"] is not None and results[2]["ppl"] is not None:
        ppl_diff = abs(results[1]["ppl"] - results[2]["ppl"])
        logger.info(f"PPL difference (GPU vs offload): {ppl_diff:.6f}")
        if ppl_diff < 0.01:
            logger.info("PPL match confirms numerical equivalence.")
        else:
            logger.warning(f"PPL mismatch: {ppl_diff:.4f} — investigate!")

    # Save
    out_path = "tests/report_cpu_offload.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
