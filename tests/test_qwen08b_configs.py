"""Comprehensive eval of Qwen3.5-0.8B-Base across quantization configurations.

Configs tested:
  1. 8-bit single pass (via quantize_model_advanced for variable bit-width path)
  2. 4-bit single pass
  3. 4-bit + 4-bit residual with independent rotation, entropy coding, norm compression

Metrics: PPL, KLD, BPW (effective bits per weight)
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
SEQ_LEN = 512
N_CHUNKS = 10


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


def eval_ppl_kld(model, ref_model, input_ids, device, seq_len=SEQ_LEN, n_chunks=N_CHUNKS):
    """Evaluate PPL and KLD."""
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

            if (i + 1) % 10 == 0:
                logger.info(f"  eval chunk {i+1}/{n_chunks}")

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    avg_kld = total_kld / total_tokens if ref_model is not None else None
    return ppl, avg_kld


def compute_bpw(model, config):
    """Compute effective BPW from memory_bytes / total quantized weight params.

    Returns: (bpw_quant, compressed_mb, n_quant_params)
    """
    from turboquant_model.module import TurboQuantLinear

    total_bytes = 0
    total_params = 0

    for module in model.modules():
        if isinstance(module, TurboQuantLinear):
            total_bytes += module.memory_bytes()
            total_params += module.out_features * module.in_features

    bpw = (total_bytes * 8) / total_params if total_params > 0 else 0
    compressed_mb = total_bytes / 1024**2
    return bpw, compressed_mb, total_params


def compute_entropy_bpw(model, config):
    """Measure theoretical entropy BPW for indices across all layers.

    Uses empirical entropy (fast) instead of full rANS compression (very slow in Python).
    Entropy gives a tight lower bound on achievable BPW with any entropy coder.
    """
    from turboquant_model.module import TurboQuantLinear
    from turboquant_model.quantize import unpack_4bit
    import numpy as np

    total_weights = 0
    total_entropy_bits = 0

    for name, module in model.named_modules():
        if not isinstance(module, TurboQuantLinear):
            continue
        M, N = module.out_features, module.in_features

        # Get indices
        if module.has_variable_bit_widths and module._indices_uint8 is not None:
            indices = module._indices_uint8.cpu()
        else:
            indices = unpack_4bit(module.indices_packed.cpu(), N)

        flat = indices.reshape(-1).numpy().astype(np.uint8)
        n_symbols = 2 ** config.bit_width
        counts = np.bincount(flat, minlength=n_symbols)
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))

        total_weights += M * N
        total_entropy_bits += entropy * M * N

        # Residual pass
        if module.has_residual and module.pass2_indices_packed is not None:
            res_bw = config.residual_bit_width or config.bit_width
            pass2_indices = unpack_4bit(module.pass2_indices_packed.cpu(), N)
            flat2 = pass2_indices.reshape(-1).numpy().astype(np.uint8)
            n_sym2 = 2 ** res_bw
            counts2 = np.bincount(flat2, minlength=n_sym2)
            probs2 = counts2 / counts2.sum()
            res_entropy = float(-np.sum(probs2[probs2 > 0] * np.log2(probs2[probs2 > 0])))
            total_entropy_bits += res_entropy * M * N

    avg_entropy = total_entropy_bits / total_weights if total_weights > 0 else 0
    # rANS typically achieves ~1-2% overhead over entropy
    estimated_rans_bpw = avg_entropy * 1.02
    return estimated_rans_bpw, avg_entropy


def quantize_and_prepare(config, model, device):
    """Quantize model with the appropriate pipeline.

    - 8-bit single pass → quantize_model_advanced (variable bit-width path)
    - 4-bit single pass → quantize_model (standard)
    - 4+4 residual → quantize_model (residual) + manual norm codec
    """
    from turboquant_model.model import quantize_model, quantize_model_advanced
    from turboquant_model.module import TurboQuantLinear

    if config.bit_width != 4 and config.residual_bit_width is None:
        # Non-4-bit single pass: use advanced pipeline (uint8 variable-width path)
        model = quantize_model_advanced(model, config)
    elif config.residual_bit_width is not None:
        # Residual: use quantize_model (advanced doesn't support residual)
        model = quantize_model(model, config)
        # Apply norm codec manually after residual quantization
        if config.norm_codec != "fp32":
            for module in model.modules():
                if isinstance(module, TurboQuantLinear):
                    if module.weight_norms.dim() >= 2:
                        module.apply_norm_codec(config.norm_codec)
    else:
        # Standard 4-bit
        model = quantize_model(model, config)

    if device in ("cpu", "mps"):
        disable_fused_kernels(model)

    return model


def run_config(name, config, base_model_fn, ref_model, tokenizer, input_ids, device):
    """Run a single configuration: quantize, evaluate, measure."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Config: {name}")
    logger.info(f"  bit_width={config.bit_width}, residual={config.residual_bit_width}, "
                f"rotation={config.rotation}, strategy={config.rotation_strategy}, "
                f"norm_codec={config.norm_codec}, entropy={config.entropy_coding}")
    logger.info(f"{'='*60}")

    # Load fresh model
    t0 = time.time()
    model = base_model_fn()
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Quantize
    t0 = time.time()
    model = quantize_and_prepare(config, model, device)
    quant_time = time.time() - t0
    logger.info(f"Quantization time: {quant_time:.1f}s")

    # BPW (from storage)
    bpw_storage, compressed_mb, n_quant = compute_bpw(model, config)
    logger.info(f"BPW (storage): {bpw_storage:.4f}")
    logger.info(f"Compressed size: {compressed_mb:.2f} MB ({n_quant:,} quantized params)")

    # Entropy coding BPW
    ec_bpw, entropy = compute_entropy_bpw(model, config)
    logger.info(f"Entropy-coded BPW (indices): {ec_bpw:.4f}")
    logger.info(f"Theoretical entropy: {entropy:.4f}")

    # Evaluate
    logger.info("Evaluating PPL and KLD...")
    t0 = time.time()
    ppl, kld = eval_ppl_kld(model, ref_model, input_ids, device)
    eval_time = time.time() - t0
    logger.info(f"PPL: {ppl:.4f}")
    logger.info(f"KLD: {kld:.6f}" if kld is not None else "KLD: N/A")
    logger.info(f"Eval time: {eval_time:.1f}s")

    result = {
        "config": name,
        "bit_width": config.bit_width,
        "residual_bit_width": config.residual_bit_width,
        "norm_codec": config.norm_codec,
        "entropy_coding": config.entropy_coding,
        "rotation_strategy": config.rotation_strategy,
        "ppl": round(ppl, 4),
        "kld": round(kld, 6) if kld is not None else None,
        "bpw_storage": round(bpw_storage, 4),
        "ec_bpw_indices": round(ec_bpw, 4),
        "entropy_indices": round(entropy, 4),
        "compressed_mb": round(compressed_mb, 2),
        "quant_time_s": round(quant_time, 1),
        "eval_time_s": round(eval_time, 1),
        "n_quant_params": n_quant,
    }

    # Free memory
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def main():
    from turboquant_model.model import TurboQuantConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = auto_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info(f"Device: {device}, dtype: {dtype}")

    # Load tokenizer + eval data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    input_ids = load_eval_data(tokenizer)
    logger.info(f"Eval data: {len(input_ids)} tokens, {N_CHUNKS} chunks × {SEQ_LEN} tokens")

    # Reference model for KLD
    logger.info("Loading reference model for KLD...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    def load_fresh():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=dtype, trust_remote_code=True
        ).to(device).eval()

    # --- Configurations ---
    configs = {
        "8-bit": TurboQuantConfig(
            bit_width=8,
            group_size=128,
            rotation="qr",
        ),
        "4-bit": TurboQuantConfig(
            bit_width=4,
            group_size=128,
            rotation="qr",
        ),
        "4+4bit independent+EC+norm": TurboQuantConfig(
            bit_width=4,
            group_size=128,
            residual_bit_width=4,
            rotation="qr",
            rotation_strategy="different",
            norm_codec="factored_int8",
            entropy_coding=True,
        ),
        "4+4bit independent+EC+norm4": TurboQuantConfig(
            bit_width=4,
            group_size=128,
            residual_bit_width=4,
            rotation="qr",
            rotation_strategy="different",
            norm_codec="factored_int4",
            entropy_coding=True,
        ),
    }

    results = []
    for name, config in configs.items():
        result = run_config(name, config, load_fresh, ref_model, tokenizer, input_ids, device)
        results.append(result)

    # --- Summary table ---
    print("\n" + "=" * 110)
    print(f"{'Config':<35} {'PPL':>8} {'KLD':>10} {'BPW(store)':>10} {'EC BPW':>8} {'Entropy':>8} {'Size MB':>8}")
    print("-" * 110)
    for r in results:
        kld_str = f"{r['kld']:>10.6f}" if r['kld'] is not None else f"{'N/A':>10}"
        print(
            f"{r['config']:<35} "
            f"{r['ppl']:>8.2f} "
            f"{kld_str} "
            f"{r['bpw_storage']:>10.4f} "
            f"{r['ec_bpw_indices']:>8.4f} "
            f"{r['entropy_indices']:>8.4f} "
            f"{r['compressed_mb']:>8.2f}"
        )
    print("=" * 110)

    # Save results
    out_path = "tests/report_qwen08b_configs.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
