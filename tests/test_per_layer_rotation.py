"""Compare per-layer rotation vs shared rotation on Qwen3.5-0.8B-Base at 4-bit.

Tests:
  1. 4-bit with shared rotation (rotation_strategy="different", all layers same seed) — baseline
  2. 4-bit with per-layer rotation (rotation_strategy="per_layer", unique seed per layer)
"""

import gc
import logging
import time

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
SEQ_LEN = 512
N_CHUNKS = 10

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from turboquant_model.model import TurboQuantConfig, quantize_model
from turboquant_model.module import TurboQuantLinear
from turboquant_model.quantize import unpack_4bit

# ---------- eval data ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
text = "\n\n".join(ds["text"])
input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
logger.info(f"Eval data: {len(input_ids)} tokens")

# ---------- reference model ----------
logger.info("Loading reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=dtype, trust_remote_code=True
).to(device).eval()


def eval_model(model, ref_model, input_ids, device):
    """Evaluate PPL and KLD."""
    n_chunks = min(N_CHUNKS, (len(input_ids) - 1) // SEQ_LEN)
    total_loss = 0.0
    total_tokens = 0
    total_kld = 0.0

    with torch.no_grad():
        for i in range(n_chunks):
            start = i * SEQ_LEN
            chunk = input_ids[start : start + SEQ_LEN + 1].unsqueeze(0).to(device)
            inp, tgt = chunk[:, :-1], chunk[:, 1:]

            logits = model(inp).logits
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += tgt.numel()

            ref_logits = ref_model(inp).logits
            log_p = nn.functional.log_softmax(logits, dim=-1)
            log_q = nn.functional.log_softmax(ref_logits, dim=-1)
            kld = nn.functional.kl_div(
                log_p.reshape(-1, logits.shape[-1]),
                log_q.reshape(-1, logits.shape[-1]),
                log_target=True,
                reduction="sum",
            )
            total_kld += kld.item()

            if (i + 1) % 5 == 0:
                logger.info(f"  chunk {i+1}/{n_chunks}")

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    avg_kld = total_kld / total_tokens
    return ppl, avg_kld


def compute_entropy(model):
    """Compute average empirical entropy of packed 4-bit indices."""
    total_bits = 0
    total_elems = 0
    for m in model.modules():
        if not isinstance(m, TurboQuantLinear):
            continue
        M, N = m.out_features, m.in_features
        idx = unpack_4bit(m.indices_packed.cpu(), N)
        flat = idx.reshape(-1).numpy().astype(np.uint8)
        counts = np.bincount(flat, minlength=16)
        probs = counts / counts.sum()
        H = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
        total_bits += H * M * N
        total_elems += M * N
    return total_bits / total_elems


configs = {
    "4-bit shared rotation": TurboQuantConfig(
        bit_width=4,
        group_size=128,
        rotation="qr",
        rotation_strategy="different",  # shared seed across layers
    ),
    "4-bit per-layer rotation": TurboQuantConfig(
        bit_width=4,
        group_size=128,
        rotation="qr",
        rotation_strategy="per_layer",  # unique seed per layer
    ),
}

results = {}

for label, config in configs.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {label}")
    logger.info(f"{'='*60}")

    # Fresh model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    t0 = time.time()
    model = quantize_model(model, config)
    quant_time = time.time() - t0

    # Disable fused kernels
    for m in model.modules():
        if isinstance(m, TurboQuantLinear):
            m.use_cutile = False
            m.use_triton = False
            m.use_metal = False

    # BPW
    total_bytes = sum(
        m.memory_bytes() for m in model.modules() if isinstance(m, TurboQuantLinear)
    )
    total_params = sum(
        m.out_features * m.in_features
        for m in model.modules()
        if isinstance(m, TurboQuantLinear)
    )
    bpw = (total_bytes * 8) / total_params

    # Entropy
    entropy = compute_entropy(model)

    # PPL + KLD
    ppl, kld = eval_model(model, ref_model, input_ids, device)

    results[label] = {
        "ppl": ppl,
        "kld": kld,
        "bpw": bpw,
        "entropy": entropy,
        "ec_bpw_est": entropy * 1.02,
        "size_mb": total_bytes / 1024**2,
        "quant_time": quant_time,
    }

    print(f"\n{label}: PPL={ppl:.4f}  KLD={kld:.6f}  BPW={bpw:.4f}  "
          f"EC_BPW={entropy * 1.02:.4f}  Size={total_bytes / 1024**2:.2f}MB  "
          f"Time={quant_time:.1f}s")

    del model
    gc.collect()

# Summary
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}")
print(f"{'Config':<30} {'PPL':>8} {'KLD':>10} {'BPW':>8} {'EC BPW':>8} {'Entropy':>8}")
print("-" * 80)
for label, r in results.items():
    print(f"{label:<30} {r['ppl']:>8.4f} {r['kld']:>10.6f} {r['bpw']:>8.4f} "
          f"{r['ec_bpw_est']:>8.4f} {r['entropy']:>8.4f}")
