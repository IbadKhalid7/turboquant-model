"""Test 4+4bit with factored_int4 norm compression on Qwen3.5-0.8B-Base."""

import gc
import json
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
from turboquant_model.model import TurboQuantConfig, quantize_model
from turboquant_model.module import TurboQuantLinear
from turboquant_model.quantize import unpack_4bit

# Load tokenizer + eval data
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
text = "\n\n".join(ds["text"])
input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
logger.info(f"Eval data: {len(input_ids)} tokens")

# Reference model
logger.info("Loading reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=dtype, trust_remote_code=True
).to(device).eval()

# Fresh model for quantization
logger.info("Loading model for quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=dtype, trust_remote_code=True
).to(device).eval()

config = TurboQuantConfig(
    bit_width=4,
    group_size=128,
    residual_bit_width=4,
    rotation="qr",
    rotation_strategy="different",
    norm_codec="factored_int4",
    entropy_coding=True,
)

t0 = time.time()
model = quantize_model(model, config)

# Apply norm codec + disable fused kernels
for m in model.modules():
    if isinstance(m, TurboQuantLinear):
        m.use_cutile = False
        m.use_triton = False
        m.use_metal = False
        if m.weight_norms.dim() >= 2:
            m.apply_norm_codec("factored_int4")

quant_time = time.time() - t0
logger.info(f"Quantization time: {quant_time:.1f}s")

# BPW
total_bytes = sum(m.memory_bytes() for m in model.modules() if isinstance(m, TurboQuantLinear))
total_params = sum(
    m.out_features * m.in_features for m in model.modules() if isinstance(m, TurboQuantLinear)
)
bpw_storage = (total_bytes * 8) / total_params
logger.info(f"BPW (storage): {bpw_storage:.4f}, Size: {total_bytes / 1024**2:.2f} MB")

# Entropy
total_weights = 0
total_entropy_bits = 0
for name, m in model.named_modules():
    if not isinstance(m, TurboQuantLinear):
        continue
    M, N = m.out_features, m.in_features
    for indices_buf, bw in [(m.indices_packed, 4), (m.pass2_indices_packed, 4)]:
        if indices_buf is None:
            continue
        idx = unpack_4bit(indices_buf.cpu(), N)
        flat = idx.reshape(-1).numpy().astype(np.uint8)
        counts = np.bincount(flat, minlength=16)
        probs = counts / counts.sum()
        H = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
        total_weights += M * N
        total_entropy_bits += H * M * N

avg_entropy = total_entropy_bits / total_weights
logger.info(f"Entropy: {avg_entropy:.4f}, EC BPW est: {avg_entropy * 1.02:.4f}")

# PPL + KLD
n_chunks = min(N_CHUNKS, (len(input_ids) - 1) // SEQ_LEN)
total_loss = 0.0
total_tokens = 0
total_kld = 0.0
model.eval()

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

print(f"\n{'='*80}")
print("4+4bit factored_int4 Results:")
print(f"  PPL:         {ppl:.4f}")
print(f"  KLD:         {avg_kld:.6f}")
print(f"  BPW storage: {bpw_storage:.4f}")
print(f"  EC BPW est:  {avg_entropy * 1.02:.4f}")
print(f"  Entropy:     {avg_entropy:.4f}")
print(f"  Size:        {total_bytes / 1024**2:.2f} MB")
print(f"{'='*80}")
