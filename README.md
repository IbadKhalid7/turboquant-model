# TurboQuant Model

Near-optimal weight quantization with on-the-fly dequantization for LLM inference.

Based on: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025)

## Features

- **4-bit weight quantization** with near-optimal MSE distortion (within 2.7x of information-theoretic lower bound)
- **Residual quantization** for fine-grained bit allocation (e.g., 4+4=8 bits, 3+2=5 bits)
- **On-the-fly dequantization** — weights stay packed as 4-bit indices, dequantized during matmul
- **3.2x GPU memory savings** vs bf16 with only 27% latency overhead
- **Drop-in replacement** for `nn.Linear` — no model architecture changes needed
- **Save/load** quantized models to disk

## Installation

```bash
uv pip install -e ".[transformers]"
```

## Quick Start

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant_model import TurboQuantConfig, quantize_model

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")

# Quantize (single-pass 4-bit)
config = TurboQuantConfig(bit_width=4, seed=42)
model = quantize_model(model, config)
# → 187 layers quantized, 1434 MB → 361 MB (4.0x compression)

# Residual quantization (4+4 = 8 total bits, near-lossless)
config = TurboQuantConfig(bit_width=4, residual_bit_width=4, seed=42)
model = quantize_model(model, config)

# Save / Load
from turboquant_model import save_quantized, load_quantized
save_quantized(model, config, "./quantized-model")
model = load_quantized("Qwen/Qwen3.5-0.8B-Base", "./quantized-model")
```

### CLI

```bash
# Quantize and save
turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized --bit-width 4

# Residual quantization
turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized \
    --bit-width 4 --residual-bit-width 4

# Evaluate PPL
turboquant eval --model Qwen/Qwen3.5-0.8B-Base --quantized ./quantized

# Generate text
turboquant generate --model Qwen/Qwen3.5-0.8B-Base --quantized ./quantized \
    --prompt "The capital of France is"

# Benchmark
turboquant benchmark --model Qwen/Qwen3.5-0.8B-Base --bit-width 4
```

## How It Works

### TurboQuant Algorithm

1. **Row-normalize** each weight row to unit norm, store the norm separately
2. **Random rotation** (QR decomposition of Gaussian matrix) — maps coordinates to near-independent N(0, 1/d)
3. **Lloyd-Max scalar quantization** per coordinate — optimal for Gaussian distribution
4. **Pack** indices into 4-bit format (2 per byte)

### On-the-fly Dequantization

Instead of inverse-rotating the weight (expensive: N×K matrix), we pre-rotate the input (cheap: B×K vector):

```
x_rot = x @ Pi.T                              # rotate input (once per layer)
output = x_rot @ codebook[indices].T           # fused lookup + matmul  
output = output * (norms / sqrt(group_size))   # rescale
```

### Residual Quantization

Apply TurboQuant twice with different rotation seeds:

```
Pass 1: W_hat = TQ(W, b1 bits, seed1)
Pass 2: R_hat = TQ(W - W_hat, b2 bits, seed2)  
Final:  W_approx = W_hat + R_hat
```

Total bits = b1 + b2, but quality is much better than single-pass at same total bits.

## Why Not QJL?

The original TurboQuant paper defines **TurboQuant_prod** — a variant that applies QJL (Quantized Johnson-Lindenstrauss) as a 1-bit correction on the residual to produce an **unbiased** inner product estimator. We do **not** use QJL. Here's why:

1. **QJL solves a different problem.** It's designed for **online inner product estimation** (e.g., KV cache attention: quantize keys once, query with many different vectors later). Weight quantization is **offline** — we compress $W$ once and compute $y = xW^T$ repeatedly. We want minimum reconstruction error $\|W - \tilde{W}\|$, not an unbiased dot-product estimator.

2. **Unbiasedness is unnecessary for weights.** A small deterministic bias from MSE-optimal quantization is absorbed by layer norms, residual connections, and softmax. An unbiased but **high-variance** estimator (QJL at 1 bit) introduces noise that changes every forward pass — worse for stable inference.

3. **Residual quantization strictly dominates.** QJL uses **1 bit** (random sign projection) for the residual. Our residual TQ uses **b₂ bits** with full Lloyd-Max codebook + independent rotation — capturing far more residual information. At 4+4 total bits, residual TQ achieves KL divergence of only 0.002 nats (practically lossless). QJL's 1-bit correction cannot compete.

4. **QJL requires the query at runtime.** The QJL correction term depends on the input activation $x$, making it incompatible with offline weight compression. You'd need to recompute corrections per forward pass — defeating the purpose.

**Summary:** QJL is elegant for streaming/KV-cache inner product preservation. For weight compression, multi-pass residual quantization with optimal scalar codebooks is the natural and superior choice.

## Benchmark Results (Qwen3.5-0.8B-Base, WikiText-103 val, 50 chunks)

| Config | Total Bits | Codebook | PPL | Δ PPL | Compressed Size | Peak GPU |
|--------|-----------|----------|-----|-------|-----------------|----------|
| Baseline bf16 | 16 | — | 14.29 | — | 1,504 MB | — |
| **4+4 residual g=128** | **8** | **16+16 (128 B)** | **14.29** | **0.00** | **762 MB** | 9.7 GB |
| 3+2 residual g=128 | 5 | 8+4 (48 B) | 15.16 | +0.87 | 762 MB | 9.7 GB |
| 4-bit g=full | 4 | 16 (64 B) | 16.23 | +1.94 | 361 MB | 9.6 GB |
| 4-bit g=128 | 4 | 16 (64 B) | 16.57 | +2.28 | 381 MB | 5.8 GB |

**Codebook** = Lloyd-Max optimal centroids for N(0,1). Size = 2^b × 4 bytes (float32). Shared globally across all layers — negligible overhead.

**Key findings:**
- **4+4 residual is lossless** — PPL 14.29 = baseline 14.29
- **4-bit g=128** fits on 8 GB GPUs (5.8 GB peak) with only 2.3 PPL degradation
- Smaller group sizes (g=128) use much less GPU memory due to smaller rotation matrices

### Triton Fused Kernel

The Triton kernel fuses 4-bit unpack + codebook lookup + matmul + norm rescale in a single kernel launch, avoiding intermediate tensor materialization. Auto-enabled when Triton is available.

| Path | Latency (ms/fwd) | Peak GPU (MB) | Speedup |
|------|-------------------|---------------|---------|
| **Triton** (fused) | 380 | **1,074** | 1.10x |
| PyTorch (fallback) | 420 | 4,408 | 1.0x |

**4x peak memory reduction** (4,408 → 1,074 MB) by never materializing the (N, K) float32 `codebook[indices]` tensor. Disable per-module with `m.use_triton = False`.

## Architecture

```
turboquant_model/
├── codebook.py          # Lloyd-Max optimal codebook (precomputed)
├── rotation.py          # QR random rotation matrices
├── quantize.py          # Single-pass quantize + pack/unpack
├── residual.py          # Residual (two-pass) quantization
├── triton_kernels.py    # Fused Triton matmul kernel (optional)
├── module.py            # TurboQuantLinear (nn.Module)
├── model.py             # quantize_model, save/load, config
└── cli.py               # Command-line interface
```

## License

MIT
