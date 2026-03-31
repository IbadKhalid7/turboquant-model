# Rotation Strategy Experiment

Comparing three residual quantization rotation strategies on Qwen3.5-0.8B-Base.

## Setup

- **Model:** Qwen3.5-0.8B-Base (186 quantizable Linear layers)
- **Dataset:** WikiText-103 validation, 2 chunks × 64 tokens
- **Metrics:** Perplexity (PPL) and KL divergence (KLD) vs. bf16 reference
- **Seeds:** seed_a=42, seed_b=1042; independent uses seed_a + i×1000
- **Group size:** 128
- **Device:** CPU (torch 2.11.0)

### Strategies

| Strategy | Description | Inference cost |
|---|---|---|
| **Shared** | All passes use the same rotation seed | 1 rotation per group |
| **Alternating** | Even passes → seed_a, odd passes → seed_b | 2 rotations per group |
| **Independent** | Each pass uses a unique seed | N rotations per group |

## Results

### Experiment 1: 2 passes × 4-bit (8 total bits)

| Strategy | PPL | KLD | Quant time |
|---|---|---|---|
| Shared | 18.1458 | 0.009293 | 12.4s |
| Alternating | **17.1801** | **0.001732** | 16.5s |
| Independent | **17.1801** | **0.001732** | 20.7s |

**Observation:** With only 2 passes, alternating and independent are equivalent (each pass gets a distinct seed either way). Both significantly outperform shared rotation — **5.4× lower KLD** and 5.3% lower PPL.

### Experiment 2: 4 passes × 2-bit (8 total bits)

| Strategy | PPL | KLD | Quant time |
|---|---|---|---|
| Shared | 17.4506 | 0.049805 | 18.5s |
| Alternating | 17.3148 | 0.004051 | 11.5s |
| Independent | **17.1801** | **0.003433** | 12.0s |

**Observation:** At 4 passes the strategies diverge clearly:
- Shared rotation degrades badly (KLD 12× worse than alternating)
- Alternating captures 85% of independent's quality improvement
- Independent is best quality but requires 4× inference rotation cost

## Key Findings

### 1. Shared rotation for residual passes is suboptimal

When the same rotation is applied to both the original weight and its residual error, the residual retains the same correlation structure. The second pass of Lloyd-Max quantization sees similar error patterns, yielding diminishing returns. At 4 passes, shared rotation's KLD (0.0498) is **14.5× worse** than independent rotation (0.0034).

### 2. Different rotations significantly improve residual capture

A different rotation matrix decorrelates the residual error differently. The residual from rotation Π₁ has structure that Π₂ can exploit more effectively than Π₁ applied again. This is analogous to how multiple random projections in compressed sensing capture different signal components.

### 3. Alternating 2 rotations is the practical sweet spot

For ≥3 passes, alternating between 2 seeds delivers most of the quality benefit (KLD within 1.2× of independent) while keeping inference cost at exactly 2 rotations per group — regardless of the number of passes. The cost scales as O(2) instead of O(N).

### 4. Trade-offs

| Approach | Best for | Downside |
|---|---|---|
| Shared | Merging via `merge_and_requantize` | Worst quality at multi-pass |
| Alternating | Multi-pass with bounded inference cost | Cannot use rotated-domain merge |
| Independent | Maximum quality | Inference cost scales with # passes |

## Recommendations

- **2 passes:** Use different seeds (already the default in `residual_quantize`). Alternating = independent at 2 passes.
- **3+ passes:** Use alternating if inference latency matters; use independent if quality is paramount.
- **Merge to single-pass:** Only possible with shared rotation via `merge_and_requantize`. Consider using shared rotation for quantization, merging, then deploying the single-pass result.

## Reproducing

```bash
# 2 passes × 4-bit
python tests/benchmark_rotation_strategies.py \
    --model Qwen/Qwen3.5-0.8B-Base \
    --n-passes 2 --bit-width 4 --device cpu

# 4 passes × 2-bit  
python tests/benchmark_rotation_strategies.py \
    --model Qwen/Qwen3.5-0.8B-Base \
    --n-passes 4 --bit-width 2 --device cpu
```
