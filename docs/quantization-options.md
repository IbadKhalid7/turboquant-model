# Quantization Options & Storage Format

This document describes the different quantization configurations available in TurboQuant, what each produces, and exactly what gets saved to disk.

## Quantization Strategies

### 1. Single-Pass (Default)

The simplest mode: one round of rotation + Lloyd-Max quantization.

```python
config = TurboQuantConfig(bit_width=4, seed=42)
model = quantize_model(model, config)
```

**CLI:**
```bash
turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized --bit-width 4
```

**Per-layer storage:**
| Artifact | Tensor shape | Dtype | Description |
|---|---|---|---|
| `indices_packed` | (M, N/2) | uint8 | Packed 4-bit quantization indices (2 per byte) |
| `weight_norms` | (M,) or (M, n_groups) | float32 | Row/group norms for rescaling |
| `codebook` | (16,) | float32 | Lloyd-Max centroids — shared across all layers |
| `rotation_seed` | scalar | int | Seed for reproducible rotation matrix generation |
| `bias` | (M,) | float32 | Optional, only if the original layer had bias |

**Effective bits per weight:** 4 (+ negligible per-row norm overhead)

---

### 2. Two-Pass Residual (Different Rotations)

Two passes with **different** rotation seeds. Pass 2 quantizes the residual error from pass 1.

```python
config = TurboQuantConfig(
    bit_width=4,
    residual_bit_width=4,
    seed=42,
    residual_seed=1042,
)
model = quantize_model(model, config)
```

**CLI:**
```bash
turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized \
    --bit-width 4 --residual-bit-width 4
```

**Per-layer storage:**
| Artifact | Tensor shape | Dtype | Description |
|---|---|---|---|
| `indices_packed` | (M, N/2) | uint8 | Pass 1 packed indices |
| `weight_norms` | (M,) or (M, n_groups) | float32 | Pass 1 norms |
| `codebook` | (16,) | float32 | Pass 1 codebook (shared across layers) |
| `pass2_indices_packed` | (M, N/2) | uint8 | Pass 2 (residual) packed indices |
| `pass2_weight_norms` | (M,) or (M, n_groups) | float32 | Pass 2 norms |
| `pass2_codebook` | (16,) | float32 | Pass 2 codebook (may differ from pass 1 if bit_width differs) |
| `rotation_seed` | scalar | int | seed_1 = 42 (pass 1 rotation) |
| `pass2_seed` | scalar | int | seed_2 = 1042 (pass 2 rotation) |

**Effective bits per weight:** bit_width_1 + bit_width_2 (e.g., 4+4 = 8)

**Inference cost:** 2 input pre-rotations per group (one per rotation seed).

---

### 3. Multi-Pass Shared Rotation

N passes all using the **same** rotation seed. Enables efficient merging in the rotated domain.

```python
from turboquant_model import multi_residual_quantize_packed, merge_and_requantize

packed = multi_residual_quantize_packed(W, n_passes=3, bit_width=4, seed=42)

# Option A: Exact dense merge (lossless)
W_merged = merge_residual_passes(packed)

# Option B: Re-quantize into single-pass format (lossy but compact)
single = merge_and_requantize(packed, target_bit_width=4)
```

**Storage before merging (N passes):**
| Artifact | Per pass | Description |
|---|---|---|
| `indices_packed` | (M, N/2) uint8 | Packed indices for this pass |
| `weight_norms` | (M,) or (M, n_groups) float32 | Norms for this pass |
| `codebook` | (16,) float32 | Shared codebook |
| `seed` | scalar int | Same seed for all passes |

**Storage after `merge_and_requantize` (single pass):**

Same format as single-pass. All N passes collapsed into one compact representation that only needs **1 rotation** at inference time.

**Key advantage:** The shared rotation lets you:
1. Sum codebook values in the rotated domain (no inverse rotation needed)
2. Re-normalize and re-quantize directly
3. Get multi-pass quality with single-pass inference cost

---

### 4. Alternating Two-Rotation

N passes alternating between two rotation seeds: even passes → `seed_a`, odd passes → `seed_b`.

```python
from turboquant_model import alternating_residual_quantize_packed

packed = alternating_residual_quantize_packed(
    W, n_passes=3, bit_width=4, seed_a=42, seed_b=1042
)

# Dense merge still works (each pass stores its own seed)
W_merged = merge_residual_passes(packed)
```

**Storage (N passes):**
| Artifact | Per pass | Description |
|---|---|---|
| `indices_packed` | (M, N/2) uint8 | Packed indices |
| `weight_norms` | (M,) or (M, n_groups) float32 | Norms |
| `codebook` | (16,) float32 | Codebook |
| `seed` | scalar int | `seed_a` for even passes, `seed_b` for odd |

**Metadata:**
- `seeds: (seed_a, seed_b)` — the two rotation seeds
- `n_passes: int` — number of passes

**Inference cost:** 2 input pre-rotations per group (one per seed), regardless of number of passes. This is O(2) vs O(N) for fully independent rotations.

**Trade-off vs shared:** Cannot use `merge_and_requantize` (different rotation domains prevent rotated-domain merging).

---

## Comparison Table

| Strategy | # Rotations | Merge in rotated domain? | `merge_and_requantize`? | Inference rotations |
|---|---|---|---|---|
| Single-pass | 1 | N/A | N/A | 1 |
| Two-pass (different seeds) | 2 | No | No | 2 |
| Multi-pass shared | 1 | Yes | Yes | 1 (after merge) |
| Alternating 2 | 2 | Partially (per seed group) | No | 2 |
| Independent (N seeds) | N | No | No | N |

---

## Disk Layout

When saving via `save_quantized()`, the directory structure is:

```
save_dir/
├── turboquant_config.json       # TurboQuantConfig as JSON
├── config.json                  # HuggingFace model config (for architecture)
├── codebook.pt                  # (16,) float32 — Lloyd-Max centroids (pass 1)
├── layers/
│   ├── {name}.indices.pt        # (M, N/2) uint8 — packed 4-bit indices
│   ├── {name}.norms.pt          # (M,) or (M, n_groups) float32 — row norms
│   ├── {name}.bias.pt           # (M,) float32 — optional, if layer had bias
│   ├── {name}.pass2_indices.pt  # (M, N/2) uint8 — residual pass indices (optional)
│   ├── {name}.pass2_norms.pt    # (M, n_groups) float32 — residual norms (optional)
│   └── {name}.pass2_codebook.pt # (16,) float32 — residual codebook (optional)
└── non_quantized.pt             # dict of non-Linear parameters (embeddings, LN, etc.)
```

**Layer naming:** Module names like `model.layers.0.self_attn.q_proj` become `model_layers_0_self_attn_q_proj` (dots → underscores).

### turboquant_config.json

```json
{
  "bit_width": 4,
  "group_size": 128,
  "seed": 42,
  "skip_embeddings": false,
  "skip_lm_head": false,
  "residual_bit_width": 4,
  "residual_seed": 1042,
  "rotation": "qr"
}
```

### Size Estimates

For an M×N weight matrix at b bits per weight:

| Component | Size | Formula |
|---|---|---|
| Packed indices | M × N / 2 bytes | 4-bit packing: 2 indices per byte |
| Row norms | M × 4 bytes (1 group) or M × n_groups × 4 bytes | float32 per row per group |
| Codebook | 64 bytes | 16 × float32, shared across all layers |

**Example: Qwen3.5-0.8B-Base**
- Original bf16: ~1434 MB
- Single-pass 4-bit: ~361 MB (4.0× compression)
- Two-pass 4+4: ~722 MB (2.0× compression, but near-lossless quality)
- After `merge_and_requantize`: ~361 MB (multi-pass quality, single-pass size)

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `bit_width` | 4 | Bits per weight for primary quantization pass |
| `group_size` | 128 | Group size for row partitioning (None = full row) |
| `seed` | 42 | Rotation seed for pass 1 |
| `residual_bit_width` | None | Bits for residual pass (None = single-pass only) |
| `residual_seed` | 1042 | Rotation seed for residual pass |
| `rotation` | "qr" | Rotation method: "qr" (Haar random) or "hadamard" (fast Walsh-Hadamard) |
| `skip_embeddings` | False | Skip embedding layers |
| `skip_lm_head` | False | Skip the language model head |
