# CPU Offload for Residual Pass 2

This document describes the pipelined CPU offload strategy for 4+4 residual quantization — halving the VRAM cost of the residual pass by keeping pass 2 data on pinned CPU memory and streaming it to the GPU asynchronously via CUDA streams.

---

## Motivation

In 4+4 residual quantization, each `TurboQuantLinear` layer stores **two full sets** of quantized data (indices, norms, codebook) — one per pass. For a model with $L$ layers of shape $(M, N)$:

| Component | Per-layer (pass 1 + pass 2) | Total |
|-----------|----------------------------|-------|
| Packed indices (uint8) | $2 \times M \times N/2$ | $L \cdot M \cdot N$ |
| Row norms (float32) | $2 \times M \times 4$ | $8LM$ |
| Codebook (float32) | $2 \times 16 \times 4$ | $128L$ |

The pass 2 data approximately doubles the VRAM footprint compared to single-pass 4-bit. For small-batch inference (batch 1), the PCIe bandwidth is underutilized — the GPU spends most of its time on matmul compute, not memory transfers. CPU offload exploits this idle bandwidth.

### VRAM Savings

With offload, pass 2 data moves to pinned CPU memory. The GPU holds:

- **Pass 1 data** (read-only, permanent — per layer)
- **Shared scratch pool** (2 sets of max-layer-size scratch, shared across all layers via double-buffering)

### VRAM Savings

For a model with $L$ offloaded layers, each with pass 2 size $S_i$ (indices + norms + codebook):

| Mode | GPU VRAM for pass 2 | CPU RAM |
|------|-------------------|---------|
| Non-offloaded | $\sum_{i=1}^{L} S_i$ | $0$ |
| CPU offload (shared scratch) | $2 \times \max_i(S_i)$ | $\sum_{i=1}^{L} S_i$ (pinned) |

The savings grow with the number of layers:

$$\text{VRAM saved} = \sum_{i=1}^{L} S_i - 2 \cdot \max_i(S_i) \approx (L - 2) \cdot \bar{S}$$

For a 24-layer model where all layers are the same size: **~92% reduction** in pass 2 VRAM.

The **double-buffered** (ping-pong) design ensures one scratch slot is written by the H2D copy stream while the other is consumed by the compute kernel, without either blocking.

---

## Architecture

### Data Layout

When `cpu_offload_pass2=True` and `enable_prefetch_chain()` has been called:

| Buffer | Location | Pinned | Scope | Purpose |
|--------|----------|--------|-------|---------|
| `indices_packed` (pass 1) | GPU | — | Per-layer | Permanent, read by kernels |
| `weight_norms` (pass 1) | GPU | — | Per-layer | Permanent |
| `codebook` (pass 1) | GPU | — | Per-layer | Permanent |
| `_pass2_cpu_indices_packed` | CPU | Yes | Per-layer | Source for async H2D copy |
| `_pass2_cpu_weight_norms` | CPU | Yes | Per-layer | Source for async H2D copy |
| `_pass2_cpu_codebook` | CPU | Yes | Per-layer | Source for async H2D copy |
| `SharedScratchPool` slot 0 | GPU | — | **Global** (shared) | Ping scratch buffer |
| `SharedScratchPool` slot 1 | GPU | — | **Global** (shared) | Pong scratch buffer |

Each scratch slot is sized to the **largest** offloaded layer. Layers with smaller pass 2 data use sliced views into the shared buffer.

Pinned (page-locked) memory enables DMA transfers that bypass the CPU page table, achieving full PCIe bandwidth (~12 GB/s for PCIe 3.0 x16, ~25 GB/s for PCIe 4.0 x16).

### CUDA Stream Pipeline

Two CUDA streams operate concurrently:

| Stream | Operations |
|--------|-----------|
| **Default stream** | Pass 1 rotations, fused matmul kernels |
| **Copy stream** (`_copy_stream`, shared) | Async H2D of pass 2 data into scratch pool |

The copy stream runs **in parallel** with compute on the default stream. The default stream only waits for the copy to finish when it actually needs the pass 2 data.

---

## Execution Timeline

### Single Layer (No Prefetch)

```
Copy stream:  ╠══ H2D pass2 ═══╣
Default:      ╠══ pass1 rotations ═══╬══ pass1 kernel ═══╬═ wait ═╬══ pass2 kernel ═══╣
                                                           ↑
                                                     wait_event()
```

The H2D copy overlaps with pass 1 compute. If the pass 1 kernel takes longer than the copy (typical for batch ≥ 1), the wait is free — the event has already been recorded.

### With Next-Layer Prefetch

```
Copy stream:  ╠═ H2D layer₀ pass2 ═╬═ H2D layer₁ pass2 ═╬═ H2D layer₂ pass2 ═╣
Default:      ╠═ L₀ pass1 ═╬═ wait ═╬═ L₀ pass2 ═╬═ L₁ pass1 ═╬═ wait ═╬═ L₁ pass2 ═╣
                               ↑          prefetch L₁ →          ↑
                          wait event₀                        wait event₁
                          (probably free)                    (usually free)
```

Each layer's `forward()` ends by calling `prefetch_pass2()` on the next offloaded layer. Since the copy started during the previous layer's kernel execution, the next layer's H2D is likely complete (or nearly so) by the time it's needed.

A single shared copy stream serializes all H2D transfers, matching the serial PCIe link.

### With Dual-Pass Fused Kernel

The dual-pass fused kernel processes both passes in a single kernel launch per group:

```
Copy stream:  ╠═ H2D pass2 ═══════╣
Default:      ╠═ rotations (both passes) ═╬═ wait ═╬═ dual_fused_kernel ═══╣
                                              ↑
                                         wait_event()
```

The dual kernel needs both passes' data upfront. The wait happens after rotation precomputation, giving the H2D more time to complete.

---

## Latency Impact

### Batch 1 (Token Generation)

For batch 1, the matmul is memory-bandwidth bound on the GPU. A typical layer with $M = 2048$, $N = 2048$:

| Metric | Value |
|--------|-------|
| Pass 2 data size | $2048 \times 1024 + 2048 \times 4 + 64 = 2.1$ MB |
| PCIe 4.0 x16 bandwidth | ~25 GB/s |
| **H2D transfer time** | **~0.08 ms** |
| Pass 1 kernel time (batch 1) | ~0.2–0.5 ms |

The H2D transfer takes ~0.08 ms while the pass 1 kernel takes 0.2–0.5 ms, so the copy is **fully hidden** behind pass 1 compute. Expected latency overhead at batch 1:

$$\Delta t_{\text{batch=1}} \approx 0 \text{ (copy hidden behind compute)}$$

With next-layer prefetch, the copy is started even earlier (during the previous layer), making it virtually guaranteed to complete before needed.

### Batch $B$ (Prompt Processing / Prefill)

For larger batches, the GPU compute time grows linearly while H2D time stays constant:

| Batch Size | Pass 1 Kernel Time | H2D Time | Overhead |
|------------|-------------------|----------|----------|
| 1 | 0.3 ms | 0.08 ms | 0% (hidden) |
| 8 | 0.8 ms | 0.08 ms | 0% (hidden) |
| 32 | 2.5 ms | 0.08 ms | 0% (hidden) |
| 128 | 9 ms | 0.08 ms | 0% (hidden) |
| 512 | 35 ms | 0.08 ms | 0% (hidden) |

The H2D copy is negligible relative to compute at any practical batch size. The overhead becomes measurable only in degenerate cases: very small layers on very high-bandwidth GPUs.

### Potential Overhead Sources

1. **First layer penalty**: The first offloaded layer has no prefetch from a previous layer, so it must start its own H2D copy and wait. This adds ~0.08 ms once per forward pass.
2. **PCIe contention**: If other operations (e.g., embedding lookups, KV cache updates) compete for PCIe bandwidth, the H2D transfer may take longer. In practice, these operations are small.
3. **CUDA event overhead**: `wait_event()` and `record_event()` calls add ~1–2 μs each — negligible.

---

## Memory Budget

### Per Layer

For a layer with `out_features=M`, `in_features=N`:

| Component | GPU (VRAM) | CPU (RAM) |
|-----------|-----------|-----------|
| Pass 1 indices (uint8) | $M \times N/2$ | — |
| Pass 1 norms (fp32) | $4M$ | — |
| Pass 1 codebook (fp32) | $64$ | — |
| Pass 2 pinned indices | — | $M \times N/2$ |
| Pass 2 pinned norms | — | $4M$ |
| Pass 2 pinned codebook | — | $64$ |
| **Per-layer total** | $MN/2 + 4M + 64$ | $MN/2 + 4M + 64$ |

Plus, the shared scratch pool (amortized across all layers):

| Shared | GPU (VRAM) |
|--------|-----------|
| 2 × scratch slots | $2 \times (M_{\max} N_{\max}/2 + 4M_{\max} + 64)$ |

### Comparison: Non-Offloaded vs Offloaded

For $L$ equal-sized layers:

| Mode | GPU total (pass 2) | CPU total |
|------|-------------------|-----------|
| Non-offloaded | $L \times (MN/2 + 4M + 64)$ | $0$ |
| CPU offload | $2 \times (MN/2 + 4M + 64)$ | $L \times (MN/2 + 4M + 64)$ |
| **Savings** | $(L - 2) \times (MN/2 + 4M + 64)$ | — |

### Full Model Example (Qwen3.5-0.8B)

Approximate for 24 transformer layers, typical dimensions $M = N = 2048$:

| Mode | Total VRAM (weights) | Total CPU (pinned) |
|------|---------------------|-------------------|
| bf16 baseline | ~1.6 GB | — |
| 4-bit single | ~0.4 GB | — |
| 4+4 non-offloaded | ~0.8 GB | — |
| 4+4 CPU offload | **~0.43 GB** | ~0.4 GB |

The GPU holds pass 1 (~0.4 GB) + shared scratch pool (~0.03 GB, 2 layers' worth) = ~0.43 GB.
Pass 2 data (~0.4 GB) lives entirely in CPU pinned memory.

---

## Usage

### CLI

```bash
# Quantize with CPU offload
turboquant quantize \
    --model Qwen/Qwen3-0.6B \
    --output ./quantized \
    --residual-bit-width 4 \
    --cpu-offload-pass2

# Evaluate
turboquant eval \
    --model Qwen/Qwen3-0.6B \
    --quantized ./quantized \
    --kld

# Generate
turboquant generate \
    --model Qwen/Qwen3-0.6B \
    --quantized ./quantized \
    --prompt "Hello, world!" \
    --cpu-offload-pass2
```

When loading a saved model that was quantized with `--cpu-offload-pass2`, the flag is stored in the config and applied automatically.

### Python API

```python
from turboquant_model.model import (
    TurboQuantConfig,
    quantize_model,
    enable_prefetch_chain,
    disable_prefetch_chain,
)

config = TurboQuantConfig(
    bit_width=4,
    residual_bit_width=4,
    cpu_offload_pass2=True,
)
model = quantize_model(model, config)
# enable_prefetch_chain is called automatically by quantize_model

# Manual control
disable_prefetch_chain(model)  # remove links
n_links = enable_prefetch_chain(model)  # re-link
```

### Per-Layer Control

```python
for module in model.modules():
    if isinstance(module, TurboQuantLinear) and module.has_residual:
        module.offload_pass2_to_cpu()   # offload
        module.reload_pass2_to_gpu()    # undo offload
```

---

## Synchronization Mechanism

The pipeline uses **CUDA events** for device-side synchronization, avoiding CPU-blocking `stream.synchronize()`:

```
1. Copy stream records event after H2D completes
2. Default stream calls wait_event() before reading scratch buffers
3. wait_event() is non-blocking to the CPU — only the GPU pauses if needed
```

This is critical for performance: `stream.synchronize()` blocks the CPU, preventing it from submitting the next kernel launch. With `wait_event()`, the CPU continues issuing work while the GPU handles the dependency internally.

### Event Lifecycle

```
Layer N forward():
  1. Check _prefetch_event (set by layer N-1) → use it
  2. OR: start H2D on copy_stream, record event
  3. Pre-compute rotations for both passes (default stream)
  4. default_stream.wait_event(copy_done_event)
  5. Launch dual-pass fused kernel
  6. Call layer N+1.prefetch_pass2(copy_stream)
     → starts N+1's H2D, records event in N+1._prefetch_event
```

---

## Dual-Pass Fused Kernel Integration

When both fused dual kernels and CPU offload are active, the execution is:

1. Rotation matrices for **both** passes are precomputed on the default stream
2. Input is rotated with both rotations (or once if seeds match / `SAME_INPUT` optimization)
3. Default stream waits on the H2D event
4. A single fused kernel consumes both passes' data, producing the combined output

The fused kernel avoids the overhead of two separate kernel launches and two separate input loads, complementing the offload pipeline.

---

## Relationship to Other Techniques

- **[Residual Quantization](residual.md)**: CPU offload is specifically designed for 4+4 residual — it offloads the pass 2 data while pass 1 remains on GPU.
- **[Fused Kernels](fused-kernels.md)**: The dual-pass fused kernel (CuTile/Triton) integrates with the offload pipeline, consuming scratch buffers directly.
- **[Norm Compression](norm-compression.md)**: Compressed norms reduce the H2D transfer size, making offload even cheaper.
- **[Entropy Coding](entropy-codec.md)**: Entropy-coded indices must be decoded before the kernel can use them — this decoding happens before the H2D copy.
