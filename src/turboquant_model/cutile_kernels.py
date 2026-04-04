"""cuTile fused dequant + matmul kernels for on-the-fly inference.

Optimized reimplementation of triton_kernels.py using NVIDIA cuTile Python
(cuda-tile). Falls back gracefully when cuTile is not available.

Optimizations over the Triton baseline:
  1. Autotune — cuda.tile_experimental.autotune_launch searches tile sizes
     (TB, TN, TK) per problem shape; cached after first run.
  2. Shared-memory codebook — the 16-entry codebook (64 B) stays in
     L1/registers throughout; ct.gather provides implicit caching.
  3. FP16/BF16 + tensor cores — ct.mma leverages tensor cores for
     half-precision inputs, delivering peak TFLOPS.
  4. TF32 tensor cores — for fp32 inputs, uses TF32 (10-bit mantissa)
     for ~2× throughput on Ampere/Ada/Blackwell.
  5. Pre-scaled norms — norms / sqrt(K) computed once on host, eliminating
     per-element division inside the kernel.
  6. Prefetching — cuTile's tile-based load model optimises the memory
     pipeline automatically.
  7. Transpose elimination — accumulates in natural (B, N) layout and
     stores directly, avoiding any transposition of the output.

Main entry points:
  cutile_fused_matmul            — static tile sizes (fast, no tuning overhead)
  cutile_fused_matmul_autotuned  — searches optimal tiles (first call slower)

Supported GPUs: Ampere (sm80), Ada (sm89), Blackwell (sm100+).
Requires NVIDIA Driver r580+ and CUDA Toolkit 13.1+.
"""

from __future__ import annotations

import math
from math import ceil

import torch

try:
    import cuda.tile as ct

    ConstInt = ct.Constant[int]
    _CUTILE_AVAILABLE = True
except ImportError:
    _CUTILE_AVAILABLE = False

# Optional experimental autotuner
_HAS_AUTOTUNE = False
if _CUTILE_AVAILABLE:
    try:
        from cuda.tile_experimental import autotune_launch

        _HAS_AUTOTUNE = True
    except ImportError:
        pass


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# cuTile kernel — defined only when cuda.tile is importable
# ---------------------------------------------------------------------------

if _CUTILE_AVAILABLE:

    @ct.kernel
    def _turboquant_cutile_kernel(
        input_ptr,       # Array: (B, K) pre-rotated activations
        indices_ptr,     # Array: (N, PACKED_K) packed uint8
        codebook_ptr,    # Array: (N_LEVELS,) float32
        norms_ptr,       # Array: (N,) float32 — pre-scaled by 1/scale
        output_ptr,      # Array: (B, N) output
        TB: ConstInt,    # tile size for batch dim
        TN: ConstInt,    # tile size for output (N) dim
        TK: ConstInt,    # tile size for reduction (K) dim, must be even
    ):
        """Fused 4-bit unpack → codebook lookup → matmul → norm rescale.

        Each CTA computes a (TB, TN) output tile by iterating over K in
        chunks of TK.  Packed uint8 indices are gathered, nibble-extracted,
        looked up in the codebook, and fed into ct.mma for tensor-core
        acceleration.
        """
        bid_b = ct.bid(0)
        bid_n = ct.bid(1)

        num_k_tiles = ct.num_tiles(input_ptr, axis=1, shape=(TB, TK))
        acc = ct.full((TB, TN), 0, dtype=ct.float32)
        zero_pad = ct.PaddingMode.ZERO

        # TF32 for fp32 inputs → tensor-core path; native dtype otherwise
        mma_dtype = (
            ct.tfloat32 if input_ptr.dtype == ct.float32 else input_ptr.dtype
        )

        # Row indices for this N-tile (constant across k-iterations)
        rn = bid_n * TN + ct.arange(TN, dtype=ct.int32)

        for k_tile in range(num_k_tiles):
            k_start = k_tile * TK

            # ---- Load input tile: (TB, TK) ----
            inp = ct.load(
                input_ptr,
                index=(bid_b, k_tile),
                shape=(TB, TK),
                padding_mode=zero_pad,
            )

            # ---- Unpack 4-bit weight indices via gather ----
            rk = ct.arange(TK, dtype=ct.int32)
            k_global = k_start + rk
            byte_col = k_global // 2            # byte column in packed array
            is_high = (k_global % 2) == 1       # high-nibble flag

            # Gather packed bytes: (TN, TK) — each byte fetched for its nibble
            packed = ct.gather(
                indices_ptr, (rn[:, None], byte_col[None, :])
            )

            # Extract the correct nibble per K position
            lo = ct.bitwise_and(packed, 0x0F)
            hi = ct.bitwise_and(ct.bitwise_rshift(packed, 4), 0x0F)
            idx = ct.where(is_high[None, :], hi, lo).astype(ct.int32)

            # ---- Codebook lookup: codebook[idx] → (TN, TK) ----
            w_tile = ct.gather(codebook_ptr, idx)

            # ---- Tensor-core MMA: (TB, TK) @ (TK, TN) → (TB, TN) ----
            a = inp.astype(mma_dtype)
            b = ct.transpose(w_tile).astype(mma_dtype)
            acc = ct.mma(a, b, acc)

        # ---- Multiply by pre-scaled norms ----
        norms = ct.load(
            norms_ptr, index=(bid_n,), shape=(TN,), padding_mode=zero_pad
        )
        acc = acc * norms[None, :]

        # ---- Store result ----
        ct.store(
            output_ptr,
            index=(bid_b, bid_n),
            tile=ct.astype(acc, output_ptr.dtype),
        )


# ---------------------------------------------------------------------------
# Python wrapper — static tile sizes
# ---------------------------------------------------------------------------


def cutile_fused_matmul(
    x_rot: torch.Tensor,          # (B, K) pre-rotated input
    indices_packed: torch.Tensor,  # (N, K//2) packed uint8
    codebook: torch.Tensor,        # (n_levels,) float32
    norms: torch.Tensor,           # (N,) float32
    K: int,                        # in_features or group_size
    scale: float | None = None,
) -> torch.Tensor:
    """Fused dequant + matmul via cuTile.

    Drop-in replacement for ``triton_fused_matmul()``.
    Pre-scales norms on host and selects power-of-2 tile sizes.

    Args:
        x_rot: (B, K) pre-rotated activations
        indices_packed: (N, K//2) packed 4-bit weight indices
        codebook: centroids
        norms: per-row weight norms (N,)
        K: dimension of this group (in_features or group_size)
        scale: norm divisor (default: sqrt(K))

    Returns:
        output: (B, N)
    """
    if not _CUTILE_AVAILABLE:
        raise RuntimeError(
            "cuda-tile is not installed. Install with: pip install cuda-tile[tileiras]"
        )

    B = x_rot.shape[0]
    N = indices_packed.shape[0]

    if scale is None:
        scale = math.sqrt(K)

    # Pre-scale norms on host (avoids per-element division in kernel)
    norms_scaled = norms / scale

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    # Power-of-2 tile sizes capped to problem dimensions
    TB = min(32, _next_power_of_2(B))
    TN = min(64, _next_power_of_2(N))
    TK = min(64, _next_power_of_2(K))
    TK = max(TK, 2)  # minimum 2 for packed nibble alignment

    grid = (ceil(B / TB), ceil(N / TN), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _turboquant_cutile_kernel,
        (x_rot, indices_packed, codebook, norms_scaled, output, TB, TN, TK),
    )
    return output


# ---------------------------------------------------------------------------
# Autotuned variant (uses cuda.tile_experimental when available)
# ---------------------------------------------------------------------------


def cutile_fused_matmul_autotuned(
    x_rot: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    K: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Autotuned fused dequant + matmul via cuTile.

    Searches tile sizes {TB, TN, TK} for the best throughput using
    ``cuda.tile_experimental.autotune_launch``.  Results are cached by
    (shape, dtype) so the search only runs once per unique problem shape.

    Falls back to ``cutile_fused_matmul()`` when the autotuner package
    is not installed.

    Args / Returns: same as ``cutile_fused_matmul()``.
    """
    if not _HAS_AUTOTUNE:
        return cutile_fused_matmul(
            x_rot, indices_packed, codebook, norms, K, scale
        )

    B = x_rot.shape[0]
    N = indices_packed.shape[0]

    if scale is None:
        scale = math.sqrt(K)
    norms_scaled = norms / scale

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    # Build search space of (TB, TN, TK) tuples
    def search_space():
        configs = []
        max_tb = min(32, _next_power_of_2(B))
        for tb in [1, 2, 4, 8, 16, 32]:
            if tb > max_tb:
                continue
            for tn in [16, 32, 64, 128]:
                if tn > N:
                    continue
                for tk in [16, 32, 64, 128]:
                    if tk > K or tk < 2:
                        continue
                    configs.append((tb, tn, tk))
        return configs

    def grid_fn(cfg):
        tb, tn, _tk = cfg
        return (ceil(B / tb), ceil(N / tn), 1)

    def args_fn(cfg):
        """Fresh output buffer for each trial (avoids clobbering)."""
        tb, tn, tk = cfg
        tmp = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)
        return (x_rot, indices_packed, codebook, norms_scaled, tmp, tb, tn, tk)

    def launch_args_fn(cfg):
        """Final launch writes into the real output buffer."""
        tb, tn, tk = cfg
        return (
            x_rot, indices_packed, codebook, norms_scaled, output, tb, tn, tk
        )

    autotune_launch(
        torch.cuda.current_stream(),
        grid_fn,
        _turboquant_cutile_kernel,
        args_fn,
        launch_args_fn=launch_args_fn,
        search_space=search_space,
        max_iter=30,
        seed=42,
    )

    return output


# ---------------------------------------------------------------------------
# Dual-pass fused kernel: both residual passes in one launch
# ---------------------------------------------------------------------------

if _CUTILE_AVAILABLE:

    @ct.kernel
    def _turboquant_cutile_dual_kernel(
        # Pass 1
        input1_ptr,       # Array: (B, K)
        indices1_ptr,     # Array: (N, PACKED_K) packed uint8
        codebook1_ptr,    # Array: (N_LEVELS,) float32
        norms1_ptr,       # Array: (N,) float32 — pre-scaled
        # Pass 2
        input2_ptr,       # Array: (B, K)
        indices2_ptr,     # Array: (N, PACKED_K) packed uint8
        codebook2_ptr,    # Array: (N_LEVELS,) float32
        norms2_ptr,       # Array: (N,) float32 — pre-scaled
        # Output
        output_ptr,       # Array: (B, N)
        TB: ConstInt,
        TN: ConstInt,
        TK: ConstInt,     # must be even
        SAME_INPUT: ConstInt,  # 1 if input1==input2 (shared rotation)
    ):
        """Dual-pass fused 4-bit unpack → codebook → matmul → combine.

        Each CTA computes a (TB, TN) output tile processing both passes,
        writing output = acc1*norms1 + acc2*norms2 in a single store.
        """
        bid_b = ct.bid(0)
        bid_n = ct.bid(1)

        num_k_tiles = ct.num_tiles(input1_ptr, axis=1, shape=(TB, TK))
        acc1 = ct.full((TB, TN), 0, dtype=ct.float32)
        acc2 = ct.full((TB, TN), 0, dtype=ct.float32)
        zero_pad = ct.PaddingMode.ZERO

        mma_dtype = (
            ct.tfloat32 if input1_ptr.dtype == ct.float32 else input1_ptr.dtype
        )

        rn = bid_n * TN + ct.arange(TN, dtype=ct.int32)

        for k_tile in range(num_k_tiles):
            k_start = k_tile * TK

            # Shared index math
            rk = ct.arange(TK, dtype=ct.int32)
            k_global = k_start + rk
            byte_col = k_global // 2
            is_high = (k_global % 2) == 1

            # ---- Pass 1 ----
            inp1 = ct.load(
                input1_ptr, index=(bid_b, k_tile),
                shape=(TB, TK), padding_mode=zero_pad,
            )
            packed1 = ct.gather(indices1_ptr, (rn[:, None], byte_col[None, :]))
            lo1 = ct.bitwise_and(packed1, 0x0F)
            hi1 = ct.bitwise_and(ct.bitwise_rshift(packed1, 4), 0x0F)
            idx1 = ct.where(is_high[None, :], hi1, lo1).astype(ct.int32)
            w1 = ct.gather(codebook1_ptr, idx1)
            acc1 = ct.mma(inp1.astype(mma_dtype), ct.transpose(w1).astype(mma_dtype), acc1)

            # ---- Pass 2 ----
            if SAME_INPUT == 1:
                inp2 = inp1
            else:
                inp2 = ct.load(
                    input2_ptr, index=(bid_b, k_tile),
                    shape=(TB, TK), padding_mode=zero_pad,
                )
            packed2 = ct.gather(indices2_ptr, (rn[:, None], byte_col[None, :]))
            lo2 = ct.bitwise_and(packed2, 0x0F)
            hi2 = ct.bitwise_and(ct.bitwise_rshift(packed2, 4), 0x0F)
            idx2 = ct.where(is_high[None, :], hi2, lo2).astype(ct.int32)
            w2 = ct.gather(codebook2_ptr, idx2)
            acc2 = ct.mma(inp2.astype(mma_dtype), ct.transpose(w2).astype(mma_dtype), acc2)

        # Epilogue: combine both passes
        n1 = ct.load(norms1_ptr, index=(bid_n,), shape=(TN,), padding_mode=zero_pad)
        n2 = ct.load(norms2_ptr, index=(bid_n,), shape=(TN,), padding_mode=zero_pad)
        result = acc1 * n1[None, :] + acc2 * n2[None, :]

        ct.store(
            output_ptr, index=(bid_b, bid_n),
            tile=ct.astype(result, output_ptr.dtype),
        )


def cutile_fused_dual_matmul(
    x_rot1: torch.Tensor,          # (B, K) pass 1
    indices1_packed: torch.Tensor,  # (N, K//2) pass 1
    codebook1: torch.Tensor,
    norms1: torch.Tensor,
    x_rot2: torch.Tensor,          # (B, K) pass 2
    indices2_packed: torch.Tensor,  # (N, K//2) pass 2
    codebook2: torch.Tensor,
    norms2: torch.Tensor,
    K: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Dual-pass fused dequant + matmul via cuTile.

    Equivalent to:
        cutile_fused_matmul(x_rot1, idx1, cb1, n1, K, scale)
      + cutile_fused_matmul(x_rot2, idx2, cb2, n2, K, scale)

    but with one kernel launch, one output write, and shared index math.
    """
    if not _CUTILE_AVAILABLE:
        raise RuntimeError(
            "cuda-tile is not installed. Install with: pip install cuda-tile[tileiras]"
        )

    B = x_rot1.shape[0]
    N = indices1_packed.shape[0]
    if scale is None:
        scale = math.sqrt(K)

    norms1_scaled = norms1 / scale
    norms2_scaled = norms2 / scale
    same_input = 1 if x_rot1.data_ptr() == x_rot2.data_ptr() else 0

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot1.device)

    TB = min(32, _next_power_of_2(B))
    TN = min(64, _next_power_of_2(N))
    TK = min(64, _next_power_of_2(K))
    TK = max(TK, 2)

    grid = (ceil(B / TB), ceil(N / TN), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _turboquant_cutile_dual_kernel,
        (
            x_rot1, indices1_packed, codebook1, norms1_scaled,
            x_rot2, indices2_packed, codebook2, norms2_scaled,
            output, TB, TN, TK, same_input,
        ),
    )
    return output
