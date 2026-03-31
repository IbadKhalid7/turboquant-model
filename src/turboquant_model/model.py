"""Model-level quantization, saving, and loading.

quantize_model:  Replace all nn.Linear → TurboQuantLinear (single-pass or residual)
save_quantized / load_quantized: Serialize/deserialize quantized models to disk
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from turboquant_model.codebook import get_codebook
from turboquant_model.rotation import generate_rotation_matrix
from turboquant_model.quantize import pack_4bit
from turboquant_model.module import TurboQuantLinear
from turboquant_model.rotation import (
    generate_rotation_matrix,
    hadamard_rotate,
    hadamard_rotate_inverse,
)

logger = logging.getLogger(__name__)


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant weight quantization."""

    bit_width: int = 4
    group_size: Optional[int] = 128
    seed: int = 42
    skip_embeddings: bool = False
    skip_lm_head: bool = False
    # Residual
    residual_bit_width: Optional[int] = None
    residual_seed: int = 1042
    # Rotation method: "qr" (Haar random orthogonal) or "hadamard" (fast Walsh-Hadamard + signs)
    rotation: str = "qr"
    # Rotation strategy for residual passes:
    #   "different" — pass 1 uses seed, pass 2 uses residual_seed (default, best quality)
    #   "shared"    — both passes use the same seed (enables merge_and_requantize)
    #   "alternating" — even passes use seed, odd passes use residual_seed (for multi-pass)
    rotation_strategy: str = "different"

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TurboQuantConfig":
        with open(path) as f:
            return cls(**json.load(f))

    @property
    def total_bits(self) -> int:
        return self.bit_width + (self.residual_bit_width or 0)


# ---------------------------------------------------------------------------
# Quantize model
# ---------------------------------------------------------------------------


@torch.no_grad()
def quantize_model(model: nn.Module, config: TurboQuantConfig) -> nn.Module:
    """Quantize all nn.Linear layers, replacing them with TurboQuantLinear.

    Supports single-pass and residual (two-pass) quantization.
    All layers use on-the-fly dequantization at inference.

    Args:
        model: HuggingFace model (or any nn.Module with Linear layers)
        config: quantization configuration

    Returns:
        model with TurboQuantLinear modules (modified in-place)
    """
    centroids, boundaries = get_codebook(config.bit_width)
    if config.residual_bit_width:
        r_centroids, r_boundaries = get_codebook(config.residual_bit_width)

    replaced = 0
    total_orig = 0
    total_compressed = 0

    # Collect modules to replace
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if config.skip_embeddings and "embed" in name.lower():
            continue
        if config.skip_lm_head and "lm_head" in name.lower():
            continue
        replacements.append((name, module))

    for name, module in replacements:
        W = module.weight.data
        M, N = W.shape
        device = W.device

        group_size = config.group_size or N

        # --- Pass 1: Quantize weight ---
        pass1_packed, pass1_norms, pass1_codebook = _quantize_weight(
            W, config.bit_width, group_size, config.seed, centroids, boundaries, device,
            rotation=config.rotation,
        )

        # --- Create TurboQuantLinear ---
        tq = TurboQuantLinear(
            in_features=N,
            out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=device,
            rotation=config.rotation,
        )
        tq.indices_packed.copy_(pass1_packed)
        tq.weight_norms.copy_(pass1_norms)
        tq.codebook.copy_(centroids.to(device))
        tq.set_rotation(config.seed)

        if module.bias is not None:
            tq.bias.copy_(module.bias.data)

        # --- Pass 2: Residual quantization ---
        if config.residual_bit_width:
            # Reconstruct pass 1 to compute residual
            W_hat1 = tq.dequantize().float()
            residual = W.float() - W_hat1

            # Determine residual rotation seed based on strategy
            if config.rotation_strategy == "shared":
                pass2_seed = config.seed
            else:  # "different" or "alternating" — both use residual_seed for pass 2
                pass2_seed = config.residual_seed

            pass2_packed, pass2_norms, pass2_codebook = _quantize_weight(
                residual, config.residual_bit_width, group_size,
                pass2_seed, r_centroids, r_boundaries, device,
                rotation=config.rotation,
            )
            tq.set_pass2(
                indices_packed=pass2_packed,
                weight_norms=pass2_norms,
                codebook=r_centroids.to(device),
                seed=pass2_seed,
            )

        # Replace in model
        _replace_module(model, name, tq)

        orig_bytes = M * N * 2  # bf16
        total_orig += orig_bytes
        total_compressed += tq.memory_bytes()
        replaced += 1

    mode = "residual" if config.residual_bit_width else "single-pass"
    bits = f"{config.bit_width}" if not config.residual_bit_width else f"{config.bit_width}+{config.residual_bit_width}"
    compression_ratio = (
        f"{total_orig / total_compressed:.1f}x" if total_compressed > 0 else "N/A"
    )
    logger.info(
        f"Quantized {replaced} layers ({mode}, {bits}-bit): "
        f"{total_orig / 1024**2:.1f}MB → {total_compressed / 1024**2:.1f}MB "
        f"({compression_ratio} compression)"
    )

    return model


def _quantize_weight(
    W: torch.Tensor,
    bit_width: int,
    group_size: int,
    seed: int,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    device: torch.device,
    rotation: str = "qr",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a single weight matrix and return packed data.

    Returns: (indices_packed, norms, codebook)
    """
    M, N = W.shape
    W = W.float()

    all_norms = []
    all_indices = []

    bnd = boundaries.to(device)
    ctr = centroids.to(device)

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        if rotation == "hadamard":
            Y = hadamard_rotate(W_norm, seed=seed + g_start)
        else:
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
            Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(bnd, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(ctr) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

    if N % 2 != 0:
        full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

    packed = pack_4bit(full_indices)
    return packed, norms_out, ctr


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_quantized(model: nn.Module, config: TurboQuantConfig, save_dir: str | Path):
    """Save quantized model to disk in safetensors format.

    Directory structure:
        save_dir/
        ├── turboquant_config.json
        ├── model.safetensors          # all quantized layer tensors + codebook
        ├── non_quantized.safetensors  # non-linear params (embeddings, norms, etc.)
        └── config.json                # (optional) HuggingFace model config
    """
    from safetensors.torch import save_file

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config.save(save_dir / "turboquant_config.json")

    # Save HF model config
    if hasattr(model, "config"):
        model.config.save_pretrained(save_dir)

    tensors = {}
    codebook_saved = False
    tq_param_prefixes = set()

    for name, module in model.named_modules():
        if isinstance(module, TurboQuantLinear):
            safe = name.replace(".", "_")
            tensors[f"{safe}.indices"] = module.indices_packed.cpu().contiguous()
            tensors[f"{safe}.norms"] = module.weight_norms.cpu().contiguous()

            if module.bias is not None:
                tensors[f"{safe}.bias"] = module.bias.cpu().contiguous()

            if module.has_residual:
                tensors[f"{safe}.pass2_indices"] = module.pass2_indices_packed.cpu().contiguous()
                tensors[f"{safe}.pass2_norms"] = module.pass2_weight_norms.cpu().contiguous()
                tensors[f"{safe}.pass2_codebook"] = module.pass2_codebook.cpu().clone()

            if not codebook_saved:
                tensors["codebook"] = module.codebook.cpu().clone()
                codebook_saved = True

            tq_param_prefixes.add(name + ".")

    save_file(tensors, save_dir / "model.safetensors")

    # Collect non-quantized parameters
    non_quantized = {}
    for pname, param in model.named_parameters():
        is_tq = any(pname.startswith(prefix) for prefix in tq_param_prefixes)
        if not is_tq:
            non_quantized[pname] = param.data.cpu().contiguous()

    for bname, buf in model.named_buffers():
        is_tq = any(bname.startswith(prefix) for prefix in tq_param_prefixes)
        if not is_tq and bname not in non_quantized:
            non_quantized[bname] = buf.cpu().contiguous()

    save_file(non_quantized, save_dir / "non_quantized.safetensors")

    total = sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file())
    logger.info(f"Saved quantized model to {save_dir} ({total / 1024**2:.1f} MB)")


@torch.no_grad()
def load_quantized(
    model_name_or_path: str,
    quantized_dir: str | Path,
    device: str = "cuda",
) -> nn.Module:
    """Load a pre-quantized model from disk.

    Supports both safetensors format (model.safetensors) and legacy
    .pt format (layers/*.pt).

    Args:
        model_name_or_path: HF model name or path (for architecture)
        quantized_dir: directory with saved quantized weights
        device: target device

    Returns:
        model with TurboQuantLinear modules, on-the-fly mode
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    quantized_dir = Path(quantized_dir)
    config = TurboQuantConfig.load(quantized_dir / "turboquant_config.json")

    # Load architecture
    if (quantized_dir / "config.json").exists():
        model_config = AutoConfig.from_pretrained(quantized_dir)
    else:
        model_config = AutoConfig.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_config(model_config).to(torch.bfloat16).to(device)

    # Detect format: safetensors vs legacy .pt
    safetensors_path = quantized_dir / "model.safetensors"
    use_safetensors = safetensors_path.exists()

    if use_safetensors:
        from safetensors.torch import load_file
        tensors = load_file(str(safetensors_path), device=device)
        codebook = tensors["codebook"]
    else:
        tensors = None
        codebook = torch.load(quantized_dir / "codebook.pt", map_location=device, weights_only=True)

    layers_dir = quantized_dir / "layers"

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if config.skip_embeddings and "embed" in name.lower():
            continue
        if config.skip_lm_head and "lm_head" in name.lower():
            continue

        safe = name.replace(".", "_")

        if use_safetensors:
            indices_key = f"{safe}.indices"
            if indices_key not in tensors:
                continue
        else:
            indices_path = layers_dir / f"{safe}.indices.pt"
            if not indices_path.exists():
                continue

        M, N = module.weight.shape
        group_size = config.group_size or N

        tq = TurboQuantLinear(
            in_features=N,
            out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=device,
            rotation=config.rotation,
        )

        if use_safetensors:
            tq.indices_packed = tensors[f"{safe}.indices"]
            tq.weight_norms = tensors[f"{safe}.norms"]
        else:
            tq.indices_packed = torch.load(layers_dir / f"{safe}.indices.pt", map_location=device, weights_only=True)
            tq.weight_norms = torch.load(layers_dir / f"{safe}.norms.pt", map_location=device, weights_only=True)

        tq.codebook = codebook

        if module.bias is not None:
            if use_safetensors:
                bias_key = f"{safe}.bias"
                if bias_key in tensors:
                    tq.bias = tensors[bias_key]
            else:
                bias_path = layers_dir / f"{safe}.bias.pt"
                if bias_path.exists():
                    tq.bias = torch.load(bias_path, map_location=device, weights_only=True)

        tq.set_rotation(config.seed)

        # Load residual pass if present
        if use_safetensors:
            pass2_key = f"{safe}.pass2_indices"
            if pass2_key in tensors:
                tq.set_pass2(
                    indices_packed=tensors[pass2_key],
                    weight_norms=tensors[f"{safe}.pass2_norms"],
                    codebook=tensors[f"{safe}.pass2_codebook"],
                    seed=config.residual_seed,
                )
        else:
            pass2_path = layers_dir / f"{safe}.pass2_indices.pt"
            if pass2_path.exists():
                tq.set_pass2(
                    indices_packed=torch.load(pass2_path, map_location=device, weights_only=True),
                    weight_norms=torch.load(layers_dir / f"{safe}.pass2_norms.pt", map_location=device, weights_only=True),
                    codebook=torch.load(layers_dir / f"{safe}.pass2_codebook.pt", map_location=device, weights_only=True),
                    seed=config.residual_seed,
                )

        _replace_module(model, name, tq)

    # Load non-quantized parameters
    non_quantized_st = quantized_dir / "non_quantized.safetensors"
    if non_quantized_st.exists():
        from safetensors.torch import load_file
        remaining = load_file(str(non_quantized_st), device=device)
    else:
        remaining = torch.load(quantized_dir / "non_quantized.pt", map_location=device, weights_only=True)

    for pname, tensor in remaining.items():
        parts = pname.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        target = getattr(parent, parts[-1], None)
        if target is not None:
            if isinstance(target, nn.Parameter):
                target.data.copy_(tensor)
            elif isinstance(target, torch.Tensor):
                target.copy_(tensor)

    model.eval()
    logger.info(f"Loaded quantized model from {quantized_dir}")
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace_module(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
