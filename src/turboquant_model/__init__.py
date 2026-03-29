"""
TurboQuant Model — Near-optimal weight quantization with on-the-fly dequantization.

Based on: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al., 2025, arXiv:2504.19874)

Usage:
    from turboquant_model import TurboQuantConfig, quantize_model, enable_fused_mode

    config = TurboQuantConfig(bit_width=4, seed=42)
    model = quantize_model(model, config)
    # On-the-fly dequant (4x memory savings):
    enable_fused_mode(model)
"""

from turboquant_model.codebook import get_codebook
from turboquant_model.rotation import generate_rotation_matrix
from turboquant_model.quantize import (
    pack_4bit,
    unpack_4bit,
    turboquant_quantize,
    turboquant_quantize_packed,
)
from turboquant_model.residual import (
    residual_quantize,
    residual_quantize_packed,
    multi_residual_quantize,
    multi_residual_quantize_packed,
    merge_residual_passes,
    merge_and_requantize,
)
from turboquant_model.module import TurboQuantLinear
from turboquant_model.model import (
    TurboQuantConfig,
    quantize_model,
    save_quantized,
    load_quantized,
)

__version__ = "0.1.0"

__all__ = [
    # Codebook
    "get_codebook",
    # Rotation
    "generate_rotation_matrix",
    # Quantize
    "pack_4bit",
    "unpack_4bit",
    "turboquant_quantize",
    "turboquant_quantize_packed",
    # Residual
    "residual_quantize",
    "residual_quantize_packed",
    "multi_residual_quantize",
    "multi_residual_quantize_packed",
    "merge_residual_passes",
    "merge_and_requantize",
    # Module
    "TurboQuantLinear",
    # Model
    "TurboQuantConfig",
    "quantize_model",
    "save_quantized",
    "load_quantized",
]
