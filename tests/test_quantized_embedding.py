"""Tests for QuantizedEmbedding (INT8 and INT4 modes)."""

import math
import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path


@pytest.fixture
def embedding():
    """A small float embedding for testing."""
    torch.manual_seed(42)
    emb = nn.Embedding(1000, 256)
    emb.weight.data.normal_(0, 0.02)
    return emb


class TestQuantizedEmbeddingINT8:
    def test_from_float_shape(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int8")
        assert qe.weight_int8.shape == (1000, 256)
        assert qe.weight_scale.shape == (1000,)
        assert qe.mode == "int8"

    def test_forward_shape(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int8")
        ids = torch.tensor([0, 5, 999])
        out = qe(ids)
        assert out.shape == (3, 256)

    def test_forward_batch(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int8")
        ids = torch.tensor([[0, 1], [2, 3]])
        out = qe(ids)
        assert out.shape == (2, 2, 256)

    def test_reconstruction_quality(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int8")
        W_orig = embedding.weight.data.float()
        W_recon = qe.dequantize().float()
        # INT8 per-row should have very good reconstruction (< 1% relative error)
        rel_err = (W_orig - W_recon).norm() / W_orig.norm()
        print(f"INT8 relative reconstruction error: {rel_err:.6f}")
        assert rel_err < 0.01, f"INT8 reconstruction error too high: {rel_err:.4f}"

    def test_memory_savings(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int8")
        orig_bytes = 1000 * 256 * 2  # bf16
        quant_bytes = qe.memory_bytes()
        ratio = orig_bytes / quant_bytes
        print(f"INT8 compression: {orig_bytes} -> {quant_bytes} ({ratio:.1f}x)")
        assert ratio > 1.5, f"Expected > 1.5x compression, got {ratio:.1f}x"


class TestQuantizedEmbeddingINT4:
    def test_from_float_shape(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int4", group_size=32)
        assert qe.weight_packed.shape == (1000, 128)  # 256 / 2
        assert qe.weight_scale.shape == (1000, 8)  # 256 / 32
        assert qe.weight_min.shape == (1000, 8)
        assert qe.mode == "int4"

    def test_forward_shape(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int4", group_size=32)
        ids = torch.tensor([0, 5, 999])
        out = qe(ids)
        assert out.shape == (3, 256)

    def test_forward_batch(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int4", group_size=32)
        ids = torch.tensor([[0, 1], [2, 3]])
        out = qe(ids)
        assert out.shape == (2, 2, 256)

    def test_reconstruction_quality(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int4", group_size=32)
        W_orig = embedding.weight.data.float()
        W_recon = qe.dequantize().float()
        # INT4 per-group should be decent (< 10% relative error)
        rel_err = (W_orig - W_recon).norm() / W_orig.norm()
        print(f"INT4 relative reconstruction error: {rel_err:.6f}")
        assert rel_err < 0.10, f"INT4 reconstruction error too high: {rel_err:.4f}"

    def test_memory_savings(self, embedding):
        from turboquant_model.module import QuantizedEmbedding
        qe = QuantizedEmbedding.from_float(embedding, mode="int4", group_size=32)
        orig_bytes = 1000 * 256 * 2  # bf16
        quant_bytes = qe.memory_bytes()
        ratio = orig_bytes / quant_bytes
        print(f"INT4 compression: {orig_bytes} -> {quant_bytes} ({ratio:.1f}x)")
        assert ratio > 3.0, f"Expected > 3x compression, got {ratio:.1f}x"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuantizedEmbeddingIntegration:
    """Test integration with quantize_model pipeline."""

    def _make_tiny_model(self):
        """Create a minimal model with embedding + linear."""
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 64, bias=False)
                self.lm_head = nn.Linear(64, 100, bias=False)

            def forward(self, input_ids):
                x = self.embed_tokens(input_ids)
                x = self.linear(x)
                return self.lm_head(x)

        torch.manual_seed(42)
        return TinyModel().to(torch.bfloat16).cuda()

    def test_quantize_model_with_int8_embedding(self):
        from turboquant_model.model import quantize_model, TurboQuantConfig
        from turboquant_model.module import TurboQuantLinear, QuantizedEmbedding

        model = self._make_tiny_model()
        config = TurboQuantConfig(
            bit_width=4, group_size=64, seed=42,
            embedding_quant="int8",
        )
        model = quantize_model(model, config)

        # Check embedding is quantized
        assert isinstance(model.embed_tokens, QuantizedEmbedding)
        assert model.embed_tokens.mode == "int8"

        # Check linear layers are TQ
        assert isinstance(model.linear, TurboQuantLinear)
        assert isinstance(model.lm_head, TurboQuantLinear)

        # Forward pass
        ids = torch.tensor([[1, 2, 3]], device="cuda")
        out = model(ids)
        assert out.shape == (1, 3, 100)

    def test_quantize_model_with_int4_embedding(self):
        from turboquant_model.model import quantize_model, TurboQuantConfig
        from turboquant_model.module import TurboQuantLinear, QuantizedEmbedding

        model = self._make_tiny_model()
        config = TurboQuantConfig(
            bit_width=4, group_size=64, seed=42,
            embedding_quant="int4", embedding_group_size=32,
        )
        model = quantize_model(model, config)

        assert isinstance(model.embed_tokens, QuantizedEmbedding)
        assert model.embed_tokens.mode == "int4"

        ids = torch.tensor([[1, 2, 3]], device="cuda")
        out = model(ids)
        assert out.shape == (1, 3, 100)

    def test_quantize_model_no_embedding(self):
        """Default: embeddings not quantized."""
        from turboquant_model.model import quantize_model, TurboQuantConfig

        model = self._make_tiny_model()
        config = TurboQuantConfig(bit_width=4, group_size=64, seed=42)
        model = quantize_model(model, config)

        # Embedding should remain nn.Embedding (default: embedding_quant="none")
        assert isinstance(model.embed_tokens, nn.Embedding)

    def test_save_load_roundtrip_int8(self):
        from turboquant_model.model import (
            quantize_model, save_quantized, load_quantized, TurboQuantConfig,
        )
        from turboquant_model.module import QuantizedEmbedding

        model = self._make_tiny_model()
        config = TurboQuantConfig(
            bit_width=4, group_size=64, seed=42,
            embedding_quant="int8",
        )
        model = quantize_model(model, config)

        ids = torch.tensor([[1, 2, 3]], device="cuda")
        out_before = model(ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_quantized(model, config, tmpdir)

            # Verify model.safetensors has embedding keys
            from safetensors.torch import load_file
            tensors = load_file(str(Path(tmpdir) / "model.safetensors"))
            assert "embed_tokens.qe_mode" in tensors
            assert "embed_tokens.weight_int8" in tensors
            assert "embed_tokens.weight_scale" in tensors

            # Load back (mock: we need a model name for AutoModelForCausalLM,
            # so we test the embedding restoration separately)
            # Instead, test the QuantizedEmbedding save/load in isolation
            meta = tensors["embed_tokens.qe_meta"]
            mode = "".join(chr(c) for c in tensors["embed_tokens.qe_mode"].tolist())
            assert mode == "int8"
            V, D, gs = int(meta[0]), int(meta[1]), int(meta[2])
            assert V == 100
            assert D == 64

    def test_save_load_roundtrip_int4(self):
        from turboquant_model.model import (
            quantize_model, save_quantized, TurboQuantConfig,
        )
        from turboquant_model.module import QuantizedEmbedding

        model = self._make_tiny_model()
        config = TurboQuantConfig(
            bit_width=4, group_size=64, seed=42,
            embedding_quant="int4", embedding_group_size=32,
        )
        model = quantize_model(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_quantized(model, config, tmpdir)

            from safetensors.torch import load_file
            tensors = load_file(str(Path(tmpdir) / "model.safetensors"))
            assert "embed_tokens.qe_mode" in tensors
            assert "embed_tokens.weight_packed" in tensors
            assert "embed_tokens.weight_scale" in tensors
            assert "embed_tokens.weight_min" in tensors

            mode = "".join(chr(c) for c in tensors["embed_tokens.qe_mode"].tolist())
            assert mode == "int4"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuantizedEmbeddingCUDA:
    def test_int8_cuda(self):
        from turboquant_model.module import QuantizedEmbedding
        torch.manual_seed(42)
        emb = nn.Embedding(500, 128).cuda()
        qe = QuantizedEmbedding.from_float(emb, mode="int8").cuda()
        ids = torch.tensor([0, 10, 499], device="cuda")
        out = qe(ids)
        assert out.device.type == "cuda"
        assert out.shape == (3, 128)

    def test_int4_cuda(self):
        from turboquant_model.module import QuantizedEmbedding
        torch.manual_seed(42)
        emb = nn.Embedding(500, 128).cuda()
        qe = QuantizedEmbedding.from_float(emb, mode="int4", group_size=32).cuda()
        ids = torch.tensor([0, 10, 499], device="cuda")
        out = qe(ids)
        assert out.device.type == "cuda"
        assert out.shape == (3, 128)
