# MLX implementation of ESM3 for Apple Silicon with KV caching
from esm.models.mlx.esm3_mlx import ESM3MLX
from esm.models.mlx.convert import convert_pytorch_to_mlx
from esm.models.mlx.layers import KVCache, FastLayerNorm
from esm.models.mlx.quantize import quantize_model, load_quantized_model
from esm.models.mlx.fused_ops import (
    fused_swiglu,
    fused_layernorm_linear,
    compiled_ffn_forward,
    FusedSwiGLUFFN,
    CompiledFFN,
    UltraFusedFFN,
    CompiledTransformerBlock,
)

__all__ = [
    # Core model
    "ESM3MLX",
    "convert_pytorch_to_mlx",
    "KVCache",
    # Quantization
    "quantize_model",
    "load_quantized_model",
    # Fused operations (F1 mode)
    "FastLayerNorm",
    "fused_swiglu",
    "fused_layernorm_linear",
    "compiled_ffn_forward",
    "FusedSwiGLUFFN",
    "CompiledFFN",
    "UltraFusedFFN",
    "CompiledTransformerBlock",
]
