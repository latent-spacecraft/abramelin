# ESM3 MLX Acceleration Guide

High-performance ESM3 inference on Apple Silicon using MLX with optimized kernels.

## Performance Summary

| Backend | Time/Step | Memory | Speedup vs CPU |
|---------|-----------|--------|----------------|
| **CPU** | 0.83s | 5.3 GB | 1.0x (baseline) |
| **MPS** | 0.37s | 5.3 GB | **2.2x** |
| **MLX FP16** | 0.32s | 5.3 GB | **2.6x** |
| **MLX INT4** | 0.32s | 0.8 GB | **2.6x** + 6.6x memory |

### End-to-End Generation (260 residues, 8 steps)

| Track | Time | Notes |
|-------|------|-------|
| Sequence | ~11s | Fill masked positions |
| Structure | ~4.5s | Fold to 3D coordinates |
| **Total** | ~16s | Full sequence + structure |

### Optimizations Applied

- **Fast SDPA**: Uses `mx.fast.scaled_dot_product_attention` for fused attention
- **Fast RoPE**: Uses `mx.fast.rope` for optimized rotary embeddings
- **Batched Generation**: Generate multiple samples in parallel for throughput
- **Fused Operations**: LayerNorm+Linear and SwiGLU fusion via `mx.compile()`

## Quick Start

### Option 1: ESM3MLX Drop-in Replacement (Recommended)

Full API compatibility with the original ESM3 class:

```python
from esm.models.mlx import ESM3MLX
from esm.sdk.api import ESMProtein, GenerationConfig

# Load model (auto-converts weights if needed)
model = ESM3MLX.from_pretrained("esm3-open")
print(f"Device: {model.device}")  # mlx

# Create protein with masked positions
protein = ESMProtein(sequence="MKTAY____QRQISFVK____RQLEERLGLIEVQ")

# Generate sequence (fills masks)
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
print(f"Generated: {protein.sequence}")

# Generate structure
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
print(f"Coordinates: {protein.coordinates.shape}")  # [L, 37, 3]

# Save PDB
protein.to_pdb("output.pdb")
```

### Option 2: Protein Alchemy Web Interface

Interactive browser-based protein design:

```bash
# Start the web server
python -m esm.web.app

# Open http://localhost:8000
```

Features:
- **3D Viewer**: 3Dmol.js with pLDDT confidence coloring
- **Drag-to-Mask**: Select residues on sequence bar
- **Click in 3D**: Toggle masks by clicking structure
- **Two-Phase Generation**: Sequence → Structure in one click
- **History Panel**: Compare previous generations

### Option 3: Low-Level MLX API

For custom pipelines:

```python
from esm.models.mlx import ESM3MLX
import mlx.core as mx

model = ESM3MLX.from_pretrained("esm3_mlx_weights.npz")

# Direct token-level generation
result = model.generate_sequence(
    sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
    function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
    num_steps=8, temperature=0.7,
)
```

### Option 4: MLX INT4 (Low Memory)

Perfect for 8GB Macs:

```python
from esm.models.mlx import ESM3MLX, quantize_model

model = ESM3MLX.from_pretrained("esm3_mlx_weights.npz")
quantize_model(model, bits=4, group_size=64)
# 6.6x memory reduction with 99.9% accuracy
```

## Sequence Generation

### Single Sequence Generation

```python
from esm.models.mlx import ESM3MLX
import mlx.core as mx

model = ESM3MLX.from_pretrained("esm3_mlx_weights.npz")

# Create masked sequence (32 = mask token)
L = 100
sequence_tokens = mx.full((1, L), 32, dtype=mx.int32)
structure_tokens = mx.full((1, L), 4100, dtype=mx.int32)
# ... other inputs ...

# Generate sequence iteratively
result = model.generate_sequence(
    sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
    function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
    num_steps=8,
    temperature=0.7,
)
```

### Batched Generation (Multiple Samples)

For ensemble generation, use batched generation for higher throughput:

```python
# Generate 16 diverse samples from the same prompt
results = model.generate_sequence_batched(
    sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
    function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
    num_steps=8,
    temperature=0.7,
    num_samples=16,  # Generate 16 samples in parallel
)
# results shape: (16, L)
```

### Generation Performance

| Batch Size | L=100 | Throughput |
|------------|-------|------------|
| 1 | 1.3s | 0.8 seq/s |
| 4 | 4.4s | 0.9 seq/s |
| 8 | 8.3s | 1.0 seq/s |
| 16 | 16.4s | 1.0 seq/s |

**Note**: Larger batches don't increase per-sample speed but allow parallel ensemble generation.

### Why KV Caching Doesn't Help ESM3

Unlike autoregressive language models (GPT, LLaMA), ESM3 uses **bidirectional attention** for iterative refinement. This means:

1. All positions attend to all other positions (not just past tokens)
2. When any position changes, the embeddings for ALL positions change
3. Cached K,V values become stale after each unmasking step

KV caching is implemented for API compatibility but provides minimal speedup (~1%) for ESM3's iterative generation. The main speedup strategies are:

- **Batched generation**: Process multiple samples in parallel
- **INT4 quantization**: 6.6x memory reduction with 99.9% accuracy
- **Fast MLX kernels**: Fused attention and RoPE operations

## Quantization Benchmarks

### Accuracy vs Speed

| Quantization | Memory | Accuracy | Cosine Sim | Best For |
|--------------|--------|----------|------------|----------|
| FP16 | 5.3 GB | 100% | 1.0000 | Maximum accuracy |
| INT8 | 1.5 GB | 100% | 1.0000 | Balanced |
| INT4 | 0.8 GB | 99.9% | 0.9999 | Memory-limited |
| INT2 | 0.5 GB | 0% | 0.9844 | ❌ Not recommended |

### Batch Throughput (L=100)

| Batch Size | FP16 (seq/s) | INT4 (seq/s) |
|------------|--------------|--------------|
| 1 | 6.4 | 6.0 |
| 4 | 7.3 | 7.4 |
| 8 | 7.6 | 7.6 |
| 16 | 7.7 | 6.7 |
| 32 | 7.7 | 4.1 |

**Recommendation:**
- Use **FP16** for maximum throughput with large batches
- Use **INT4** for memory-constrained scenarios (8GB Macs)
- Use **INT8** for balanced memory/speed

## Convert Weights to MLX Format

```bash
# One-time conversion from PyTorch weights
python -m esm.models.mlx.convert --source huggingface --input esm3-open --output esm3_mlx_weights.npz
```

## Files

```
esm/models/mlx/
├── __init__.py      # Module exports (ESM3MLX, quantize_model, fused ops)
├── esm3_mlx.py      # Full MLX ESM3 model - drop-in replacement for ESM3
├── layers.py        # MLX transformer layers with fast kernels
├── fused_ops.py     # Fused operations: SwiGLU, LayerNorm+Linear, compiled blocks
├── convert.py       # PyTorch → MLX weight converter
└── quantize.py      # Quantization utilities (INT4/INT8)

esm/web/
├── app.py           # FastAPI server with WebSocket streaming
├── api/             # REST API routes
├── core/            # Session management, generation service
└── static/
    ├── index.html   # Protein Alchemy SPA
    ├── css/alchemy.css  # Dark theme with pLDDT coloring
    └── js/
        ├── app.js       # Main controller
        ├── viewer3d.js  # 3Dmol.js wrapper with click handlers
        ├── sequenceBar.js   # Drag-to-mask sequence bar
        ├── maskSync.js  # Bidirectional mask state sync
        └── websocket.js # Generation progress streaming

Scripts:
├── test_mlx_api.py        # ESM3MLX drop-in replacement test
├── benchmark_esm3.py      # CPU vs MPS vs MLX benchmark
└── benchmark_kv_cache.py  # Generation throughput benchmark
```

## Deep Profiling Results

### Where Time Is Spent

Per forward pass (L=100, B=1):

| Component | Time | % |
|-----------|------|---|
| **Transformer (48 layers)** | 153ms | 98% |
| Encoder | 1.8ms | 1% |
| Output heads | 1.5ms | 1% |

Per transformer layer (3.4ms):

| Component | Time | % |
|-----------|------|---|
| **FFN W1+W2** | 2.0ms | 60% |
| Attention (SDPA) | 0.4ms | 12% |
| QKV projection | 1.0ms | 28% |

### Precision Analysis

M4 GPU is **FP32-optimized** - lower precision doesn't help speed:

| Precision | Time | Notes |
|-----------|------|-------|
| FP32 | 157ms | Fastest |
| INT4 | 172ms | 6x memory savings |
| BF16 | 244ms | Slower (conversion overhead) |
| FP16 | 241ms | Slower |

### Batch Scaling

| Batch | Total | Per Sample | Throughput |
|-------|-------|------------|------------|
| 1 | 158ms | 158ms | 6.3 fwd/s |
| 8 | 1040ms | 130ms | 7.7 fwd/s |
| 64 | 8348ms | 130ms | 7.7 fwd/s |

Batching improves per-sample efficiency by ~18% (158ms → 130ms).

## F1 Mode: Maximum Performance 🏎️

### Key Findings

**M4 GPU Architecture:**
- FP32 is fastest (not FP16/BF16)
- INT4 quantization is ~8% slower than FP32
- Peak efficiency achieved: ~21% of theoretical 14 TFLOP/s

**Optimal Configuration:**
- Batch size: B=16 (128ms/sample)
- Precision: FP32 (not quantized)
- Generation steps: 2-4 for speed vs 8 for quality

### Throughput Results

| Steps | Samples | Time | Throughput |
|-------|---------|------|------------|
| 2 | 64 | 16.5s | 3.9 seq/s |
| 2 | 128 | 35.5s | 3.6 seq/s |
| 4 | 64 | 32.6s | 2.0 seq/s |
| 8 | 64 | 80.2s | 0.8 seq/s |

### Fused Operations

Implemented in `esm/models/mlx/fused_ops.py`:

```python
from esm.models.mlx.fused_ops import (
    fused_swiglu,           # Inline SwiGLU activation
    fused_layernorm_linear, # LayerNorm + Linear
    compiled_ffn_forward,   # JIT-compiled FFN
    CompiledTransformerBlock,  # Fully compiled block
)
```

**Performance gains from fusion:**
- Compiled FFN: ~6% faster
- Compiled transformer block: ~21% faster at B=1
- Marginal improvement at larger batches (compute-bound)

### Hardware Efficiency Analysis

```
Forward pass breakdown (per layer):
├── FFN W1+W2:     60% (memory-bound matmuls)
├── Attention:     40%
│   ├── QKV proj:  28%
│   ├── SDPA:      12%
│   └── Out proj:  varies
└── LayerNorm:     <1% (using mx.fast.layer_norm)

Memory bandwidth utilization: ~11%
Compute utilization: ~21%
```

### Future Optimization Paths

1. **Model Distillation**: Smaller model with similar quality
2. **Speculative Decoding**: Predict multiple positions
3. **Custom Metal Matmul**: Tiled implementation for better cache utilization
4. **Reduced Step Training**: Train model to converge in 2-4 steps

## Requirements

```bash
pip install mlx mlx-lm
```

Tested on:
- macOS 15 (Sequoia)
- M4 Max with 64GB RAM
- MLX 0.30.0
- PyTorch 2.9.1
