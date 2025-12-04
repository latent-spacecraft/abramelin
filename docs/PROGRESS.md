# ESM3 MLX Fork - Progress Summary

This fork adds high-performance Apple Silicon support to ESM3 via MLX, plus an interactive web interface for protein design.

## Project Overview

**Goal**: Make ESM3 run fast on Apple Silicon M-series chips, with an intuitive interface for interactive protein design.

**Status**: Fully functional - sequence and structure generation working with 2.6x speedup over CPU.

## What's Been Built

### 1. ESM3MLX - Drop-in Replacement for ESM3

A complete MLX implementation of ESM3 that's API-compatible with the original:

```python
from esm.models.mlx import ESM3MLX
from esm.sdk.api import ESMProtein, GenerationConfig

model = ESM3MLX.from_pretrained("esm3-open")
protein = ESMProtein(sequence="MKTAY____QRQISFVK")
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("output.pdb")
```

**Key files:**
- `esm/models/mlx/esm3_mlx.py` - Main model with `encode()`, `decode()`, `generate()` methods
- `esm/models/mlx/layers.py` - Transformer layers using `mx.fast.*` kernels
- `esm/models/mlx/fused_ops.py` - Fused operations for additional speedup

### 2. Protein Alchemy - Interactive Web Interface

A browser-based protein design workbench:

```bash
python -m esm.web.app
# Open http://localhost:8000
```

**Features:**
- 3Dmol.js 3D viewer with pLDDT confidence coloring
- Drag-to-mask on sequence bar
- Click residues in 3D to toggle masks
- Two-phase generation: Sequence → Structure
- History panel for comparing generations
- Dark theme aesthetic

**Key files:**
- `esm/web/app.py` - FastAPI server with WebSocket streaming
- `esm/web/static/js/viewer3d.js` - 3Dmol.js wrapper with click handlers
- `esm/web/static/js/sequenceBar.js` - Interactive sequence bar
- `esm/web/static/js/maskSync.js` - Bidirectional mask state sync

### 3. Performance Optimizations

| Optimization | Implementation | Speedup |
|-------------|----------------|---------|
| Fast SDPA | `mx.fast.scaled_dot_product_attention` | ~2x attention |
| Fast RoPE | `mx.fast.rope` | ~1.5x embeddings |
| Fast LayerNorm | `mx.fast.layer_norm` | ~2x normalization |
| Fused SwiGLU | Inline activation | ~6% FFN |
| Compiled blocks | `mx.compile()` | ~21% at B=1 |

**Benchmarks (M4 Max, 260 residues, 8 steps):**
- Sequence generation: ~11s
- Structure generation: ~4.5s
- Total: ~16s

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Protein Alchemy Web UI                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   3Dmol.js      │  │  Sequence Bar   │  │  History Panel  │ │
│  │   Viewer        │  │  (drag-to-mask) │  │                 │ │
│  └────────┬────────┘  └────────┬────────┘  └─────────────────┘ │
│           │                    │                                 │
│           └──────────┬─────────┘                                 │
│                      │ MaskSync                                  │
│                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  FastAPI + WebSocket                         ││
│  │                  /ws/generate/{session_id}                   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ESM3MLX Model                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   encode()  │ →  │  generate() │ →  │   decode()  │         │
│  │  ESMProtein │    │  MLX Core   │    │  ESMProtein │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              TransformerStack (48 layers)                    ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        ││
│  │  │ Attn    │→ │ FFN     │→ │ Attn    │→ │ FFN     │→ ...   ││
│  │  │ (SDPA)  │  │(SwiGLU) │  │ (SDPA)  │  │(SwiGLU) │        ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Key Technical Insights

### Why KV Caching Doesn't Help ESM3

Unlike autoregressive LLMs, ESM3 uses **bidirectional attention** for iterative refinement:
- All positions attend to all other positions
- When any position changes, ALL embeddings change
- Cached K,V values become stale after each step

KV caching is implemented for API compatibility but provides only ~1% speedup.

### M4 GPU is FP32-Optimized

Surprising finding: FP32 is faster than FP16/BF16 on M4:

| Precision | Time | Notes |
|-----------|------|-------|
| FP32 | 157ms | Fastest |
| INT4 | 172ms | 6x memory savings |
| BF16/FP16 | ~240ms | Conversion overhead |

### Structure Token BOS/EOS Fix

Structure generation requires preserving special tokens:
- BOS token (4098) at position 0
- EOS token (4097) at position -1
- Only unmask positions with STRUCTURE_MASK_TOKEN (4100)

Fixed in `generate_structure()` method using `mx.where()` masks.

## Files Changed/Added

```
esm/models/mlx/           # NEW - MLX implementation
├── __init__.py
├── esm3_mlx.py           # Main model (850+ lines)
├── layers.py             # Transformer layers
├── fused_ops.py          # Fused operations
├── convert.py            # Weight converter
└── quantize.py           # INT4/INT8 quantization

esm/web/                  # NEW - Web interface
├── app.py                # FastAPI server
├── api/__init__.py
├── core/__init__.py
└── static/
    ├── index.html
    ├── css/alchemy.css
    └── js/*.js           # 5 JavaScript modules

docs/
├── MLX_ACCELERATION.md   # Updated guide
└── PROGRESS.md           # This file
```

## Dependencies Added

```
mlx
mlx-lm
fastapi
uvicorn
websockets
```

## Usage Commands

```bash
# Run tests
python test_mlx_api.py

# Start web interface
python -m esm.web.app

# Convert weights (one-time)
python -m esm.models.mlx.convert --input esm3-open --output esm3_mlx_weights.npz

# Benchmark
python benchmark_esm3.py
```

## Future Work

1. **Model Distillation** - Smaller model with similar quality
2. **Speculative Decoding** - Predict multiple positions per step
3. **Custom Metal Kernels** - Tiled matmul for better cache utilization
4. **Reduced Step Training** - Train model to converge in 2-4 steps
5. **Multi-chain Support** - Handle protein complexes in web UI

## Environment

Tested on:
- macOS 15 (Sequoia)
- M4 Max with 64GB RAM
- MLX 0.30.0
- Python 3.11
