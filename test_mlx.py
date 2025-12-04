#!/usr/bin/env python3
"""
Quick test script for MLX ESM3 inference.

Run: python test_mlx.py
"""

import time
import mlx.core as mx
from mlx.utils import tree_unflatten
from esm.models.mlx.esm3_mlx import ESM3MLX

print("=" * 60)
print("ESM3 MLX Quick Test")
print("=" * 60)

# Load model
print("\nLoading MLX model...")
start = time.time()
model = ESM3MLX()
weights = mx.load("esm3_mlx_weights.npz")
model.update(tree_unflatten(list(weights.items())))
mx.eval(model.parameters())
print(f"Model loaded in {time.time() - start:.2f}s")
print(f"Device: {mx.default_device()}")

# Create a test sequence (similar to user's test)
L = 200  # Sequence length
B = 1    # Batch size

print(f"\nTest sequence length: {L}")

# All mask tokens (for generation)
sequence_tokens = mx.full((B, L), 32, dtype=mx.int32)
structure_tokens = mx.full((B, L), 4100, dtype=mx.int32)
ss8_tokens = mx.full((B, L), 10, dtype=mx.int32)
sasa_tokens = mx.full((B, L), 18, dtype=mx.int32)
function_tokens = mx.zeros((B, L, 8), dtype=mx.int32)
residue_annotation_tokens = mx.zeros((B, L, 16), dtype=mx.int32)
average_plddt = mx.ones((B, L))
per_res_plddt = mx.zeros((B, L))

# Run forward pass
print("\nRunning forward pass...")
start = time.time()
outputs, _ = model(
    sequence_tokens=sequence_tokens,
    structure_tokens=structure_tokens,
    ss8_tokens=ss8_tokens,
    sasa_tokens=sasa_tokens,
    function_tokens=function_tokens,
    residue_annotation_tokens=residue_annotation_tokens,
    average_plddt=average_plddt,
    per_res_plddt=per_res_plddt,
)
mx.eval(outputs)
forward_time = time.time() - start

print(f"Forward pass: {forward_time:.3f}s")

# Get predictions
sequence_logits = outputs["sequence_logits"]
predicted_tokens = mx.argmax(sequence_logits, axis=-1)
mx.eval(predicted_tokens)

print(f"\nOutput shapes:")
for k, v in outputs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape}")

# Simple token-to-AA mapping (for display)
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
predicted_seq = ""
for token in predicted_tokens[0].tolist():
    if 4 <= token < 24:  # Standard amino acid tokens
        predicted_seq += AA_VOCAB[token - 4]
    elif token == 32:  # Mask
        predicted_seq += "_"
    else:
        predicted_seq += "X"

print(f"\nPredicted sequence (first 50): {predicted_seq[:50]}...")

print("\n" + "=" * 60)
print("SUCCESS!")
print("=" * 60)
