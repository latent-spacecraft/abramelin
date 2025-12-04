#!/usr/bin/env python3
"""
Benchmark KV caching for ESM3 MLX generation.

Compares cached vs non-cached generation performance.
"""

import time
import mlx.core as mx
from mlx.utils import tree_unflatten
from esm.models.mlx.esm3_mlx import ESM3MLX

print("=" * 70)
print("ESM3 MLX KV CACHE BENCHMARK")
print("=" * 70)

# Load model
print("\nLoading MLX model...")
start = time.time()
model = ESM3MLX()
weights = mx.load("esm3_mlx_weights.npz")
model.update(tree_unflatten(list(weights.items())))
mx.eval(model.parameters())
print(f"Model loaded in {time.time() - start:.2f}s")
print(f"Device: {mx.default_device()}")

# Test configurations
configs = [
    (100, 8),   # L=100, 8 steps
    (200, 8),   # L=200, 8 steps
    (200, 16),  # L=200, 16 steps
]

for L, num_steps in configs:
    print(f"\n{'='*70}")
    print(f"Sequence Length: {L}, Generation Steps: {num_steps}")
    print("=" * 70)

    B = 1

    # Create inputs - fully masked sequence for generation
    sequence_tokens = mx.full((B, L), 32, dtype=mx.int32)  # All mask tokens
    structure_tokens = mx.full((B, L), 4100, dtype=mx.int32)
    ss8_tokens = mx.full((B, L), 10, dtype=mx.int32)
    sasa_tokens = mx.full((B, L), 18, dtype=mx.int32)
    function_tokens = mx.zeros((B, L, 8), dtype=mx.int32)
    residue_annotation_tokens = mx.zeros((B, L, 16), dtype=mx.int32)
    average_plddt = mx.ones((B, L))
    per_res_plddt = mx.zeros((B, L))

    # Warmup
    print("\nWarming up...")
    _ = model.generate_sequence(
        sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
        function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        num_steps=2, use_cache=True,
    )
    mx.eval(_)

    # Benchmark WITHOUT cache
    print("\n[Without KV Cache]")
    times_no_cache = []
    for trial in range(3):
        # Reset to masked
        seq = mx.full((B, L), 32, dtype=mx.int32)
        mx.eval(seq)

        start = time.time()
        result = model.generate_sequence(
            seq, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
            num_steps=num_steps, use_cache=False, temperature=0.0,
        )
        mx.eval(result)
        elapsed = time.time() - start
        times_no_cache.append(elapsed)
        print(f"  Trial {trial+1}: {elapsed:.3f}s ({elapsed/num_steps*1000:.1f}ms/step)")

    avg_no_cache = sum(times_no_cache) / len(times_no_cache)
    print(f"  Average: {avg_no_cache:.3f}s")

    # Benchmark WITH cache
    print("\n[With KV Cache]")
    times_with_cache = []
    for trial in range(3):
        # Reset to masked
        seq = mx.full((B, L), 32, dtype=mx.int32)
        mx.eval(seq)

        start = time.time()
        result = model.generate_sequence(
            seq, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
            num_steps=num_steps, use_cache=True, temperature=0.0,
        )
        mx.eval(result)
        elapsed = time.time() - start
        times_with_cache.append(elapsed)
        print(f"  Trial {trial+1}: {elapsed:.3f}s ({elapsed/num_steps*1000:.1f}ms/step)")

    avg_with_cache = sum(times_with_cache) / len(times_with_cache)
    print(f"  Average: {avg_with_cache:.3f}s")

    # Summary
    speedup = avg_no_cache / avg_with_cache
    savings = (1 - avg_with_cache / avg_no_cache) * 100
    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Time savings: {savings:.1f}%")

# Batch generation test
print(f"\n{'='*70}")
print("BATCH GENERATION TEST")
print("=" * 70)

L = 100
num_steps = 8

for num_samples in [4, 8, 16, 32]:
    B = 1
    sequence_tokens = mx.full((B, L), 32, dtype=mx.int32)
    structure_tokens = mx.full((B, L), 4100, dtype=mx.int32)
    ss8_tokens = mx.full((B, L), 10, dtype=mx.int32)
    sasa_tokens = mx.full((B, L), 18, dtype=mx.int32)
    function_tokens = mx.zeros((B, L, 8), dtype=mx.int32)
    residue_annotation_tokens = mx.zeros((B, L, 16), dtype=mx.int32)
    average_plddt = mx.ones((B, L))
    per_res_plddt = mx.zeros((B, L))

    print(f"\n[{num_samples} samples, L={L}, steps={num_steps}]")

    start = time.time()
    results = model.generate_sequence_batched(
        sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
        function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        num_steps=num_steps, num_samples=num_samples, temperature=0.7,
    )
    mx.eval(results)
    elapsed = time.time() - start

    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Per sample: {elapsed/num_samples*1000:.1f}ms")
    print(f"  Throughput: {num_samples/elapsed:.1f} seq/s")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
