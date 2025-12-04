#!/usr/bin/env python3
"""
Batch generation benchmark: MPS vs MLX for 64 samples.

Establishes baseline throughput for ensemble generation.
"""

import time
import torch

NUM_SAMPLES = 64
SEQUENCE_LENGTH = 100
NUM_STEPS = 8
TEMPERATURE = 0.7


def benchmark_mps(num_samples: int = NUM_SAMPLES):
    """Benchmark MPS (PyTorch) batch generation."""
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig

    print(f"\n{'='*60}")
    print("MPS (PyTorch) Batch Generation")
    print(f"{'='*60}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = ESM3.from_pretrained("esm3-open", device=device)

    # Create prompt
    prompt = "_" * SEQUENCE_LENGTH
    protein = ESMProtein(sequence=prompt)

    # Warmup
    print("Warming up...")
    _ = model.generate(
        ESMProtein(sequence="_" * 20),
        GenerationConfig(track="sequence", num_steps=2, temperature=TEMPERATURE),
    )
    if device.type == "mps":
        torch.mps.synchronize()

    # Generate samples sequentially (MPS doesn't batch well for generation)
    print(f"\nGenerating {num_samples} samples (L={SEQUENCE_LENGTH}, steps={NUM_STEPS})...")
    results = []
    start = time.time()

    for i in range(num_samples):
        protein_in = ESMProtein(sequence=prompt)
        result = model.generate(
            protein_in,
            GenerationConfig(track="sequence", num_steps=NUM_STEPS, temperature=TEMPERATURE),
        )
        results.append(result.sequence)
        if (i + 1) % 16 == 0:
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.time() - start
            print(f"  {i+1}/{num_samples} complete ({elapsed:.1f}s)")

    if device.type == "mps":
        torch.mps.synchronize()
    total_time = time.time() - start

    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Per sample: {total_time/num_samples*1000:.1f}ms")
    print(f"  Throughput: {num_samples/total_time:.2f} seq/s")
    print(f"  Sample[0]: {results[0][:40]}...")

    return total_time, results


def benchmark_mlx(num_samples: int = NUM_SAMPLES):
    """Benchmark MLX batch generation."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from esm.models.mlx import ESM3MLX

    print(f"\n{'='*60}")
    print("MLX Batch Generation")
    print(f"{'='*60}")

    print(f"Device: {mx.default_device()}")

    # Load model
    model = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    # Create inputs
    L = SEQUENCE_LENGTH
    B = 1

    sequence_tokens = mx.full((B, L), 32, dtype=mx.int32)  # All masked
    structure_tokens = mx.full((B, L), 4100, dtype=mx.int32)
    ss8_tokens = mx.full((B, L), 10, dtype=mx.int32)
    sasa_tokens = mx.full((B, L), 18, dtype=mx.int32)
    function_tokens = mx.zeros((B, L, 8), dtype=mx.int32)
    residue_annotation_tokens = mx.zeros((B, L, 16), dtype=mx.int32)
    average_plddt = mx.ones((B, L))
    per_res_plddt = mx.zeros((B, L))

    # Warmup
    print("Warming up...")
    _ = model.generate_sequence(
        mx.full((1, 20), 32, dtype=mx.int32),
        mx.full((1, 20), 4100, dtype=mx.int32),
        mx.full((1, 20), 10, dtype=mx.int32),
        mx.full((1, 20), 18, dtype=mx.int32),
        mx.zeros((1, 20, 8), dtype=mx.int32),
        mx.zeros((1, 20, 16), dtype=mx.int32),
        mx.ones((1, 20)),
        mx.zeros((1, 20)),
        num_steps=2,
        temperature=TEMPERATURE,
    )
    mx.eval(_)

    # Batch generation
    print(f"\nGenerating {num_samples} samples (L={SEQUENCE_LENGTH}, steps={NUM_STEPS})...")
    start = time.time()

    results = model.generate_sequence_batched(
        sequence_tokens,
        structure_tokens,
        ss8_tokens,
        sasa_tokens,
        function_tokens,
        residue_annotation_tokens,
        average_plddt,
        per_res_plddt,
        num_steps=NUM_STEPS,
        temperature=TEMPERATURE,
        num_samples=num_samples,
    )
    mx.eval(results)

    total_time = time.time() - start

    # Convert first result to sequence
    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
    seq = ""
    for token in results[0].tolist():
        if 4 <= token < 24:
            seq += AA_VOCAB[token - 4]
        elif token == 32:
            seq += "_"
        else:
            seq += "X"

    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Per sample: {total_time/num_samples*1000:.1f}ms")
    print(f"  Throughput: {num_samples/total_time:.2f} seq/s")
    print(f"  Sample[0]: {seq[:40]}...")

    return total_time, results


def benchmark_mlx_int4(num_samples: int = NUM_SAMPLES):
    """Benchmark MLX INT4 quantized batch generation."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from esm.models.mlx import ESM3MLX, quantize_model

    print(f"\n{'='*60}")
    print("MLX INT4 Quantized Batch Generation")
    print(f"{'='*60}")

    print(f"Device: {mx.default_device()}")

    # Load and quantize model
    print("Loading and quantizing model...")
    model = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    quantize_model(model, bits=4, group_size=64)
    mx.eval(model.parameters())

    # Create inputs
    L = SEQUENCE_LENGTH
    B = 1

    sequence_tokens = mx.full((B, L), 32, dtype=mx.int32)
    structure_tokens = mx.full((B, L), 4100, dtype=mx.int32)
    ss8_tokens = mx.full((B, L), 10, dtype=mx.int32)
    sasa_tokens = mx.full((B, L), 18, dtype=mx.int32)
    function_tokens = mx.zeros((B, L, 8), dtype=mx.int32)
    residue_annotation_tokens = mx.zeros((B, L, 16), dtype=mx.int32)
    average_plddt = mx.ones((B, L))
    per_res_plddt = mx.zeros((B, L))

    # Warmup
    print("Warming up...")
    _ = model.generate_sequence(
        mx.full((1, 20), 32, dtype=mx.int32),
        mx.full((1, 20), 4100, dtype=mx.int32),
        mx.full((1, 20), 10, dtype=mx.int32),
        mx.full((1, 20), 18, dtype=mx.int32),
        mx.zeros((1, 20, 8), dtype=mx.int32),
        mx.zeros((1, 20, 16), dtype=mx.int32),
        mx.ones((1, 20)),
        mx.zeros((1, 20)),
        num_steps=2,
        temperature=TEMPERATURE,
    )
    mx.eval(_)

    # Batch generation
    print(f"\nGenerating {num_samples} samples (L={SEQUENCE_LENGTH}, steps={NUM_STEPS})...")
    start = time.time()

    results = model.generate_sequence_batched(
        sequence_tokens,
        structure_tokens,
        ss8_tokens,
        sasa_tokens,
        function_tokens,
        residue_annotation_tokens,
        average_plddt,
        per_res_plddt,
        num_steps=NUM_STEPS,
        temperature=TEMPERATURE,
        num_samples=num_samples,
    )
    mx.eval(results)

    total_time = time.time() - start

    # Convert first result to sequence
    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
    seq = ""
    for token in results[0].tolist():
        if 4 <= token < 24:
            seq += AA_VOCAB[token - 4]
        elif token == 32:
            seq += "_"
        else:
            seq += "X"

    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Per sample: {total_time/num_samples*1000:.1f}ms")
    print(f"  Throughput: {num_samples/total_time:.2f} seq/s")
    print(f"  Sample[0]: {seq[:40]}...")

    return total_time, results


def main():
    print("=" * 60)
    print(f"BATCH GENERATION BENCHMARK: {NUM_SAMPLES} SAMPLES")
    print(f"Sequence Length: {SEQUENCE_LENGTH}, Steps: {NUM_STEPS}")
    print("=" * 60)

    results = {}

    # MPS benchmark
    try:
        mps_time, _ = benchmark_mps()
        results["MPS"] = mps_time
    except Exception as e:
        print(f"MPS benchmark failed: {e}")

    # MLX FP16 benchmark
    try:
        mlx_time, _ = benchmark_mlx()
        results["MLX FP16"] = mlx_time
    except Exception as e:
        print(f"MLX benchmark failed: {e}")

    # MLX INT4 benchmark
    try:
        mlx_int4_time, _ = benchmark_mlx_int4()
        results["MLX INT4"] = mlx_int4_time
    except Exception as e:
        print(f"MLX INT4 benchmark failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Backend':<15} {'Total Time':>12} {'Per Sample':>12} {'Throughput':>12}")
    print("-" * 60)

    baseline = results.get("MPS", 1)
    for name, total_time in results.items():
        per_sample = total_time / NUM_SAMPLES * 1000
        throughput = NUM_SAMPLES / total_time
        speedup = baseline / total_time
        print(f"{name:<15} {total_time:>10.2f}s {per_sample:>10.1f}ms {throughput:>10.2f}/s  ({speedup:.1f}x)")

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
