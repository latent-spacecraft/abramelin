#!/usr/bin/env python3
"""
Benchmark ESM3 inference across CPU, MPS, and MLX.

This script compares inference performance for:
- CPU (PyTorch)
- MPS (PyTorch on Apple Metal)
- MLX (Apple's ML framework)
"""

import time
import torch

def benchmark_pytorch(device_name: str, sequence_length: int = 200, num_steps: int = 4):
    """Benchmark PyTorch ESM3 on specified device."""
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig

    device = torch.device(device_name)
    print(f"\n[{device_name.upper()}]")

    model = ESM3.from_pretrained("esm3-open", device=device)
    print(f"  Model loaded on: {model.device}")

    prompt = "_" * sequence_length
    protein = ESMProtein(sequence=prompt)

    # Warmup
    _ = model.generate(
        ESMProtein(sequence="___AAA___"),
        GenerationConfig(track="sequence", num_steps=1, temperature=0.7),
    )
    if device_name == "mps":
        torch.mps.synchronize()

    # Benchmark
    start = time.time()
    protein_out = model.generate(
        protein, GenerationConfig(track="sequence", num_steps=num_steps, temperature=0.7)
    )
    if device_name == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - start

    return elapsed, elapsed / num_steps


def benchmark_mlx(sequence_length: int = 200, num_steps: int = 4):
    """Benchmark MLX ESM3."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from esm.models.mlx.esm3_mlx import ESM3MLX

    print("\n[MLX]")

    model = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    print(f"  Model loaded on: {mx.default_device()}")

    B, L = 1, sequence_length
    sequence_tokens = mx.full((B, L), 32, dtype=mx.int32)
    structure_tokens = mx.full((B, L), 4100, dtype=mx.int32)
    ss8_tokens = mx.full((B, L), 10, dtype=mx.int32)
    sasa_tokens = mx.full((B, L), 18, dtype=mx.int32)
    function_tokens = mx.zeros((B, L, 8), dtype=mx.int32)
    residue_annotation_tokens = mx.zeros((B, L, 16), dtype=mx.int32)
    average_plddt = mx.ones((B, L))
    per_res_plddt = mx.zeros((B, L))

    # Warmup
    for _ in range(2):
        outputs = model(
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        )
        mx.eval(outputs)

    # Benchmark
    start = time.time()
    for _ in range(num_steps):
        outputs = model(
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        )
        mx.eval(outputs)
    elapsed = time.time() - start

    return elapsed, elapsed / num_steps


def main():
    print("=" * 60)
    print("ESM3 INFERENCE BENCHMARK")
    print("M4 Mac - CPU vs MPS vs MLX")
    print("=" * 60)

    sequence_length = 200
    num_steps = 4

    print(f"\nTest config: L={sequence_length}, steps={num_steps}")

    results = {}

    # CPU benchmark
    cpu_total, cpu_per_step = benchmark_pytorch("cpu", sequence_length, num_steps)
    results["CPU"] = (cpu_total, cpu_per_step)
    print(f"  {num_steps} steps: {cpu_total:.2f}s ({cpu_per_step:.2f}s/step)")

    # MPS benchmark
    if torch.backends.mps.is_available():
        mps_total, mps_per_step = benchmark_pytorch("mps", sequence_length, num_steps)
        results["MPS"] = (mps_total, mps_per_step)
        print(f"  {num_steps} steps: {mps_total:.2f}s ({mps_per_step:.2f}s/step)")
    else:
        print("  MPS not available")

    # MLX benchmark
    try:
        mlx_total, mlx_per_step = benchmark_mlx(sequence_length, num_steps)
        results["MLX"] = (mlx_total, mlx_per_step)
        print(f"  {num_steps} forward passes: {mlx_total:.2f}s ({mlx_per_step:.2f}s/pass)")
    except Exception as e:
        print(f"  MLX error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    baseline = results.get("CPU", (1, 1))[1]
    for name, (total, per_step) in results.items():
        speedup = baseline / per_step
        print(f"{name:5s}: {per_step:.3f}s/step  ({speedup:.1f}x vs CPU)")

    print("\n" + "=" * 60)
    print("SPEEDUPS")
    print("=" * 60)

    if "MPS" in results and "CPU" in results:
        print(f"MPS vs CPU: {results['CPU'][1] / results['MPS'][1]:.2f}x faster")
    if "MLX" in results and "CPU" in results:
        print(f"MLX vs CPU: {results['CPU'][1] / results['MLX'][1]:.2f}x faster")
    if "MLX" in results and "MPS" in results:
        print(f"MLX vs MPS: {results['MPS'][1] / results['MLX'][1]:.2f}x faster")


if __name__ == "__main__":
    main()
