"""
MLX Quantization utilities for ESM3.

Supports 2-bit, 4-bit, and 8-bit quantization with accuracy benchmarking.
"""

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten

from esm.models.mlx.esm3_mlx import ESM3MLX


def quantize_model(
    model: ESM3MLX,
    bits: int = 4,
    group_size: int = 64,
    exclude_embeddings: bool = False,
    verbose: bool = False,
) -> ESM3MLX:
    """Quantize an ESM3MLX model.

    Args:
        model: The ESM3MLX model to quantize
        bits: Number of bits (2, 4, or 8)
        group_size: Quantization group size (larger = less accurate but faster)
        exclude_embeddings: If True, don't quantize embedding layers
        verbose: Print which layers are being quantized

    Returns:
        The quantized model (modified in-place)
    """
    def can_quantize(module: nn.Module, group_size: int) -> bool:
        """Check if a module can be quantized with given group_size."""
        if isinstance(module, nn.Linear):
            # Check if input dimension is divisible by group_size
            weight = module.weight
            # MLX Linear weight is (out_features, in_features)
            in_features = weight.shape[1]
            return in_features % group_size == 0
        elif isinstance(module, nn.Embedding):
            # Embedding dim must be divisible
            return module.weight.shape[1] % group_size == 0
        return False

    def predicate(path: str, module: nn.Module):
        """Determine if module should be quantized and with what params."""
        if exclude_embeddings and isinstance(module, nn.Embedding):
            return False

        if isinstance(module, (nn.Linear, nn.Embedding)):
            # MLX only supports group sizes 32, 64, 128
            for gs in [group_size, 64, 32]:
                if gs > group_size:
                    continue
                if can_quantize(module, gs):
                    if verbose:
                        print(f"  Quantizing {path} with group_size={gs}")
                    return {"group_size": gs, "bits": bits}

            if verbose:
                print(f"  Skipping {path} (dimensions not divisible by 32)")
            return False

        return False

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    return model


def get_model_size(model: nn.Module) -> Tuple[int, float]:
    """Get the number of parameters and approximate size in MB."""
    params = tree_flatten(model.parameters())
    total_params = 0
    total_bytes = 0

    for name, param in params:
        total_params += param.size
        # Estimate bytes based on dtype
        if param.dtype == mx.float32:
            total_bytes += param.size * 4
        elif param.dtype == mx.float16 or param.dtype == mx.bfloat16:
            total_bytes += param.size * 2
        elif param.dtype == mx.uint32:
            # Quantized weights are packed
            total_bytes += param.size * 4
        else:
            total_bytes += param.size * 4  # Default estimate

    return total_params, total_bytes / (1024 * 1024)


def benchmark_inference(
    model: ESM3MLX,
    sequence_length: int = 100,
    batch_size: int = 1,
    num_warmup: int = 2,
    num_iterations: int = 5,
) -> Dict[str, float]:
    """Benchmark model inference speed.

    Returns:
        Dict with timing statistics
    """
    B, L = batch_size, sequence_length

    # Create inputs
    sequence_tokens = mx.full((B, L), 32, dtype=mx.int32)
    structure_tokens = mx.full((B, L), 4100, dtype=mx.int32)
    ss8_tokens = mx.full((B, L), 10, dtype=mx.int32)
    sasa_tokens = mx.full((B, L), 18, dtype=mx.int32)
    function_tokens = mx.zeros((B, L, 8), dtype=mx.int32)
    residue_annotation_tokens = mx.zeros((B, L, 16), dtype=mx.int32)
    average_plddt = mx.ones((B, L))
    per_res_plddt = mx.zeros((B, L))

    # Warmup
    for _ in range(num_warmup):
        outputs, _ = model(
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        )
        mx.eval(outputs)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        outputs, _ = model(
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        )
        mx.eval(outputs)
        times.append(time.time() - start)

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "throughput": batch_size / (sum(times) / len(times)),
    }


def compare_outputs(
    outputs_ref: Dict[str, mx.array],
    outputs_quant: Dict[str, mx.array],
) -> Dict[str, float]:
    """Compare outputs between reference and quantized models.

    Returns:
        Dict with accuracy metrics (cosine similarity, max diff, etc.)
    """
    metrics = {}

    for key in ["sequence_logits", "structure_logits", "embeddings"]:
        if key not in outputs_ref or key not in outputs_quant:
            continue

        ref = outputs_ref[key].astype(mx.float32)
        quant = outputs_quant[key].astype(mx.float32)

        # Flatten for comparison
        ref_flat = ref.reshape(-1)
        quant_flat = quant.reshape(-1)

        # Cosine similarity
        dot = mx.sum(ref_flat * quant_flat)
        norm_ref = mx.sqrt(mx.sum(ref_flat * ref_flat))
        norm_quant = mx.sqrt(mx.sum(quant_flat * quant_flat))
        cos_sim = dot / (norm_ref * norm_quant + 1e-8)

        # Max absolute difference
        max_diff = mx.max(mx.abs(ref_flat - quant_flat))

        # Mean absolute difference
        mean_diff = mx.mean(mx.abs(ref_flat - quant_flat))

        # Relative error
        rel_error = mx.mean(mx.abs(ref_flat - quant_flat) / (mx.abs(ref_flat) + 1e-8))

        mx.eval(cos_sim, max_diff, mean_diff, rel_error)

        metrics[f"{key}_cosine_sim"] = float(cos_sim)
        metrics[f"{key}_max_diff"] = float(max_diff)
        metrics[f"{key}_mean_diff"] = float(mean_diff)
        metrics[f"{key}_rel_error"] = float(rel_error)

    # Check if predictions match
    ref_preds = mx.argmax(outputs_ref["sequence_logits"], axis=-1)
    quant_preds = mx.argmax(outputs_quant["sequence_logits"], axis=-1)
    match_rate = mx.mean((ref_preds == quant_preds).astype(mx.float32))
    mx.eval(match_rate)
    metrics["prediction_match_rate"] = float(match_rate)

    return metrics


def save_quantized_weights(
    model: ESM3MLX,
    output_path: str,
    bits: int,
):
    """Save quantized model weights."""
    params = dict(tree_flatten(model.parameters()))
    mx.savez(output_path, **params)
    print(f"Saved {bits}-bit quantized weights to {output_path}")


def load_quantized_model(
    weights_path: str,
    bits: int = 4,
    group_size: int = 64,
) -> ESM3MLX:
    """Load a quantized ESM3MLX model.

    If loading from original weights, quantizes on the fly.
    If loading from pre-quantized weights, loads directly.
    """
    # Create model
    model = ESM3MLX()

    # Load weights
    weights = mx.load(weights_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    # Check if already quantized by looking at weight dtypes
    params = dict(tree_flatten(model.parameters()))
    sample_weight = list(params.values())[0]

    # If not quantized, quantize now
    if sample_weight.dtype in [mx.float32, mx.float16, mx.bfloat16]:
        print(f"Quantizing to {bits}-bit...")
        quantize_model(model, bits=bits, group_size=group_size)
        mx.eval(model.parameters())

    return model


def run_quantization_benchmark(
    weights_path: str = "esm3_mlx_weights.npz",
    sequence_length: int = 100,
    batch_size: int = 1,
):
    """Run comprehensive quantization benchmark.

    Tests fp16 (baseline), 8-bit, 4-bit, and 2-bit quantization.
    """
    print("=" * 70)
    print("ESM3 MLX QUANTIZATION BENCHMARK")
    print("=" * 70)
    print(f"\nConfig: L={sequence_length}, B={batch_size}")

    results = {}

    # Test configurations: (name, bits, group_size)
    configs = [
        ("FP16 (baseline)", None, None),
        ("INT8 (g=64)", 8, 64),
        ("INT4 (g=64)", 4, 64),
        ("INT4 (g=128)", 4, 128),
        ("INT2 (g=64)", 2, 64),
    ]

    # Create test inputs once
    B, L = batch_size, sequence_length
    test_inputs = {
        "sequence_tokens": mx.full((B, L), 32, dtype=mx.int32),
        "structure_tokens": mx.full((B, L), 4100, dtype=mx.int32),
        "ss8_tokens": mx.full((B, L), 10, dtype=mx.int32),
        "sasa_tokens": mx.full((B, L), 18, dtype=mx.int32),
        "function_tokens": mx.zeros((B, L, 8), dtype=mx.int32),
        "residue_annotation_tokens": mx.zeros((B, L, 16), dtype=mx.int32),
        "average_plddt": mx.ones((B, L)),
        "per_res_plddt": mx.zeros((B, L)),
    }

    # Get reference outputs from FP16 model
    print("\n[Loading FP16 baseline...]")
    model_fp16 = ESM3MLX()
    weights = mx.load(weights_path)
    model_fp16.update(tree_unflatten(list(weights.items())))
    mx.eval(model_fp16.parameters())

    ref_outputs, _ = model_fp16(**test_inputs)
    mx.eval(ref_outputs)

    for name, bits, group_size in configs:
        print(f"\n[{name}]")

        if bits is None:
            # FP16 baseline
            model = model_fp16
        else:
            # Create fresh model and quantize
            model = ESM3MLX()
            model.update(tree_unflatten(list(weights.items())))
            mx.eval(model.parameters())
            quantize_model(model, bits=bits, group_size=group_size)
            mx.eval(model.parameters())

        # Get model size
        num_params, size_mb = get_model_size(model)

        # Benchmark speed
        timing = benchmark_inference(
            model, sequence_length, batch_size,
            num_warmup=2, num_iterations=5
        )

        # Compare accuracy (skip for baseline)
        if bits is not None:
            quant_outputs, _ = model(**test_inputs)
            mx.eval(quant_outputs)
            accuracy = compare_outputs(ref_outputs, quant_outputs)
        else:
            accuracy = {"prediction_match_rate": 1.0, "sequence_logits_cosine_sim": 1.0}

        results[name] = {
            "bits": bits,
            "group_size": group_size,
            "size_mb": size_mb,
            "mean_time": timing["mean_time"],
            "throughput": timing["throughput"],
            "pred_match": accuracy["prediction_match_rate"],
            "cos_sim": accuracy.get("sequence_logits_cosine_sim", 1.0),
        }

        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Speed: {timing['mean_time']*1000:.1f}ms ({timing['throughput']:.1f} seq/s)")
        print(f"  Pred match: {accuracy['prediction_match_rate']*100:.1f}%")
        print(f"  Cosine sim: {accuracy.get('sequence_logits_cosine_sim', 1.0):.4f}")

        # Clean up to save memory
        if bits is not None:
            del model

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Size':>8} {'Speed':>10} {'Match':>8} {'CosSim':>8} {'Speedup':>8}")
    print("-" * 70)

    baseline_time = results["FP16 (baseline)"]["mean_time"]
    for name, r in results.items():
        speedup = baseline_time / r["mean_time"]
        print(f"{name:<20} {r['size_mb']:>6.0f}MB {r['mean_time']*1000:>8.1f}ms {r['pred_match']*100:>7.1f}% {r['cos_sim']:>8.4f} {speedup:>7.2f}x")

    return results


if __name__ == "__main__":
    run_quantization_benchmark()
