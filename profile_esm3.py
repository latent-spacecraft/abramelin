#!/usr/bin/env python3
"""
Deep profiling of ESM3 generation pipeline.

Identifies bottlenecks beyond the forward pass:
- Tokenization
- Embedding lookup
- Sampling/decoding
- Data transfer
- Python overhead
"""

import time
import cProfile
import pstats
import io
from functools import wraps
from collections import defaultdict

import mlx.core as mx
from mlx.utils import tree_unflatten

# Timing decorator
timings = defaultdict(list)

def timed(name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            timings[name].append(elapsed)
            return result
        return wrapper
    return decorator


def profile_mlx_generation():
    """Profile MLX generation step by step."""
    from esm.models.mlx import ESM3MLX

    print("=" * 70)
    print("MLX GENERATION PROFILING")
    print("=" * 70)

    # Load model
    print("\n[1] Loading model...")
    start = time.perf_counter()
    model = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    load_time = time.perf_counter() - start
    print(f"    Model load: {load_time:.2f}s")

    # Test config
    L = 100
    B = 1
    NUM_STEPS = 8

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
    print("\n[2] Warming up...")
    for _ in range(2):
        outputs, _ = model(
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        )
        mx.eval(outputs)

    # Profile individual components
    print("\n[3] Profiling components...")
    print("-" * 70)

    # 3a. Input creation
    start = time.perf_counter()
    for _ in range(100):
        seq = mx.full((B, L), 32, dtype=mx.int32)
        mx.eval(seq)
    input_create_time = (time.perf_counter() - start) / 100
    print(f"    Input tensor creation: {input_create_time*1000:.3f}ms")

    # 3b. Forward pass only
    start = time.perf_counter()
    for _ in range(10):
        outputs, _ = model(
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        )
        mx.eval(outputs)
    forward_time = (time.perf_counter() - start) / 10
    print(f"    Forward pass: {forward_time*1000:.1f}ms")

    # 3c. Logits extraction
    start = time.perf_counter()
    for _ in range(100):
        logits = outputs["sequence_logits"]
        mx.eval(logits)
    logits_time = (time.perf_counter() - start) / 100
    print(f"    Logits extraction: {logits_time*1000:.3f}ms")

    # 3d. Softmax
    start = time.perf_counter()
    for _ in range(100):
        probs = mx.softmax(outputs["sequence_logits"] / 0.7, axis=-1)
        mx.eval(probs)
    softmax_time = (time.perf_counter() - start) / 100
    print(f"    Softmax: {softmax_time*1000:.3f}ms")

    # 3e. Sampling (Gumbel-max)
    start = time.perf_counter()
    for _ in range(100):
        probs = mx.softmax(outputs["sequence_logits"] / 0.7, axis=-1)
        gumbel = -mx.log(-mx.log(mx.random.uniform(shape=probs.shape) + 1e-10) + 1e-10)
        sampled = mx.argmax(mx.log(probs + 1e-10) + gumbel, axis=-1)
        mx.eval(sampled)
    sample_time = (time.perf_counter() - start) / 100
    print(f"    Sampling (Gumbel-max): {sample_time*1000:.3f}ms")

    # 3f. Argmax only
    start = time.perf_counter()
    for _ in range(100):
        sampled = mx.argmax(outputs["sequence_logits"], axis=-1)
        mx.eval(sampled)
    argmax_time = (time.perf_counter() - start) / 100
    print(f"    Argmax only: {argmax_time*1000:.3f}ms")

    # 3g. Mask operations
    start = time.perf_counter()
    for _ in range(100):
        is_masked = sequence_tokens == 32
        confidence = mx.max(mx.softmax(outputs["sequence_logits"], axis=-1), axis=-1)
        confidence = mx.where(is_masked, confidence, -1e9)
        sorted_idx = mx.argsort(-confidence, axis=-1)
        mx.eval(sorted_idx)
    mask_ops_time = (time.perf_counter() - start) / 100
    print(f"    Mask/confidence ops: {mask_ops_time*1000:.3f}ms")

    # 3h. Token update
    start = time.perf_counter()
    for _ in range(100):
        is_masked = sequence_tokens == 32
        sampled = mx.argmax(outputs["sequence_logits"], axis=-1)
        new_seq = mx.where(is_masked, sampled, sequence_tokens)
        mx.eval(new_seq)
    update_time = (time.perf_counter() - start) / 100
    print(f"    Token update (where): {update_time*1000:.3f}ms")

    # Profile full generation loop
    print("\n[4] Full generation loop breakdown...")
    print("-" * 70)

    step_times = {
        "forward": [],
        "sampling": [],
        "masking": [],
        "update": [],
    }

    sequence_tokens = mx.full((B, L), 32, dtype=mx.int32)
    is_fixed = sequence_tokens != 32

    for step in range(NUM_STEPS):
        # Forward
        t0 = time.perf_counter()
        outputs, _ = model(
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        )
        mx.eval(outputs)
        step_times["forward"].append(time.perf_counter() - t0)

        # Sampling
        t0 = time.perf_counter()
        logits = outputs["sequence_logits"]
        probs = mx.softmax(logits / 0.7, axis=-1)
        gumbel = -mx.log(-mx.log(mx.random.uniform(shape=probs.shape) + 1e-10) + 1e-10)
        sampled = mx.argmax(mx.log(probs + 1e-10) + gumbel, axis=-1)
        mx.eval(sampled)
        step_times["sampling"].append(time.perf_counter() - t0)

        # Masking logic
        t0 = time.perf_counter()
        is_masked = sequence_tokens == 32
        num_masked = mx.sum(is_masked, axis=-1, keepdims=True)
        num_to_unmask = mx.maximum(num_masked // (NUM_STEPS - step), 1)
        confidence = mx.max(probs, axis=-1)
        confidence = mx.where(is_masked, confidence, -1e9)
        sorted_indices = mx.argsort(-confidence, axis=-1)
        position_ranks = mx.argsort(sorted_indices, axis=-1)
        unmask_mask = (position_ranks < num_to_unmask) & is_masked
        mx.eval(unmask_mask)
        step_times["masking"].append(time.perf_counter() - t0)

        # Update
        t0 = time.perf_counter()
        sequence_tokens = mx.where(unmask_mask, sampled, sequence_tokens)
        is_fixed = is_fixed | unmask_mask
        mx.eval(sequence_tokens, is_fixed)
        step_times["update"].append(time.perf_counter() - t0)

    # Print breakdown
    total_gen_time = sum(sum(v) for v in step_times.values())
    print(f"\n    Total generation time: {total_gen_time*1000:.1f}ms ({NUM_STEPS} steps)")
    print(f"\n    Per-step breakdown:")
    for name, times in step_times.items():
        avg = sum(times) / len(times) * 1000
        total = sum(times) * 1000
        pct = total / (total_gen_time * 1000) * 100
        print(f"      {name:12s}: {avg:6.1f}ms avg, {total:7.1f}ms total ({pct:5.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Forward pass dominates at {step_times['forward'][0]*1000:.0f}ms/step")
    print(f"  Sampling + masking overhead: {(sample_time + mask_ops_time)*1000:.1f}ms/step")
    print(f"  Overhead ratio: {(sample_time + mask_ops_time) / forward_time * 100:.1f}%")

    return step_times


def profile_pytorch_mps():
    """Profile PyTorch MPS generation for comparison."""
    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig

    print("\n" + "=" * 70)
    print("PYTORCH MPS GENERATION PROFILING")
    print("=" * 70)

    device = torch.device("mps")
    model = ESM3.from_pretrained("esm3-open", device=device)

    # Profile with cProfile
    print("\n[1] Running cProfile on single generation...")

    prompt = "_" * 100
    protein = ESMProtein(sequence=prompt)

    # Warmup
    _ = model.generate(
        ESMProtein(sequence="_" * 20),
        GenerationConfig(track="sequence", num_steps=2),
    )
    torch.mps.synchronize()

    # Profile
    pr = cProfile.Profile()
    pr.enable()

    result = model.generate(
        protein,
        GenerationConfig(track="sequence", num_steps=4, temperature=0.7),
    )
    torch.mps.synchronize()

    pr.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    return result


def profile_encoder_components():
    """Profile the input encoder separately."""
    from esm.models.mlx.esm3_mlx import ESM3MLX, EncodeInputs

    print("\n" + "=" * 70)
    print("ENCODER COMPONENT PROFILING")
    print("=" * 70)

    # Load just encoder weights
    model = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    encoder = model.encoder

    L = 100
    B = 1

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
    for _ in range(3):
        x = encoder(
            sequence_tokens, structure_tokens, average_plddt, per_res_plddt,
            ss8_tokens, sasa_tokens, function_tokens, residue_annotation_tokens,
        )
        mx.eval(x)

    # Profile
    print("\n  Encoder components:")

    # Sequence embedding
    start = time.perf_counter()
    for _ in range(100):
        x = encoder.sequence_embed(sequence_tokens)
        mx.eval(x)
    seq_time = (time.perf_counter() - start) / 100
    print(f"    sequence_embed: {seq_time*1000:.3f}ms")

    # Structure embedding
    start = time.perf_counter()
    for _ in range(100):
        x = encoder.structure_tokens_embed(structure_tokens)
        mx.eval(x)
    struct_time = (time.perf_counter() - start) / 100
    print(f"    structure_embed: {struct_time*1000:.3f}ms")

    # Function embedding (8 separate)
    start = time.perf_counter()
    for _ in range(100):
        embeds = []
        for i, embed_fn in enumerate(encoder.function_embed):
            embeds.append(embed_fn(function_tokens[..., i]))
        x = mx.concatenate(embeds, axis=-1)
        mx.eval(x)
    func_time = (time.perf_counter() - start) / 100
    print(f"    function_embed (8x): {func_time*1000:.3f}ms")

    # Residue annotation (embedding bag)
    start = time.perf_counter()
    for _ in range(100):
        B, L, N = residue_annotation_tokens.shape
        flat = residue_annotation_tokens.reshape(-1, N)
        embedded = encoder.residue_embed(flat)
        x = embedded.sum(axis=1).reshape(B, L, -1)
        mx.eval(x)
    residue_time = (time.perf_counter() - start) / 100
    print(f"    residue_embed (sum): {residue_time*1000:.3f}ms")

    # Full encoder
    start = time.perf_counter()
    for _ in range(100):
        x = encoder(
            sequence_tokens, structure_tokens, average_plddt, per_res_plddt,
            ss8_tokens, sasa_tokens, function_tokens, residue_annotation_tokens,
        )
        mx.eval(x)
    full_time = (time.perf_counter() - start) / 100
    print(f"    FULL encoder: {full_time*1000:.3f}ms")


def main():
    print("=" * 70)
    print("ESM3 DEEP PROFILING")
    print("=" * 70)

    # MLX profiling
    profile_mlx_generation()

    # Encoder breakdown
    profile_encoder_components()

    # PyTorch comparison (optional - takes longer)
    print("\n\nRun PyTorch MPS profiling? (takes ~30s)")
    # Uncomment to enable:
    # profile_pytorch_mps()


if __name__ == "__main__":
    main()
