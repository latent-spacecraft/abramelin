#!/usr/bin/env python3
"""
Deep profiling of the ESM3 transformer layers.

Since forward pass is 99% of generation time, we need to understand
what's happening inside those 48 layers.
"""

import time
import mlx.core as mx
from mlx.utils import tree_unflatten
from esm.models.mlx import ESM3MLX


def profile_transformer_layers():
    """Profile individual transformer layer components."""
    print("=" * 70)
    print("TRANSFORMER LAYER PROFILING")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    L = 100
    B = 1

    # Create dummy embedding input (what goes into transformer)
    x = mx.random.normal(shape=(B, L, 1536))
    mx.eval(x)

    # Get a single layer
    layer = model.transformer.layers[0]

    print("\n[1] Single layer breakdown (Layer 0)...")
    print("-" * 70)

    # Warmup
    for _ in range(3):
        out, _ = layer(x)
        mx.eval(out)

    # Full layer
    start = time.perf_counter()
    for _ in range(20):
        out, _ = layer(x)
        mx.eval(out)
    layer_time = (time.perf_counter() - start) / 20
    print(f"    Full layer: {layer_time*1000:.2f}ms")

    # Attention component
    attn = layer.attn
    start = time.perf_counter()
    for _ in range(20):
        out, _ = attn(x)
        mx.eval(out)
    attn_time = (time.perf_counter() - start) / 20
    print(f"    Attention: {attn_time*1000:.2f}ms ({attn_time/layer_time*100:.1f}%)")

    # FFN component
    ffn = layer.ffn
    start = time.perf_counter()
    for _ in range(20):
        out = ffn(x)
        mx.eval(out)
    ffn_time = (time.perf_counter() - start) / 20
    print(f"    FFN: {ffn_time*1000:.2f}ms ({ffn_time/layer_time*100:.1f}%)")

    # Breakdown within attention
    print("\n[2] Attention internals...")
    print("-" * 70)

    # LayerNorm
    start = time.perf_counter()
    for _ in range(100):
        x_norm = attn.layernorm(x)
        mx.eval(x_norm)
    ln_time = (time.perf_counter() - start) / 100
    print(f"    LayerNorm: {ln_time*1000:.3f}ms")

    # QKV projection
    start = time.perf_counter()
    for _ in range(100):
        x_norm = attn.layernorm(x)
        qkv = attn.qkv_proj(x_norm)
        mx.eval(qkv)
    qkv_time = (time.perf_counter() - start) / 100 - ln_time
    print(f"    QKV projection: {qkv_time*1000:.3f}ms")

    # Q/K LayerNorm
    x_norm = attn.layernorm(x)
    qkv = attn.qkv_proj(x_norm)
    q, k, v = mx.split(qkv, 3, axis=-1)
    start = time.perf_counter()
    for _ in range(100):
        q_ln = attn.q_ln(q)
        k_ln = attn.k_ln(k)
        mx.eval(q_ln, k_ln)
    qk_ln_time = (time.perf_counter() - start) / 100
    print(f"    Q/K LayerNorm: {qk_ln_time*1000:.3f}ms")

    # Reshape + RoPE
    q = attn.q_ln(q)
    k = attn.k_ln(k)
    n_heads = attn.n_heads
    d_head = attn.d_head

    start = time.perf_counter()
    for _ in range(100):
        q_r = q.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
        k_r = k.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
        v_r = v.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
        q_rope = mx.fast.rope(q_r, dims=d_head, traditional=False, base=10000.0, scale=1.0, offset=0)
        k_rope = mx.fast.rope(k_r, dims=d_head, traditional=False, base=10000.0, scale=1.0, offset=0)
        mx.eval(q_rope, k_rope, v_r)
    rope_time = (time.perf_counter() - start) / 100
    print(f"    Reshape + RoPE: {rope_time*1000:.3f}ms")

    # SDPA
    q_r = q.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
    k_r = k.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
    v_r = v.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
    q_rope = mx.fast.rope(q_r, dims=d_head, traditional=False, base=10000.0, scale=1.0, offset=0)
    k_rope = mx.fast.rope(k_r, dims=d_head, traditional=False, base=10000.0, scale=1.0, offset=0)

    import math
    scale = math.sqrt(1.0 / d_head)

    start = time.perf_counter()
    for _ in range(100):
        attn_out = mx.fast.scaled_dot_product_attention(q_rope, k_rope, v_r, scale=scale)
        mx.eval(attn_out)
    sdpa_time = (time.perf_counter() - start) / 100
    print(f"    Scaled dot-product attention: {sdpa_time*1000:.3f}ms")

    # Output projection
    attn_out = mx.fast.scaled_dot_product_attention(q_rope, k_rope, v_r, scale=scale)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)

    start = time.perf_counter()
    for _ in range(100):
        out = attn.out_proj(attn_out)
        mx.eval(out)
    out_proj_time = (time.perf_counter() - start) / 100
    print(f"    Output projection: {out_proj_time*1000:.3f}ms")

    # FFN breakdown
    print("\n[3] FFN internals...")
    print("-" * 70)

    start = time.perf_counter()
    for _ in range(100):
        x_norm = ffn.layernorm(x)
        mx.eval(x_norm)
    ffn_ln_time = (time.perf_counter() - start) / 100
    print(f"    LayerNorm: {ffn_ln_time*1000:.3f}ms")

    x_norm = ffn.layernorm(x)
    start = time.perf_counter()
    for _ in range(100):
        h = ffn.w1(x_norm)
        mx.eval(h)
    w1_time = (time.perf_counter() - start) / 100
    print(f"    W1 (expand to {ffn.w1.weight.shape[0]}): {w1_time*1000:.3f}ms")

    h = ffn.w1(x_norm)
    start = time.perf_counter()
    for _ in range(100):
        h1, h2 = mx.split(h, 2, axis=-1)
        h_act = mx.sigmoid(h1) * h1 * h2  # SwiGLU
        mx.eval(h_act)
    swiglu_time = (time.perf_counter() - start) / 100
    print(f"    SwiGLU activation: {swiglu_time*1000:.3f}ms")

    h1, h2 = mx.split(h, 2, axis=-1)
    h_act = mx.sigmoid(h1) * h1 * h2
    start = time.perf_counter()
    for _ in range(100):
        out = ffn.w2(h_act)
        mx.eval(out)
    w2_time = (time.perf_counter() - start) / 100
    print(f"    W2 (project back): {w2_time*1000:.3f}ms")

    # Full transformer stack timing
    print("\n[4] Full transformer stack...")
    print("-" * 70)

    start = time.perf_counter()
    for _ in range(5):
        out, emb, _ = model.transformer(x)
        mx.eval(out, emb)
    stack_time = (time.perf_counter() - start) / 5
    print(f"    48 layers total: {stack_time*1000:.1f}ms")
    print(f"    Per layer average: {stack_time/48*1000:.2f}ms")

    # Layer timing variance
    print("\n[5] Per-layer timing (first 10 layers)...")
    print("-" * 70)

    layer_times = []
    for i in range(10):
        layer = model.transformer.layers[i]
        start = time.perf_counter()
        for _ in range(10):
            out, _ = layer(x)
            mx.eval(out)
            x = out  # Use output for next layer (more realistic)
        t = (time.perf_counter() - start) / 10
        layer_times.append(t)
        print(f"    Layer {i}: {t*1000:.2f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Single layer: {layer_time*1000:.2f}ms")
    print(f"    - Attention: {attn_time*1000:.2f}ms ({attn_time/layer_time*100:.0f}%)")
    print(f"    - FFN: {ffn_time*1000:.2f}ms ({ffn_time/layer_time*100:.0f}%)")
    print(f"\n  Attention breakdown:")
    print(f"    - QKV projection: {qkv_time*1000:.2f}ms")
    print(f"    - SDPA: {sdpa_time*1000:.2f}ms")
    print(f"    - Output projection: {out_proj_time*1000:.2f}ms")
    print(f"\n  FFN breakdown:")
    print(f"    - W1 (expand): {w1_time*1000:.2f}ms")
    print(f"    - W2 (project): {w2_time*1000:.2f}ms")

    # Estimate theoretical limits
    print(f"\n  48 layers × {layer_time*1000:.1f}ms = {48*layer_time*1000:.0f}ms theoretical")
    print(f"  Actual: {stack_time*1000:.0f}ms")


def profile_batch_scaling():
    """Profile how batch size affects throughput."""
    print("\n" + "=" * 70)
    print("BATCH SIZE SCALING")
    print("=" * 70)

    model = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    L = 100

    for B in [1, 2, 4, 8, 16, 32]:
        x = mx.random.normal(shape=(B, L, 1536))
        mx.eval(x)

        # Warmup
        for _ in range(2):
            out, emb, _ = model.transformer(x)
            mx.eval(out)

        # Benchmark
        start = time.perf_counter()
        for _ in range(5):
            out, emb, _ = model.transformer(x)
            mx.eval(out)
        elapsed = (time.perf_counter() - start) / 5

        throughput = B / elapsed
        per_sample = elapsed / B * 1000
        print(f"  B={B:2d}: {elapsed*1000:6.1f}ms total, {per_sample:5.1f}ms/sample, {throughput:5.1f} samples/s")


if __name__ == "__main__":
    profile_transformer_layers()
    profile_batch_scaling()
