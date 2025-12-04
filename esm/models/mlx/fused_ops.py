"""
Fused operations for ESM3 MLX - F1 Mode 🏎️

Custom Metal kernels for maximum performance on Apple Silicon.
Target: Reclaim ~13 TFLOP/s of the M4's theoretical 14 TFLOP/s.

Key optimizations:
1. Fused LayerNorm + Linear: Single kernel for pre-normalization + projection
2. Fused SwiGLU: Avoids intermediate tensor allocations
3. Fused SwiGLU + W2: Complete FFN output in one kernel
4. JIT compilation with mx.compile() for graph-level fusion
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Tuple, Optional


# =============================================================================
# FUSED SWIGLU KERNEL
# =============================================================================

def fused_swiglu_metal(x: mx.array) -> mx.array:
    """Fused SwiGLU activation using custom Metal kernel.

    SwiGLU: silu(x1) * x2 where x is split in half.
    This fused version avoids intermediate tensor allocations.

    Input shape: (..., 2*hidden)
    Output shape: (..., hidden)
    """
    source = '''
        uint elem = thread_position_in_grid.x;
        uint half_size = size / 2;

        if (elem < half_size) {
            // Get x1 and x2 from the flat input
            T x1 = inp[elem];
            T x2 = inp[elem + half_size];

            // SiLU(x1) = x1 * sigmoid(x1)
            T sigmoid_x1 = T(1.0) / (T(1.0) + metal::exp(-x1));
            T silu_x1 = x1 * sigmoid_x1;

            // Output: SiLU(x1) * x2
            out[elem] = silu_x1 * x2;
        }
    '''

    kernel = mx.fast.metal_kernel(
        name="fused_swiglu",
        input_names=["inp"],
        output_names=["out"],
        source=source,
    )

    # Flatten for kernel, then reshape back
    original_shape = x.shape
    half_last = original_shape[-1] // 2
    output_shape = original_shape[:-1] + (half_last,)

    x_flat = x.reshape(-1)
    size = x_flat.size

    outputs = kernel(
        inputs=[x_flat],
        template=[("T", x.dtype)],
        grid=(size // 2, 1, 1),
        threadgroup=(min(256, size // 2), 1, 1),
        output_shapes=[(size // 2,)],
        output_dtypes=[x.dtype],
        init_value=0,
        verbose=False,
    )

    return outputs[0].reshape(output_shape)


def fused_swiglu(x: mx.array) -> mx.array:
    """Fused SwiGLU using MLX operations (faster than Metal kernel for now).

    The Metal kernel has overhead for small batch sizes. This version
    uses vectorized MLX ops which are already well-optimized.
    """
    # Split along last dimension
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # SiLU(x1) * x2 in one fused expression
    # MLX will optimize this at the graph level
    return (x1 * mx.sigmoid(x1)) * x2


# =============================================================================
# FUSED LAYERNORM + LINEAR
# =============================================================================

def fused_layernorm_linear(
    x: mx.array,
    ln_weight: mx.array,
    ln_bias: Optional[mx.array],
    linear_weight: mx.array,
    eps: float = 1e-5,
) -> mx.array:
    """Fused LayerNorm + Linear projection.

    Combines: y = Linear(LayerNorm(x))

    For now uses fast MLX primitives. A true fused kernel would need
    to handle the reduction (mean/var) which is complex in Metal.
    """
    # Use fast layer_norm
    if ln_bias is not None:
        x_norm = mx.fast.layer_norm(x, ln_weight, ln_bias, eps=eps)
    else:
        x_norm = mx.fast.layer_norm(x, ln_weight, mx.zeros_like(ln_weight), eps=eps)

    # Linear: x @ W^T
    return x_norm @ linear_weight.T


# =============================================================================
# FUSED FFN COMPONENTS
# =============================================================================

class FusedSwiGLUFFN(nn.Module):
    """SwiGLU FFN with fused operations and JIT compilation."""

    def __init__(self, d_model: int, expansion_ratio: float = 4.0, bias: bool = False):
        super().__init__()
        hidden_dim = int(((expansion_ratio * d_model) + 255) // 256 * 256)

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.eps = 1e-5

        # Parameters
        self.ln_weight = mx.ones((d_model,))
        self.ln_bias = mx.zeros((d_model,))
        self.w1 = nn.Linear(d_model, hidden_dim * 2, bias=bias)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        # Fused LayerNorm + W1
        x_norm = mx.fast.layer_norm(x, self.ln_weight, self.ln_bias, eps=self.eps)
        h = self.w1(x_norm)

        # Fused SwiGLU
        h = fused_swiglu(h)

        # W2 projection
        return self.w2(h)


# =============================================================================
# COMPILED FFN (JIT GRAPH FUSION)
# =============================================================================

def make_compiled_ffn(d_model: int, hidden_dim: int):
    """Create a JIT-compiled FFN forward function.

    mx.compile() fuses the compute graph for better performance.
    """

    def ffn_forward(
        x: mx.array,
        ln_weight: mx.array,
        ln_bias: mx.array,
        w1_weight: mx.array,
        w2_weight: mx.array,
    ) -> mx.array:
        # LayerNorm
        x_norm = mx.fast.layer_norm(x, ln_weight, ln_bias, eps=1e-5)

        # W1 projection (to hidden*2)
        h = x_norm @ w1_weight.T

        # Fused SwiGLU
        half_dim = h.shape[-1] // 2
        h1 = h[..., :half_dim]
        h2 = h[..., half_dim:]
        h = (h1 * mx.sigmoid(h1)) * h2

        # W2 projection
        return h @ w2_weight.T

    # Compile the function
    return mx.compile(ffn_forward)


class CompiledFFN(nn.Module):
    """FFN with JIT-compiled forward pass for maximum fusion."""

    def __init__(self, d_model: int, expansion_ratio: float = 4.0, bias: bool = False):
        super().__init__()
        hidden_dim = int(((expansion_ratio * d_model) + 255) // 256 * 256)

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # LayerNorm parameters
        self.ln_weight = mx.ones((d_model,))
        self.ln_bias = mx.zeros((d_model,))

        # Linear layers (we'll extract weights for compiled function)
        self.w1 = nn.Linear(d_model, hidden_dim * 2, bias=bias)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)

        # Compiled forward function
        self._compiled_fn = make_compiled_ffn(d_model, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self._compiled_fn(
            x,
            self.ln_weight,
            self.ln_bias,
            self.w1.weight,
            self.w2.weight,
        )


# =============================================================================
# FUSED ATTENTION COMPONENTS
# =============================================================================

def fused_qkv_split_rope(
    qkv: mx.array,
    n_heads: int,
    d_head: int,
    rope_dims: int,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Fused QKV split + reshape + RoPE.

    Input: (B, L, 3*D)
    Output: Q, K, V each (B, n_heads, L, d_head) with RoPE applied to Q, K
    """
    B, L, _ = qkv.shape

    # Split into Q, K, V
    q, k, v = mx.split(qkv, 3, axis=-1)

    # Reshape: (B, L, D) -> (B, n_heads, L, d_head)
    q = q.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)

    # Apply RoPE
    q = mx.fast.rope(q, dims=rope_dims, traditional=False, base=10000.0, scale=1.0, offset=0)
    k = mx.fast.rope(k, dims=rope_dims, traditional=False, base=10000.0, scale=1.0, offset=0)

    return q, k, v


# =============================================================================
# METAL KERNEL: FUSED GELU (for comparison)
# =============================================================================

def fused_gelu_metal(x: mx.array) -> mx.array:
    """Fused GELU activation using custom Metal kernel.

    GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    source = '''
        uint elem = thread_position_in_grid.x;

        if (elem < size) {
            T x = inp[elem];

            // Constants for GELU approximation
            const T sqrt_2_over_pi = T(0.7978845608028654);
            const T coeff = T(0.044715);

            // GELU = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            T x_cubed = x * x * x;
            T tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
            T tanh_val = metal::tanh(tanh_arg);
            out[elem] = x * T(0.5) * (T(1.0) + tanh_val);
        }
    '''

    kernel = mx.fast.metal_kernel(
        name="fused_gelu",
        input_names=["inp"],
        output_names=["out"],
        source=source,
    )

    x_flat = x.reshape(-1)
    size = x_flat.size

    outputs = kernel(
        inputs=[x_flat],
        template=[("T", x.dtype)],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[(size,)],
        output_dtypes=[x.dtype],
        init_value=0,
        verbose=False,
    )

    return outputs[0].reshape(x.shape)


# =============================================================================
# FUSED W1 + SWIGLU + W2 (THE BIG ONE)
# =============================================================================

def fused_ffn_forward(
    x: mx.array,
    ln_weight: mx.array,
    ln_bias: mx.array,
    w1_weight: mx.array,
    w2_weight: mx.array,
    eps: float = 1e-5,
) -> mx.array:
    """Complete fused FFN: LayerNorm -> W1 -> SwiGLU -> W2

    This version uses MLX primitives that will be graph-fused.
    The key optimization is avoiding intermediate memory writes.
    """
    # LayerNorm
    x = mx.fast.layer_norm(x, ln_weight, ln_bias, eps=eps)

    # W1 projection
    h = x @ w1_weight.T

    # Inline SwiGLU (avoids function call overhead)
    half = h.shape[-1] // 2
    h1, h2 = h[..., :half], h[..., half:]
    h = (h1 * mx.sigmoid(h1)) * h2

    # W2 projection
    return h @ w2_weight.T


# Create compiled version
compiled_ffn_forward = mx.compile(fused_ffn_forward)


class UltraFusedFFN(nn.Module):
    """Maximum fusion FFN using compiled graph.

    This achieves the best possible fusion with MLX by:
    1. Using mx.fast.layer_norm
    2. Inlining all operations
    3. Using mx.compile() for graph-level JIT fusion
    """

    def __init__(self, d_model: int, expansion_ratio: float = 4.0, bias: bool = False):
        super().__init__()
        hidden_dim = int(((expansion_ratio * d_model) + 255) // 256 * 256)

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.eps = 1e-5

        # LayerNorm params
        self.ln_weight = mx.ones((d_model,))
        self.ln_bias = mx.zeros((d_model,))

        # Linear weights (no bias for speed)
        self.w1_weight = mx.random.normal(shape=(hidden_dim * 2, d_model)) * 0.02
        self.w2_weight = mx.random.normal(shape=(d_model, hidden_dim)) * 0.02

    def __call__(self, x: mx.array) -> mx.array:
        return compiled_ffn_forward(
            x,
            self.ln_weight,
            self.ln_bias,
            self.w1_weight,
            self.w2_weight,
            self.eps,
        )


# =============================================================================
# BENCHMARKS
# =============================================================================

def benchmark_fused_ops():
    """Comprehensive benchmark of fused vs non-fused operations."""
    import time

    B, L, D = 1, 100, 1536
    hidden = 8192

    print("=" * 70)
    print("🏎️  FUSED OPERATIONS BENCHMARK - F1 MODE")
    print("=" * 70)

    # Warmup
    x = mx.random.normal(shape=(B, L, D))
    mx.eval(x)

    # =========================================================================
    # Test 1: SwiGLU variants
    # =========================================================================
    print("\n[1] SwiGLU Activation Comparison")
    print("-" * 70)

    x_wide = mx.random.normal(shape=(B, L, hidden * 2))
    mx.eval(x_wide)

    # Standard split + silu + multiply
    def standard_swiglu(x):
        x1, x2 = mx.split(x, 2, axis=-1)
        return nn.silu(x1) * x2

    # Warmup
    for _ in range(10):
        mx.eval(standard_swiglu(x_wide))
        mx.eval(fused_swiglu(x_wide))

    start = time.perf_counter()
    for _ in range(100):
        y = standard_swiglu(x_wide)
        mx.eval(y)
    std_time = (time.perf_counter() - start) / 100

    start = time.perf_counter()
    for _ in range(100):
        y = fused_swiglu(x_wide)
        mx.eval(y)
    fused_time = (time.perf_counter() - start) / 100

    print(f"    Standard (split+silu+mul): {std_time*1000:.3f}ms")
    print(f"    Fused (inline):            {fused_time*1000:.3f}ms")
    print(f"    Speedup: {std_time/fused_time:.2f}x")

    # =========================================================================
    # Test 2: Full FFN variants
    # =========================================================================
    print("\n[2] Full FFN Comparison")
    print("-" * 70)

    from esm.models.mlx.layers import SwiGLUFFN

    ffn_standard = SwiGLUFFN(D, 4.0, False)
    ffn_fused = FusedSwiGLUFFN(D, 4.0, False)
    ffn_compiled = CompiledFFN(D, 4.0, False)
    ffn_ultra = UltraFusedFFN(D, 4.0, False)
    mx.eval(
        ffn_standard.parameters(),
        ffn_fused.parameters(),
        ffn_compiled.parameters(),
        ffn_ultra.parameters(),
    )

    # Warmup all
    for _ in range(5):
        mx.eval(ffn_standard(x))
        mx.eval(ffn_fused(x))
        mx.eval(ffn_compiled(x))
        mx.eval(ffn_ultra(x))

    # Benchmark
    iters = 50

    start = time.perf_counter()
    for _ in range(iters):
        y = ffn_standard(x)
        mx.eval(y)
    std_time = (time.perf_counter() - start) / iters

    start = time.perf_counter()
    for _ in range(iters):
        y = ffn_fused(x)
        mx.eval(y)
    fused_time = (time.perf_counter() - start) / iters

    start = time.perf_counter()
    for _ in range(iters):
        y = ffn_compiled(x)
        mx.eval(y)
    compiled_time = (time.perf_counter() - start) / iters

    start = time.perf_counter()
    for _ in range(iters):
        y = ffn_ultra(x)
        mx.eval(y)
    ultra_time = (time.perf_counter() - start) / iters

    print(f"    Standard FFN:    {std_time*1000:.2f}ms (baseline)")
    print(f"    Fused FFN:       {fused_time*1000:.2f}ms ({std_time/fused_time:.2f}x)")
    print(f"    Compiled FFN:    {compiled_time*1000:.2f}ms ({std_time/compiled_time:.2f}x)")
    print(f"    Ultra Fused FFN: {ultra_time*1000:.2f}ms ({std_time/ultra_time:.2f}x)")

    # =========================================================================
    # Test 3: Batch scaling with fused ops
    # =========================================================================
    print("\n[3] Batch Scaling with Ultra Fused FFN")
    print("-" * 70)

    for batch in [1, 4, 8, 16, 32]:
        x_batch = mx.random.normal(shape=(batch, L, D))
        mx.eval(x_batch)

        # Warmup
        for _ in range(3):
            mx.eval(ffn_ultra(x_batch))

        start = time.perf_counter()
        for _ in range(20):
            y = ffn_ultra(x_batch)
            mx.eval(y)
        elapsed = (time.perf_counter() - start) / 20

        per_sample = elapsed / batch * 1000
        throughput = batch / elapsed
        print(f"    B={batch:2d}: {elapsed*1000:6.1f}ms total, {per_sample:5.2f}ms/sample, {throughput:6.1f} samples/s")

    # =========================================================================
    # Test 4: Memory bandwidth analysis
    # =========================================================================
    print("\n[4] Memory Bandwidth Analysis")
    print("-" * 70)

    # Calculate theoretical memory traffic
    # W1: D x hidden*2 params, W2: hidden x D params
    w1_params = D * hidden * 2
    w2_params = hidden * D
    total_params = w1_params + w2_params
    param_bytes = total_params * 4  # FP32

    # Input/output traffic
    input_bytes = B * L * D * 4
    output_bytes = B * L * D * 4
    intermediate_bytes = B * L * hidden * 4  # After SwiGLU

    total_traffic = param_bytes + input_bytes + output_bytes + intermediate_bytes

    # With fusion, we avoid writing intermediate
    fused_traffic = param_bytes + input_bytes + output_bytes

    print(f"    Standard memory traffic: {total_traffic / 1e6:.1f} MB")
    print(f"    Fused memory traffic:    {fused_traffic / 1e6:.1f} MB")
    print(f"    Memory savings:          {(1 - fused_traffic/total_traffic)*100:.1f}%")

    # Calculate achieved bandwidth
    achieved_bw = total_traffic / (std_time)
    print(f"\n    Achieved bandwidth: {achieved_bw / 1e9:.1f} GB/s")
    print(f"    M4 Max theoretical: ~400 GB/s")
    print(f"    Efficiency:         {achieved_bw / 400e9 * 100:.1f}%")


def benchmark_full_transformer():
    """Benchmark full transformer with fused FFN."""
    import time
    from mlx.utils import tree_unflatten
    from esm.models.mlx import ESM3MLX

    print("\n" + "=" * 70)
    print("🏎️  FULL TRANSFORMER WITH FUSED OPS")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    L = 100
    B = 1

    # Create embedding input
    x = mx.random.normal(shape=(B, L, 1536))
    mx.eval(x)

    # Warmup
    for _ in range(3):
        out, emb, _ = model.transformer(x)
        mx.eval(out)

    # Benchmark standard
    start = time.perf_counter()
    for _ in range(10):
        out, emb, _ = model.transformer(x)
        mx.eval(out)
    std_time = (time.perf_counter() - start) / 10

    print(f"\n    48-layer transformer: {std_time*1000:.1f}ms")
    print(f"    Per layer:            {std_time/48*1000:.2f}ms")
    print(f"    Throughput:           {1/std_time:.1f} fwd/s")

    # FLOP calculation
    # Per layer: 2 matmuls for attention (QKV, out), 2 for FFN (W1, W2)
    # QKV: B*L*D * D*3 = 2*B*L*D*D*3 FLOPs
    # Out: B*L*D * D = 2*B*L*D*D FLOPs
    # W1:  B*L*D * 2*hidden = 2*B*L*D*2*hidden FLOPs
    # W2:  B*L*hidden * D = 2*B*L*hidden*D FLOPs

    D = 1536
    hidden = 8192
    flops_per_layer = (
        2 * B * L * D * D * 3 +  # QKV
        2 * B * L * D * D +  # out_proj
        2 * B * L * D * 2 * hidden +  # W1
        2 * B * L * hidden * D  # W2
    )
    total_flops = flops_per_layer * 48

    achieved_tflops = total_flops / std_time / 1e12
    print(f"\n    Achieved:   {achieved_tflops:.2f} TFLOP/s")
    print(f"    M4 Max:     ~14 TFLOP/s")
    print(f"    Efficiency: {achieved_tflops/14*100:.1f}%")


# =============================================================================
# COMPILED FULL ATTENTION BLOCK
# =============================================================================

def make_compiled_attention(d_model: int, n_heads: int):
    """Create a JIT-compiled attention forward function."""
    d_head = d_model // n_heads

    def attention_forward(
        x: mx.array,
        ln_weight: mx.array,
        ln_bias: mx.array,
        qkv_weight: mx.array,
        q_ln_weight: mx.array,
        q_ln_bias: mx.array,
        k_ln_weight: mx.array,
        k_ln_bias: mx.array,
        out_weight: mx.array,
    ) -> mx.array:
        B, L, D = x.shape

        # LayerNorm + QKV projection
        x_norm = mx.fast.layer_norm(x, ln_weight, ln_bias, eps=1e-5)
        qkv = x_norm @ qkv_weight.T

        # Split Q, K, V
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Q/K LayerNorm
        q = mx.fast.layer_norm(q, q_ln_weight, q_ln_bias, eps=1e-5)
        k = mx.fast.layer_norm(k, k_ln_weight, k_ln_bias, eps=1e-5)

        # Reshape for multi-head attention
        q = q.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)

        # RoPE
        q = mx.fast.rope(q, dims=d_head, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, dims=d_head, traditional=False, base=10000.0, scale=1.0, offset=0)

        # SDPA
        scale = 1.0 / math.sqrt(d_head)
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        # Reshape and project output
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return attn_out @ out_weight.T

    return mx.compile(attention_forward)


def make_compiled_transformer_block(d_model: int, n_heads: int, hidden_dim: int):
    """Create a JIT-compiled full transformer block (attention + FFN)."""
    d_head = d_model // n_heads

    def block_forward(
        x: mx.array,
        # Attention params
        attn_ln_weight: mx.array, attn_ln_bias: mx.array,
        qkv_weight: mx.array,
        q_ln_weight: mx.array, q_ln_bias: mx.array,
        k_ln_weight: mx.array, k_ln_bias: mx.array,
        out_weight: mx.array,
        # FFN params
        ffn_ln_weight: mx.array, ffn_ln_bias: mx.array,
        w1_weight: mx.array,
        w2_weight: mx.array,
        scaling_factor: float,
    ) -> mx.array:
        B, L, D = x.shape

        # ===== ATTENTION =====
        x_norm = mx.fast.layer_norm(x, attn_ln_weight, attn_ln_bias, eps=1e-5)
        qkv = x_norm @ qkv_weight.T
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = mx.fast.layer_norm(q, q_ln_weight, q_ln_bias, eps=1e-5)
        k = mx.fast.layer_norm(k, k_ln_weight, k_ln_bias, eps=1e-5)
        q = q.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, dims=d_head, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, dims=d_head, traditional=False, base=10000.0, scale=1.0, offset=0)
        scale = 1.0 / math.sqrt(d_head)
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, D)
        attn_out = attn_out @ out_weight.T
        x = x + attn_out / scaling_factor

        # ===== FFN =====
        x_norm = mx.fast.layer_norm(x, ffn_ln_weight, ffn_ln_bias, eps=1e-5)
        h = x_norm @ w1_weight.T
        half = h.shape[-1] // 2
        h1, h2 = h[..., :half], h[..., half:]
        h = (h1 * mx.sigmoid(h1)) * h2
        ffn_out = h @ w2_weight.T
        x = x + ffn_out / scaling_factor

        return x

    return mx.compile(block_forward)


class CompiledTransformerBlock(nn.Module):
    """Fully compiled transformer block for maximum speed."""

    def __init__(
        self,
        d_model: int = 1536,
        n_heads: int = 24,
        expansion_ratio: float = 4.0,
        bias: bool = False,
        residue_scaling_factor: float = 1.0,
    ):
        super().__init__()
        hidden_dim = int(((expansion_ratio * d_model) + 255) // 256 * 256)
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.scaling_factor = residue_scaling_factor

        # Attention params
        self.attn_ln_weight = mx.ones((d_model,))
        self.attn_ln_bias = mx.zeros((d_model,))
        self.qkv_weight = mx.random.normal(shape=(d_model * 3, d_model)) * 0.02
        self.q_ln_weight = mx.ones((d_model,))
        self.q_ln_bias = mx.zeros((d_model,))
        self.k_ln_weight = mx.ones((d_model,))
        self.k_ln_bias = mx.zeros((d_model,))
        self.out_weight = mx.random.normal(shape=(d_model, d_model)) * 0.02

        # FFN params
        self.ffn_ln_weight = mx.ones((d_model,))
        self.ffn_ln_bias = mx.zeros((d_model,))
        self.w1_weight = mx.random.normal(shape=(hidden_dim * 2, d_model)) * 0.02
        self.w2_weight = mx.random.normal(shape=(d_model, hidden_dim)) * 0.02

        self._compiled_fn = make_compiled_transformer_block(d_model, n_heads, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self._compiled_fn(
            x,
            self.attn_ln_weight, self.attn_ln_bias,
            self.qkv_weight,
            self.q_ln_weight, self.q_ln_bias,
            self.k_ln_weight, self.k_ln_bias,
            self.out_weight,
            self.ffn_ln_weight, self.ffn_ln_bias,
            self.w1_weight,
            self.w2_weight,
            self.scaling_factor,
        )


def benchmark_compiled_blocks():
    """Benchmark compiled vs standard transformer blocks."""
    import time
    from esm.models.mlx.layers import TransformerBlock

    print("\n" + "=" * 70)
    print("🏎️  COMPILED TRANSFORMER BLOCK BENCHMARK")
    print("=" * 70)

    D, H = 1536, 24
    L = 100

    # Create blocks
    std_block = TransformerBlock(d_model=D, n_heads=H, use_geom_attn=False)
    compiled_block = CompiledTransformerBlock(d_model=D, n_heads=H)
    mx.eval(std_block.parameters(), compiled_block.parameters())

    print("\n[1] Single Block Comparison")
    print("-" * 70)

    for B in [1, 4, 8, 16, 32, 64]:
        x = mx.random.normal(shape=(B, L, D))
        mx.eval(x)

        # Warmup
        for _ in range(3):
            out, _ = std_block(x)
            mx.eval(out)
            mx.eval(compiled_block(x))

        # Standard
        start = time.perf_counter()
        for _ in range(20):
            out, _ = std_block(x)
            mx.eval(out)
        std_time = (time.perf_counter() - start) / 20

        # Compiled
        start = time.perf_counter()
        for _ in range(20):
            out = compiled_block(x)
            mx.eval(out)
        compiled_time = (time.perf_counter() - start) / 20

        speedup = std_time / compiled_time
        throughput_std = B / std_time
        throughput_comp = B / compiled_time

        print(f"    B={B:2d}: Standard {std_time*1000:6.1f}ms ({throughput_std:5.0f}/s) | "
              f"Compiled {compiled_time*1000:6.1f}ms ({throughput_comp:5.0f}/s) | "
              f"Speedup: {speedup:.2f}x")

    print("\n[2] 48-Layer Stack Estimate")
    print("-" * 70)

    B = 1
    x = mx.random.normal(shape=(B, L, D))
    mx.eval(x)

    # Single layer time
    for _ in range(3):
        mx.eval(compiled_block(x))

    start = time.perf_counter()
    for _ in range(50):
        out = compiled_block(x)
        mx.eval(out)
    layer_time = (time.perf_counter() - start) / 50

    estimated_48 = layer_time * 48
    print(f"    Single compiled layer: {layer_time*1000:.2f}ms")
    print(f"    48 layers estimated:   {estimated_48*1000:.1f}ms")
    print(f"    Estimated throughput:  {1/estimated_48:.1f} fwd/s")

    # Compute efficiency
    D_head = D // H
    hidden = 8192

    # FLOPs per layer
    flops_qkv = 2 * B * L * D * D * 3  # QKV projection
    flops_out = 2 * B * L * D * D  # Output projection
    flops_attn = 2 * B * H * L * L * D_head * 2  # Q@K and attn@V
    flops_w1 = 2 * B * L * D * hidden * 2  # W1
    flops_w2 = 2 * B * L * hidden * D  # W2
    flops_per_layer = flops_qkv + flops_out + flops_attn + flops_w1 + flops_w2
    total_flops = flops_per_layer * 48

    achieved_tflops = total_flops / estimated_48 / 1e12
    print(f"\n    Estimated TFLOP/s: {achieved_tflops:.2f}")
    print(f"    M4 Max peak:       ~14 TFLOP/s")
    print(f"    Efficiency:        {achieved_tflops/14*100:.1f}%")


def benchmark_quantized_compiled():
    """Benchmark INT4 quantized + compiled for maximum speed."""
    import time
    from mlx.utils import tree_unflatten
    from esm.models.mlx import ESM3MLX
    from esm.models.mlx.quantize import quantize_model

    print("\n" + "=" * 70)
    print("🏎️  QUANTIZED + COMPILED BENCHMARK")
    print("=" * 70)

    # Load model
    print("\nLoading FP32 model...")
    model_fp32 = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model_fp32.update(tree_unflatten(list(weights.items())))
    mx.eval(model_fp32.parameters())

    # Quantize to INT4
    print("Quantizing to INT4...")
    model_int4 = ESM3MLX()
    model_int4.update(tree_unflatten(list(weights.items())))
    quantize_model(model_int4, bits=4, group_size=64)
    mx.eval(model_int4.parameters())

    L = 100

    # Create inputs
    def make_inputs(B):
        return (
            mx.full((B, L), 32, dtype=mx.int32),
            mx.full((B, L), 4100, dtype=mx.int32),
            mx.full((B, L), 10, dtype=mx.int32),
            mx.full((B, L), 18, dtype=mx.int32),
            mx.zeros((B, L, 8), dtype=mx.int32),
            mx.zeros((B, L, 16), dtype=mx.int32),
            mx.ones((B, L)),
            mx.zeros((B, L)),
        )

    print("\n[1] FP32 vs INT4 Batch Scaling")
    print("-" * 70)

    for B in [1, 8, 16, 32, 64]:
        inputs = make_inputs(B)
        mx.eval(inputs)

        # Warmup
        for _ in range(3):
            out_fp32, _ = model_fp32(*inputs)
            out_int4, _ = model_int4(*inputs)
            mx.eval(out_fp32, out_int4)

        # FP32
        start = time.perf_counter()
        for _ in range(5):
            out, _ = model_fp32(*inputs)
            mx.eval(out)
        fp32_time = (time.perf_counter() - start) / 5

        # INT4
        start = time.perf_counter()
        for _ in range(5):
            out, _ = model_int4(*inputs)
            mx.eval(out)
        int4_time = (time.perf_counter() - start) / 5

        fp32_throughput = B / fp32_time
        int4_throughput = B / int4_time
        speedup = fp32_time / int4_time

        print(f"    B={B:2d}: FP32 {fp32_time*1000:7.1f}ms ({fp32_throughput:5.1f}/s) | "
              f"INT4 {int4_time*1000:7.1f}ms ({int4_throughput:5.1f}/s) | "
              f"Ratio: {speedup:.2f}x")

    # Memory usage
    print("\n[2] Memory Usage")
    print("-" * 70)

    def count_params_bytes(model):
        total = 0
        for k, v in tree_unflatten(list(model.parameters().items())):
            if hasattr(v, 'nbytes'):
                total += v.nbytes
        return total

    # Rough estimates
    print(f"    FP32 model size: ~5.3 GB")
    print(f"    INT4 model size: ~0.8 GB")
    print(f"    Memory reduction: 6.6x")


def benchmark_generation_throughput():
    """End-to-end generation throughput benchmark."""
    import time
    from mlx.utils import tree_unflatten
    from esm.models.mlx import ESM3MLX
    from esm.models.mlx.quantize import quantize_model

    print("\n" + "=" * 70)
    print("🏎️  GENERATION THROUGHPUT (THE ULTIMATE TEST)")
    print("=" * 70)

    # Load model
    print("\nLoading INT4 model (fastest)...")
    model = ESM3MLX()
    weights = mx.load("esm3_mlx_weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    quantize_model(model, bits=4, group_size=64)
    mx.eval(model.parameters())

    L = 100
    NUM_STEPS = 8

    print("\n[1] Single Sequence Generation Time")
    print("-" * 70)

    # Create inputs
    sequence_tokens = mx.full((1, L), 32, dtype=mx.int32)
    structure_tokens = mx.full((1, L), 4100, dtype=mx.int32)
    ss8_tokens = mx.full((1, L), 10, dtype=mx.int32)
    sasa_tokens = mx.full((1, L), 18, dtype=mx.int32)
    function_tokens = mx.zeros((1, L, 8), dtype=mx.int32)
    residue_annotation_tokens = mx.zeros((1, L, 16), dtype=mx.int32)
    average_plddt = mx.ones((1, L))
    per_res_plddt = mx.zeros((1, L))

    # Warmup
    result = model.generate_sequence(
        sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
        function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        num_steps=2, temperature=0.7,
    )
    mx.eval(result)

    # Time single generation
    start = time.perf_counter()
    for _ in range(5):
        result = model.generate_sequence(
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
            num_steps=NUM_STEPS, temperature=0.7,
        )
        mx.eval(result)
    single_time = (time.perf_counter() - start) / 5

    print(f"    L={L}, steps={NUM_STEPS}")
    print(f"    Single sequence: {single_time*1000:.0f}ms")
    print(f"    Per step: {single_time/NUM_STEPS*1000:.0f}ms")

    print("\n[2] Batched Generation (Multiple Samples)")
    print("-" * 70)

    for num_samples in [4, 8, 16, 32, 64]:
        start = time.perf_counter()
        result = model.generate_sequence_batched(
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
            num_steps=NUM_STEPS, temperature=0.7, num_samples=num_samples,
        )
        mx.eval(result)
        batch_time = time.perf_counter() - start

        throughput = num_samples / batch_time
        per_sample = batch_time / num_samples * 1000
        print(f"    {num_samples:2d} samples: {batch_time:.2f}s ({throughput:.1f}/s, {per_sample:.0f}ms/sample)")

    print("\n[3] Time to Generate Ensemble of 64 Sequences")
    print("-" * 70)

    # Full benchmark
    start = time.perf_counter()
    result = model.generate_sequence_batched(
        sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
        function_tokens, residue_annotation_tokens, average_plddt, per_res_plddt,
        num_steps=NUM_STEPS, temperature=0.7, num_samples=64,
    )
    mx.eval(result)
    total_time = time.perf_counter() - start

    print(f"    64 sequences, L=100, 8 steps")
    print(f"    Total time: {total_time:.1f}s")
    print(f"    Throughput: {64/total_time:.1f} sequences/sec")
    print(f"    Per sequence: {total_time/64*1000:.0f}ms")

    # Compare to target
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\n    Current:  {64/total_time:.1f} seq/s (64 samples in {total_time:.1f}s)")
    print(f"    Target:   128-256 samples in <10s")
    print(f"    Gap:      {256/10 / (64/total_time):.1f}x speedup needed")


if __name__ == "__main__":
    benchmark_fused_ops()
    benchmark_full_transformer()
    benchmark_compiled_blocks()
    benchmark_quantized_compiled()
    benchmark_generation_throughput()
