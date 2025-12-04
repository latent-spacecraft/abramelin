"""MLX layers for ESM3 with fast kernels and optimizations."""

import math
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn


# Type alias for KV cache: (keys, values) per layer
KVCache = Tuple[mx.array, mx.array]


class FastLayerNorm(nn.Module):
    """LayerNorm using mx.fast.layer_norm for ~30% better performance."""

    def __init__(self, dims: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,)) if bias else None
        self._dims = dims
        self._has_bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        if self._has_bias:
            return mx.fast.layer_norm(x, self.weight, self.bias, eps=self.eps)
        else:
            return mx.fast.layer_norm(x, self.weight, mx.zeros((self._dims,)), eps=self.eps)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE, QK LayerNorm, using MLX fast kernels."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Pre-attention LayerNorm + QKV projection combined (using fast LN)
        self.layernorm = FastLayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Optional Q/K LayerNorm (using fast LN)
        self.qk_layernorm = qk_layernorm
        if qk_layernorm:
            self.q_ln = FastLayerNorm(d_model, bias=bias)
            self.k_ln = FastLayerNorm(d_model, bias=bias)

        # RoPE parameters for mx.fast.rope
        self.rope_dims = self.d_head
        self._rope_freqs = None  # Lazy init

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        update_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[KVCache]]:
        """Forward pass using MLX's optimized scaled dot-product attention.

        Args:
            x: Input tensor (B, L, D)
            mask: Attention mask
            cache: Optional (K, V) cache (unused, kept for API compatibility)
            update_mask: (B, L) bool mask (unused, kept for API compatibility)

        Returns:
            output: Attention output (B, L, D)
            new_cache: None (caching not beneficial for bidirectional attention)
        """
        B, L, D = x.shape

        # LayerNorm + QKV projection
        x_norm = self.layernorm(x)
        qkv = self.qkv_proj(x_norm)

        # Split into Q, K, V
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Apply Q/K LayerNorm if enabled
        if self.qk_layernorm:
            q = self.q_ln(q)
            k = self.k_ln(k)

        # Reshape for multi-head attention: (B, L, D) -> (B, n_heads, L, d_head)
        q = q.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # Apply RoPE using fast kernel
        q = mx.fast.rope(q, dims=self.rope_dims, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, dims=self.rope_dims, traditional=False, base=10000.0, scale=1.0, offset=0)

        # Use MLX's fused scaled dot-product attention kernel
        # This is significantly faster than manual matmul + softmax
        scale = math.sqrt(1.0 / self.d_head)
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=scale,
            mask=mask,
        )

        # Reshape back: (B, n_heads, L, d_head) -> (B, L, D)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, D)

        return self.out_proj(attn_out), None


def swiglu(x: mx.array) -> mx.array:
    """SwiGLU activation function."""
    x1, x2 = mx.split(x, 2, axis=-1)
    return nn.silu(x1) * x2


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    """Calculate hidden dimension for SwiGLU FFN."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with fast LayerNorm."""

    def __init__(self, d_model: int, expansion_ratio: float = 4.0, bias: bool = False):
        super().__init__()
        hidden_dim = swiglu_correction_fn(expansion_ratio, d_model)

        self.layernorm = FastLayerNorm(d_model)  # Use fast LayerNorm
        self.w1 = nn.Linear(d_model, hidden_dim * 2, bias=bias)  # Projects to gate + value
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layernorm(x)
        x = self.w1(x)
        x = swiglu(x)  # Use function instead of module
        return self.w2(x)


class GeometricAttention(nn.Module):
    """Geometric attention for structural reasoning.

    This implements the geometric attention from ESM3 that reasons about
    3D structural relationships using rotation and distance terms.
    """

    def __init__(
        self,
        d_model: int,
        v_heads: int,
        num_vector_messages: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.v_heads = v_heads
        self.num_vector_messages = num_vector_messages

        self.layernorm = FastLayerNorm(d_model, bias=bias)

        # Projection dimension: 4 * v_heads * 3 (q_rot, k_rot, v) + v_heads * 3 (q_dist, k_dist)
        dim_rot = 2 * v_heads * 3 + v_heads * 3 * num_vector_messages  # q_rot, k_rot, value
        dim_dist = 2 * v_heads * 3  # q_dist, k_dist
        self.proj = nn.Linear(d_model, dim_rot + dim_dist, bias=bias)

        # Output projection
        channels_out = v_heads * 3 * num_vector_messages
        self.out_proj = nn.Linear(channels_out, d_model, bias=bias)

        # Per-head scaling parameters (initialized to zeros, will be softplus'd)
        self.distance_scale = mx.zeros((v_heads,))
        self.rotation_scale = mx.zeros((v_heads,))

    def __call__(
        self,
        x: mx.array,
        rotation_matrices: mx.array,  # (B, L, 3, 3)
        translations: mx.array,  # (B, L, 3)
        frame_mask: mx.array,  # (B, L) bool
        sequence_mask: Optional[mx.array] = None,  # (B, L, L) bool
    ) -> mx.array:
        B, L, D = x.shape

        # Normalize input
        x_norm = self.layernorm(x)

        # Project to rotation and distance vectors
        proj = self.proj(x_norm)
        dim_rot = 2 * self.v_heads * 3 + self.v_heads * 3 * self.num_vector_messages
        vec_rot, vec_dist = mx.split(proj, [dim_rot], axis=-1)

        # Reshape vectors: (B, L, v_heads * 3) -> (B, L, v_heads, 3)
        vec_rot = vec_rot.reshape(B, L, -1, 3)  # (B, L, 2*v_heads + v_heads*msg, 3)
        vec_dist = vec_dist.reshape(B, L, 2 * self.v_heads, 3)  # (B, L, 2*v_heads, 3)

        # Apply rotation to rotation vectors
        # R: (B, L, 3, 3), vec: (B, L, H, 3) -> (B, L, H, 3)
        vec_rot_transformed = mx.einsum("blij,blhj->blhi", rotation_matrices, vec_rot)

        # Split into q_rot, k_rot, value
        q_rot = vec_rot_transformed[:, :, :self.v_heads, :]  # (B, L, v_heads, 3)
        k_rot = vec_rot_transformed[:, :, self.v_heads:2*self.v_heads, :]
        value = vec_rot_transformed[:, :, 2*self.v_heads:, :]  # (B, L, v_heads*msg, 3)

        # Apply full affine transform (rotation + translation) to distance vectors
        vec_dist_rotated = mx.einsum("blij,blhj->blhi", rotation_matrices, vec_dist)
        vec_dist_transformed = vec_dist_rotated + translations[:, :, None, :]

        q_dist = vec_dist_transformed[:, :, :self.v_heads, :]  # (B, L, v_heads, 3)
        k_dist = vec_dist_transformed[:, :, self.v_heads:, :]

        # Rearrange for attention computation
        q_rot = q_rot.transpose(0, 2, 1, 3)  # (B, v_heads, L, 3)
        k_rot = k_rot.transpose(0, 2, 3, 1)  # (B, v_heads, 3, L)

        q_dist = q_dist.transpose(0, 2, 1, 3)[:, :, :, None, :]  # (B, v_heads, L, 1, 3)
        k_dist = k_dist.transpose(0, 2, 1, 3)[:, :, None, :, :]  # (B, v_heads, 1, L, 3)

        value = value.reshape(B, L, self.v_heads, self.num_vector_messages, 3)
        value = value.transpose(0, 2, 1, 3, 4)  # (B, v_heads, L, msg, 3)
        value = value.reshape(B, self.v_heads, L, -1)  # (B, v_heads, L, msg*3)

        # Compute attention scores
        sqrt3 = math.sqrt(3)
        rotation_term = (q_rot @ k_rot) / sqrt3  # (B, v_heads, L, L)
        distance_term = mx.sqrt(mx.sum((q_dist - k_dist) ** 2, axis=-1) + 1e-8) / sqrt3  # (B, v_heads, L, L)

        # Apply per-head scaling (softplus for positive values)
        rot_scale = nn.softplus(self.rotation_scale).reshape(1, -1, 1, 1)
        dist_scale = nn.softplus(self.distance_scale).reshape(1, -1, 1, 1)

        attn_weight = rotation_term * rot_scale - distance_term * dist_scale

        # Apply masks
        if sequence_mask is not None:
            mask_value = -1e9
            attn_weight = mx.where(sequence_mask[:, None, :, :], attn_weight, mask_value)

        frame_mask_2d = frame_mask[:, None, None, :]  # (B, 1, 1, L)
        attn_weight = mx.where(frame_mask_2d, attn_weight, -1e9)

        # Softmax attention
        attn_weight = mx.softmax(attn_weight, axis=-1)

        # Apply attention to values
        attn_out = attn_weight @ value  # (B, v_heads, L, msg*3)

        # Inverse rotation on output
        attn_out = attn_out.reshape(B, self.v_heads, L, self.num_vector_messages, 3)
        attn_out = attn_out.transpose(0, 2, 1, 3, 4)  # (B, L, v_heads, msg, 3)

        # Apply inverse rotation: R^T * v
        rotation_inv = rotation_matrices.transpose(0, 1, 3, 2)  # (B, L, 3, 3) transposed
        attn_out_flat = attn_out.reshape(B, L, -1, 3)  # (B, L, v_heads*msg, 3)
        attn_out_rotated = mx.einsum("blij,blhj->blhi", rotation_inv, attn_out_flat)

        # Reshape and project output
        attn_out_final = attn_out_rotated.reshape(B, L, -1)  # (B, L, v_heads*msg*3)

        # Zero out frameless positions
        attn_out_final = mx.where(frame_mask[:, :, None], attn_out_final, 0.0)

        return self.out_proj(attn_out_final)


class TransformerBlock(nn.Module):
    """Unified transformer block with optional geometric attention and KV caching."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: Optional[int] = None,
        use_geom_attn: bool = False,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1.0,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.use_geom_attn = use_geom_attn
        self.scaling_factor = residue_scaling_factor

        # Standard multi-head attention
        self.attn = MultiHeadAttention(d_model, n_heads, bias, qk_layernorm)

        # Optional geometric attention
        if use_geom_attn:
            if v_heads is None:
                raise ValueError("v_heads must be specified when use_geom_attn is True")
            self.geom_attn = GeometricAttention(d_model, v_heads, bias=bias)

        # Feed-forward network
        self.ffn = SwiGLUFFN(d_model, expansion_ratio, bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        rotation_matrices: Optional[mx.array] = None,
        translations: Optional[mx.array] = None,
        frame_mask: Optional[mx.array] = None,
        sequence_mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        update_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[KVCache]]:
        """Forward pass with optional KV caching.

        Args:
            x: Input tensor (B, L, D)
            mask: Attention mask
            rotation_matrices, translations, frame_mask, sequence_mask: Geometric attention args
            cache: Optional KV cache for this layer
            update_mask: (B, L) bool mask for positions to recompute

        Returns:
            output: Block output (B, L, D)
            new_cache: Updated KV cache for this layer
        """
        # Self-attention with caching
        r1, new_cache = self.attn(x, mask, cache=cache, update_mask=update_mask)
        x = x + r1 / self.scaling_factor

        # Geometric attention (if enabled) - no caching for geometric attention
        if self.use_geom_attn and rotation_matrices is not None:
            r2 = self.geom_attn(x, rotation_matrices, translations, frame_mask, sequence_mask)
            x = x + r2 / self.scaling_factor

        # Feed-forward
        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3

        return x, new_cache


class TransformerStack(nn.Module):
    """Stack of transformer blocks with KV caching support."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: Optional[int],
        n_layers: int,
        n_layers_geom: int = 1,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads

        self.layers = []
        for i in range(n_layers):
            use_geom = i < n_layers_geom
            self.layers.append(
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    v_heads=v_heads if use_geom else None,
                    use_geom_attn=use_geom,
                    bias=bias,
                    expansion_ratio=expansion_ratio,
                    qk_layernorm=qk_layernorm,
                )
            )

        # Final norm has no bias (matching PyTorch ESM3)
        self.norm = FastLayerNorm(d_model, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        rotation_matrices: Optional[mx.array] = None,
        translations: Optional[mx.array] = None,
        frame_mask: Optional[mx.array] = None,
        sequence_mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
        update_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, Optional[List[KVCache]]]:
        """Forward pass with optional KV caching.

        Args:
            x: Input tensor (B, L, D)
            mask: Attention mask
            rotation_matrices, translations, frame_mask, sequence_mask: Geometric attention args
            cache: Optional list of KV caches, one per layer
            update_mask: (B, L) bool mask for positions to recompute

        Returns:
            output: Normalized output (B, L, D)
            embedding: Pre-norm embedding (B, L, D)
            new_cache: Updated list of KV caches
        """
        new_cache = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = layer(
                x,
                mask=mask,
                rotation_matrices=rotation_matrices,
                translations=translations,
                frame_mask=frame_mask,
                sequence_mask=sequence_mask,
                cache=layer_cache,
                update_mask=update_mask,
            )
            new_cache.append(layer_new_cache)

        embedding = x
        x = self.norm(x)
        return x, embedding, new_cache
