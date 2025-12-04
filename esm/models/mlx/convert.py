"""Convert PyTorch ESM3 weights to MLX format."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def map_key(key: str) -> Tuple[Optional[str], bool]:
    """Map PyTorch weight keys to MLX keys.

    Returns:
        Tuple of (new_key, should_transpose)
        new_key is None if the key should be skipped
    """
    # Skip rotary embedding buffers (MLX computes these on the fly)
    if "rotary" in key or "inv_freq" in key or "_cos_cached" in key or "_sin_cached" in key:
        return None, False

    # Transformer mappings
    key = key.replace("transformer.blocks", "transformer.layers")

    # Attention mappings within blocks
    # PyTorch: layernorm_qkv is Sequential(LayerNorm[0], Linear[1])
    key = key.replace(".attn.layernorm_qkv.0.", ".attn.layernorm.")
    key = key.replace(".attn.layernorm_qkv.1.", ".attn.qkv_proj.")

    # Geometric attention mappings
    key = key.replace(".geom_attn.s_norm.", ".geom_attn.layernorm.")
    key = key.replace(".geom_attn.distance_scale_per_head", ".geom_attn.distance_scale")
    key = key.replace(".geom_attn.rotation_scale_per_head", ".geom_attn.rotation_scale")

    # FFN mappings (SwiGLU)
    # PyTorch: ffn = Sequential(LayerNorm[0], Linear(d, hidden*2)[1], SwiGLU[2], Linear(hidden, d)[3])
    # MLX: ffn.layernorm, ffn.w1, ffn.w2
    key = key.replace(".ffn.0.", ".ffn.layernorm.")
    key = key.replace(".ffn.1.", ".ffn.w1.")
    key = key.replace(".ffn.3.", ".ffn.w2.")

    # Output heads - RegressionHead is Sequential(Linear, GELU, LayerNorm, Linear)
    # PyTorch indices: 0=linear1, 1=gelu(no params), 2=norm, 3=linear2
    for head in ["sequence_head", "structure_head", "ss8_head", "sasa_head", "function_head", "residue_head"]:
        key = key.replace(f"output_heads.{head}.0.", f"output_heads.{head}.linear1.")
        key = key.replace(f"output_heads.{head}.2.", f"output_heads.{head}.norm.")
        key = key.replace(f"output_heads.{head}.3.", f"output_heads.{head}.linear2.")

    # Determine if we need to transpose (Linear layers in MLX expect different shape)
    # MLX Linear weights are (out_features, in_features), same as PyTorch
    # But mlx.nn.Linear expects weights in (out, in) format already
    # Actually MLX stores as (out, in) and PyTorch stores as (out, in), so no transpose needed
    # The issue is that in PyTorch the shapes are (out, in) which is what we want
    should_transpose = False

    return key, should_transpose


def convert_pytorch_to_mlx(
    pytorch_weights_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Convert PyTorch ESM3 weights to MLX format.

    Args:
        pytorch_weights_path: Path to PyTorch .pth weights file
        output_path: Optional path to save .npz file. If None, returns dict only.
        verbose: Print conversion progress

    Returns:
        Dictionary of numpy arrays ready for MLX
    """
    import torch

    if verbose:
        print(f"Loading PyTorch weights from {pytorch_weights_path}...")

    state_dict = torch.load(pytorch_weights_path, map_location="cpu", weights_only=True)

    mlx_weights = {}
    skipped = []
    converted = []

    for key, value in state_dict.items():
        new_key, should_transpose = map_key(key)

        if new_key is None:
            skipped.append(key)
            continue

        # Convert to numpy
        np_value = value.numpy()

        # Transpose linear weights if needed
        if should_transpose and len(np_value.shape) == 2:
            np_value = np_value.T

        mlx_weights[new_key] = np_value
        converted.append((key, new_key))

    if verbose:
        print(f"Converted {len(converted)} weights")
        print(f"Skipped {len(skipped)} weights (rotary caches, etc.)")

        # Show some example mappings
        print("\nExample mappings:")
        for old, new in converted[:5]:
            print(f"  {old} -> {new}")

    if output_path:
        if verbose:
            print(f"\nSaving to {output_path}...")
        np.savez(output_path, **mlx_weights)
        if verbose:
            print("Done!")

    return mlx_weights


def convert_from_huggingface(
    model_name: str = "esm3-open",
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Convert ESM3 weights from HuggingFace to MLX format.

    This loads the model using the ESM3 API and extracts weights.

    Args:
        model_name: Name of the ESM3 model (e.g., "esm3-open")
        output_path: Optional path to save .npz file
        verbose: Print conversion progress

    Returns:
        Dictionary of numpy arrays ready for MLX
    """
    import torch
    from esm.models.esm3 import ESM3

    if verbose:
        print(f"Loading ESM3 model '{model_name}'...")

    # Load model on CPU
    model = ESM3.from_pretrained(model_name, device=torch.device("cpu"))

    if verbose:
        print("Extracting state dict...")

    state_dict = model.state_dict()

    mlx_weights = {}
    skipped = []
    converted = []

    for key, value in state_dict.items():
        new_key, should_transpose = map_key(key)

        if new_key is None:
            skipped.append(key)
            continue

        # Convert to numpy
        np_value = value.numpy()

        # Transpose linear weights if needed
        if should_transpose and len(np_value.shape) == 2:
            np_value = np_value.T

        mlx_weights[new_key] = np_value
        converted.append((key, new_key))

    if verbose:
        print(f"Converted {len(converted)} weights")
        print(f"Skipped {len(skipped)} weights")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving to {output_path}...")
        np.savez(output_path, **mlx_weights)
        if verbose:
            print("Done!")

    return mlx_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ESM3 weights to MLX format")
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "file"],
        help="Source of weights (huggingface or file path)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="esm3-open",
        help="Model name (for huggingface) or path to .pth file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="esm3_mlx_weights.npz",
        help="Output .npz file path",
    )
    args = parser.parse_args()

    if args.source == "huggingface":
        convert_from_huggingface(args.input, args.output)
    else:
        convert_pytorch_to_mlx(args.input, args.output)
