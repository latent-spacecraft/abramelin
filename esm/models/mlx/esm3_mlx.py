"""MLX implementation of ESM3 for Apple Silicon with KV caching.

This is a drop-in replacement for esm.models.esm3.ESM3 that runs on MLX.
Supports the same API: generate(ESMProtein, GenerationConfig).
"""

from functools import partial
from typing import Optional, List, Tuple, Dict

import mlx.core as mx
import mlx.nn as nn

from esm.models.mlx.layers import TransformerStack, KVCache
from esm.sdk.api import ESMProtein, ESMProteinTensor, GenerationConfig, ProteinType
from esm.tokenization import get_esm3_model_tokenizers, TokenizerCollection
from esm.utils import encoding
from esm.utils.constants import esm3 as C
from esm.utils.decoding import decode_protein_tensor


def rbf(values: mx.array, v_min: float, v_max: float, n_bins: int) -> mx.array:
    """Radial basis function encoding."""
    bin_centers = mx.linspace(v_min, v_max, n_bins)
    sigma = (v_max - v_min) / (2 * n_bins)
    diff = values[..., None] - bin_centers
    return mx.exp(-0.5 * (diff / sigma) ** 2)


class EncodeInputs(nn.Module):
    """Encode multi-modal inputs for ESM3."""

    def __init__(self, d_model: int):
        super().__init__()

        # Sequence embedding
        self.sequence_embed = nn.Embedding(64, d_model)

        # pLDDT projections
        self.plddt_projection = nn.Linear(16, d_model)
        self.structure_per_res_plddt_projection = nn.Linear(16, d_model)

        # Structure tokens
        self.structure_tokens_embed = nn.Embedding(4096 + 5, d_model)

        # Secondary structure and SASA
        self.ss8_embed = nn.Embedding(8 + 3, d_model)
        self.sasa_embed = nn.Embedding(16 + 3, d_model)

        # Function embeddings (8 separate embeddings for InterPro)
        self.function_embed = [nn.Embedding(260, d_model // 8) for _ in range(8)]

        # Residue annotations (embedding bag equivalent - sum of embeddings)
        self.residue_embed = nn.Embedding(1478, d_model)

    def __call__(
        self,
        sequence_tokens: mx.array,
        structure_tokens: mx.array,
        average_plddt: mx.array,
        per_res_plddt: mx.array,
        ss8_tokens: mx.array,
        sasa_tokens: mx.array,
        function_tokens: mx.array,
        residue_annotation_tokens: mx.array,
    ) -> mx.array:
        # Sequence embedding
        seq_embed = self.sequence_embed(sequence_tokens)

        # pLDDT embeddings
        rbf_fn = partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        plddt_embed = self.plddt_projection(rbf_fn(average_plddt))
        per_res_plddt_embed = self.structure_per_res_plddt_projection(rbf_fn(per_res_plddt))

        # Structure embeddings
        struct_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)

        # Function embeddings (concatenate 8 separate embeddings)
        func_embeds = []
        for i, embed_fn in enumerate(self.function_embed):
            func_embeds.append(embed_fn(function_tokens[..., i]))
        function_embed = mx.concatenate(func_embeds, axis=-1)

        # Residue annotation embeddings (sum over annotations per position)
        # residue_annotation_tokens: (B, L, N) -> embed each and sum
        B, L, N = residue_annotation_tokens.shape
        residue_flat = residue_annotation_tokens.reshape(-1, N)  # (B*L, N)
        residue_embedded = self.residue_embed(residue_flat)  # (B*L, N, D)
        residue_embed = residue_embedded.sum(axis=1)  # (B*L, D)
        residue_embed = residue_embed.reshape(B, L, -1)  # (B, L, D)

        # Sum all embeddings
        return (
            seq_embed
            + plddt_embed
            + per_res_plddt_embed
            + struct_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )


class RegressionHead(nn.Module):
    """Output head for logits prediction.

    This is a 2-layer MLP: Linear -> GELU -> LayerNorm -> Linear
    """

    def __init__(self, d_model: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else d_model
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.gelu(x)  # Use function instead of module
        x = self.norm(x)
        return self.linear2(x)


class OutputHeads(nn.Module):
    """All output heads for ESM3."""

    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, 64)
        self.structure_head = RegressionHead(d_model, 4096)
        self.ss8_head = RegressionHead(d_model, 8 + 3)
        self.sasa_head = RegressionHead(d_model, 16 + 3)
        self.function_head = RegressionHead(d_model, 260 * 8)
        self.residue_head = RegressionHead(d_model, 1478)

    def __call__(self, x: mx.array):
        return {
            "sequence_logits": self.sequence_head(x),
            "structure_logits": self.structure_head(x),
            "secondary_structure_logits": self.ss8_head(x),
            "sasa_logits": self.sasa_head(x),
            "function_logits": self.function_head(x).reshape(*x.shape[:-1], 8, 260),
            "residue_logits": self.residue_head(x),
        }


class ESM3MLX(nn.Module):
    """MLX implementation of ESM3 for Apple Silicon.

    This is a drop-in replacement for esm.models.esm3.ESM3 that runs on MLX.
    Supports the same API: generate(ESMProtein, GenerationConfig).

    Optimized for M-series chips with unified memory architecture.
    """

    def __init__(
        self,
        d_model: int = 1536,
        n_heads: int = 24,
        v_heads: int = 256,
        n_layers: int = 48,  # ESM3 Open Small has 48 layers
        n_layers_geom: int = 1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.v_heads = v_heads
        self.n_layers = n_layers

        # Input encoder
        self.encoder = EncodeInputs(d_model)

        # Transformer stack
        self.transformer = TransformerStack(
            d_model=d_model,
            n_heads=n_heads,
            v_heads=v_heads,
            n_layers=n_layers,
            n_layers_geom=n_layers_geom,
        )

        # Output heads
        self.output_heads = OutputHeads(d_model)

        # Tokenizers for encode/decode
        self._tokenizers = None

    @property
    def device(self) -> str:
        """Return device string for compatibility."""
        return "mlx"

    @property
    def tokenizers(self) -> TokenizerCollection:
        """Lazy-load tokenizers."""
        if self._tokenizers is None:
            self._tokenizers = get_esm3_model_tokenizers()
        return self._tokenizers

    def __call__(
        self,
        sequence_tokens: mx.array,
        structure_tokens: mx.array,
        ss8_tokens: mx.array,
        sasa_tokens: mx.array,
        function_tokens: mx.array,
        residue_annotation_tokens: mx.array,
        average_plddt: mx.array,
        per_res_plddt: mx.array,
        rotation_matrices: Optional[mx.array] = None,
        translations: Optional[mx.array] = None,
        frame_mask: Optional[mx.array] = None,
        sequence_mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
        update_mask: Optional[mx.array] = None,
    ) -> Tuple[Dict[str, mx.array], Optional[List[KVCache]]]:
        """Forward pass through ESM3 with optional KV caching.

        Args:
            sequence_tokens: (B, L) amino acid tokens
            structure_tokens: (B, L) structure tokens
            ss8_tokens: (B, L) secondary structure tokens
            sasa_tokens: (B, L) SASA tokens
            function_tokens: (B, L, 8) function annotation tokens
            residue_annotation_tokens: (B, L, N) residue annotation tokens
            average_plddt: (B, L) average pLDDT scores
            per_res_plddt: (B, L) per-residue pLDDT scores
            rotation_matrices: (B, L, 3, 3) rotation matrices for geometric attention
            translations: (B, L, 3) translation vectors for geometric attention
            frame_mask: (B, L) mask for valid frames
            sequence_mask: (B, L, L) mask for sequence attention
            cache: Optional list of KV caches from previous forward pass
            update_mask: (B, L) bool mask - True for positions to recompute

        Returns:
            outputs: Dictionary with logits for all tracks
            new_cache: Updated KV cache for next iteration
        """
        # Encode inputs
        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )

        # Create sequence mask for attention if not provided
        if sequence_mask is None:
            B, L = sequence_tokens.shape
            sequence_mask = mx.ones((B, L, L), dtype=mx.bool_)

        # Transform through layers with caching
        x, embedding, new_cache = self.transformer(
            x,
            mask=None,
            rotation_matrices=rotation_matrices,
            translations=translations,
            frame_mask=frame_mask,
            sequence_mask=sequence_mask,
            cache=cache,
            update_mask=update_mask,
        )

        # Get output logits
        outputs = self.output_heads(x)
        outputs["embeddings"] = embedding

        return outputs, new_cache

    @classmethod
    def from_pretrained(cls, model_name: str = "esm3-open") -> "ESM3MLX":
        """Load pretrained weights.

        Args:
            model_name: Model name ("esm3-open") or path to .npz weights file

        Returns:
            ESM3MLX model with loaded weights
        """
        from mlx.utils import tree_unflatten
        import os

        # Create model with default ESM3 Open Small config
        model = cls(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            n_layers_geom=1,
        )

        # Determine weights path
        if model_name in ("esm3-open", "esm3_open", "esm3-open-small"):
            # Look for converted weights in common locations
            search_paths = [
                "esm3_mlx_weights.npz",
                os.path.expanduser("~/.cache/esm/esm3_mlx_weights.npz"),
                os.path.join(os.path.dirname(__file__), "esm3_mlx_weights.npz"),
            ]
            weights_path = None
            for path in search_paths:
                if os.path.exists(path):
                    weights_path = path
                    break

            if weights_path is None:
                # Auto-convert from PyTorch weights
                print("MLX weights not found. Converting from PyTorch...")
                from esm.models.mlx.convert import convert_pytorch_to_mlx
                weights_path = "esm3_mlx_weights.npz"
                convert_pytorch_to_mlx("esm3-open", weights_path)
                print(f"Saved MLX weights to {weights_path}")
        else:
            weights_path = model_name

        # Load weights
        weights = mx.load(weights_path)
        model.update(tree_unflatten(list(weights.items())))

        return model

    def generate_sequence(
        self,
        sequence_tokens: mx.array,
        structure_tokens: mx.array,
        ss8_tokens: mx.array,
        sasa_tokens: mx.array,
        function_tokens: mx.array,
        residue_annotation_tokens: mx.array,
        average_plddt: mx.array,
        per_res_plddt: mx.array,
        mask_token: int = 32,  # ESM3 mask token
        num_steps: int = 8,
        temperature: float = 1.0,
        use_cache: bool = True,
    ) -> mx.array:
        """Iteratively generate sequence by unmasking with KV caching.

        This implements the iterative refinement strategy used in ESM3,
        optimized with KV caching to avoid redundant computation for
        positions that have already been unmasked.

        Args:
            sequence_tokens: (B, L) initial sequence with mask tokens
            structure_tokens: (B, L) structure tokens
            ss8_tokens: (B, L) secondary structure tokens
            sasa_tokens: (B, L) SASA tokens
            function_tokens: (B, L, 8) function tokens
            residue_annotation_tokens: (B, L, N) residue annotations
            average_plddt: (B, L) average pLDDT
            per_res_plddt: (B, L) per-residue pLDDT
            mask_token: Token ID for masked positions
            num_steps: Number of unmasking steps
            temperature: Sampling temperature
            use_cache: Whether to use KV caching (default True)

        Returns:
            Generated sequence tokens (B, L)
        """
        cache = None

        # Track which positions have been unmasked (fixed)
        # Initially, non-mask positions are "fixed"
        is_fixed = sequence_tokens != mask_token

        for step in range(num_steps):
            # Determine update mask: positions that changed since last step
            # First step: update all positions; subsequent steps: only masked positions
            if step == 0 or not use_cache:
                update_mask = None  # Compute all positions
            else:
                # Only recompute positions that are still masked
                update_mask = ~is_fixed

            # Forward pass with caching
            outputs, cache = self(
                sequence_tokens,
                structure_tokens,
                ss8_tokens,
                sasa_tokens,
                function_tokens,
                residue_annotation_tokens,
                average_plddt,
                per_res_plddt,
                cache=cache if use_cache else None,
                update_mask=update_mask,
            )

            # Get sequence logits and sample
            logits = outputs["sequence_logits"]  # (B, L, vocab_size)

            # Find masked positions
            is_masked = sequence_tokens == mask_token

            if not mx.any(is_masked):
                break

            # Sample from logits at masked positions
            if temperature > 0:
                probs = mx.softmax(logits / temperature, axis=-1)
                # Gumbel-max trick for sampling
                gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=probs.shape) + 1e-10) + 1e-10)
                sampled = mx.argmax(mx.log(probs + 1e-10) + gumbel_noise, axis=-1)
            else:
                sampled = mx.argmax(logits, axis=-1)

            # Determine which positions to unmask this step
            num_masked = mx.sum(is_masked, axis=-1, keepdims=True)
            num_to_unmask = mx.maximum(num_masked // (num_steps - step), 1)

            # Get confidence scores (max probability at each position)
            confidence = mx.max(mx.softmax(logits, axis=-1), axis=-1)
            confidence = mx.where(is_masked, confidence, -1e9)

            # Unmask highest confidence positions
            sorted_indices = mx.argsort(-confidence, axis=-1)
            position_ranks = mx.argsort(sorted_indices, axis=-1)
            unmask_mask = (position_ranks < num_to_unmask) & is_masked

            # Update sequence tokens and fixed mask
            sequence_tokens = mx.where(unmask_mask, sampled, sequence_tokens)
            is_fixed = is_fixed | unmask_mask

            mx.eval(sequence_tokens, is_fixed)
            if use_cache:
                mx.eval(cache)

        return sequence_tokens

    def generate_structure(
        self,
        sequence_tokens: mx.array,
        structure_tokens: mx.array,
        ss8_tokens: mx.array,
        sasa_tokens: mx.array,
        function_tokens: mx.array,
        residue_annotation_tokens: mx.array,
        average_plddt: mx.array,
        per_res_plddt: mx.array,
        num_steps: int = 8,
        temperature: float = 1.0,
        use_cache: bool = True,
    ) -> mx.array:
        """Iteratively generate structure tokens by unmasking.

        Similar to generate_sequence but operates on structure tokens and
        preserves BOS/EOS tokens at positions 0 and -1.

        Args:
            sequence_tokens: (B, L) amino acid tokens (should be unmasked)
            structure_tokens: (B, L) structure tokens with masks to fill
            ... (other args same as generate_sequence)

        Returns:
            Generated structure tokens (B, L) with BOS/EOS preserved
        """
        cache = None
        mask_token = C.STRUCTURE_MASK_TOKEN  # 4100
        bos_token = C.STRUCTURE_BOS_TOKEN    # 4098
        eos_token = C.STRUCTURE_EOS_TOKEN    # 4097

        # Ensure BOS/EOS are set correctly using mx.where
        structure_tokens = mx.array(structure_tokens)
        batch_size, seq_len = structure_tokens.shape
        pos_indices = mx.arange(seq_len)[None, :]  # (1, L)
        bos_mask = pos_indices == 0
        eos_mask = pos_indices == (seq_len - 1)
        structure_tokens = mx.where(bos_mask, bos_token, structure_tokens)
        structure_tokens = mx.where(eos_mask, eos_token, structure_tokens)

        # Track which positions have been unmasked (fixed)
        # BOS/EOS positions are always fixed
        is_fixed = structure_tokens != mask_token

        for step in range(num_steps):
            # Determine update mask
            if step == 0 or not use_cache:
                update_mask = None
            else:
                update_mask = ~is_fixed

            # Forward pass with caching
            outputs, cache = self(
                sequence_tokens,
                structure_tokens,
                ss8_tokens,
                sasa_tokens,
                function_tokens,
                residue_annotation_tokens,
                average_plddt,
                per_res_plddt,
                cache=cache if use_cache else None,
                update_mask=update_mask,
            )

            # Get structure logits and sample
            logits = outputs["structure_logits"]  # (B, L, 4096)

            # Find masked positions (excluding BOS/EOS)
            is_masked = structure_tokens == mask_token

            if not mx.any(is_masked):
                break

            # Sample from logits at masked positions
            if temperature > 0:
                probs = mx.softmax(logits / temperature, axis=-1)
                gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=probs.shape) + 1e-10) + 1e-10)
                sampled = mx.argmax(mx.log(probs + 1e-10) + gumbel_noise, axis=-1)
            else:
                sampled = mx.argmax(logits, axis=-1)

            # Determine which positions to unmask this step
            num_masked = mx.sum(is_masked, axis=-1, keepdims=True)
            num_to_unmask = mx.maximum(num_masked // (num_steps - step), 1)

            # Get confidence scores (max probability at each position)
            confidence = mx.max(mx.softmax(logits, axis=-1), axis=-1)
            confidence = mx.where(is_masked, confidence, -1e9)

            # Unmask highest confidence positions
            sorted_indices = mx.argsort(-confidence, axis=-1)
            position_ranks = mx.argsort(sorted_indices, axis=-1)
            unmask_mask = (position_ranks < num_to_unmask) & is_masked

            # Update structure tokens and fixed mask
            structure_tokens = mx.where(unmask_mask, sampled, structure_tokens)
            is_fixed = is_fixed | unmask_mask

            mx.eval(structure_tokens, is_fixed)
            if use_cache:
                mx.eval(cache)

        # Ensure BOS/EOS are still correct after generation
        structure_tokens = mx.where(bos_mask, bos_token, structure_tokens)
        structure_tokens = mx.where(eos_mask, eos_token, structure_tokens)

        return structure_tokens

    def generate_sequence_batched(
        self,
        sequence_tokens: mx.array,
        structure_tokens: mx.array,
        ss8_tokens: mx.array,
        sasa_tokens: mx.array,
        function_tokens: mx.array,
        residue_annotation_tokens: mx.array,
        average_plddt: mx.array,
        per_res_plddt: mx.array,
        mask_token: int = 32,
        num_steps: int = 8,
        temperature: float = 1.0,
        num_samples: int = 1,
    ) -> mx.array:
        """Generate multiple sequence samples in parallel.

        Efficiently generates multiple samples by batching them together.

        Args:
            sequence_tokens: (B, L) or (L,) initial sequence
            ... (same as generate_sequence)
            num_samples: Number of samples to generate per input

        Returns:
            Generated sequences (B * num_samples, L) or (num_samples, L)
        """
        # Expand inputs for parallel sampling
        if len(sequence_tokens.shape) == 1:
            sequence_tokens = sequence_tokens[None, :]  # Add batch dim

        # Repeat each input num_samples times
        sequence_tokens = mx.repeat(sequence_tokens, num_samples, axis=0)
        structure_tokens = mx.repeat(structure_tokens, num_samples, axis=0)
        ss8_tokens = mx.repeat(ss8_tokens, num_samples, axis=0)
        sasa_tokens = mx.repeat(sasa_tokens, num_samples, axis=0)
        function_tokens = mx.repeat(function_tokens, num_samples, axis=0)
        residue_annotation_tokens = mx.repeat(residue_annotation_tokens, num_samples, axis=0)
        average_plddt = mx.repeat(average_plddt, num_samples, axis=0)
        per_res_plddt = mx.repeat(per_res_plddt, num_samples, axis=0)

        # Generate with caching
        return self.generate_sequence(
            sequence_tokens,
            structure_tokens,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
            average_plddt,
            per_res_plddt,
            mask_token=mask_token,
            num_steps=num_steps,
            temperature=temperature,
            use_cache=True,
        )

    # =========================================================================
    # ESM3InferenceClient Interface (drop-in replacement for ESM3)
    # =========================================================================

    def encode(self, protein: ESMProtein) -> ESMProteinTensor:
        """Encode ESMProtein to ESMProteinTensor (tokenized form).

        This matches the interface of esm.models.esm3.ESM3.encode().
        """
        import torch

        sequence_tokens = None
        structure_tokens = None
        secondary_structure_tokens = None
        sasa_tokens = None
        function_tokens = None
        residue_annotation_tokens = None
        coordinates = None

        if protein.sequence is not None:
            sequence_tokens = encoding.tokenize_sequence(
                protein.sequence, self.tokenizers.sequence, add_special_tokens=True
            )
        if protein.secondary_structure is not None:
            secondary_structure_tokens = encoding.tokenize_secondary_structure(
                protein.secondary_structure,
                self.tokenizers.secondary_structure,
                add_special_tokens=True,
            )
        if protein.sasa is not None:
            sasa_tokens = encoding.tokenize_sasa(
                protein.sasa, self.tokenizers.sasa, add_special_tokens=True
            )

        # Infer input length
        sequence_length = -1
        if sequence_tokens is not None:
            sequence_length = len(sequence_tokens)
        elif secondary_structure_tokens is not None:
            sequence_length = len(secondary_structure_tokens)
        elif sasa_tokens is not None:
            sequence_length = len(sasa_tokens)

        # Handle coordinates/structure
        if protein.coordinates is not None:
            # For structure generation, we need to encode the coordinates
            # This requires the structure encoder which uses PyTorch
            from esm.pretrained import load_local_model
            from esm.utils.constants.models import ESM3_OPEN_SMALL

            # Get structure encoder from a PyTorch model (lazy load)
            if not hasattr(self, '_structure_encoder') or self._structure_encoder is None:
                torch_model = load_local_model(ESM3_OPEN_SMALL, device=torch.device("cpu"))
                self._structure_encoder = torch_model.get_structure_encoder()

            coordinates, _, structure_tokens = encoding.tokenize_structure(
                protein.coordinates,
                self._structure_encoder,
                structure_tokenizer=self.tokenizers.structure,
                reference_sequence=protein.sequence or "",
                add_special_tokens=True,
            )
            if sequence_length == -1:
                sequence_length = len(structure_tokens)

        if sequence_length == -1:
            raise ValueError(
                "Cannot infer input length. Provide sequence, structure, secondary_structure, or sasa."
            )

        # Function and residue annotations
        if protein.function_annotations is not None:
            reference_sequence = protein.sequence or encoding.get_default_sequence(sequence_length - 2)
            function_tokens, residue_annotation_tokens = encoding.tokenize_function_annotations(
                protein.function_annotations,
                reference_sequence=reference_sequence,
                function_tokenizer=self.tokenizers.function,
                residue_annotation_tokenizer=self.tokenizers.residue_annotations,
                add_special_tokens=True,
            )

        return ESMProteinTensor(
            sequence=sequence_tokens,
            structure=structure_tokens,
            secondary_structure=secondary_structure_tokens,
            sasa=sasa_tokens,
            function=function_tokens,
            residue_annotations=residue_annotation_tokens,
            coordinates=coordinates,
            potential_sequence_of_concern=protein.potential_sequence_of_concern,
        )

    def decode(self, protein_tensor: ESMProteinTensor) -> ESMProtein:
        """Decode ESMProteinTensor back to ESMProtein.

        This matches the interface of esm.models.esm3.ESM3.decode().
        """
        import torch
        from esm.pretrained import load_local_model
        from esm.utils.constants.models import ESM3_OPEN_SMALL

        # Get structure decoder (need PyTorch model for this)
        if not hasattr(self, '_structure_decoder') or self._structure_decoder is None:
            torch_model = load_local_model(ESM3_OPEN_SMALL, device=torch.device("cpu"))
            self._structure_decoder = torch_model.get_structure_decoder()

        return decode_protein_tensor(
            protein_tensor,
            self.tokenizers,
            structure_token_decoder=self._structure_decoder,
        )

    def generate(self, protein: ProteinType, config: GenerationConfig) -> ProteinType:
        """Generate protein sequence/structure using the ESM3 API.

        This is the main entry point matching esm.models.esm3.ESM3.generate().

        Args:
            protein: ESMProtein or ESMProteinTensor input
            config: GenerationConfig specifying track, num_steps, temperature, etc.

        Returns:
            ESMProtein or ESMProteinTensor with generated track filled in
        """
        import torch

        # Handle ESMProtein input
        if isinstance(protein, ESMProtein):
            protein_tensor = self.encode(protein)
            result_tensor = self._generate_tensor(protein_tensor, config)
            result = self.decode(result_tensor)
            # Copy over any input tracks that weren't generated
            if config.track == "sequence" and protein.coordinates is not None:
                result.coordinates = protein.coordinates
            return result
        elif isinstance(protein, ESMProteinTensor):
            return self._generate_tensor(protein, config)
        else:
            raise ValueError(f"Unknown protein type: {type(protein)}")

    def _generate_tensor(
        self, protein_tensor: ESMProteinTensor, config: GenerationConfig
    ) -> ESMProteinTensor:
        """Generate on tokenized protein tensor."""
        import attr
        import torch

        # Get sequence length from tensor
        seq_len = None
        for field in [protein_tensor.sequence, protein_tensor.structure,
                      protein_tensor.secondary_structure, protein_tensor.sasa]:
            if field is not None:
                seq_len = len(field)
                break

        if seq_len is None:
            raise ValueError("Cannot determine sequence length from protein tensor")

        def get_or_default(tensor, default_token, dtype=mx.int32):
            if tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    return mx.array(tensor.numpy(), dtype=dtype)
                return mx.array(tensor, dtype=dtype)
            return mx.full((1, seq_len), default_token, dtype=dtype)

        sequence_tokens = get_or_default(protein_tensor.sequence, C.SEQUENCE_MASK_TOKEN)
        structure_tokens = get_or_default(protein_tensor.structure, C.STRUCTURE_MASK_TOKEN)
        ss8_tokens = get_or_default(protein_tensor.secondary_structure, C.SS8_PAD_TOKEN)
        sasa_tokens = get_or_default(protein_tensor.sasa, C.SASA_PAD_TOKEN)

        # Handle function tokens (shape: seq_len, 8)
        if protein_tensor.function is not None:
            if isinstance(protein_tensor.function, torch.Tensor):
                function_tokens = mx.array(protein_tensor.function.numpy(), dtype=mx.int32)
            else:
                function_tokens = mx.array(protein_tensor.function, dtype=mx.int32)
            if len(function_tokens.shape) == 2:
                function_tokens = function_tokens[None, ...]  # Add batch dim
        else:
            function_tokens = mx.zeros((1, seq_len, 8), dtype=mx.int32)

        # Handle residue annotations (shape: seq_len, 16)
        if protein_tensor.residue_annotations is not None:
            if isinstance(protein_tensor.residue_annotations, torch.Tensor):
                residue_tokens = mx.array(protein_tensor.residue_annotations.numpy(), dtype=mx.int32)
            else:
                residue_tokens = mx.array(protein_tensor.residue_annotations, dtype=mx.int32)
            if len(residue_tokens.shape) == 2:
                residue_tokens = residue_tokens[None, ...]
        else:
            residue_tokens = mx.zeros((1, seq_len, 16), dtype=mx.int32)

        # Ensure batch dimension
        if len(sequence_tokens.shape) == 1:
            sequence_tokens = sequence_tokens[None, :]
        if len(structure_tokens.shape) == 1:
            structure_tokens = structure_tokens[None, :]
        if len(ss8_tokens.shape) == 1:
            ss8_tokens = ss8_tokens[None, :]
        if len(sasa_tokens.shape) == 1:
            sasa_tokens = sasa_tokens[None, :]

        average_plddt = mx.ones((1, seq_len))
        per_res_plddt = mx.zeros((1, seq_len))

        # Generate based on track
        if config.track == "sequence":
            result = self.generate_sequence(
                sequence_tokens,
                structure_tokens,
                ss8_tokens,
                sasa_tokens,
                function_tokens,
                residue_tokens,
                average_plddt,
                per_res_plddt,
                mask_token=C.SEQUENCE_MASK_TOKEN,
                num_steps=config.num_steps,
                temperature=config.temperature if config.temperature else 1.0,
                use_cache=True,
            )
            # Convert result back to torch tensor
            result_np = result.tolist()
            result_torch = torch.tensor(result_np, dtype=torch.long).squeeze(0)
            output = attr.evolve(protein_tensor)
            output.sequence = result_torch

        elif config.track == "structure":
            result = self.generate_structure(
                sequence_tokens,
                structure_tokens,
                ss8_tokens,
                sasa_tokens,
                function_tokens,
                residue_tokens,
                average_plddt,
                per_res_plddt,
                num_steps=config.num_steps,
                temperature=config.temperature if config.temperature else 1.0,
                use_cache=True,
            )
            # Convert result back to torch tensor
            result_np = result.tolist()
            result_torch = torch.tensor(result_np, dtype=torch.long).squeeze(0)
            output = attr.evolve(protein_tensor)
            output.structure = result_torch

        else:
            raise ValueError(f"Unsupported track: {config.track}. Use 'sequence' or 'structure'.")

        return output

    def predict_function(
        self,
        protein: ESMProtein,
        temperature: float = 1.0,
        p_none_threshold: float = 0.05,
    ) -> list:
        """Predict function annotations from sequence/structure.

        This runs a forward pass through the model and samples function tokens
        from the function logits, then decodes them to FunctionAnnotation objects.

        Args:
            protein: ESMProtein with sequence (and optionally structure)
            temperature: Sampling temperature for function tokens
            p_none_threshold: Threshold for predicting "no function" at a position

        Returns:
            List of FunctionAnnotation objects (label, start, end)
        """
        import torch
        from esm.pretrained import load_local_model
        from esm.utils.constants.models import ESM3_OPEN_SMALL
        from esm.utils.function.encode_decode import decode_function_tokens
        from esm.utils.types import FunctionAnnotation

        # Encode protein
        protein_tensor = self.encode(protein)
        seq_len = len(protein_tensor.sequence)

        def get_or_default(tensor, default_token, dtype=mx.int32):
            if tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    return mx.array(tensor.numpy(), dtype=dtype)
                return mx.array(tensor, dtype=dtype)
            return mx.full((1, seq_len), default_token, dtype=dtype)

        sequence_tokens = get_or_default(protein_tensor.sequence, C.SEQUENCE_MASK_TOKEN)
        structure_tokens = get_or_default(protein_tensor.structure, C.STRUCTURE_MASK_TOKEN)
        ss8_tokens = get_or_default(protein_tensor.secondary_structure, C.SS8_PAD_TOKEN)
        sasa_tokens = get_or_default(protein_tensor.sasa, C.SASA_PAD_TOKEN)

        # Function tokens - use zeros/mask
        function_tokens = mx.zeros((1, seq_len, 8), dtype=mx.int32)
        residue_tokens = mx.zeros((1, seq_len, 16), dtype=mx.int32)

        # Ensure batch dimension
        if len(sequence_tokens.shape) == 1:
            sequence_tokens = sequence_tokens[None, :]
        if len(structure_tokens.shape) == 1:
            structure_tokens = structure_tokens[None, :]
        if len(ss8_tokens.shape) == 1:
            ss8_tokens = ss8_tokens[None, :]
        if len(sasa_tokens.shape) == 1:
            sasa_tokens = sasa_tokens[None, :]

        average_plddt = mx.ones((1, seq_len))
        per_res_plddt = mx.zeros((1, seq_len))

        # Forward pass - returns (outputs_dict, cache)
        print(f"[predict_function] Running forward pass for seq_len={seq_len}")
        output, _ = self(
            sequence_tokens,
            structure_tokens,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_tokens,
            average_plddt,
            per_res_plddt,
        )

        # Get function logits: (B, L, 8, 260)
        function_logits = output["function_logits"]
        mx.eval(function_logits)
        print(f"[predict_function] function_logits shape: {function_logits.shape}")

        # Sample from function logits using temperature and p_none threshold
        # Convert to numpy/torch for compatibility with existing decoder
        func_logits_np = function_logits[0].tolist()  # (L, 8, 260)
        func_logits_torch = torch.tensor(func_logits_np, dtype=torch.float32)
        print(f"[predict_function] func_logits_torch shape: {func_logits_torch.shape}")

        # Sample function tokens using argmax with none-threshold logic
        # Shape: (L, 8, 260) -> (L, 8)
        log_p = torch.nn.functional.log_softmax(func_logits_torch / temperature, dim=-1)

        # Check for none predictions
        none_index = 0  # <none> is typically at index 0 in the function tokenizer
        log_p_nones = log_p[..., none_index]  # (L, 8)
        p_none = torch.exp(log_p_nones).mean(dim=-1)  # (L,)
        where_none = p_none > p_none_threshold

        num_none = where_none.sum().item()
        print(f"[predict_function] Positions with p_none > {p_none_threshold}: {num_none}/{len(where_none)}")

        # Sample tokens - use argmax for deterministic prediction
        function_ids = torch.argmax(log_p, dim=-1)  # (L, 8)
        function_ids[where_none, :] = none_index

        # Remove BOS/EOS
        function_ids = function_ids[1:-1]  # (L-2, 8)
        print(f"[predict_function] function_ids shape (after BOS/EOS removal): {function_ids.shape}")

        # Debug: check unique token values
        unique_tokens = torch.unique(function_ids).tolist()
        print(f"[predict_function] Unique token values: {unique_tokens[:20]}{'...' if len(unique_tokens) > 20 else ''}")

        # Decode function tokens to annotations
        # Get function token decoder from PyTorch model (lazy load)
        if not hasattr(self, '_function_decoder') or self._function_decoder is None:
            print("[predict_function] Loading function token decoder...")
            torch_model = load_local_model(ESM3_OPEN_SMALL, device=torch.device("cpu"))
            self._function_decoder = torch_model.get_function_decoder()
            print("[predict_function] Function token decoder loaded")

        try:
            print(f"[predict_function] Decoding function tokens...")
            annotations = decode_function_tokens(
                function_ids,
                function_token_decoder=self._function_decoder,
                function_tokens_tokenizer=self.tokenizers.function,
                decoder_annotation_threshold=0.1,
                annotation_min_length=5,
                annotation_gap_merge_max=3,
            )
            print(f"[predict_function] Decoded {len(annotations)} annotations")
            for i, ann in enumerate(annotations[:5]):  # Show first 5
                print(f"  [{i}] {ann.label} ({ann.start}-{ann.end})")
            if len(annotations) > 5:
                print(f"  ... and {len(annotations) - 5} more")
            return annotations
        except Exception as e:
            import traceback
            print(f"[predict_function] Decoding error: {e}")
            traceback.print_exc()
            return []
