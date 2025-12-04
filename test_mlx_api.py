#!/usr/bin/env python3
"""Test ESM3MLX as a drop-in replacement for ESM3."""

from esm.models.mlx import ESM3MLX
from esm.sdk.api import ESMProtein, GenerationConfig
import time

# Test with MLX
print("=" * 70)
print("Testing ESM3MLX (MLX Implementation)")
print("=" * 70)

# Load model
print("\n[1] Loading MLX model...")
start = time.perf_counter()
model = ESM3MLX.from_pretrained("esm3-open")
print(f"    Loaded in {time.perf_counter() - start:.1f}s")
print(f"    Device: {model.device}")

# Generate a completion for a partial sequence
prompt = "______STTCWAQGWFISTGDLASSAFITTIAIHTYLSVVRDYKLPTWAFWCMIGSVWFFIYALAIAGVIITN____________RAAAWCWVNVRYEAMRLYLHYLWMFVSFFITAVLYVLIFNHIRRTDPSLQLPSSSNNTTSSASQSNR_____________________TAPLALGRVITMAGKSVSLEYFCLAGAMIASNGWLDVLLFSTTRHVIIFNASPDYEETGIETFAFMRTPANRRYGNMVWVQGAGSAPNNLSADEGTGGWLWKLFHRGRGAGDLKRDRRRSG_____________"

print(f"\n[2] Creating protein from prompt...")
print(f"    Prompt length: {len(prompt)}")
print(f"    Masked positions: {prompt.count('_')}")

protein = ESMProtein(sequence=prompt)
"""
# Generate sequence
print("\n[3] Generating sequence...")
start = time.perf_counter()
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
seq_time = time.perf_counter() - start
print(f"    Time: {seq_time:.1f}s")
print(f"    Generated sequence: {protein.sequence[:50]}...")
"""
# Generate structure
print("\n[4] Generating structure...")
start = time.perf_counter()
try:
    protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
    struct_time = time.perf_counter() - start
    print(f"    Time: {struct_time:.1f}s")
    if protein.coordinates is not None:
        print(f"    Coordinates shape: {protein.coordinates.shape}")
except Exception as e:
    print(f"    Error: {e}")
    struct_time = None

# Save PDB
print("\n[5] Saving PDB...")
try:
    protein.to_pdb("./generation_mlx.pdb")
    print("    Saved to generation_mlx.pdb")
except Exception as e:
    print(f"    Error: {e}")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
#print(f"    Sequence generation: {seq_time:.1f}s")
if struct_time:
    print(f"    Structure generation: {struct_time:.1f}s")
print(f"    Total: {(struct_time or 0):.1f}s")
