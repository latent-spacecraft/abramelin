"""Protein Alchemy - Interactive ESM3 Web Interface.

A mesmerizing protein design workbench powered by ESM3MLX.
"""

import asyncio
import io
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# ESM3MLX imports
from esm.models.mlx import ESM3MLX
from esm.sdk.api import ESMProtein, GenerationConfig

# Global model reference
model: Optional[ESM3MLX] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    print("Loading ESM3MLX model...")
    model = ESM3MLX.from_pretrained("esm3_mlx_weights.npz")
    print(f"Model loaded on device: {model.device}")
    yield
    # Cleanup if needed
    model = None


app = FastAPI(
    title="Protein Alchemy",
    description="Interactive protein design with ESM3MLX",
    lifespan=lifespan,
)

# Paths
STATIC_DIR = Path(__file__).parent / "static"

# Global state
sessions: dict[str, dict] = {}


# ============================================================================
# Pydantic Models
# ============================================================================

class GenerateRequest(BaseModel):
    sequence: str
    num_steps: int = 8
    temperature: float = 1.0
    session_id: Optional[str] = None


class MaskRequest(BaseModel):
    session_id: str
    action: str  # "add", "remove", "toggle", "set", "clear"
    indices: list[int] = []


class ProteinResponse(BaseModel):
    session_id: str
    sequence: str
    pdb: Optional[str] = None
    plddt: Optional[list[float]] = None
    masked_indices: list[int] = []


# ============================================================================
# Static Files & HTML
# ============================================================================

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main application."""
    return FileResponse(STATIC_DIR / "index.html")


# ============================================================================
# REST API
# ============================================================================

@app.post("/api/generate")
async def start_generation(request: GenerateRequest) -> dict:
    """Start a generation job (always runs sequence then structure)."""
    session_id = request.session_id or str(uuid.uuid4())

    sessions[session_id] = {
        "sequence": request.sequence,
        "num_steps": request.num_steps,
        "temperature": request.temperature,
        "status": "pending",
        "result": None,
        "masked_indices": [i for i, c in enumerate(request.sequence) if c == "_"],
    }

    return {"session_id": session_id, "status": "pending"}


@app.get("/api/protein/{session_id}")
async def get_protein(session_id: str) -> ProteinResponse:
    """Get current protein state."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return ProteinResponse(
        session_id=session_id,
        sequence=session.get("sequence", ""),
        pdb=session.get("pdb"),
        plddt=session.get("plddt"),
        masked_indices=session.get("masked_indices", []),
    )


@app.post("/api/mask")
async def update_mask(request: MaskRequest) -> dict:
    """Update mask state for a session."""
    if request.session_id not in sessions:
        sessions[request.session_id] = {"masked_indices": [], "sequence": ""}

    session = sessions[request.session_id]
    current_masks = set(session.get("masked_indices", []))

    if request.action == "add":
        current_masks.update(request.indices)
    elif request.action == "remove":
        current_masks.difference_update(request.indices)
    elif request.action == "toggle":
        for idx in request.indices:
            if idx in current_masks:
                current_masks.remove(idx)
            else:
                current_masks.add(idx)
    elif request.action == "set":
        current_masks = set(request.indices)
    elif request.action == "clear":
        current_masks = set()

    session["masked_indices"] = sorted(current_masks)

    return {"masked_indices": session["masked_indices"]}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str) -> dict:
    """Get generation history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"history": sessions[session_id].get("history", [])}


# ============================================================================
# WebSocket for Generation Streaming
# ============================================================================

@app.websocket("/ws/generate/{session_id}")
async def websocket_generate(websocket: WebSocket, session_id: str):
    """Stream generation progress via WebSocket.

    Always runs:
    1. Sequence generation (if there are masks)
    2. Structure generation (always, to fold the sequence)
    """
    await websocket.accept()

    if session_id not in sessions:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    session = sessions[session_id]
    sequence = session["sequence"]
    num_steps = session.get("num_steps", 8)
    temperature = session.get("temperature", 1.0)
    masked_count = sequence.count("_")

    try:
        # Send initial status
        await websocket.send_json({
            "type": "start",
            "sequence_length": len(sequence),
            "masked_count": masked_count,
            "num_steps": num_steps,
        })

        # Create protein
        protein = ESMProtein(sequence=sequence)

        # =====================================================================
        # Phase 1: Sequence Generation (if there are masks)
        # =====================================================================
        if masked_count > 0:
            await websocket.send_json({
                "type": "phase",
                "phase": "sequence",
                "message": "Generating sequence...",
            })

            # Progress updates for sequence generation
            for step in range(num_steps):
                await websocket.send_json({
                    "type": "progress",
                    "phase": "sequence",
                    "step": step + 1,
                    "total_steps": num_steps,
                })
                await asyncio.sleep(0.02)

            # Run sequence generation
            seq_config = GenerationConfig(
                track="sequence",
                num_steps=num_steps,
                temperature=temperature,
            )
            protein = model.generate(protein, seq_config)

            await websocket.send_json({
                "type": "sequence_complete",
                "sequence": protein.sequence,
            })

        # =====================================================================
        # Phase 2: Structure Generation (always run)
        # =====================================================================
        await websocket.send_json({
            "type": "phase",
            "phase": "structure",
            "message": "Folding structure...",
        })

        # Progress updates for structure generation
        for step in range(num_steps):
            await websocket.send_json({
                "type": "progress",
                "phase": "structure",
                "step": step + 1,
                "total_steps": num_steps,
            })
            await asyncio.sleep(0.02)

        # Run structure generation
        struct_config = GenerationConfig(
            track="structure",
            num_steps=num_steps,
            temperature=0.7,  # Lower temp for structure
        )
        protein = model.generate(protein, struct_config)

        # Get PDB string
        pdb_string = None
        plddt_list = None

        if protein.coordinates is not None:
            try:
                pdb_string = protein.to_protein_chain().to_pdb_string()
            except Exception as e:
                print(f"PDB generation error: {e}")
                # Try alternative: write to buffer
                try:
                    buffer = io.StringIO()
                    protein.to_pdb(buffer)
                    pdb_string = buffer.getvalue()
                except Exception as e2:
                    print(f"Fallback PDB error: {e2}")

        if protein.plddt is not None:
            plddt_list = protein.plddt.tolist()

        # Update session
        session["result"] = protein
        session["pdb"] = pdb_string
        session["plddt"] = plddt_list
        session["sequence"] = protein.sequence
        session["status"] = "complete"

        # Add to history
        if "history" not in session:
            session["history"] = []
        session["history"].append({
            "sequence": protein.sequence,
            "pdb": pdb_string,
            "plddt": plddt_list,
        })

        # Send completion
        await websocket.send_json({
            "type": "complete",
            "sequence": protein.sequence,
            "pdb": pdb_string,
            "plddt": plddt_list,
        })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
