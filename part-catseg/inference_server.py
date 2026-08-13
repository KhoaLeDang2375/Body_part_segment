"""
inference_server.py
-------------------
FastAPI server wrapping PartCATSeg InferenceEngine.
Runs in conda env `partcatseg` on port 8001.

Endpoints:
    POST /segment_parts   → per-part binary masks (base64 PNG)
    GET  /health          → service health check
    GET  /classes         → available object classes & parts

Usage:
    conda activate partcatseg
    cd part-catseg
    python inference_server.py                     # default port 8001
    python inference_server.py --port 8001 --device cuda
"""

import argparse
import base64
import io
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="PartCATSeg Inference Server")
parser.add_argument("--port", type=int, default=8001, help="Server port")
parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
cli_args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────
# Global engine — loaded in background thread at startup
# ─────────────────────────────────────────────────────────────────
_engine = None
_model_ready = False  # True only after weights fully loaded

def _load_model_background():
    """Run model loading in a background thread so uvicorn starts immediately."""
    global _engine, _model_ready
    try:
        from inference_engine import InferenceEngine
        logger.info(f"[Background] Loading PartCATSeg model on {cli_args.device} ...")
        _engine = InferenceEngine(device=cli_args.device)
        _engine._ensure_loaded()
        _model_ready = True
        logger.info("[Background] PartCATSeg model loaded. Ready to serve requests.")
    except Exception as e:
        logger.error(f"[Background] Model loading failed: {e}", exc_info=True)

def _get_engine():
    """Return engine. Raises RuntimeError if model not ready yet."""
    if not _model_ready or _engine is None:
        raise RuntimeError("Model not ready yet. Please wait.")
    return _engine


def _mask_to_base64_png(mask: np.ndarray) -> str:
    """Convert a boolean mask (H, W) to base64-encoded PNG string."""
    mask_uint8 = (mask.astype(np.uint8)) * 255
    img = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────
# FastAPI App — model loads in background at startup
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start model loading in background thread when uvicorn starts."""
    thread = threading.Thread(target=_load_model_background, daemon=True)
    thread.start()
    logger.info("uvicorn started. Model loading in background thread...")
    yield  # server runs here
    logger.info("Shutting down PartCATSeg server.")

app = FastAPI(
    title="PartCATSeg Inference Server",
    description="Body part segmentation API using PartCATSeg model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """
    Lightweight health check — responds immediately without blocking.
    Returns {"status": "loading"} while model weights are loading,
    and {"status": "ok"} once ready.
    """
    if not _model_ready:
        return JSONResponse(
            status_code=200,
            content={"status": "loading", "model_loaded": False, "device": cli_args.device},
        )
    return {"status": "ok", "model_loaded": True, "device": cli_args.device}


@app.get("/classes")
def get_classes():
    """Return available object classes and their parts."""
    from app_config import VOC_OBJ_CLASSES, OBJ_TO_PARTS
    return {
        "object_classes": VOC_OBJ_CLASSES,
        "obj_to_parts": OBJ_TO_PARTS,
    }


@app.post("/segment_parts")
def segment_parts(
    image_file: UploadFile = File(...),
    obj_class: str = Form(default="person"),
    conf_threshold: float = Form(default=0.3),
    selected_parts: Optional[str] = Form(default=None),
):
    """
    Segment body parts from an uploaded image.

    Args:
        image_file:     Uploaded image file (JPEG/PNG).
        obj_class:      Object class to segment (default: "person").
        conf_threshold: Minimum confidence threshold (default: 0.3).
        selected_parts: Comma-separated part names to filter (optional).
                        E.g. "head,torso,hand"

    Returns:
        JSON with:
            - parts: Dict[part_name → base64 PNG mask]
            - scores: Dict[part_name → confidence]
            - detected_count: number of parts detected
            - image_size: [width, height]
    """
    try:
        t0 = time.time()

        # Read image
        image_data = await image_file.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Parse selected parts
        parts_filter = None
        if selected_parts and selected_parts.strip():
            parts_filter = [p.strip() for p in selected_parts.split(",")]

        # Run inference
        engine = _get_engine()
        result = engine.predict_raw_masks(
            pil_image,
            obj_class_name=obj_class,
            conf_threshold=conf_threshold,
            selected_parts=parts_filter,
        )

        # Convert masks to base64 PNG
        parts_b64 = {}
        for part_name, mask in result["masks"].items():
            parts_b64[part_name] = _mask_to_base64_png(mask)

        elapsed = time.time() - t0
        logger.info(
            f"Segmented {len(parts_b64)} parts in {elapsed:.2f}s "
            f"(class={obj_class}, conf={conf_threshold})"
        )

        return JSONResponse(content={
            "parts": parts_b64,
            "scores": result["scores"],
            "detected_count": len(parts_b64),
            "image_size": [pil_image.width, pil_image.height],
            "inference_time_s": round(elapsed, 3),
        })

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Error in segment_parts: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting PartCATSeg server on {cli_args.host}:{cli_args.port}")
    logger.info(f"Device: {cli_args.device}")
    logger.info("Model will load in background thread after uvicorn starts.")
    # Model loading happens in background via lifespan event
    uvicorn.run(
        app,
        host=cli_args.host,
        port=cli_args.port,
        log_level="info",
    )
