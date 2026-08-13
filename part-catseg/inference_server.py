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
import time
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
# Global engine (lazy-loaded)
# ─────────────────────────────────────────────────────────────────
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        from inference_engine import InferenceEngine
        logger.info(f"Loading PartCATSeg model on {cli_args.device} ...")
        _engine = InferenceEngine(device=cli_args.device)
        _engine._ensure_loaded()
        logger.info("Model loaded. Ready to serve requests.")
    return _engine


def _mask_to_base64_png(mask: np.ndarray) -> str:
    """Convert a boolean mask (H, W) to base64-encoded PNG string."""
    # Convert bool mask to uint8 (0 or 255)
    mask_uint8 = (mask.astype(np.uint8)) * 255
    img = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PartCATSeg Inference Server",
    description="Body part segmentation API using PartCATSeg model",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Health check endpoint."""
    engine = _get_engine()
    model_loaded = engine._model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": cli_args.device,
    }


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

    # Pre-load model at startup
    _get_engine()

    uvicorn.run(
        app,
        host=cli_args.host,
        port=cli_args.port,
        log_level="info",
    )
