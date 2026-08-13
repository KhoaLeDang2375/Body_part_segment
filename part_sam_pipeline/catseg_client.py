"""
catseg_client.py
----------------
HTTP client for the PartCATSeg inference server.
Communicates with the FastAPI server running on port 8001.

Usage:
    client = CatSegClient("http://localhost:8001")
    result = client.get_person_parts(pil_image, conf=0.3)
    # result["masks"] → Dict[str, np.ndarray (H,W) bool]
    # result["scores"] → Dict[str, float]
"""

import base64
import io
import logging
from typing import Dict, List, Optional

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Timeout for HTTP requests (model inference can take a while on first call)
DEFAULT_TIMEOUT = 120.0


class CatSegClient:
    """HTTP client for the PartCATSeg inference server."""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """Check if the server is alive and model is loaded."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    def get_classes(self) -> dict:
        """Get available object classes and their parts."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{self.base_url}/classes")
            resp.raise_for_status()
            return resp.json()

    def get_person_parts(
        self,
        pil_image: Image.Image,
        obj_class: str = "person",
        conf_threshold: float = 0.3,
        selected_parts: Optional[List[str]] = None,
    ) -> dict:
        """
        Send image to PartCATSeg server and get per-part binary masks.

        Args:
            pil_image:       PIL image to segment.
            obj_class:       Object class (default: "person").
            conf_threshold:  Confidence threshold.
            selected_parts:  Optional list of part names to filter.

        Returns:
            Dict with:
                "masks":  Dict[str, np.ndarray]  — {part_name: binary mask (H,W) bool}
                "scores": Dict[str, float]       — {part_name: confidence}
                "image_size": [width, height]
        """
        # Encode image as JPEG bytes for efficient transfer
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        # Build form data
        files = {"image_file": ("image.jpg", buf, "image/jpeg")}
        data = {
            "obj_class": obj_class,
            "conf_threshold": str(conf_threshold),
        }
        if selected_parts:
            data["selected_parts"] = ",".join(selected_parts)

        # Send request
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/segment_parts",
                files=files,
                data=data,
            )
            resp.raise_for_status()
            result = resp.json()

        # Decode base64 PNG masks → numpy arrays
        masks_dict = {}
        for part_name, b64_str in result.get("parts", {}).items():
            mask_bytes = base64.b64decode(b64_str)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            mask_np = np.array(mask_img) > 127  # threshold to bool
            masks_dict[part_name] = mask_np

        logger.info(
            f"CATSeg returned {len(masks_dict)} parts: {list(masks_dict.keys())} "
            f"(inference: {result.get('inference_time_s', '?')}s)"
        )

        return {
            "masks": masks_dict,
            "scores": result.get("scores", {}),
            "image_size": result.get("image_size", [pil_image.width, pil_image.height]),
        }

    def wait_for_server(self, max_retries: int = 30, interval: float = 2.0) -> bool:
        """
        Wait for the PartCATSeg server to become available.

        Args:
            max_retries: Maximum number of retry attempts.
            interval:    Seconds between retries.

        Returns:
            True if server is ready, False if timed out.
        """
        import time
        for i in range(max_retries):
            try:
                status = self.health()
                if status.get("status") == "ok":
                    logger.info(f"PartCATSeg server ready (attempt {i+1})")
                    return True
            except Exception as e:
                logger.warning(f"Health check attempt {i+1}/{max_retries} failed: {e}")
            if i < max_retries - 1:
                time.sleep(interval)
        logger.error(f"PartCATSeg server not available after {max_retries} attempts")
        return False
