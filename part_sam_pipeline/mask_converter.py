"""
mask_converter.py
-----------------
Convert PartCATSeg output masks to SAM3 prompt format.

SAM3's Sam3Processor.add_geometric_prompt() expects boxes in
[center_x, center_y, width, height] format, normalized to [0, 1].

This module handles:
    - Binary mask → bounding box conversion
    - Normalized cxcywh format conversion
    - Building SAM3 prompt specifications per body part
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def mask_to_bbox_xyxy(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract tight bounding box from a binary mask.

    Args:
        mask: Binary mask (H, W), dtype bool.

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates, or None if mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def bbox_xyxy_to_cxcywh_normalized(
    bbox: Tuple[int, int, int, int],
    img_width: int,
    img_height: int,
    padding: float = 0.05,
) -> List[float]:
    """
    Convert pixel bbox (x_min, y_min, x_max, y_max) to normalized [cx, cy, w, h].

    SAM3's add_geometric_prompt expects this format with values in [0, 1].

    Args:
        bbox:       (x_min, y_min, x_max, y_max) in pixel coordinates.
        img_width:  Image width in pixels.
        img_height: Image height in pixels.
        padding:    Fractional padding to add around the bbox (default: 5%).
                    Helps SAM3 capture context around the part.

    Returns:
        [center_x, center_y, width, height] normalized to [0, 1].
    """
    x_min, y_min, x_max, y_max = bbox

    # Add padding
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_x = box_w * padding
    pad_y = box_h * padding

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(img_width, x_max + pad_x)
    y_max = min(img_height, y_max + pad_y)

    # Convert to center + size, normalized
    cx = ((x_min + x_max) / 2) / img_width
    cy = ((y_min + y_max) / 2) / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height

    return [cx, cy, w, h]


def build_sam3_prompts(
    part_masks: Dict[str, np.ndarray],
    img_width: int,
    img_height: int,
    target_parts: Optional[List[str]] = None,
    padding: float = 0.05,
) -> Dict[str, dict]:
    """
    Build SAM3 prompt specifications from PartCATSeg masks.

    For each detected body part, generates:
        - box: [cx, cy, w, h] normalized — for add_geometric_prompt()
        - text: "person's {part_name}" — for set_text_prompt()
        - bbox_xyxy: [x_min, y_min, x_max, y_max] in pixels — for reference

    Args:
        part_masks:   Dict[part_name → binary mask (H,W) bool].
        img_width:    Original image width.
        img_height:   Original image height.
        target_parts: Optional filter — only include these parts.
        padding:      Fractional bbox padding for SAM3 prompt.

    Returns:
        Dict[part_name → {
            "box": [cx, cy, w, h],
            "text": "person's {part_name}",
            "bbox_xyxy": [x_min, y_min, x_max, y_max],
            "mask_pixels": int  — number of mask pixels
        }]
    """
    prompts = {}

    for part_name, mask in part_masks.items():
        # Filter by target parts if specified
        if target_parts is not None:
            if part_name.lower() not in {p.lower() for p in target_parts}:
                continue

        # Get bounding box
        bbox = mask_to_bbox_xyxy(mask)
        if bbox is None:
            continue

        # Convert to SAM3 format
        box_cxcywh = bbox_xyxy_to_cxcywh_normalized(
            bbox, img_width, img_height, padding=padding
        )

        prompts[part_name] = {
            "box": box_cxcywh,
            "text": f"person's {part_name}",
            "bbox_xyxy": list(bbox),
            "mask_pixels": int(mask.sum()),
        }

    return prompts
