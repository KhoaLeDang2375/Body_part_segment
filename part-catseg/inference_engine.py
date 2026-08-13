"""
inference_engine.py
-------------------
Wraps PartCATSeg model loading and single-image inference.
Decoupled from Gradio so it can be tested independently.

Usage:
    engine = InferenceEngine(device="cuda")
    overlay_img, part_table = engine.predict(pil_image, "person", conf_threshold=0.3, alpha=0.6)
"""

import os
import sys
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Detectron2 imports ────────────────────────────────────────────
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import ColorMode

# ── PartCATSeg imports ────────────────────────────────────────────
# Register datasets (side-effect import — must happen before model build)
import baselines.data.datasets  # noqa: F401 — triggers dataset registration
from baselines import add_mask_former_config, PartCATSeg  # noqa: F401
from baselines.utils.misc import random_seed
from baselines.utils.visualizer import CustomVisualizer

from app_config import MODEL_CONFIG, VOC_OBJ_CLASSES, TRAIN_DATASET, TEST_DATASET

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _build_cfg(config_path: str, weights_path: str, device: str):
    """Build a frozen Detectron2 config for inference-only use."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg.merge_from_list([
        "MODEL.WEIGHTS",    weights_path,
        "MODEL.DEVICE",     device,
    ])
    cfg.freeze()
    return cfg


def _pil_to_tensor(pil_img: Image.Image, max_size: int = 768) -> torch.Tensor:
    """
    Convert a PIL image to a float32 CHW tensor in [0, 255] range.
    Resizes so the longer edge ≤ max_size (preserving aspect ratio).
    """
    w, h = pil_img.size
    scale = min(max_size / max(h, w), 1.0)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32)   # H W C
    tensor = torch.from_numpy(arr).permute(2, 0, 1)             # C H W
    return tensor


def _render_overlay(
    pil_img: Image.Image,
    sem_seg: torch.Tensor,
    metadata,
    conf_threshold: float,
    alpha: float,
    selected_parts: Optional[List[str]] = None,
) -> Tuple[Image.Image, List[Tuple[str, float]]]:
    """
    Render coloured part masks over the original image.

    Args:
        pil_img:        Original PIL image (H, W, RGB).
        sem_seg:        (K, H, W) sigmoid output from model — K part classes.
        metadata:       Detectron2 MetadataCatalog entry with stuff_classes & stuff_colors.
        conf_threshold: Minimum per-pixel confidence to colour a part.
        alpha:          Overlay opacity.
        selected_parts: Optional list of part names to filter (e.g., ["head", "hand"]).
                        If provided, only these parts will be displayed.

    Returns:
        overlay_pil:    PIL image with coloured mask overlay.
        part_table:     List of (part_name, max_confidence) sorted descending.
    """
    # ── 1. Resize sem_seg to original image size ──────────────────
    orig_h, orig_w = pil_img.size[1], pil_img.size[0]
    sem_seg_rs = F.interpolate(
        sem_seg.unsqueeze(0).float(),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)                                    # (K, H, W)

    # ── 1b. Filter by selected parts if specified ─────────────────
    if selected_parts:
        # Convert selected part names to lowercase for robust matching
        sel_set = {p.lower().strip() for p in selected_parts}
        for idx, cls_name in enumerate(metadata.stuff_classes):
            # cls_name is e.g. "person's head" -> part_name is "head"
            part_name = cls_name.split("'s")[1].strip() if "'s" in cls_name else cls_name
            if part_name.lower() not in sel_set and cls_name.lower() not in sel_set:
                sem_seg_rs[idx] = -100.0

    # ── 2. Argmax → class map, mask low-confidence pixels ─────────
    max_conf, class_map = sem_seg_rs.max(dim=0)    # (H, W) each
    class_map[max_conf < conf_threshold] = len(metadata.stuff_classes)  # "background" index

    # ── 3. Draw with CustomVisualizer ─────────────────────────────
    vis = CustomVisualizer(
        np.array(pil_img.convert("RGB")),
        metadata=metadata,
        instance_mode=ColorMode.IMAGE,
    )
    vis_output = vis.draw_sem_seg(class_map.numpy(), alpha=alpha)
    overlay_pil = Image.fromarray(vis_output.get_image())

    # ── 4. Build part confidence table ────────────────────────────
    classes = metadata.stuff_classes
    part_table = []
    for idx, cls_name in enumerate(classes):
        if idx < sem_seg_rs.shape[0]:
            max_score = float(sem_seg_rs[idx].max())
            if max_score >= conf_threshold:
                part_table.append((cls_name, round(max_score, 3)))
    part_table.sort(key=lambda x: x[1], reverse=True)

    return overlay_pil, part_table


# ─────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────

class InferenceEngine:
    """Lazy-loading inference wrapper for PartCATSeg."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model: Optional[torch.nn.Module] = None
        self._cfg = None
        self._metadata = None

    # ── Model loading ─────────────────────────────────────────────

    def _ensure_loaded(self):
        """Load model on first call (lazy init)."""
        if self._model is not None:
            return

        logger.info("Loading PartCATSeg model … (first-time only)")
        cfg = _build_cfg(
            MODEL_CONFIG["config"],
            MODEL_CONFIG["weights"],
            self.device,
        )
        self._cfg = cfg

        # Metadata (used for visualisation)
        self._metadata = MetadataCatalog.get(TEST_DATASET)

        # Build model + load weights
        from detectron2.modeling import build_model
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval()
        self._model = model
        logger.info("Model loaded successfully.")

    # ── Inference ─────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        pil_image: Image.Image,
        obj_class_name: str,
        conf_threshold: float = 0.3,
        alpha: float = 0.6,
        selected_parts: Optional[List[str]] = None,
    ) -> Tuple[Image.Image, List[Tuple[str, float]]]:
        """
        Run part segmentation on a single PIL image.

        Args:
            pil_image:      Input image (any size; will be resized ≤ 768 px).
            obj_class_name: Object class to segment (must be in VOC_OBJ_CLASSES).
            conf_threshold: Minimum confidence to display a part mask.
            alpha:          Overlay opacity (0 = transparent, 1 = opaque).
            selected_parts: Optional list of part names to filter.

        Returns:
            overlay_image:  PIL image with part masks drawn.
            part_table:     List of (part_name, confidence) tuples, sorted desc.
        """
        self._ensure_loaded()

        # ── Validate object class ─────────────────────────────────
        if obj_class_name not in VOC_OBJ_CLASSES:
            raise ValueError(
                f"'{obj_class_name}' is not in the supported object class list. "
                f"Choose from: {VOC_OBJ_CLASSES}"
            )
        obj_class_idx = VOC_OBJ_CLASSES.index(obj_class_name)

        # ── Prepare image tensor ──────────────────────────────────
        orig_w, orig_h = pil_image.size
        img_tensor = _pil_to_tensor(pil_image)      # float32 CHW [0,255]
        _, resized_h, resized_w = img_tensor.shape

        # ── Build Detectron2-style batched input ──────────────────
        # Even in eval mode, forward() accesses "obj_part_sem_seg" and "sem_seg"
        # at the top of the function, so we pass dummy zero tensors.
        dummy_part_seg = torch.zeros(1, resized_h, resized_w, dtype=torch.long)
        dummy_obj_seg  = torch.zeros(1, resized_h, resized_w, dtype=torch.long)

        # Instances must carry the object class index
        instances = Instances(image_size=(resized_h, resized_w))
        instances.gt_classes = torch.tensor([obj_class_idx], dtype=torch.long)
        instances.gt_boxes   = Boxes(torch.tensor([[0, 0, resized_w, resized_h]], dtype=torch.float32))

        batched_input = {
            "image":            img_tensor,
            "height":           orig_h,
            "width":            orig_w,
            "instances":        instances,
            "obj_part_sem_seg": dummy_part_seg,
            "sem_seg":          dummy_obj_seg,
        }

        # ── Run inference ─────────────────────────────────────────
        outputs = self._model([batched_input])
        sem_seg = outputs[0]["sem_seg"].cpu()       # (K, H, W)

        # ── Render overlay + build table ──────────────────────────
        overlay_img, part_table = _render_overlay(
            pil_image, sem_seg, self._metadata, conf_threshold, alpha, selected_parts
        )

        torch.cuda.empty_cache()
        return overlay_img, part_table

    # ── Raw mask extraction (for API / pipeline use) ──────────

    @torch.no_grad()
    def predict_raw_masks(
        self,
        pil_image: Image.Image,
        obj_class_name: str,
        conf_threshold: float = 0.3,
        selected_parts: Optional[List[str]] = None,
    ) -> dict:
        """
        Run part segmentation and return raw binary masks per body part.

        Args:
            pil_image:      Input image (any size; will be resized ≤ 768 px).
            obj_class_name: Object class to segment (must be in VOC_OBJ_CLASSES).
            conf_threshold: Minimum confidence to include a part mask.
            selected_parts: Optional list of part names to filter.

        Returns:
            Dict with keys:
                "masks":  Dict[str, np.ndarray]  — {part_name: binary mask (H, W) bool}
                "scores": Dict[str, float]       — {part_name: max_confidence}
        """
        self._ensure_loaded()

        # ── Validate object class ─────────────────────────────────
        if obj_class_name not in VOC_OBJ_CLASSES:
            raise ValueError(
                f"'{obj_class_name}' is not in the supported object class list. "
                f"Choose from: {VOC_OBJ_CLASSES}"
            )
        obj_class_idx = VOC_OBJ_CLASSES.index(obj_class_name)

        # ── Prepare image tensor ──────────────────────────────────
        orig_w, orig_h = pil_image.size
        img_tensor = _pil_to_tensor(pil_image)
        _, resized_h, resized_w = img_tensor.shape

        # ── Build Detectron2-style batched input ──────────────────
        dummy_part_seg = torch.zeros(1, resized_h, resized_w, dtype=torch.long)
        dummy_obj_seg  = torch.zeros(1, resized_h, resized_w, dtype=torch.long)

        instances = Instances(image_size=(resized_h, resized_w))
        instances.gt_classes = torch.tensor([obj_class_idx], dtype=torch.long)
        instances.gt_boxes   = Boxes(torch.tensor(
            [[0, 0, resized_w, resized_h]], dtype=torch.float32
        ))

        batched_input = {
            "image":            img_tensor,
            "height":           orig_h,
            "width":            orig_w,
            "instances":        instances,
            "obj_part_sem_seg": dummy_part_seg,
            "sem_seg":          dummy_obj_seg,
        }

        # ── Run inference ─────────────────────────────────────────
        outputs = self._model([batched_input])
        sem_seg = outputs[0]["sem_seg"].cpu()  # (K, H, W)

        # ── Resize to original image size ─────────────────────────
        sem_seg_rs = F.interpolate(
            sem_seg.unsqueeze(0).float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (K, H, W)

        # ── Extract per-part binary masks ─────────────────────────
        classes = self._metadata.stuff_classes
        sel_set = None
        if selected_parts:
            sel_set = {p.lower().strip() for p in selected_parts}

        masks_dict = {}
        scores_dict = {}
        for idx, cls_name in enumerate(classes):
            if idx >= sem_seg_rs.shape[0]:
                continue
            # Extract short part name: "person's head" → "head"
            part_name = cls_name.split("'s")[1].strip() if "'s" in cls_name else cls_name

            # Filter by selected parts if specified
            if sel_set is not None:
                if part_name.lower() not in sel_set and cls_name.lower() not in sel_set:
                    continue

            max_score = float(sem_seg_rs[idx].max())
            if max_score < conf_threshold:
                continue

            # Build binary mask via argmax: pixel belongs to this part if
            # this class has the highest score AND exceeds threshold
            max_conf, class_map = sem_seg_rs.max(dim=0)
            binary_mask = (class_map == idx) & (max_conf >= conf_threshold)
            binary_mask_np = binary_mask.numpy().astype(np.bool_)

            if binary_mask_np.sum() == 0:
                continue

            masks_dict[part_name] = binary_mask_np
            scores_dict[part_name] = round(max_score, 4)

        torch.cuda.empty_cache()
        return {"masks": masks_dict, "scores": scores_dict}
