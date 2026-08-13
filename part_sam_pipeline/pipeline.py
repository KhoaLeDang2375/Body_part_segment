"""
pipeline.py
-----------
Main orchestrator: PartCATSeg → Mask Conversion → SAM3 Refinement.

This module:
    1. Calls PartCATSeg server to get coarse body-part masks
    2. Converts masks to SAM3 prompt format (box + text)
    3. Runs SAM3 per body part to get refined high-quality masks

Usage:
    pipeline = PartSamPipeline(sam3_checkpoint="/workspace/checkpoints/sam3.pt")
    results = pipeline.segment_all_parts(pil_image)
    # results["refined_masks"] → Dict[str, np.ndarray (H,W) bool]
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .catseg_client import CatSegClient
from .mask_converter import build_sam3_prompts

logger = logging.getLogger(__name__)

# 14 human body parts from PartCATSeg (person class)
PERSON_PARTS = [
    "head", "eye", "torso", "neck", "leg", "foot",
    "nose", "ear", "eyebrow", "mouth", "hair",
    "lower arm", "upper arm", "hand",
]


class PartSamPipeline:
    """
    Orchestrates PartCATSeg (coarse part segmentation) → SAM3 (refinement).

    Architecture:
        - PartCATSeg runs in a separate conda env as a FastAPI server (port 8001)
        - SAM3 runs in this process (same conda env as this pipeline)
        - Communication is via HTTP localhost
    """

    def __init__(
        self,
        sam3_checkpoint: Optional[str] = None,
        catseg_url: str = "http://localhost:8001",
        device: str = "cuda",
        sam3_confidence: float = 0.3,
    ):
        """
        Args:
            sam3_checkpoint: Path to SAM3 .pt checkpoint. If None, downloads from HF.
            catseg_url:      URL of the PartCATSeg inference server.
            device:          Device for SAM3 model (cuda/cpu).
            sam3_confidence: Confidence threshold for SAM3 mask output.
        """
        self.device = device
        self.sam3_checkpoint = sam3_checkpoint
        self.sam3_confidence = sam3_confidence

        # PartCATSeg client
        self.catseg = CatSegClient(base_url=catseg_url)

        # SAM3 components (lazy loaded)
        self._sam3_model = None
        self._sam3_processor = None

    def _ensure_sam3_loaded(self):
        """Lazy-load SAM3 model on first use."""
        if self._sam3_processor is not None:
            return

        logger.info("Loading SAM3 model ...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self._sam3_model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=self.sam3_checkpoint,
            load_from_HF=(self.sam3_checkpoint is None),
            enable_inst_interactivity=False,  # Text + Box only, save VRAM
        )
        self._sam3_processor = Sam3Processor(
            self._sam3_model,
            device=self.device,
            confidence_threshold=self.sam3_confidence,
        )
        logger.info("SAM3 model loaded successfully.")

    @torch.inference_mode()
    def segment_all_parts(
        self,
        pil_image: Image.Image,
        obj_class: str = "person",
        catseg_conf: float = 0.3,
        target_parts: Optional[List[str]] = None,
        bbox_padding: float = 0.05,
    ) -> dict:
        """
        Full pipeline: PartCATSeg → SAM3 per-part refinement.

        Args:
            pil_image:    Input image (PIL RGB).
            obj_class:    Object class for PartCATSeg (default: "person").
            catseg_conf:  Confidence threshold for PartCATSeg.
            target_parts: Optional list of part names to segment.
                          If None, segments all 14 human parts.
            bbox_padding: Fractional padding for SAM3 bounding box prompts.

        Returns:
            Dict with:
                "coarse_masks":  Dict[str, np.ndarray]  — raw PartCATSeg masks
                "refined_masks": Dict[str, np.ndarray]  — SAM3-refined masks
                "scores":        Dict[str, dict]        — {part: {catseg, sam3}}
                "prompts":       Dict[str, dict]        — SAM3 prompt specs
                "timing":        dict                   — performance metrics
        """
        t_total = time.time()

        # ── Step 1: Get coarse masks from PartCATSeg ──────────────
        t0 = time.time()
        logger.info(f"Step 1: Querying PartCATSeg for '{obj_class}' parts ...")

        catseg_result = self.catseg.get_person_parts(
            pil_image,
            obj_class=obj_class,
            conf_threshold=catseg_conf,
            selected_parts=target_parts,
        )
        coarse_masks = catseg_result["masks"]
        catseg_scores = catseg_result["scores"]
        t_catseg = time.time() - t0

        logger.info(
            f"  PartCATSeg returned {len(coarse_masks)} parts: "
            f"{list(coarse_masks.keys())} ({t_catseg:.2f}s)"
        )

        if not coarse_masks:
            return {
                "coarse_masks": {},
                "refined_masks": {},
                "scores": {},
                "prompts": {},
                "timing": {"catseg_s": t_catseg, "sam3_s": 0, "total_s": time.time() - t_total},
            }

        # ── Step 2: Build SAM3 prompts from coarse masks ──────────
        img_w, img_h = pil_image.size
        prompts = build_sam3_prompts(
            coarse_masks, img_w, img_h,
            target_parts=target_parts,
            padding=bbox_padding,
        )
        logger.info(f"Step 2: Built {len(prompts)} SAM3 prompts")

        # ── Step 3: Run SAM3 refinement per part ──────────────────
        t0 = time.time()
        logger.info("Step 3: Running SAM3 refinement ...")

        self._ensure_sam3_loaded()

        # Set image once (backbone encoding is expensive)
        state = self._sam3_processor.set_image(pil_image)

        refined_masks = {}
        scores_combined = {}

        for part_name, prompt in prompts.items():
            logger.info(f"  Refining '{part_name}' with SAM3 ...")

            # Reset prompts for each part
            self._sam3_processor.reset_all_prompts(state)

            # Add box prompt from PartCATSeg mask
            state = self._sam3_processor.add_geometric_prompt(
                box=prompt["box"],
                label=True,
                state=state,
            )

            # Add text prompt for context
            state = self._sam3_processor.set_text_prompt(
                prompt["text"],
                state=state,
            )

            # Extract refined mask
            if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
                # Take the best mask (highest score)
                mask_tensor = state["masks"][0].squeeze()  # (H, W) bool
                refined_mask = mask_tensor.cpu().numpy()
                refined_masks[part_name] = refined_mask

                sam3_score = 0.0
                if "scores" in state and state["scores"] is not None and len(state["scores"]) > 0:
                    sam3_score = float(state["scores"][0].item())

                scores_combined[part_name] = {
                    "catseg": catseg_scores.get(part_name, 0.0),
                    "sam3": round(sam3_score, 4),
                }
            else:
                # SAM3 didn't detect anything — fall back to coarse mask
                logger.warning(f"  SAM3 returned no mask for '{part_name}', using coarse mask")
                refined_masks[part_name] = coarse_masks[part_name]
                scores_combined[part_name] = {
                    "catseg": catseg_scores.get(part_name, 0.0),
                    "sam3": 0.0,
                }

        t_sam3 = time.time() - t0
        t_total_elapsed = time.time() - t_total

        logger.info(
            f"Pipeline complete: {len(refined_masks)} parts refined "
            f"(CATSeg: {t_catseg:.2f}s, SAM3: {t_sam3:.2f}s, Total: {t_total_elapsed:.2f}s)"
        )

        return {
            "coarse_masks": coarse_masks,
            "refined_masks": refined_masks,
            "scores": scores_combined,
            "prompts": prompts,
            "timing": {
                "catseg_s": round(t_catseg, 3),
                "sam3_s": round(t_sam3, 3),
                "total_s": round(t_total_elapsed, 3),
            },
        }

    def segment_single_part(
        self,
        pil_image: Image.Image,
        part_name: str,
        obj_class: str = "person",
        catseg_conf: float = 0.3,
    ) -> Optional[np.ndarray]:
        """
        Convenience method: segment a single body part.

        Returns:
            np.ndarray (H, W) bool — refined mask, or None if not detected.
        """
        result = self.segment_all_parts(
            pil_image,
            obj_class=obj_class,
            catseg_conf=catseg_conf,
            target_parts=[part_name],
        )
        return result["refined_masks"].get(part_name)
