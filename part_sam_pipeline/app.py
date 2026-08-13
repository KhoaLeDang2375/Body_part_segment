"""
app.py — Gradio UI for PartCATSeg + SAM3 Pipeline
--------------------------------------------------
Interactive web interface for body-part segmentation pipeline.

Features:
    - Upload image → segment all body parts
    - Compare coarse (PartCATSeg) vs refined (SAM3) masks
    - Select specific parts to display
    - Export all masks as ZIP

Usage:
    conda activate sam3env
    python -m part_sam_pipeline.app
    python -m part_sam_pipeline.app --checkpoint /workspace/checkpoints/sam3.pt
"""

import argparse
import io
import logging
import os
import tempfile
import zipfile
from typing import Dict, List, Optional

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="PartSAM Pipeline Gradio App")
parser.add_argument("--checkpoint", type=str, default=None, help="SAM3 checkpoint path")
parser.add_argument("--catseg-url", type=str, default="http://localhost:8001", help="PartCATSeg server URL")
parser.add_argument("--device", type=str, default="cuda", help="Device")
parser.add_argument("--port", type=int, default=7860, help="Gradio port")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio host")
parser.add_argument("--share", action="store_true", help="Create public Gradio link")
app_args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────
# Colors for overlay
# ─────────────────────────────────────────────────────────────────
PART_COLORS = {
    "head":      (255, 80,  80),
    "eye":       (80,  255, 80),
    "torso":     (80,  80,  255),
    "neck":      (255, 200, 80),
    "leg":       (200, 80,  255),
    "foot":      (80,  255, 200),
    "nose":      (255, 128, 0),
    "ear":       (0,   200, 200),
    "eyebrow":   (200, 200, 0),
    "mouth":     (255, 0,   128),
    "hair":      (128, 0,   255),
    "lower arm": (0,   128, 255),
    "upper arm": (128, 255, 0),
    "hand":      (255, 0,   255),
}

DEFAULT_COLORS = [
    (255, 56, 56), (56, 182, 255), (56, 255, 100),
    (255, 182, 56), (200, 56, 255), (56, 255, 232),
]

PERSON_PARTS = [
    "head", "eye", "torso", "neck", "leg", "foot",
    "nose", "ear", "eyebrow", "mouth", "hair",
    "lower arm", "upper arm", "hand",
]


# ─────────────────────────────────────────────────────────────────
# Pipeline (lazy loaded)
# ─────────────────────────────────────────────────────────────────
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from .pipeline import PartSamPipeline
        logger.info("Initializing PartSAM pipeline ...")
        _pipeline = PartSamPipeline(
            sam3_checkpoint=app_args.checkpoint,
            catseg_url=app_args.catseg_url,
            device=app_args.device,
        )
    return _pipeline


# ─────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────

def _get_color(part_name: str, idx: int = 0) -> tuple:
    """Get color for a body part."""
    return PART_COLORS.get(part_name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


def draw_masks_overlay(
    image: Image.Image,
    masks: Dict[str, np.ndarray],
    alpha: float = 0.5,
    show_labels: bool = True,
) -> Image.Image:
    """Draw coloured mask overlays on image with part labels."""
    result = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))

    for idx, (part_name, mask) in enumerate(masks.items()):
        color = _get_color(part_name, idx)
        color_rgba = color + (int(alpha * 255),)

        # Draw filled mask
        mask_img = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_img[mask] = color_rgba
        mask_pil = Image.fromarray(mask_img, mode="RGBA")
        overlay = Image.alpha_composite(overlay, mask_pil)

        # Draw label
        if show_labels and mask.any():
            draw = ImageDraw.Draw(overlay)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if rows.any() and cols.any():
                y_min = np.where(rows)[0][0]
                x_min = np.where(cols)[0][0]
                label = part_name
                # Background for readability
                draw.rectangle(
                    [x_min, max(0, y_min - 18), x_min + len(label) * 8 + 4, y_min],
                    fill=color + (200,)
                )
                draw.text((x_min + 2, max(0, y_min - 16)), label, fill=(255, 255, 255, 255))

    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def create_comparison_image(
    original: Image.Image,
    coarse_masks: Dict[str, np.ndarray],
    refined_masks: Dict[str, np.ndarray],
) -> Image.Image:
    """Create side-by-side comparison: Original | CATSeg (coarse) | SAM3 (refined)."""
    w, h = original.size

    coarse_overlay = draw_masks_overlay(original, coarse_masks, alpha=0.5)
    refined_overlay = draw_masks_overlay(original, refined_masks, alpha=0.5)

    # Create side-by-side canvas
    canvas = Image.new("RGB", (w * 3 + 20, h + 30), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Labels
    labels = ["Original", "PartCATSeg (Coarse)", "SAM3 (Refined)"]
    images = [original, coarse_overlay, refined_overlay]

    for i, (label, img) in enumerate(zip(labels, images)):
        x_offset = i * (w + 10)
        canvas.paste(img, (x_offset, 25))
        draw.text((x_offset + w // 2 - len(label) * 4, 5), label, fill=(200, 200, 200))

    return canvas


def masks_to_zip(masks: Dict[str, np.ndarray]) -> Optional[str]:
    """Save all masks as a ZIP file of PNGs. Returns the file path."""
    if not masks:
        return None

    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "body_part_masks.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for part_name, mask in masks.items():
            mask_uint8 = (mask.astype(np.uint8)) * 255
            img = Image.fromarray(mask_uint8, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            # Sanitize filename
            safe_name = part_name.replace(" ", "_").replace("'", "")
            zf.writestr(f"{safe_name}.png", buf.read())

    return zip_path


# ─────────────────────────────────────────────────────────────────
# Gradio callbacks
# ─────────────────────────────────────────────────────────────────

def run_pipeline(
    image: Image.Image,
    selected_parts: List[str],
    catseg_conf: float,
    overlay_alpha: float,
):
    """Main Gradio callback — runs the full pipeline."""
    if image is None:
        return None, None, None, "⚠️ Vui lòng upload ảnh trước."

    try:
        pipeline = _get_pipeline()

        # Filter parts
        parts_filter = selected_parts if selected_parts else None

        # Run pipeline
        result = pipeline.segment_all_parts(
            pil_image=image,
            obj_class="person",
            catseg_conf=catseg_conf,
            target_parts=parts_filter,
        )

        coarse = result["coarse_masks"]
        refined = result["refined_masks"]
        scores = result["scores"]
        timing = result["timing"]

        if not refined:
            return image, image, None, "⚠️ Không phát hiện bộ phận cơ thể nào. Thử giảm Confidence Threshold."

        # Create visualizations
        refined_overlay = draw_masks_overlay(image, refined, alpha=overlay_alpha)
        comparison = create_comparison_image(image, coarse, refined)

        # Create ZIP for download
        zip_path = masks_to_zip(refined)

        # Build summary text
        lines = [
            f"✅ **Phát hiện {len(refined)} bộ phận:**\n",
            "| Bộ phận | CATSeg Score | SAM3 Score |",
            "|---|---|---|",
        ]
        for part_name, part_scores in scores.items():
            catseg_s = part_scores.get("catseg", 0)
            sam3_s = part_scores.get("sam3", 0)
            lines.append(f"| {part_name} | {catseg_s:.3f} | {sam3_s:.3f} |")

        lines.append(f"\n⏱️ **Thời gian:** CATSeg={timing['catseg_s']:.2f}s | SAM3={timing['sam3_s']:.2f}s | Tổng={timing['total_s']:.2f}s")

        summary = "\n".join(lines)

        return refined_overlay, comparison, zip_path, summary

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return image, None, None, f"❌ Lỗi: {str(e)}"


# ─────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Body Part Segmentation — PartCATSeg + SAM3",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🧬 Body Part Segmentation Pipeline
            ### PartCATSeg (coarse detection) → SAM3 (refined segmentation)

            **Workflow:**
            1. Upload ảnh nhân vật
            2. Chọn bộ phận cần segment (hoặc để trống = tất cả)
            3. Nhấn **Run Pipeline** → nhận mask chất lượng cao cho từng bộ phận
            4. Download ZIP chứa tất cả masks
            """
        )

        with gr.Row():
            # ── Left column: Input ─────────────────────────────
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="📷 Upload Ảnh",
                    height=400,
                )

                selected_parts = gr.CheckboxGroup(
                    choices=PERSON_PARTS,
                    label="🎯 Chọn bộ phận (để trống = tất cả 14 parts)",
                    value=[],
                )

                with gr.Row():
                    catseg_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                        label="Confidence Threshold",
                    )
                    overlay_alpha = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.1,
                        label="Overlay Opacity",
                    )

                run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")

            # ── Right column: Output ───────────────────────────
            with gr.Column(scale=2):
                with gr.Tab("Refined Masks"):
                    output_refined = gr.Image(
                        type="pil",
                        label="SAM3 Refined Segmentation",
                        height=500,
                    )

                with gr.Tab("Comparison"):
                    output_comparison = gr.Image(
                        type="pil",
                        label="Original | CATSeg (Coarse) | SAM3 (Refined)",
                    )

                with gr.Row():
                    output_zip = gr.File(label="📦 Download Masks (ZIP)")
                    output_summary = gr.Markdown(label="📊 Results")

        # ── Connect callbacks ──────────────────────────────────
        run_btn.click(
            fn=run_pipeline,
            inputs=[input_image, selected_parts, catseg_conf, overlay_alpha],
            outputs=[output_refined, output_comparison, output_zip, output_summary],
        )

    return demo


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check if PartCATSeg server is available
    from .catseg_client import CatSegClient
    client = CatSegClient(app_args.catseg_url)
    logger.info(f"Checking PartCATSeg server at {app_args.catseg_url} ...")
    if not client.wait_for_server(max_retries=30, interval=2.0):
        logger.error(
            f"PartCATSeg server not available at {app_args.catseg_url}. "
            f"Make sure to start it first:\n"
            f"  conda activate partcatseg\n"
            f"  cd part-catseg && python inference_server.py"
        )
        exit(1)
    logger.info("PartCATSeg server is ready!")

    # Build and launch
    demo = build_ui()
    demo.launch(
        server_name=app_args.host,
        server_port=app_args.port,
        share=app_args.share,
    )
