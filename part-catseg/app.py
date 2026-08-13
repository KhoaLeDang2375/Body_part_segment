"""
app.py
------
Gradio 4.x demo for PartCATSeg — Open-Vocabulary Part Segmentation.
Optimised for RunPod deployment (server_name="0.0.0.0", port=7860).

Launch:
    python app.py
    python app.py --device cpu          # CPU-only fallback
    python app.py --port 8080           # Custom port
"""

import argparse
import logging
import os

import gradio as gr
from PIL import Image

from app_config import VOC_OBJ_CLASSES, OBJ_TO_PARTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Global engine (lazy-loaded on first prediction)
# ─────────────────────────────────────────────────────────────────
_engine = None

def _get_engine(device: str):
    global _engine
    if _engine is None:
        from inference_engine import InferenceEngine
        logger.info(f"Initialising InferenceEngine on {device} …")
        _engine = InferenceEngine(device=device)
    return _engine


# ─────────────────────────────────────────────────────────────────
# Gradio callback functions
# ─────────────────────────────────────────────────────────────────

def update_part_choices(obj_class: str):
    """Update available target parts dropdown based on selected object class."""
    parts = OBJ_TO_PARTS.get(obj_class, [])
    if not parts:
        preview = f"*No dedicated part labels for **{obj_class}** in this dataset.*"
    else:
        parts_fmt = ", ".join(f"`{p}`" for p in parts)
        preview = f"**Detectable parts for {obj_class}:** {parts_fmt}"
    return gr.Dropdown(choices=parts, value=[]), preview


def run_segmentation(
    image,
    obj_class: str,
    target_parts: list,
    conf_threshold: float,
    alpha: float,
    device: str,
):
    """Main Gradio predict callback."""
    if image is None:
        gr.Warning("Please upload an image first.")
        return None, [], "⚠️  No image provided."

    if obj_class not in VOC_OBJ_CLASSES:
        gr.Warning("Please select a valid object class.")
        return None, [], "⚠️  Invalid object class."

    pil_image = Image.fromarray(image).convert("RGB")
    try:
        engine = _get_engine(device)
        gr.Info("Running segmentation …")
        overlay, part_table = engine.predict(
            pil_image,
            obj_class_name=obj_class,
            conf_threshold=conf_threshold,
            alpha=alpha,
            selected_parts=target_parts if target_parts else None,
        )
        filter_msg = f" (filtered to: {', '.join(target_parts)})" if target_parts else " (All Parts)"
        status = f"✅  Done — {len(part_table)} part(s) detected above threshold {conf_threshold:.2f}{filter_msg}"
        return overlay, part_table, status

    except Exception as e:
        logger.exception("Inference failed")
        return None, [], f"❌  Error: {str(e)}"


# ─────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────

CSS = """
/* General */
body { font-family: 'Inter', sans-serif; }

/* Header gradient */
#app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 16px;
    color: white;
}
#app-header h1 { font-size: 1.8rem; margin: 0; }
#app-header p  { margin: 4px 0 0; opacity: 0.8; font-size: 0.95rem; }

/* Run button */
#run-btn {
    background: linear-gradient(90deg, #e94560 0%, #f5a623 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    color: white !important;
    transition: opacity 0.2s ease !important;
}
#run-btn:hover { opacity: 0.88 !important; }

/* Part preview text */
#part-preview { border-radius: 8px; padding: 10px 14px; min-height: 50px; }

/* Status bar */
#status-bar textarea {
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    border-radius: 8px;
}

/* Result table */
#part-table table { font-size: 0.88rem; }
"""

HEADER_HTML = """
<div id="app-header">
  <h1>🧩 PartCATSeg — Part Segmentation Demo</h1>
  <p>Open-Vocabulary Part Segmentation · PascalPart116 · CLIP + DINOv2 backbone</p>
</div>
"""

EXAMPLES_DIR = "assets"
_EXAMPLE_LIST = []
if os.path.isdir(EXAMPLES_DIR):
    for fname in os.listdir(EXAMPLES_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            _EXAMPLE_LIST.append([os.path.join(EXAMPLES_DIR, fname), "person", 0.3, 0.6])


def build_ui(device: str) -> gr.Blocks:
    with gr.Blocks(css=CSS, title="PartCATSeg Demo") as demo:

        # ── Header ────────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        # ── Body ──────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # LEFT: Input panel
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📤 Input")

                inp_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=320,
                )

                inp_obj_class = gr.Dropdown(
                    label="Object Class",
                    choices=VOC_OBJ_CLASSES,
                    value="person",
                    filterable=True,
                    info="Select the main object in the image.",
                )

                inp_target_parts = gr.Dropdown(
                    label="Target Parts Filter (Optional)",
                    choices=OBJ_TO_PARTS.get("person", []),
                    value=[],
                    multiselect=True,
                    info="Leave empty to segment ALL parts, or select specific parts (e.g. head, hand).",
                )

                part_preview = gr.Markdown(
                    value=update_part_choices("person")[1],
                    elem_id="part-preview",
                )

                with gr.Accordion("⚙️ Settings", open=False):
                    inp_conf = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.3,
                        info="Parts with max confidence below this value are hidden.",
                    )
                    inp_alpha = gr.Slider(
                        label="Overlay Opacity (α)",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=0.6,
                        info="Transparency of the coloured mask overlay.",
                    )

                btn_run = gr.Button(
                    "🚀  Run Segmentation",
                    variant="primary",
                    elem_id="run-btn",
                )

            # RIGHT: Output panel
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 🎨 Output")

                out_image = gr.Image(
                    label="Segmentation Result",
                    type="pil",
                    height=320,
                    interactive=False,
                )

                out_table = gr.Dataframe(
                    headers=["Part", "Confidence"],
                    datatype=["str", "number"],
                    label="Detected Parts",
                    elem_id="part-table",
                    interactive=False,
                    wrap=True,
                )

                out_status = gr.Textbox(
                    label="Status",
                    value="Waiting for input …",
                    interactive=False,
                    elem_id="status-bar",
                    lines=1,
                )

        # ── Examples ──────────────────────────────────────────────
        if _EXAMPLE_LIST:
            gr.Markdown("### 🖼️ Examples")
            gr.Examples(
                examples=_EXAMPLE_LIST,
                inputs=[inp_image, inp_obj_class, inp_conf, inp_alpha],
                outputs=[out_image, out_table, out_status],
                fn=lambda img, cls, conf, a: run_segmentation(img, cls, [], conf, a, device),
                cache_examples=False,
            )

        # ── Footer ────────────────────────────────────────────────
        gr.Markdown(
            """
---
**Model**: [PartCATSeg (CVPR 2025)](https://arxiv.org/abs/2501.09688) ·
**Checkpoint**: `partcatseg_voc.pth` (PascalPart116) ·
**Backbone**: DINOv2-ViT-S/14 + CLIP ViT-B/16
            """,
            elem_id="footer",
        )

        # ── Event wiring ──────────────────────────────────────────

        # Update part choices and preview when object class changes
        inp_obj_class.change(
            fn=update_part_choices,
            inputs=[inp_obj_class],
            outputs=[inp_target_parts, part_preview],
        )

        # Run segmentation
        btn_run.click(
            fn=lambda img, cls, parts, conf, a: run_segmentation(img, cls, parts, conf, a, device),
            inputs=[inp_image, inp_obj_class, inp_target_parts, inp_conf, inp_alpha],
            outputs=[out_image, out_table, out_status],
            api_name="segment",
        )

    return demo


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PartCATSeg Gradio Demo")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run inference on: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to serve the Gradio app on (default: 7860).",
    )
    parser.add_argument(
        "--share", action="store_true", default=False,
        help="Create a public Gradio share link (useful for quick testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Starting PartCATSeg Gradio Demo | device={args.device} | port={args.port}")

    demo = build_ui(device=args.device)
    demo.launch(
        server_name="0.0.0.0",   # Required on RunPod (listen on all interfaces)
        server_port=args.port,
        share=args.share,
        show_error=True,
        favicon_path=None,
    )
