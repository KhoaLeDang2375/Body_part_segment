"""
SAM 3 Interactive Segmentation Demo
Gradio-based web interface supporting Text, Bounding Box, and Point Click prompts.

Usage:
    python app.py                         # Run full app
    python app.py --check                 # Check imports and GPU only, no model load
    python app.py --checkpoint /path/to/sam3.pt  # Use local checkpoint
"""
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import sys
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CLI flags (parsed early so --check can skip heavy imports)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SAM 3 Gradio Demo")
parser.add_argument("--check", action="store_true", help="Check env only, don't load model or launch UI")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to local SAM 3 checkpoint (.pt)")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio server host")
parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Environment / GPU check
# ---------------------------------------------------------------------------
import torch

def get_gpu_info() -> dict:
    """Return GPU metadata as a dict."""
    if not torch.cuda.is_available():
        return {"available": False, "name": "CPU only", "total_vram": "N/A", "free_vram": "N/A"}
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / (1024 ** 3)
    free = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024 ** 3)
    return {
        "available": True,
        "name": props.name,
        "total_vram": f"{total:.1f} GB",
        "free_vram": f"{free:.1f} GB",
        "compute_capability": f"{props.major}.{props.minor}",
    }

if args.check:
    gpu = get_gpu_info()
    print("=" * 50)
    print("SAM 3 Demo — Environment Check")
    print("=" * 50)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if gpu["available"]:
        print(f"  GPU: {gpu['name']}")
        print(f"  VRAM Total: {gpu['total_vram']}")
        print(f"  Compute Capability: {gpu['compute_capability']}")
    print("\n✅ Environment check passed. Run without --check to launch the app.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Heavy imports (only after --check bypass)
# ---------------------------------------------------------------------------
import gradio as gr
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------------------------------------------------------------------
# Model Manager (Singleton + Lazy Loading)
# ---------------------------------------------------------------------------
COLORS_RGBA = [
    (255, 56, 56, 120),
    (56, 182, 255, 120),
    (56, 255, 100, 120),
    (255, 182, 56, 120),
    (200, 56, 255, 120),
    (56, 255, 232, 120),
    (255, 128, 0, 120),
    (128, 0, 255, 120),
]

class ModelManager:
    """Singleton that holds the loaded SAM 3 model and processors."""

    _instance: Optional["ModelManager"] = None
    _loaded: bool = False
    processor: Optional[Sam3Processor] = None
    inst_predictor = None           # SAM3InteractiveImagePredictor
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, checkpoint_path: Optional[str] = None):
        """Load SAM 3 model with interactive mode enabled (supports Text + Box + Point)."""
        if self._loaded:
            return
        print("\n[ModelManager] Loading SAM 3 model (enable_inst_interactivity=True)...")
        print("[ModelManager] This may take 30-60 seconds on first run...")

        model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=(checkpoint_path is None),
            enable_inst_interactivity=True,  # enables Point Click mode
        )

        self.processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=0.5,
        )
        # The interactive predictor is attached to the model
        self.inst_predictor = model.inst_interactive_predictor

        self._loaded = True
        gpu = get_gpu_info()
        print(f"[ModelManager] ✅ Model loaded on {self.device.upper()} ({gpu.get('name','')}, {gpu.get('free_vram','')} free)")

_manager = ModelManager()


def ensure_model_loaded():
    """Load model on first inference call."""
    if not _manager._loaded:
        _manager.load(checkpoint_path=args.checkpoint if args.checkpoint else None)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def draw_results(
    image: Image.Image,
    masks: Optional[torch.Tensor],
    boxes: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
) -> Image.Image:
    """
    Overlay masks (semi-transparent fill) and bounding boxes on the image.

    Args:
        image: PIL RGB image (original size).
        masks: (N, 1, H, W) bool tensor or (N, H, W).
        boxes: (N, 4) tensor in XYXY pixel coords.
        scores: (N,) float tensor.

    Returns:
        Annotated PIL image.
    """
    result = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    n = 0 if masks is None else (masks.shape[0] if masks is not None else 0)

    for i in range(n):
        color = COLORS_RGBA[i % len(COLORS_RGBA)]

        # Draw mask
        if masks is not None:
            mask_np = masks[i].squeeze().cpu().numpy().astype(bool)
            mask_img = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
            mask_img[mask_np] = color
            mask_pil = Image.fromarray(mask_img, mode="RGBA")
            overlay = Image.alpha_composite(overlay, mask_pil)

        # Draw bounding box
        if boxes is not None:
            x0, y0, x1, y1 = boxes[i].cpu().tolist()
            box_color = color[:3] + (255,)
            box_draw = ImageDraw.Draw(overlay)
            box_draw.rectangle([x0, y0, x1, y1], outline=box_color, width=3)

            # Score label
            if scores is not None:
                score_val = scores[i].item()
                label = f"#{i+1} {score_val:.2f}"
                # Background rectangle for text readability
                box_draw.rectangle([x0, y0 - 20, x0 + len(label) * 8, y0], fill=box_color)
                box_draw.text((x0 + 2, y0 - 18), label, fill=(255, 255, 255, 255))

    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def build_summary(n_found: int, scores: Optional[torch.Tensor]) -> str:
    """Build a human-readable text summary of results."""
    if n_found == 0:
        return "⚠️ Không tìm thấy đối tượng nào. Thử giảm Confidence Threshold hoặc đổi prompt."
    lines = [f"✅ Tìm thấy **{n_found}** đối tượng:\n"]
    if scores is not None:
        for i, s in enumerate(scores.cpu().tolist()):
            lines.append(f"  • Object #{i+1}: score = {s:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------
def run_text_inference(image_np: np.ndarray, text: str, threshold: float):
    """Run Text Prompt segmentation."""
    ensure_model_loaded()
    if image_np is None:
        return None, "❌ Vui lòng upload ảnh trước."
    if not text or not text.strip():
        return None, "❌ Vui lòng nhập text prompt."

    _manager.processor.confidence_threshold = threshold
    image_pil = Image.fromarray(image_np.astype(np.uint8))
    
    with torch.inference_mode():
        if _manager.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = _manager.processor.set_image(image_pil)
                state = _manager.processor.set_text_prompt(prompt=text.strip(), state=state)
        else:
            state = _manager.processor.set_image(image_pil)
            state = _manager.processor.set_text_prompt(prompt=text.strip(), state=state)

    masks = state.get("masks")
    boxes = state.get("boxes")
    scores = state.get("scores")

    n_found = 0 if masks is None else masks.shape[0]
    annotated = draw_results(image_pil, masks, boxes, scores)
    summary = build_summary(n_found, scores)
    return annotated, summary


def run_box_inference(image_np: np.ndarray, box_data: dict, threshold: float):
    """
    Run Bounding Box prompt segmentation.
    box_data comes from gr.ImageEditor as dict with 'background' and 'layers'.
    We extract the drawn rectangle from the first layer.
    """
    ensure_model_loaded()
    if image_np is None:
        return None, "❌ Vui lòng upload ảnh trước."

    # Extract the base image from ImageEditor data
    if isinstance(box_data, dict):
        bg = box_data.get("background")
        if bg is None:
            return None, "❌ Vui lòng upload ảnh và vẽ bounding box."
        base_image_np = np.array(bg)
        # Try to find drawn region from layers (composite layer minus background)
        layers = box_data.get("layers", [])
        drawn_box = None
        if layers:
            layer_arr = np.array(layers[0])
            # Find non-zero alpha pixels (the drawn stroke)
            if layer_arr.shape[-1] == 4:
                alpha = layer_arr[:, :, 3]
                ys, xs = np.where(alpha > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    img_h, img_w = base_image_np.shape[:2]
                    # Convert to normalized cx,cy,w,h
                    cx = ((x0 + x1) / 2) / img_w
                    cy = ((y0 + y1) / 2) / img_h
                    w = (x1 - x0) / img_w
                    h = (y1 - y0) / img_h
                    drawn_box = [cx, cy, w, h]

        if drawn_box is None:
            return None, "❌ Không phát hiện được vùng vẽ. Hãy vẽ một hình chữ nhật trên ảnh."

        image_pil = Image.fromarray(base_image_np[:, :, :3].astype(np.uint8))
    else:
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        drawn_box = None

    if drawn_box is None:
        return None, "❌ Không phát hiện được bounding box."

    _manager.processor.confidence_threshold = threshold

    with torch.inference_mode():
        if _manager.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = _manager.processor.set_image(image_pil)
                state = _manager.processor.add_geometric_prompt(
                    box=drawn_box, label=True, state=state
                )
        else:
            state = _manager.processor.set_image(image_pil)
            state = _manager.processor.add_geometric_prompt(
                box=drawn_box, label=True, state=state
            )

    masks = state.get("masks")
    boxes = state.get("boxes")
    scores = state.get("scores")

    n_found = 0 if masks is None else masks.shape[0]
    annotated = draw_results(image_pil, masks, boxes, scores)
    summary = build_summary(n_found, scores)
    return annotated, summary


def run_point_inference(
    image_np: np.ndarray,
    point_coords_str: str,
    point_labels_str: str,
    threshold: float,
):
    """
    Run Point Click segmentation via SAM3InteractiveImagePredictor.

    point_coords_str: comma-separated pairs, e.g. "100,200;300,400"
    point_labels_str: comma-separated labels (1=foreground, 0=background), e.g. "1;0"
    """
    ensure_model_loaded()
    if image_np is None:
        return None, "❌ Vui lòng upload ảnh trước."

    if _manager.inst_predictor is None:
        return None, "❌ Interactive predictor chưa được khởi tạo. Restart app."

    # Parse point coordinates
    try:
        coords = []
        for pair in point_coords_str.strip().split(";"):
            pair = pair.strip()
            if not pair:
                continue
            x, y = pair.split(",")
            coords.append([float(x.strip()), float(y.strip())])
        if not coords:
            return None, "❌ Không parse được tọa độ điểm. Định dạng: 'x1,y1;x2,y2'"
        point_coords = np.array(coords)

        labels = [int(l.strip()) for l in point_labels_str.strip().split(";") if l.strip()]
        if len(labels) != len(coords):
            return None, f"❌ Số lượng labels ({len(labels)}) không khớp với số điểm ({len(coords)})."
        point_labels = np.array(labels)
    except Exception as e:
        return None, f"❌ Lỗi parse input: {e}\nĐịnh dạng đúng: coords='x1,y1;x2,y2', labels='1;0'"

    image_pil = Image.fromarray(image_np.astype(np.uint8))

    with torch.inference_mode():
        if _manager.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _manager.inst_predictor.set_image(image_pil)
                masks_np, iou_scores, _ = _manager.inst_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
        else:
            _manager.inst_predictor.set_image(image_pil)
            masks_np, iou_scores, _ = _manager.inst_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

    # Select best mask by IOU score
    best_idx = int(np.argmax(iou_scores))
    best_mask = masks_np[best_idx]  # (H, W) bool
    best_score = iou_scores[best_idx]

    # Convert to tensor format for draw_results
    mask_tensor = torch.from_numpy(best_mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    score_tensor = torch.tensor([best_score])

    # Build bbox from mask
    ys, xs = np.where(best_mask)
    if len(xs) > 0:
        box_tensor = torch.tensor([[
            float(xs.min()), float(ys.min()),
            float(xs.max()), float(ys.max()),
        ]])
    else:
        box_tensor = None

    annotated = draw_results(image_pil, mask_tensor, box_tensor, score_tensor)
    n_found = 1 if best_mask.any() else 0
    summary = build_summary(n_found, score_tensor)
    return annotated, summary


# ---------------------------------------------------------------------------
# Point visualization helper (draw user-clicked points on image)
# ---------------------------------------------------------------------------
def visualize_points(image_np: np.ndarray, point_coords_str: str, point_labels_str: str):
    """Return image with drawn point markers (green=fg, red=bg)."""
    if image_np is None:
        return None
    image_pil = Image.fromarray(image_np.astype(np.uint8)).copy()
    draw = ImageDraw.Draw(image_pil)

    try:
        coords = []
        for pair in point_coords_str.strip().split(";"):
            pair = pair.strip()
            if pair:
                x, y = pair.split(",")
                coords.append((float(x), float(y)))
        labels = [int(l.strip()) for l in point_labels_str.strip().split(";") if l.strip()]

        for i, (x, y) in enumerate(coords):
            label = labels[i] if i < len(labels) else 1
            color = (0, 220, 0) if label == 1 else (220, 0, 0)
            r = 8
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=2)
    except Exception:
        pass

    return image_pil


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    gpu = get_gpu_info()
    gpu_str = f"{gpu['name']} | VRAM: {gpu['total_vram']}" if gpu["available"] else "CPU mode"

    with gr.Blocks(
        title="SAM 3 Interactive Segmentation Demo",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.indigo,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        #header { text-align: center; padding: 16px 0 8px 0; }
        #gpu-info { text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 8px; }
        .gr-button-primary { background: linear-gradient(90deg, #3b82f6, #6366f1); }
        """,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            "# 🔬 SAM 3 — Interactive Segmentation Demo",
            elem_id="header",
        )
        gr.Markdown(
            f"**{gpu_str}** | Model: SAM 3.0 (`facebook/sam3`) | "
            "Hỗ trợ: Text Prompt · Bounding Box · Point Click",
            elem_id="gpu-info",
        )

        # ── Prompt Mode Selector ─────────────────────────────────────────────
        with gr.Row():
            prompt_mode = gr.Radio(
                choices=["Text Prompt", "Bounding Box", "Point Click"],
                value="Text Prompt",
                label="🎯 Chọn Chế Độ Prompt",
                interactive=True,
            )
            confidence_slider = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                label="Confidence Threshold",
                info="Ngưỡng tin cậy — thấp hơn = tìm nhiều hơn nhưng có thể sai",
            )

        gr.Markdown("---")

        # ── Main 2-column layout ─────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT: Input Panel ────────────────────────────────────────────
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📥 Input")

                # --- Text Mode ---
                with gr.Group(visible=True) as group_text:
                    input_image_text = gr.Image(
                        label="Upload Ảnh",
                        type="numpy",
                        sources=["upload", "clipboard"],
                        height=380,
                        elem_id="input-image-text",
                    )
                    text_prompt = gr.Textbox(
                        label="Text Prompt",
                        placeholder='Nhập mô tả đối tượng... ví dụ: "dog", "person in red shirt", "car"',
                        lines=2,
                    )

                # --- Box Mode ---
                with gr.Group(visible=False) as group_box:
                    gr.Markdown(
                        "**Hướng dẫn:** Chọn công cụ ✏️ (Sketch) rồi vẽ hình chữ nhật bao quanh đối tượng cần segment."
                    )
                    input_image_box = gr.ImageEditor(
                        label="Upload & Vẽ Bounding Box",
                        type="numpy",
                        brush=gr.Brush(colors=["#FF3838"], default_size=3),
                        height=380,
                        elem_id="input-image-box",
                    )

                # --- Point Click Mode ---
                with gr.Group(visible=False) as group_point:
                    gr.Markdown(
                        "**Hướng dẫn:** Nhập tọa độ điểm và nhãn thủ công.\n"
                        "- **Coords** (x,y): `100,200;300,400` (mỗi điểm cách nhau bởi `;`)\n"
                        "- **Labels**: `1;0` — `1` = foreground (xanh), `0` = background (đỏ)"
                    )
                    input_image_point = gr.Image(
                        label="Upload Ảnh",
                        type="numpy",
                        sources=["upload", "clipboard"],
                        height=280,
                        elem_id="input-image-point",
                    )
                    point_coords_input = gr.Textbox(
                        label="Point Coordinates (x,y;x,y;...)",
                        placeholder="Ví dụ: 150,200;300,400",
                        lines=1,
                    )
                    point_labels_input = gr.Textbox(
                        label="Point Labels (1=fg, 0=bg, cách nhau bởi ;)",
                        placeholder="Ví dụ: 1;0",
                        value="1",
                        lines=1,
                    )
                    preview_points_btn = gr.Button("👁️ Preview Điểm Trên Ảnh", variant="secondary", size="sm")
                    point_preview_img = gr.Image(label="Preview Points", height=280, interactive=False)

                # --- Action Buttons ---
                with gr.Row():
                    run_btn = gr.Button("▶ Run Segmentation", variant="primary", scale=3)
                    reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1)

            # ── RIGHT: Output Panel ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📤 Output")
                output_image = gr.Image(
                    label="Kết Quả Segmentation",
                    type="pil",
                    height=420,
                    interactive=False,
                    elem_id="output-image",
                )
                output_summary = gr.Markdown("*Kết quả sẽ hiển thị tại đây sau khi chạy...*")

        # ── Examples section ─────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 💡 Gợi ý Text Prompts")
        gr.Markdown(
            "Thử các prompt sau với ảnh tương ứng:\n"
            "- `dog` · `cat` · `person` · `car` · `bicycle`\n"
            "- `person wearing red shirt` · `player in white uniform`\n"
            "- `all chairs in the room` · `trees in the background`"
        )

        # ── Interactions ─────────────────────────────────────────────────────

        # Switch visible groups when prompt mode changes
        def switch_mode(mode):
            return (
                gr.update(visible=(mode == "Text Prompt")),
                gr.update(visible=(mode == "Bounding Box")),
                gr.update(visible=(mode == "Point Click")),
            )

        prompt_mode.change(
            fn=switch_mode,
            inputs=[prompt_mode],
            outputs=[group_text, group_box, group_point],
        )

        # Run button — dispatch to correct inference function
        def run_inference(mode, img_text, text, img_box_data, img_point, coords_str, labels_str, threshold):
            if mode == "Text Prompt":
                return run_text_inference(img_text, text, threshold)
            elif mode == "Bounding Box":
                return run_box_inference(None, img_box_data, threshold)
            elif mode == "Point Click":
                return run_point_inference(img_point, coords_str, labels_str, threshold)
            return None, "❌ Chế độ không hợp lệ."

        run_btn.click(
            fn=run_inference,
            inputs=[
                prompt_mode,
                input_image_text, text_prompt,
                input_image_box,
                input_image_point, point_coords_input, point_labels_input,
                confidence_slider,
            ],
            outputs=[output_image, output_summary],
        )

        # Preview points
        preview_points_btn.click(
            fn=visualize_points,
            inputs=[input_image_point, point_coords_input, point_labels_input],
            outputs=[point_preview_img],
        )

        # Reset button
        def reset_all():
            return None, None, None, None, "", "", "1", 0.5, "*Kết quả sẽ hiển thị tại đây sau khi chạy...*"

        reset_btn.click(
            fn=reset_all,
            inputs=[],
            outputs=[
                input_image_text, output_image, input_image_box, input_image_point,
                text_prompt, point_coords_input, point_labels_input,
                confidence_slider, output_summary,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  SAM 3 Interactive Segmentation Demo")
    print("=" * 55)
    gpu = get_gpu_info()
    print(f"  GPU    : {gpu['name']}")
    print(f"  VRAM   : {gpu['total_vram']} total, {gpu['free_vram']} free")
    print(f"  Device : {_manager.device.upper()}")
    if args.checkpoint:
        print(f"  Ckpt   : {args.checkpoint} (local)")
    else:
        print("  Ckpt   : Auto-download from facebook/sam3 on first inference")
    print(f"  Host   : {args.host}:{args.port}")
    print("=" * 55)
    print("  Model sẽ được load khi bạn submit lần đầu tiên.")
    print("=" * 55)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=True,        # Tạo public URL qua Gradio tunnel (gradio.live)
        show_error=True,
        favicon_path=None,
    )
