# 🚀 Hướng Dẫn Triển Khai Body Part Segmentation Pipeline Trên RunPod

Tài liệu chi tiết để triển khai pipeline **PartCATSeg + SAM3** trên **RunPod**, cho phép segment **14 bộ phận cơ thể người** với chất lượng cao.

---

## 📌 0. Quick Reference

| Thông số | Giá trị |
|:---|:---|
| **GPU tối thiểu** | 20 GB VRAM (RTX A5000, RTX 3090, L40) |
| **GPU khuyến nghị** | 24 GB+ (RTX A6000, L40S, A100) |
| **Template RunPod** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| **Container Disk** | ≥ 30 GB |
| **Volume Disk** | ≥ 60 GB (mount tại `/workspace`) |
| **HTTP Ports** | `7860` (Pipeline UI), `8001` (CATSeg API — internal) |
| **Conda Envs** | `partcatseg` (Python 3.11), `sam3env` (Python 3.12) |
| **VRAM Usage** | CATSeg ~4 GB + SAM3 ~6 GB = **~10 GB** (chạy song song) |

### Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────┐
│  RunPod Pod — GPU 20+ GB VRAM                               │
│                                                             │
│  ┌────────────────────────────────────┐                     │
│  │  conda env: partcatseg             │                     │
│  │  PartCATSeg FastAPI Server :8001   │                     │
│  │  → Nhận ảnh, trả body part masks   │                     │
│  └──────────────┬─────────────────────┘                     │
│                 │ HTTP localhost                             │
│  ┌──────────────▼─────────────────────┐                     │
│  │  conda env: sam3env                │                     │
│  │  Pipeline + SAM3 + Gradio UI :7860 │ ← Public Access     │
│  │  → Nhận CATSeg masks              │                     │
│  │  → SAM3 refine từng body part     │                     │
│  │  → Hiển thị + Export ZIP           │                     │
│  └────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📥 1. Tạo Pod Trên RunPod

### 1.1. Chọn GPU
- **Tối thiểu:** GPU **20 GB VRAM** — RTX A5000, RTX 3090, L40.
- **Khuyến nghị:** GPU **24 GB+ VRAM** — RTX A6000, L40S, A100.
- **KHÔNG NÊN dùng:** T4, V100 (compute capability thấp, VRAM không đủ).

### 1.2. Cấu Hình Pod
- **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container Disk:** ≥ 30 GB
- **Volume Disk:** ≥ **60 GB** (lưu 2 model + 2 conda env + cache)
- **HTTP Port:** Thêm port **`7860`** vào Expose HTTP Ports

> ⚠️ **QUAN TRỌNG:** Tất cả file PHẢI đặt trong `/workspace`. Các thư mục khác sẽ bị **XÓA SẠCH** khi Stop/Restart Pod.

---

## 🔧 2. Setup Tự Động (Khuyến Nghị)

### Bước 1: Kết nối Pod
Sau khi Pod chạy, vào **Web Terminal** hoặc **SSH**.

### Bước 2: Clone repo và chạy setup
```bash
cd /workspace

# Clone repo
git clone https://github.com/KhoaLeDang2375/Body_part_segment.git
cd Body_part_segment

# Chạy setup (truyền HF Token cho SAM3)
HF_TOKEN="hf_xxxxxxxxxxxxxxxx" bash startup.sh
```

> 💡 **HF Token:** SAM3 checkpoint nằm trên Hugging Face (Gated repo). Cần:
> 1. Vào [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) → **Request Access**
> 2. Tạo token tại [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Script `startup.sh` sẽ tự động:
1. ✅ Cài Miniconda (nếu chưa có)
2. ✅ Tạo 2 conda env: `partcatseg` (Python 3.11) + `sam3env` (Python 3.12)
3. ✅ Cài PyTorch, Detectron2, SAM3 và tất cả dependencies
4. ✅ Tải PartCATSeg weights (~885 MB) + SAM3 checkpoint (~4 GB)
5. ✅ Cấu hình persistent storage

**⏱ Thời gian setup:** ~10-15 phút (phụ thuộc tốc độ mạng).

### Bước 3: Chạy pipeline
```bash
bash start_pipeline.sh
```

### Bước 4: Truy cập giao diện
- **Từ RunPod Dashboard:** Nhấn **Connect** → **HTTP Service [7860]**
- **Từ URL:** `https://<pod-id>-7860.proxy.runpod.net`

---

## 🔧 3. Setup Thủ Công (Từng Bước)

<details>
<summary>Mở nếu muốn cài từng bước thay vì dùng startup.sh</summary>

### 3.1. Cài Conda
```bash
cd /workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /workspace/miniconda3
export PATH="/workspace/miniconda3/bin:$PATH"
echo 'export PATH="/workspace/miniconda3/bin:$PATH"' >> ~/.bashrc
conda init bash
source ~/.bashrc

# Accept ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 3.2. Clone Repo
```bash
cd /workspace
git clone https://github.com/KhoaLeDang2375/Body_part_segment.git
cd Body_part_segment
```

### 3.3. Setup PartCATSeg Environment
```bash
# Tạo env
conda create -n partcatseg -c conda-forge python=3.11 -y
conda activate partcatseg

# PyTorch + CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# Dependencies
cd part-catseg
pip install -r requirements.txt
pip install fastapi uvicorn[standard]

# Detectron2
git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2
pip install --no-build-isolation --no-deps -e /tmp/detectron2

# Download weights
pip install gdown
mkdir -p weights
gdown "https://drive.google.com/uc?id=1JUJjJQLMKE96H5SLNs4EMm4jiU6fPgRb" -O weights/partcatseg_voc.pth

conda deactivate
```

### 3.4. Setup SAM3 Environment
```bash
# Tạo env
conda create -n sam3env -c conda-forge python=3.12 -y
conda activate sam3env

# PyTorch + CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# SAM3
cd /workspace/Body_part_segment/sam3
pip install -e .

# Pipeline dependencies
pip install -r /workspace/Body_part_segment/part_sam_pipeline/requirements.txt

# (Optional) FlashAttention
pip install einops ninja
pip install flash-attn --no-build-isolation

# Download SAM3 checkpoint
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"
mkdir -p /workspace/checkpoints
hf download facebook/sam3 sam3.pt --local-dir /workspace/checkpoints --token $HF_TOKEN

conda deactivate
```

</details>

---

## 🚀 4. Chạy Pipeline

### 4.1. Chạy Tự Động (1 lệnh)
```bash
cd /workspace/Body_part_segment
bash start_pipeline.sh
```

Script này sẽ:
1. Start PartCATSeg server trên port 8001 (nền)
2. Đợi server sẵn sàng (~30-60 giây lần đầu)
3. Start Pipeline Gradio UI trên port 7860

### 4.2. Chạy Thủ Công (2 terminal)

**Terminal 1 — PartCATSeg Server:**
```bash
conda activate partcatseg
export PYTHONPATH="/tmp/detectron2:$PYTHONPATH"
cd /workspace/Body_part_segment/part-catseg
python inference_server.py --port 8001 --device cuda
```

**Terminal 2 — Pipeline UI:**
```bash
conda activate sam3env
cd /workspace/Body_part_segment
python -m part_sam_pipeline.app --checkpoint /workspace/checkpoints/sam3.pt
```

### 4.3. Chạy Nền (Background)
```bash
cd /workspace/Body_part_segment

# Start CATSeg server
conda activate partcatseg
export PYTHONPATH="/tmp/detectron2:$PYTHONPATH"
nohup python part-catseg/inference_server.py > /workspace/catseg.log 2>&1 &

# Start Pipeline
conda activate sam3env
nohup python -m part_sam_pipeline.app \
    --checkpoint /workspace/checkpoints/sam3.pt \
    > /workspace/pipeline.log 2>&1 &

# Xem logs
tail -f /workspace/catseg.log
tail -f /workspace/pipeline.log
```

---

## 🎨 5. Sử Dụng Giao Diện

### Workflow Cơ Bản
1. **Upload ảnh** nhân vật (khuyến nghị: 512px - 1080px)
2. **Chọn body parts** muốn segment (hoặc để trống = tất cả 14 parts)
3. **Điều chỉnh Confidence** (giảm nếu muốn detect nhiều hơn)
4. Nhấn **🚀 Run Pipeline**

### 14 Body Parts Hỗ Trợ
| Nhóm | Parts |
|---|---|
| **Đầu & Mặt** | head, eye, nose, ear, eyebrow, mouth, hair |
| **Thân** | torso, neck |
| **Tay** | upper arm, lower arm, hand |
| **Chân** | leg, foot |

### Tab Kết Quả
- **Refined Masks:** Masks đã refined bởi SAM3 (chất lượng cao)
- **Comparison:** So sánh side-by-side: Original | CATSeg (thô) | SAM3 (refined)
- **Download:** ZIP chứa tất cả masks dạng PNG (mỗi part 1 file)

---

## ♻️ 6. Restart Pod

Khi restart Pod, cần restore lại môi trường:

```bash
cd /workspace

# Restore Conda PATH
export PATH="/workspace/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

# Reinstall Detectron2 (bị xóa khỏi /tmp khi restart)
conda activate partcatseg
git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2
pip install --no-build-isolation --no-deps -e /tmp/detectron2
conda deactivate

# Start pipeline
cd /workspace/Body_part_segment
bash start_pipeline.sh
```

> 💡 **Tip:** Conda envs, checkpoints, code trong `/workspace` sẽ **TỒN TẠI** qua restart. Chỉ có `/tmp/detectron2` cần clone lại.

---

## 🛠️ 7. Xử Lý Lỗi (Troubleshooting)

| Lỗi | Nguyên nhân | Cách khắc phục |
|:---|:---|:---|
| `bash: conda: command not found` | Conda chưa có trong PATH | `export PATH="/workspace/miniconda3/bin:$PATH"` |
| `CondaToSNonInteractiveError` | Chưa chấp nhận Anaconda ToS | Chạy 2 lệnh `conda tos accept` (xem Bước 3.1) |
| `Connection refused (port 8001)` | PartCATSeg server chưa start | Kiểm tra log: `tail -f /workspace/catseg_server.log` |
| `ModuleNotFoundError: detectron2` | Detectron2 bị xóa sau restart | Clone lại: `git clone ... /tmp/detectron2` rồi `pip install -e` |
| `CUDA Out of Memory` | Không đủ VRAM cho 2 model | Dùng GPU 24GB+, hoặc giảm kích thước ảnh |
| `401 / Gated Repo` (SAM3) | HF Token chưa set hoặc chưa approved | Kiểm tra `echo $HF_TOKEN`, xin access trên HF |
| `PartCATSeg returns 0 parts` | Ảnh không chứa người hoặc conf quá cao | Giảm Confidence Threshold, chọn obj_class đúng |
| Port 7860 không truy cập | Chưa expose port trên RunPod | Stop Pod → Edit → Thêm HTTP port 7860 → Restart |
| `timm version conflict` | Cài nhầm env | Kiểm tra `conda info --envs`, đảm bảo đúng env |

---

## 📂 8. Cấu Trúc Dự Án

```
Body_part_segment/
├── startup.sh                      # One-command setup script
├── start_pipeline.sh               # Launch both services
├── RUNPOD_GUIDE.md                 # Tài liệu này
├── README.md                       # Tổng quan dự án
│
├── part-catseg/                    # PartCATSeg model (conda: partcatseg)
│   ├── inference_engine.py         # Model wrapper + predict_raw_masks()
│   ├── inference_server.py         # FastAPI server (port 8001)
│   ├── app.py                      # Gradio demo gốc (standalone)
│   ├── app_config.py               # Object/part class config
│   ├── environment.yml             # Conda env spec
│   ├── requirements.txt            # Python dependencies
│   └── start_demo.sh               # Standalone startup script
│
├── sam3/                           # SAM3 model (conda: sam3env)
│   ├── sam3/                       # SAM3 Python package
│   ├── app.py                      # Gradio demo gốc (standalone)
│   ├── pyproject.toml              # Package config
│   └── startup.sh                  # Standalone startup script
│
└── part_sam_pipeline/              # Pipeline module (conda: sam3env)
    ├── __init__.py
    ├── catseg_client.py            # HTTP client → PartCATSeg API
    ├── mask_converter.py           # Mask → SAM3 prompt conversion
    ├── pipeline.py                 # Main orchestrator
    ├── app.py                      # Gradio UI cho pipeline
    └── requirements.txt            # Pipeline dependencies
```

---

## 🔬 9. API Endpoints (Cho Lập Trình Viên)

### PartCATSeg Server (port 8001)

```bash
# Health check
curl http://localhost:8001/health

# Get available classes
curl http://localhost:8001/classes

# Segment body parts
curl -X POST http://localhost:8001/segment_parts \
  -F "image_file=@photo.jpg" \
  -F "obj_class=person" \
  -F "conf_threshold=0.3"
```

### Pipeline Python API

```python
from part_sam_pipeline.pipeline import PartSamPipeline
from PIL import Image

pipeline = PartSamPipeline(
    sam3_checkpoint="/workspace/checkpoints/sam3.pt",
    catseg_url="http://localhost:8001",
)

image = Image.open("photo.jpg")
result = pipeline.segment_all_parts(image)

# result["refined_masks"] → Dict[str, np.ndarray (H,W) bool]
# result["coarse_masks"]  → Dict[str, np.ndarray (H,W) bool]
# result["scores"]        → Dict[str, {"catseg": float, "sam3": float}]
```
