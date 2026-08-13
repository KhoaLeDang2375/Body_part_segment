# Hướng Dẫn Triển Khai và Chạy SAM 3 Trên RunPod

Tài liệu này hướng dẫn chi tiết từng bước để triển khai, cài đặt và chạy mô hình **SAM 3 (Segment Anything Model 3)** trên nền tảng Cloud GPU **RunPod**, bao gồm giao diện Gradio để test mô hình.

---

## 0. Quick Reference

| Thông số | Giá trị khuyến nghị |
| :--- | :--- |
| **GPU tối thiểu** | 16 GB VRAM (RTX A4000, RTX 4090, RTX A5000, RTX 3090) |
| **GPU thay thế khi hết A4000** | RTX 4090, RTX 3090, RTX 3090 Ti, L4, RTX A5000 (tất cả đều dùng Ampere/Ada trở lên) |
| **GPU KHÔNG NÊN dùng** | T4, V100, RTX 2080 Ti (Compute Capability < 8.0) |
| **Template RunPod** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` hoặc mới hơn |
| **Container Disk** | ≥ 30 GB |
| **Volume Disk** | ≥ 50 GB (mount tại `/workspace`) |
| **Python** | 3.12 |
| **PyTorch** | ≥ 2.7 với CUDA 12.4+ |
| **Gradio Port** | 7860 (HTTP) |
| **HF Repo checkpoint** | `facebook/sam3` (Gated — cần xin quyền) |

---

## 1. Khởi Tạo Pod Trên RunPod

### 1.1. Chọn GPU
* **Tối thiểu:** GPU **16 GB VRAM** — RTX A4000, RTX 4090, RTX A5000, L4, RTX 3090.
* **Khuyên dùng (Image + Point Interactive):** GPU **24 GB+ VRAM** — RTX A6000, L40S, A100 80GB, H100.
  > ⚠️ Chế độ Point Click (`enable_inst_interactivity=True`) cần thêm ~2 GB VRAM so với chỉ Text Prompt.
* **Nếu không thuê được RTX 4000 Ada:** Ưu tiên **RTX 4090** hoặc **RTX 3090** — giá rẻ, 24GB VRAM, cùng kiến trúc Ampere/Ada, chạy code 100% không xung đột.

### 1.2. Chọn Template & Dung Lượng Đĩa
* **Template:** Chọn `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (hoặc mới hơn).
* **Container Disk:** Tối thiểu **30 GB**.
* **Volume Disk (Persistent Storage):** Tối thiểu **50 GB**, gắn tại `/workspace`.

> ⚠️ **Lưu ý cực kỳ quan trọng:** Toàn bộ code, checkpoint, conda environment PHẢI đặt trong `/workspace`. Các thư mục khác như `/root` hay `/` sẽ bị **XÓA SẠCH** khi Stop/Restart Pod.

### 1.3. Cấu Hình Network Port cho Gradio
Trước khi tạo Pod, thêm port HTTP:
* Vào mục **Expose HTTP Ports** → Thêm port **`7860`**.
* RunPod sẽ tạo ra URL public dạng `https://<pod-id>-7860.proxy.runpod.net`.

---

## 2. Kết Nối và Thiết Lập Môi Trường

Sau khi Pod trạng thái `Running`, kết nối qua **Web Terminal** hoặc **SSH**.

**Lệnh đầu tiên LUÔN phải là:**
```bash
cd /workspace
```

### 2.1. Thiết Lập Lưu Trữ Bền Vững (Persistent Storage)

```bash
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/.cache/huggingface' >> ~/.bashrc
source ~/.bashrc
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/checkpoints
```

### 2.2. Cài Đặt Conda (Nếu chưa có)

> ℹ️ Nhiều template RunPod **không có sẵn Conda**. Thực hiện các bước sau để cài Miniconda vào `/workspace` (bền vững qua các lần restart):

```bash
# 1. Tải Miniconda installer
cd /workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 2. Cài vào /workspace/miniconda3
bash miniconda.sh -b -p /workspace/miniconda3

# 3. Nạp đường dẫn conda và lưu vào bashrc
export PATH="/workspace/miniconda3/bin:$PATH"
echo 'export PATH="/workspace/miniconda3/bin:$PATH"' >> ~/.bashrc

# 4. Khởi tạo conda và reload shell
conda init bash
source ~/.bashrc

# 5. Kiểm tra
conda --version
```

### 2.3. Chấp Nhận Anaconda Terms of Service

> ⚠️ Anaconda phiên bản mới **bắt buộc** phải accept ToS trước khi tạo environment. Nếu bỏ qua bước này sẽ gặp lỗi `CondaToSNonInteractiveError`.

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 2.4. Clone Repository

> ⚠️ Clone từ **Repo của bạn** (nhánh `dev1`), KHÔNG phải repo gốc của Meta — vì repo của bạn mới chứa `app.py`, `startup.sh`.

```bash
cd /workspace

# Nếu thư mục sam3 đã tồn tại, xóa trước để tránh lỗi
rm -rf sam3

# Clone từ repo của bạn, nhánh dev1
git clone -b dev1 https://github.com/KhoaLeDang2375/sam3.git sam3
cd sam3
```

### 2.5. Tạo Môi Trường Conda

```bash
# Dùng conda-forge để tránh vấn đề ToS với channel mặc định
conda create -n sam3 -c conda-forge python=3.12 -y

# Kích hoạt môi trường
conda activate sam3

# Tự động kích hoạt khi mở terminal mới
echo "conda activate sam3" >> ~/.bashrc
```

### 2.6. Cài Đặt PyTorch & SAM 3

```bash
# Cài PyTorch với CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Cài SAM 3 package
pip install -e .

# Cài Gradio và HF CLI
pip install "gradio>=4.0" huggingface_hub
```

### 2.7. (Tùy Chọn) Cài FlashAttention Tăng Tốc Suy Luận

```bash
# GPU Ampere / Ada (RTX 3090, 4090, A100, A6000, L40S)
pip install einops ninja
pip install flash-attn --no-build-isolation

# GPU Hopper (H100) — KHÔNG dùng cho GPU khác!
# pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
```

---

## 3. Xác Thực Hugging Face (Bắt Buộc)

SAM 3 checkpoint là mô hình **Gated** — cần xin quyền truy cập trước.

### 3.1. Xin Quyền Truy Cập
1. Đăng nhập tại [huggingface.co](https://huggingface.co).
2. Vào [facebook/sam3](https://huggingface.co/facebook/sam3) → nhấn **Request Access**.

### 3.2. Tạo Access Token
1. Vào [Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Nhấn **New Token** → quyền **Read** → **Generate**.
3. Sao chép token (dạng `hf_xxxxxxxx...`).

### 3.3. Đưa Token Vào RunPod (Chọn 1 trong 3 cách)

**Cách 1 — Export trong Terminal (Đơn giản nhất):**
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# Lưu luôn vào bashrc để không mất khi mở terminal mới
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
```

**Cách 2 — Cài HF CLI rồi đăng nhập:**
```bash
pip install huggingface_hub
huggingface-cli login
# Dán token khi được hỏi → Enter → Gõ 'n' khi hỏi git credential
```

**Cách 3 — RunPod Secrets (Bảo mật nhất):**
* RunPod Dashboard → **Secrets** → Thêm `HF_TOKEN` = giá trị token → Pod tự inject khi khởi động.

**Kiểm tra token đã được nạp chưa:**
```bash
echo $HF_TOKEN
```

---

## 4. Tải Checkpoint (Pre-download)

```bash
# Đảm bảo token đã được set
echo $HF_TOKEN

# Tải checkpoint SAM 3.0 (~4 GB)
huggingface-cli download facebook/sam3 sam3.pt \
    --local-dir /workspace/checkpoints \
    --token $HF_TOKEN

# Kiểm tra
ls -lh /workspace/checkpoints/
```

> 💡 Lần sau khi restart Pod, checkpoint vẫn còn trong `/workspace/checkpoints/`. Chạy app bằng lệnh: `python app.py --checkpoint /workspace/checkpoints/sam3.pt`

---

## 5. Chạy Gradio App Demo

### 5.1. Chạy App

```bash
cd /workspace/sam3
conda activate sam3

# Dùng checkpoint local (khuyên dùng, tránh tải lại)
python app.py --checkpoint /workspace/checkpoints/sam3.pt

# Hoặc auto-download từ HF (cần HF_TOKEN)
python app.py
```

### 5.2. Truy Cập Giao Diện

* **Từ RunPod Dashboard:** Nhấn **Connect** → **HTTP Service [7860]**.
* **Từ URL public:** `https://<pod-id>-7860.proxy.runpod.net`.

### 5.3. Hướng Dẫn Sử Dụng 3 Chế Độ Prompt

| Chế Độ | Cách Dùng | Phù Hợp Khi |
| :--- | :--- | :--- |
| **Text Prompt** | Nhập mô tả đối tượng vào ô text | Muốn tìm tất cả instances của một concept |
| **Bounding Box** | Vẽ hình chữ nhật bao quanh đối tượng | Muốn segment đối tượng cụ thể trong ảnh |
| **Point Click** | Nhập tọa độ foreground/background | Muốn segment chính xác bằng chấm điểm |

### 5.4. Dừng và Chạy Nền (Background)

```bash
# Chạy nền
nohup python app.py --checkpoint /workspace/checkpoints/sam3.pt > /workspace/gradio.log 2>&1 &

# Xem log
tail -f /workspace/gradio.log

# Dừng app
pkill -f "python app.py"
```

---

## 6. Script Khởi Động Tự Động (`startup.sh`)

Script này tự động hóa toàn bộ quá trình setup. Chạy **một lần duy nhất** sau khi tạo Pod mới:

```bash
# Bước 1: Set HF Token trước
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Bước 2: Vào workspace và clone repo của bạn
cd /workspace
git clone -b dev1 https://github.com/KhoaLeDang2375/sam3.git sam3
cd sam3

# Bước 3: Chạy startup script
chmod +x startup.sh
./startup.sh
```

---

## 7. Xử Lý Lỗi (Troubleshooting)

| Lỗi / Vấn đề | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **`bash: conda: command not found`** | Template không có Conda, chưa cài | Cài Miniconda theo Phần 2.2 |
| **`CondaToSNonInteractiveError`** | Anaconda yêu cầu accept ToS | Chạy 2 lệnh `conda tos accept` theo Phần 2.3 |
| **`fatal: destination path 'sam3' already exists`** | Thư mục sam3 đã tồn tại | Chạy `rm -rf /workspace/sam3` rồi clone lại |
| **`bash: hf: command not found`** | Chưa cài thư viện huggingface_hub | Chạy `pip install huggingface_hub` |
| **CUDA Out of Memory (OOM)** | Ảnh quá lớn, Point Mode tốn thêm VRAM | Giảm confidence threshold, dùng bfloat16 |
| **401 Unauthorized / Gated Repo** | Token chưa set hoặc chưa được chấp thuận | Kiểm tra `echo $HF_TOKEN`, đảm bảo đã Request Access tại `facebook/sam3` |
| **Mất data/checkpoint khi Stop Pod** | Đặt file ngoài `/workspace` | Luôn dùng `/workspace` cho mọi thứ |
| **`flash-attn-3` build failed** | Cài cho GPU không phải H100 | Dùng `pip install flash-attn` cho Ampere/Ada |
| **Port 7860 không truy cập được** | Chưa expose port khi tạo Pod | Stop Pod → Edit → Thêm HTTP port 7860 → Restart |
| **`ModuleNotFoundError: sam3`** | Chưa cài package | Chạy `pip install -e .` trong `/workspace/sam3` |
