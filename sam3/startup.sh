#!/bin/bash
# =============================================================================
# SAM 3 RunPod Startup Script — Phiên bản đã fix tất cả lỗi thực tế
#
# CÁCH SỬ DỤNG:
#   Bước 1: Vào workspace
#           cd /workspace
#
#   Bước 2: Clone repo và vào thư mục
#           git clone -b dev1 https://github.com/KhoaLeDang2375/sam3.git sam3
#           cd sam3
#
#   Bước 3: Chạy script (truyền HF Token trực tiếp vào lệnh)
#           HF_TOKEN="hf_xxx..." bash startup.sh
#           HOẶC
#           export HF_TOKEN="hf_xxx..."
#           bash startup.sh
# =============================================================================

set -e  # Dừng nếu có lỗi

echo "=============================================="
echo " SAM 3 RunPod Setup Script"
echo "=============================================="
echo ""

# --- Cấu hình ---
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/sam3"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
CONDA_ENV="sam3"
PYTHON_VERSION="3.12"
MINICONDA_DIR="$WORKSPACE/miniconda3"

# =============================================================================
# BƯỚC 0: Kiểm tra HF Token
# =============================================================================
echo "[0/7] Kiểm tra Hugging Face Token..."

# Hỏi token nếu chưa được cung cấp
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "  HF_TOKEN chưa được set."
    echo "  Nhập Hugging Face Token của bạn (hf_xxx...):"
    read -r -s HF_TOKEN
    echo ""
fi

if [ -z "$HF_TOKEN" ]; then
    echo "  ⚠️  Không có HF Token! Sẽ bỏ qua bước tải checkpoint."
    echo "     (Chạy lại script với: HF_TOKEN='hf_xxx...' bash startup.sh)"
else
    echo "  ✓ HF Token đã được cung cấp (${#HF_TOKEN} ký tự)"
fi

# =============================================================================
# BƯỚC 1: Thiết lập Persistent Storage — TOÀN BỘ cache vào /workspace
#          (tránh lỗi Disk Quota Exceeded trên container disk nhỏ)
# =============================================================================
echo ""
echo "[1/7] Thiết lập Persistent Storage (tránh đầy đĩa)..."

mkdir -p "$WORKSPACE/.cache/huggingface"
mkdir -p "$WORKSPACE/.cache/torch"
mkdir -p "$CHECKPOINT_DIR"

# Đặt biến môi trường ngay trong session này
export HF_HOME="$WORKSPACE/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORKSPACE/.cache/huggingface"
export TORCH_HOME="$WORKSPACE/.cache/torch"
export HF_HUB_CACHE="$WORKSPACE/.cache/huggingface/hub"

# Ghi vĩnh viễn vào bashrc (mỗi lần restart Pod tự áp dụng)
grep -qF 'HF_HOME' ~/.bashrc || echo "export HF_HOME=\"$WORKSPACE/.cache/huggingface\"" >> ~/.bashrc
grep -qF 'TRANSFORMERS_CACHE' ~/.bashrc || echo "export TRANSFORMERS_CACHE=\"$WORKSPACE/.cache/huggingface\"" >> ~/.bashrc
grep -qF 'TORCH_HOME' ~/.bashrc || echo "export TORCH_HOME=\"$WORKSPACE/.cache/torch\"" >> ~/.bashrc
grep -qF 'HF_HUB_CACHE' ~/.bashrc || echo "export HF_HUB_CACHE=\"$WORKSPACE/.cache/huggingface/hub\"" >> ~/.bashrc
[ -n "$HF_TOKEN" ] && grep -qF 'HF_TOKEN' ~/.bashrc || ([ -n "$HF_TOKEN" ] && echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc)

# Xóa rác cache cũ trên container disk (giải phóng đĩa tạm)
rm -rf ~/.cache/huggingface/hub/tmp* 2>/dev/null || true

echo "  ✓ HF_HOME = $HF_HOME"
echo "  ✓ TORCH_HOME = $TORCH_HOME"
echo "  ✓ CHECKPOINT_DIR = $CHECKPOINT_DIR"
echo "  ✓ Container disk space:"
df -h / | tail -1 | awk '{print "     Tổng: " $2 " | Đã dùng: " $3 " | Còn lại: " $4}'

# =============================================================================
# BƯỚC 2: Cài hoặc khởi tạo Conda
# =============================================================================
echo ""
echo "[2/7] Kiểm tra Conda..."

if [ -f "$MINICONDA_DIR/bin/conda" ]; then
    export PATH="$MINICONDA_DIR/bin:$PATH"
    echo "  ✓ Tìm thấy Miniconda tại $MINICONDA_DIR"
elif [ -f "/opt/conda/bin/conda" ]; then
    export PATH="/opt/conda/bin:$PATH"
    echo "  ✓ Tìm thấy Conda tại /opt/conda"
else
    echo "  Conda chưa có. Đang tải Miniconda vào $MINICONDA_DIR..."
    cd "$WORKSPACE"
    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm miniconda.sh
    export PATH="$MINICONDA_DIR/bin:$PATH"
    echo "  ✓ Đã cài Miniconda tại $MINICONDA_DIR"
fi

# Ghi PATH conda vào bashrc nếu chưa có
grep -qF "$MINICONDA_DIR/bin" ~/.bashrc || echo "export PATH=\"$MINICONDA_DIR/bin:\$PATH\"" >> ~/.bashrc

# Khởi tạo conda shell hook (cho phép conda activate trong script)
eval "$(conda shell.bash hook)"
echo "  ✓ Conda: $(conda --version)"

# =============================================================================
# BƯỚC 3: Chấp nhận Anaconda Terms of Service (tránh CondaToSNonInteractiveError)
# =============================================================================
echo ""
echo "[3/7] Chấp nhận Anaconda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
echo "  ✓ Anaconda ToS đã chấp nhận"

# =============================================================================
# BƯỚC 4: Tạo và kích hoạt Conda Environment
# =============================================================================
echo ""
echo "[4/7] Thiết lập Conda Environment ($CONDA_ENV)..."
cd "$REPO_DIR"

if conda info --envs | grep -q "^$CONDA_ENV "; then
    echo "  Env '$CONDA_ENV' đã tồn tại, bỏ qua bước tạo."
else
    echo "  Đang tạo conda env '$CONDA_ENV' với Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV" -c conda-forge python="$PYTHON_VERSION" -y
fi

conda activate "$CONDA_ENV"
echo "  ✓ Conda env '$CONDA_ENV' đang active"

# Ghi vào bashrc để tự động activate
grep -qF "conda activate $CONDA_ENV" ~/.bashrc || echo "conda activate $CONDA_ENV" >> ~/.bashrc

# =============================================================================
# BƯỚC 5: Cài đặt Dependencies
# =============================================================================
echo ""
echo "[5/7] Cài đặt Dependencies..."

echo "  [5/4] Installing PyTorch với CUDA 12.8..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "  [5/4] Installing SAM 3 package..."
pip install --quiet -e .

echo "  [5/4] Installing Gradio + all required dependencies..."
pip install --quiet "gradio>=4.0" einops pycocotools opencv-python \
    tqdm timm scipy scikit-image scikit-learn ftfy pandas torchmetrics iopath \
    psutil pyyaml omegaconf hydra-core \
    huggingface_hub

echo "  ✓ Tất cả dependencies đã cài xong"

# =============================================================================
# BƯỚC 6: Tải Checkpoint từ Hugging Face
# =============================================================================
echo ""
echo "[6/7] Kiểm tra và Tải Checkpoint..."

CKPT_FILE="$CHECKPOINT_DIR/sam3.pt"

if [ -f "$CKPT_FILE" ]; then
    FILESIZE=$(du -sh "$CKPT_FILE" | cut -f1)
    echo "  ✓ Checkpoint đã có: $CKPT_FILE ($FILESIZE) — Bỏ qua tải lại."
elif [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "  ⚠️  Bỏ qua tải checkpoint vì không có HF_TOKEN."
    echo "     Tải thủ công sau khi set token:"
    echo "       export HF_TOKEN='hf_your_token_here'"
    echo "       hf download facebook/sam3 sam3.pt --local-dir $CHECKPOINT_DIR --token \$HF_TOKEN"
else
    echo "  Đang tải SAM 3 checkpoint từ Hugging Face (~4 GB)..."
    echo "  (Cache sẽ được lưu vào $HF_HOME — không tràn container disk)"

    # Dùng lệnh 'hf download' (lệnh mới thay thế huggingface-cli đã deprecated)
    hf download facebook/sam3 sam3.pt \
        --local-dir "$CHECKPOINT_DIR" \
        --token "$HF_TOKEN"

    echo "  ✓ Checkpoint đã tải: $CKPT_FILE"
    ls -lh "$CKPT_FILE"
fi

# =============================================================================
# BƯỚC 7: Kiểm tra cuối
# =============================================================================
echo ""
echo "[7/7] Kiểm tra cuối..."
df -h "$WORKSPACE" | tail -1 | awk '{print "  Workspace Disk — Tổng: " $2 " | Đã dùng: " $3 " | Còn lại: " $4}'
echo ""
echo "=============================================="
echo " ✅ Setup hoàn tất!"
echo "=============================================="
echo ""
echo "Để chạy Gradio App:"
echo "  cd $REPO_DIR"
echo "  conda activate $CONDA_ENV"
echo "  python app.py --checkpoint $CHECKPOINT_DIR/sam3.pt"
echo ""
echo "Để chạy nền:"
echo "  nohup python app.py --checkpoint $CHECKPOINT_DIR/sam3.pt > $WORKSPACE/gradio.log 2>&1 &"
echo "  tail -f $WORKSPACE/gradio.log"
echo ""
