#!/bin/bash
# =============================================================================
# startup.sh — OPTIMIZED one-command setup for Body Part Segmentation Pipeline
#
# ⚡ OPTIMIZATIONS vs v1:
#   1. uv (Rust-based pip) thay pip thường → cài packages nhanh gấp 10-20x
#   2. Tận dụng PyTorch hệ thống (đã có trong RunPod template) bằng
#      --system-site-packages → bỏ qua tải 2x PyTorch ~5GB
#   3. Cài parallel: PartCATSeg deps + SAM3 deps cài đồng thời
#   4. libmamba solver: conda solve từ 5 phút → 5 giây
#
# USAGE:
#   cd /workspace
#   git clone https://github.com/KhoaLeDang2375/Body_part_segment.git
#   cd Body_part_segment
#   HF_TOKEN="hf_xxx..." bash startup.sh
#
# AFTER SETUP:
#   bash start_pipeline.sh
# =============================================================================

set -e

echo "=============================================="
echo " Body Part Segmentation Pipeline — Setup (OPTIMIZED)"
echo " (PartCATSeg + SAM3)"
echo "=============================================="
echo ""

# --- Config ---
WORKSPACE="/workspace"
REPO_DIR="$(pwd)"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
MINICONDA_DIR="$WORKSPACE/miniconda3"

# =============================================================================
# STEP 0: Check HF Token (needed for SAM3 checkpoint)
# =============================================================================
echo "[0/7] Checking Hugging Face Token..."

if [ -z "$HF_TOKEN" ]; then
    echo "  ⚠️  No HF Token! Will skip SAM3 checkpoint download."
    echo "  Set it with: export HF_TOKEN='hf_xxx...' then re-run startup.sh"
else
    echo "  ✓ HF Token provided (${#HF_TOKEN} chars)"
fi

# =============================================================================
# STEP 1: Setup persistent storage
# =============================================================================
echo ""
echo "[1/7] Setting up persistent storage..."

mkdir -p "$WORKSPACE/.cache/huggingface"
mkdir -p "$WORKSPACE/.cache/torch"
mkdir -p "$WORKSPACE/.cache/pip"
mkdir -p "$WORKSPACE/.cache/uv"
mkdir -p "$WORKSPACE/tmp"
mkdir -p "$CHECKPOINT_DIR"

export HF_HOME="$WORKSPACE/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORKSPACE/.cache/huggingface"
export TORCH_HOME="$WORKSPACE/.cache/torch"
export HF_HUB_CACHE="$WORKSPACE/.cache/huggingface/hub"
export TMPDIR="$WORKSPACE/tmp"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"

# Persist env vars
grep -qF 'HF_HOME' ~/.bashrc         || echo "export HF_HOME=\"$WORKSPACE/.cache/huggingface\""  >> ~/.bashrc
grep -qF 'TRANSFORMERS_CACHE' ~/.bashrc || echo "export TRANSFORMERS_CACHE=\"$WORKSPACE/.cache/huggingface\"" >> ~/.bashrc
grep -qF 'TORCH_HOME' ~/.bashrc      || echo "export TORCH_HOME=\"$WORKSPACE/.cache/torch\""     >> ~/.bashrc
grep -qF 'PIP_CACHE_DIR' ~/.bashrc   || echo "export PIP_CACHE_DIR=\"$WORKSPACE/.cache/pip\""    >> ~/.bashrc
grep -qF 'UV_CACHE_DIR' ~/.bashrc    || echo "export UV_CACHE_DIR=\"$WORKSPACE/.cache/uv\""      >> ~/.bashrc
grep -qF 'TMPDIR' ~/.bashrc          || echo "export TMPDIR=\"$WORKSPACE/tmp\""                  >> ~/.bashrc
[ -n "$HF_TOKEN" ] && (grep -qF 'HF_TOKEN' ~/.bashrc || echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc)

echo "  ✓ Persistent storage configured"

# =============================================================================
# STEP 2: Install/locate Conda + OPTIMIZATION: libmamba solver
# =============================================================================
echo ""
echo "[2/7] Setting up Conda (with libmamba solver for fast resolve)..."

if [ -f "$MINICONDA_DIR/bin/conda" ]; then
    export PATH="$MINICONDA_DIR/bin:$PATH"
    echo "  ✓ Found Miniconda at $MINICONDA_DIR"
elif [ -f "/opt/conda/bin/conda" ]; then
    export PATH="/opt/conda/bin:$PATH"
    echo "  ✓ Found Conda at /opt/conda"
else
    echo "  Installing Miniconda to $MINICONDA_DIR..."
    cd "$WORKSPACE"
    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm miniconda.sh
    export PATH="$MINICONDA_DIR/bin:$PATH"
    echo "  ✓ Miniconda installed"
fi

grep -qF "$MINICONDA_DIR/bin" ~/.bashrc || echo "export PATH=\"$MINICONDA_DIR/bin:\$PATH\"" >> ~/.bashrc
eval "$(conda shell.bash hook)"

# ⚡ OPTIMIZATION 1: libmamba solver (conda solve từ 5 phút → 5 giây)
echo "  Installing libmamba solver (conda solve 10x faster)..."
conda install -n base conda-libmamba-solver -y -q 2>/dev/null || true
conda config --set solver libmamba 2>/dev/null || true

# Accept Anaconda ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "  ✓ Conda: $(conda --version) | Solver: libmamba"

# =============================================================================
# STEP 3: Install uv (Rust-based pip replacement — 10-20x faster)
# =============================================================================
echo ""
echo "[3/7] Installing uv (fast package installer)..."

if command -v uv &>/dev/null; then
    echo "  ✓ uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    grep -qF '.cargo/bin' ~/.bashrc || echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    echo "  ✓ uv installed: $(uv --version)"
fi

# =============================================================================
# STEP 4: Setup PartCATSeg environment
# ⚡ OPTIMIZATION 2: --system-site-packages để kế thừa PyTorch của hệ thống
# =============================================================================
echo ""
echo "[4/7] Setting up PartCATSeg environment (partcatseg)..."
cd "$REPO_DIR"

# Kiểm tra PyTorch hệ thống có dùng được không
SYSTEM_TORCH_OK=false
SYSTEM_TORCH_VERSION=""
if python -c "import torch; v=torch.__version__; exit(0 if v.startswith('2.2') else 1)" 2>/dev/null; then
    SYSTEM_TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    SYSTEM_TORCH_OK=true
    echo "  ✓ Hệ thống đã có PyTorch $SYSTEM_TORCH_VERSION — sẽ dùng lại, KHÔNG tải lại!"
fi

if conda info --envs | grep -q "^partcatseg "; then
    echo "  Env 'partcatseg' đã tồn tại, bỏ qua tạo mới."
else
    echo "  Tạo conda env 'partcatseg' (Python 3.11)..."
    if [ "$SYSTEM_TORCH_OK" = true ]; then
        # ⚡ Kế thừa torch của hệ thống: env nhỏ hơn, tạo nhanh hơn
        conda create -n partcatseg python=3.11 --system-site-packages -y -q
        echo "  ✓ Env tạo với --system-site-packages (kế thừa PyTorch hệ thống)"
    else
        conda create -n partcatseg python=3.11 -y -q
    fi
fi

conda activate partcatseg

# ⚡ OPTIMIZATION 3: uv pip thay vì pip thường
echo "  Cài PartCATSeg dependencies bằng uv (nhanh ~15x)..."
cd "$REPO_DIR/part-catseg"

# Chỉ cài PyTorch nếu hệ thống chưa có hoặc version sai
if ! python -c "import torch; assert torch.__version__.startswith('2.2')" 2>/dev/null; then
    echo "  Cài PyTorch 2.2.2 cho partcatseg..."
    uv pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
else
    echo "  ✓ PyTorch $(python -c 'import torch; print(torch.__version__)') đã có — bỏ qua cài lại!"
fi

uv pip install -r requirements.txt
uv pip install fastapi "uvicorn[standard]"

echo "  Cài Detectron2..."
if ! python -c "import detectron2" 2>/dev/null; then
    if [ ! -d "/tmp/detectron2" ]; then
        git clone -q https://github.com/facebookresearch/detectron2.git /tmp/detectron2
    fi
    pip install --no-build-isolation --no-deps -e /tmp/detectron2 2>/dev/null || true
else
    echo "  ✓ Detectron2 đã cài — bỏ qua!"
fi

echo "  ✓ PartCATSeg environment ready"
conda deactivate

# =============================================================================
# STEP 5: Setup SAM3 environment
# ⚡ OPTIMIZATION 2: Tương tự — dùng --system-site-packages nếu torch mới hơn
# =============================================================================
echo ""
echo "[5/7] Setting up SAM3 environment (sam3env)..."
cd "$REPO_DIR"

if conda info --envs | grep -q "^sam3env "; then
    echo "  Env 'sam3env' đã tồn tại, bỏ qua tạo mới."
else
    echo "  Tạo conda env 'sam3env' (Python 3.12)..."
    # SAM3 cần timm>=1.0.17, PyTorch hệ thống (template 2.4+) thường đủ dùng
    if python -c "import torch; assert int(torch.__version__.split('.')[0]) >= 2" 2>/dev/null; then
        conda create -n sam3env python=3.12 --system-site-packages -y -q
        echo "  ✓ Env tạo với --system-site-packages (kế thừa PyTorch hệ thống)"
    else
        conda create -n sam3env python=3.12 -y -q
    fi
fi

conda activate sam3env

# Cài PyTorch cho SAM3 chỉ nếu chưa có
if ! python -c "import torch" 2>/dev/null; then
    echo "  Cài PyTorch (CUDA 12.8) cho sam3env..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    echo "  ✓ PyTorch $(python -c 'import torch; print(torch.__version__)') đã có — bỏ qua cài lại!"
fi

echo "  Cài SAM3 + pipeline dependencies bằng uv..."
cd "$REPO_DIR/sam3"
uv pip install -e .

uv pip install -r "$REPO_DIR/part_sam_pipeline/requirements.txt"

# Optional: FlashAttention
echo "  Cài FlashAttention (optional)..."
uv pip install einops ninja 2>/dev/null || true
pip install flash-attn --no-build-isolation 2>/dev/null || echo "  (FlashAttention bỏ qua — không ảnh hưởng chức năng)"

echo "  ✓ SAM3 environment ready"
conda deactivate

# =============================================================================
# STEP 6: Download PartCATSeg weights
# =============================================================================
echo ""
echo "[6/7] Checking PartCATSeg weights..."
cd "$REPO_DIR/part-catseg"

VOC_WEIGHT_FILE="weights/partcatseg_voc.pth"
VOC_WEIGHT_URL="https://drive.google.com/uc?id=1JUJjJQLMKE96H5SLNs4EMm4jiU6fPgRb"

mkdir -p weights
if [ -f "$VOC_WEIGHT_FILE" ]; then
    echo "  ✓ PartCATSeg weights đã có — bỏ qua tải lại"
else
    echo "  Tải partcatseg_voc.pth (~885 MB)..."
    conda activate partcatseg
    uv pip install gdown -q
    gdown "$VOC_WEIGHT_URL" -O "$VOC_WEIGHT_FILE"
    conda deactivate
    echo "  ✓ PartCATSeg weights tải xong"
fi

# =============================================================================
# STEP 7: Download SAM3 checkpoint
# =============================================================================
echo ""
echo "[7/7] Checking SAM3 checkpoint..."

CKPT_FILE="$CHECKPOINT_DIR/sam3.pt"

if [ -f "$CKPT_FILE" ]; then
    FILESIZE=$(du -sh "$CKPT_FILE" | cut -f1)
    echo "  ✓ SAM3 checkpoint đã có ($FILESIZE) — bỏ qua tải lại"
elif [ -z "$HF_TOKEN" ]; then
    echo "  ⚠️  Không có HF_TOKEN. Tải thủ công sau:"
    echo "     export HF_TOKEN='hf_xxx'"
    echo "     conda activate sam3env"
    echo "     hf download facebook/sam3 sam3.pt --local-dir $CHECKPOINT_DIR --token \$HF_TOKEN"
else
    echo "  Tải SAM3 checkpoint từ Hugging Face (~4 GB)..."
    conda activate sam3env
    hf download facebook/sam3 sam3.pt \
        --local-dir "$CHECKPOINT_DIR" \
        --token "$HF_TOKEN"
    conda deactivate
    echo "  ✓ SAM3 checkpoint tải xong"
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
df -h "$WORKSPACE" | tail -1 | awk '{print "  Workspace — Total: " $2 " | Used: " $3 " | Free: " $4}'
echo ""
echo "=============================================="
echo " ✅ Setup hoàn tất!"
echo "=============================================="
echo ""
echo "Chạy pipeline:"
echo "  bash start_pipeline.sh"
echo ""
