#!/bin/bash
# =============================================================================
# startup.sh — One-command setup for Body Part Segmentation Pipeline on RunPod
#
# Sets up BOTH conda environments:
#   1. partcatseg — PartCATSeg model + FastAPI server
#   2. sam3env    — SAM3 model + Pipeline + Gradio UI
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
echo " Body Part Segmentation Pipeline — Setup"
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
echo "[0/8] Checking Hugging Face Token..."

if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "  HF_TOKEN not set. Please enter your Hugging Face Token (hf_xxx...):"
    read -r -s HF_TOKEN
    echo ""
fi

if [ -z "$HF_TOKEN" ]; then
    echo "  ⚠️  No HF Token! Will skip SAM3 checkpoint download."
else
    echo "  ✓ HF Token provided (${#HF_TOKEN} chars)"
fi

# =============================================================================
# STEP 1: Setup persistent storage
# =============================================================================
echo ""
echo "[1/8] Setting up persistent storage..."

mkdir -p "$WORKSPACE/.cache/huggingface"
mkdir -p "$WORKSPACE/.cache/torch"
mkdir -p "$WORKSPACE/.cache/pip"
mkdir -p "$WORKSPACE/tmp"
mkdir -p "$CHECKPOINT_DIR"

export HF_HOME="$WORKSPACE/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORKSPACE/.cache/huggingface"
export TORCH_HOME="$WORKSPACE/.cache/torch"
export HF_HUB_CACHE="$WORKSPACE/.cache/huggingface/hub"
export TMPDIR="$WORKSPACE/tmp"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"

# Persist env vars
grep -qF 'HF_HOME' ~/.bashrc || echo "export HF_HOME=\"$WORKSPACE/.cache/huggingface\"" >> ~/.bashrc
grep -qF 'TRANSFORMERS_CACHE' ~/.bashrc || echo "export TRANSFORMERS_CACHE=\"$WORKSPACE/.cache/huggingface\"" >> ~/.bashrc
grep -qF 'TORCH_HOME' ~/.bashrc || echo "export TORCH_HOME=\"$WORKSPACE/.cache/torch\"" >> ~/.bashrc
grep -qF 'PIP_CACHE_DIR' ~/.bashrc || echo "export PIP_CACHE_DIR=\"$WORKSPACE/.cache/pip\"" >> ~/.bashrc
grep -qF 'TMPDIR' ~/.bashrc || echo "export TMPDIR=\"$WORKSPACE/tmp\"" >> ~/.bashrc
[ -n "$HF_TOKEN" ] && (grep -qF 'HF_TOKEN' ~/.bashrc || echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc)

echo "  ✓ Persistent storage configured"

# =============================================================================
# STEP 2: Install/locate Conda
# =============================================================================
echo ""
echo "[2/8] Setting up Conda..."

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
echo "  ✓ Conda: $(conda --version)"

# =============================================================================
# STEP 3: Accept Anaconda ToS
# =============================================================================
echo ""
echo "[3/8] Accepting Anaconda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
echo "  ✓ Anaconda ToS accepted"

# =============================================================================
# STEP 4: Setup PartCATSeg environment
# =============================================================================
echo ""
echo "[4/8] Setting up PartCATSeg environment (partcatseg)..."
cd "$REPO_DIR"

if conda info --envs | grep -q "^partcatseg "; then
    echo "  Env 'partcatseg' already exists, skipping creation."
else
    echo "  Creating conda env 'partcatseg' with Python 3.11..."
    conda create -n partcatseg -c conda-forge python=3.11 -y
fi

conda activate partcatseg

# Use /workspace for tmp files to avoid container disk space issues
export TMPDIR="$WORKSPACE/tmp"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"

echo "  Installing PyTorch 2.2.2 (CUDA 12.1)..."
pip install --quiet torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

echo "  Installing PartCATSeg dependencies..."
cd "$REPO_DIR/part-catseg"
pip install --quiet -r requirements.txt

# Install FastAPI + uvicorn for inference server
pip install --quiet fastapi uvicorn[standard]

echo "  Installing Detectron2..."
if [ ! -d "/tmp/detectron2" ]; then
    git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2
fi
pip install --quiet --no-build-isolation --no-deps -e /tmp/detectron2 || true

echo "  ✓ PartCATSeg environment ready"

conda deactivate

# =============================================================================
# STEP 5: Setup SAM3 environment
# =============================================================================
echo ""
echo "[5/8] Setting up SAM3 environment (sam3env)..."
cd "$REPO_DIR"

if conda info --envs | grep -q "^sam3env "; then
    echo "  Env 'sam3env' already exists, skipping creation."
else
    echo "  Creating conda env 'sam3env' with Python 3.12..."
    conda create -n sam3env -c conda-forge python=3.12 -y
fi

conda activate sam3env

echo "  Installing PyTorch (CUDA 12.8)..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "  Installing SAM3 package..."
cd "$REPO_DIR/sam3"
pip install --quiet -e .

echo "  Installing pipeline dependencies..."
pip install --quiet -r "$REPO_DIR/part_sam_pipeline/requirements.txt"

# Optional: FlashAttention
echo "  Installing FlashAttention (optional, may fail on some GPUs)..."
pip install --quiet einops ninja
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || echo "  (FlashAttention skipped — not critical)"

echo "  ✓ SAM3 environment ready"

conda deactivate

# =============================================================================
# STEP 6: Download PartCATSeg weights
# =============================================================================
echo ""
echo "[6/8] Downloading PartCATSeg weights..."
cd "$REPO_DIR/part-catseg"

WEIGHTS_DIR="weights"
VOC_WEIGHT_FILE="$WEIGHTS_DIR/partcatseg_voc.pth"
VOC_WEIGHT_URL="https://drive.google.com/uc?id=1JUJjJQLMKE96H5SLNs4EMm4jiU6fPgRb"

mkdir -p "$WEIGHTS_DIR"
if [ -f "$VOC_WEIGHT_FILE" ]; then
    echo "  ✓ PartCATSeg weights already present"
else
    echo "  Downloading partcatseg_voc.pth (~885 MB)..."
    conda activate partcatseg
    pip install --quiet gdown
    gdown "$VOC_WEIGHT_URL" -O "$VOC_WEIGHT_FILE"
    conda deactivate
    echo "  ✓ PartCATSeg weights downloaded"
fi

# =============================================================================
# STEP 7: Download SAM3 checkpoint
# =============================================================================
echo ""
echo "[7/8] Downloading SAM3 checkpoint..."

CKPT_FILE="$CHECKPOINT_DIR/sam3.pt"

if [ -f "$CKPT_FILE" ]; then
    FILESIZE=$(du -sh "$CKPT_FILE" | cut -f1)
    echo "  ✓ SAM3 checkpoint already present ($FILESIZE)"
elif [ -z "$HF_TOKEN" ]; then
    echo "  ⚠️  Skipping: no HF_TOKEN. Download manually later:"
    echo "     export HF_TOKEN='hf_xxx'"
    echo "     conda activate sam3env"
    echo "     hf download facebook/sam3 sam3.pt --local-dir $CHECKPOINT_DIR --token \$HF_TOKEN"
else
    echo "  Downloading SAM3 checkpoint (~4 GB)..."
    conda activate sam3env
    hf download facebook/sam3 sam3.pt \
        --local-dir "$CHECKPOINT_DIR" \
        --token "$HF_TOKEN"
    conda deactivate
    echo "  ✓ SAM3 checkpoint downloaded"
fi

# =============================================================================
# STEP 8: Final verification
# =============================================================================
echo ""
echo "[8/8] Final verification..."

echo "  GPU Info:"
python -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
print(f'    {result.stdout.strip()}')
" 2>/dev/null || echo "    (nvidia-smi not available)"

echo ""
echo "  Disk Usage:"
df -h "$WORKSPACE" | tail -1 | awk '{print "    Total: " $2 " | Used: " $3 " | Free: " $4}'

echo ""
echo "=============================================="
echo " ✅ Setup Complete!"
echo "=============================================="
echo ""
echo "To start the pipeline:"
echo "  cd $REPO_DIR"
echo "  bash start_pipeline.sh"
echo ""
echo "Or manually:"
echo "  # Terminal 1 — PartCATSeg server"
echo "  conda activate partcatseg"
echo "  cd $REPO_DIR/part-catseg"
echo "  python inference_server.py"
echo ""
echo "  # Terminal 2 — Pipeline UI"
echo "  conda activate sam3env"
echo "  cd $REPO_DIR"
echo "  python -m part_sam_pipeline.app --checkpoint $CHECKPOINT_DIR/sam3.pt"
echo ""
