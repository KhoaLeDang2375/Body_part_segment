#!/bin/bash
# =============================================================
# start_demo.sh — RunPod startup script for PartCATSeg Demo
#
# Usage on RunPod terminal:
#   bash start_demo.sh
#
# The Gradio UI will be accessible at:
#   https://<pod-id>-7860.proxy.runpod.net
# =============================================================

set -e

CONDA_ENV="partcatseg"
PORT=7860
WEIGHTS_DIR="weights"
VOC_WEIGHT_URL="https://drive.google.com/uc?id=1JUJjJQLMKE96H5SLNs4EMm4jiU6fPgRb"
VOC_WEIGHT_FILE="$WEIGHTS_DIR/partcatseg_voc.pth"

echo "========================================"
echo " PartCATSeg Gradio Demo — RunPod Setup"
echo "========================================"

# ── 1. Activate conda environment ────────────────────────────
echo ""
echo "[1/4] Activating conda environment: $CONDA_ENV"

# Try to find and source conda.sh for subshell activation support
CONDA_SH=""
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
elif [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/root/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
fi

if command -v conda &> /dev/null; then
    conda activate "$CONDA_ENV" 2>/dev/null || source activate "$CONDA_ENV" 2>/dev/null || echo "  (Using base/active environment)"
else
    echo "  (Conda not found, using system Python)"
fi

# Use /workspace for temporary files & pip cache to avoid running out of disk space on /
mkdir -p /workspace/tmp /workspace/.cache/pip
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/.cache/pip

# Ensure PYTHONPATH includes /tmp/detectron2 if present
if [ -d "/tmp/detectron2" ]; then
    export PYTHONPATH="/tmp/detectron2:$PYTHONPATH"
fi

# Auto-install PyTorch if missing in current environment
if ! python -c "import torch" &>/dev/null; then
    echo "  ⚠️  PyTorch not found in environment '$CONDA_ENV'. Installing PyTorch 2.2.2 (CUDA 12.1) …"
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir -r requirements.txt
fi

# Auto-install detectron2 if needed
if ! python -c "import detectron2" &>/dev/null; then
    echo "  ⚠️  Detectron2 not found. Installing Detectron2 automatically …"
    if [ ! -d "/tmp/detectron2" ]; then
        git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2
    fi
    pip install --no-cache-dir --no-build-isolation --no-deps -e /tmp/detectron2 || true
fi

# ── 2. Download checkpoint if not already present ────────────
echo ""
echo "[2/4] Checking model weights …"
mkdir -p "$WEIGHTS_DIR"
if [ ! -f "$VOC_WEIGHT_FILE" ]; then
    echo "  Downloading partcatseg_voc.pth (~885 MB) …"
    gdown "$VOC_WEIGHT_URL" -O "$VOC_WEIGHT_FILE"
    echo "  ✅  Download complete."
else
    echo "  ✅  Weights already present: $VOC_WEIGHT_FILE"
fi

# ── 3. Verify GPU availability ────────────────────────────────
echo ""
echo "[3/4] Checking GPU …"

python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  ✅  GPU: {name} ({vram:.1f} GB VRAM)')
else:
    print('  ⚠️   No GPU detected — falling back to CPU (slower)')
"

# ── 4. Launch Gradio app ──────────────────────────────────────
echo ""
echo "[4/4] Starting Gradio demo on port $PORT …"
echo "  Access at: https://\$(hostname)-${PORT}.proxy.runpod.net"
echo ""

python app.py --device cuda --port "$PORT" "$@"
