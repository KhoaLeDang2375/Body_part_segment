#!/bin/bash
# =============================================================================
# start_pipeline.sh — Launch both services for the Body Part Segmentation Pipeline
#
# This script starts:
#   1. PartCATSeg inference server (conda env: partcatseg, port 8001)
#   2. Pipeline Gradio UI with SAM3 (conda env: sam3env, port 7860)
#
# USAGE:
#   cd /workspace/Body_part_segment
#   bash start_pipeline.sh
#
# OPTIONS:
#   --share       Create a public Gradio link
#   --checkpoint  Path to SAM3 checkpoint (default: /workspace/checkpoints/sam3.pt)
# =============================================================================

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT="${CHECKPOINT:-/workspace/checkpoints/sam3.pt}"
CATSEG_PORT=8001
PIPELINE_PORT=7860
SHARE_FLAG=""
WORKSPACE="/workspace"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --share)
            SHARE_FLAG="--share"
            ;;
        --checkpoint=*)
            CHECKPOINT="${arg#*=}"
            ;;
    esac
done

echo "=============================================="
echo " Body Part Segmentation Pipeline — Start"
echo "=============================================="
echo ""

# --- Find conda ---
MINICONDA_DIR="$WORKSPACE/miniconda3"
if [ -f "$MINICONDA_DIR/bin/conda" ]; then
    export PATH="$MINICONDA_DIR/bin:$PATH"
    CONDA_ENVS_DIR="$MINICONDA_DIR/envs"
elif [ -f "/opt/conda/bin/conda" ]; then
    export PATH="/opt/conda/bin:$PATH"
    CONDA_ENVS_DIR="/opt/conda/envs"
else
    echo "❌ Conda not found! Run startup.sh first."
    exit 1
fi
eval "$(conda shell.bash hook)"

# --- Resolve Python paths for each conda env ---
# Using direct paths avoids the 'conda activate' bug in non-interactive shells
PARTCATSEG_PYTHON="$CONDA_ENVS_DIR/partcatseg/bin/python"
SAM3_PYTHON="$CONDA_ENVS_DIR/sam3env/bin/python"

if [ ! -f "$PARTCATSEG_PYTHON" ]; then
    echo "❌ partcatseg env not found at $PARTCATSEG_PYTHON"
    echo "   Run startup.sh first to create conda environments."
    exit 1
fi
if [ ! -f "$SAM3_PYTHON" ]; then
    echo "❌ sam3env env not found at $SAM3_PYTHON"
    echo "   Run startup.sh first to create conda environments."
    exit 1
fi

echo "  ✓ partcatseg Python: $PARTCATSEG_PYTHON"
echo "  ✓ sam3env Python:    $SAM3_PYTHON"

# --- Ensure Detectron2 is available ---
D2_FOUND=false
for D2_DIR in "$WORKSPACE/detectron2" "/tmp/detectron2"; do
    if [ -d "$D2_DIR" ]; then
        export PYTHONPATH="$D2_DIR:$PYTHONPATH"
        echo "  ✓ Detectron2 found at $D2_DIR"
        D2_FOUND=true
        break
    fi
done

# Auto-recover Detectron2 if missing (e.g. after pod restart cleared /tmp)
if [ "$D2_FOUND" = false ]; then
    echo "  ⚠️  Detectron2 not found — cloning to $WORKSPACE/detectron2 ..."
    D2_DIR="$WORKSPACE/detectron2"
    git clone -q https://github.com/facebookresearch/detectron2.git "$D2_DIR"
    "$PARTCATSEG_PYTHON" -m pip install --quiet --no-build-isolation --no-deps -e "$D2_DIR"
    export PYTHONPATH="$D2_DIR:$PYTHONPATH"
    echo "  ✓ Detectron2 installed to $D2_DIR"
fi

# Use /workspace for temp files
export TMPDIR="$WORKSPACE/tmp"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# =============================================================================
# Step 1: Start PartCATSeg server (background)
# =============================================================================
echo ""
echo "[1/2] Starting PartCATSeg server on port $CATSEG_PORT ..."

# Kill any existing server on that port
pkill -f "inference_server.py" 2>/dev/null || true
sleep 1

cd "$REPO_DIR/part-catseg"

# Start using the partcatseg env's Python directly (no conda activate needed)
nohup "$PARTCATSEG_PYTHON" inference_server.py \
    --port $CATSEG_PORT \
    --device cuda \
    > "$WORKSPACE/catseg_server.log" 2>&1 &

CATSEG_PID=$!
echo "  ✓ PartCATSeg server started (PID: $CATSEG_PID)"
echo "  Log: $WORKSPACE/catseg_server.log"

# =============================================================================
# Step 2: Wait for CATSeg server to be FULLY ready (model loaded)
# =============================================================================
echo ""
echo "  Waiting for PartCATSeg server to load model ..."
echo "  (This takes 30-60 seconds on first run)"

MAX_WAIT=180
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    # Check JSON body for {"status":"ok"}, not just HTTP 200
    # The /health endpoint returns 200 with {"status":"loading"} while model loads
    HEALTH=$(curl -s "http://localhost:$CATSEG_PORT/health" 2>/dev/null || echo "")
    if echo "$HEALTH" | grep -q '"status".*:.*"ok"'; then
        echo "  ✓ PartCATSeg server is ready! (model loaded)"
        break
    elif echo "$HEALTH" | grep -q '"status".*:.*"loading"'; then
        # Server is up but model still loading — keep waiting
        if [ $((WAITED % 15)) -eq 0 ] && [ $WAITED -gt 0 ]; then
            echo "    ... model loading ($WAITED/${MAX_WAIT}s)"
        fi
    else
        # Server not responding yet
        if [ $((WAITED % 15)) -eq 0 ] && [ $WAITED -gt 0 ]; then
            echo "    ... waiting for server ($WAITED/${MAX_WAIT}s)"
        fi
    fi
    sleep 3
    WAITED=$((WAITED + 3))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  ⚠️  PartCATSeg server didn't respond in time."
    echo "  Check logs: tail -f $WORKSPACE/catseg_server.log"
    echo "  Starting pipeline anyway (will retry connection)..."
fi

# =============================================================================
# Step 3: Start Pipeline Gradio UI (foreground)
# =============================================================================
echo ""
echo "[2/2] Starting Pipeline Gradio UI on port $PIPELINE_PORT ..."
echo ""

cd "$REPO_DIR"

echo "=============================================="
echo " Access the UI at:"
echo "   http://localhost:$PIPELINE_PORT"
# RunPod injects $RUNPOD_POD_ID automatically — use it for the correct proxy URL
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "   https://${RUNPOD_POD_ID}-${PIPELINE_PORT}.proxy.runpod.net"
else
    # Fallback: Derive from hostname (works on most RunPod setups)
    POD_HOSTNAME=$(hostname 2>/dev/null || echo "your-pod-id")
    echo "   https://${POD_HOSTNAME}-${PIPELINE_PORT}.proxy.runpod.net"
fi
echo ""
echo "💡 Hoặc từ RunPod Dashboard: Connect → HTTP Service [${PIPELINE_PORT}]"
echo "=============================================="
echo ""

# Start using the sam3env's Python directly (no conda activate needed)
"$SAM3_PYTHON" -m part_sam_pipeline.app \
    --checkpoint "$CHECKPOINT" \
    --catseg-url "http://localhost:$CATSEG_PORT" \
    --port $PIPELINE_PORT \
    --device cuda \
    $SHARE_FLAG
