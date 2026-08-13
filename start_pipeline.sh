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
elif [ -f "/opt/conda/bin/conda" ]; then
    export PATH="/opt/conda/bin:$PATH"
fi
eval "$(conda shell.bash hook)"

# --- Ensure Detectron2 is on PYTHONPATH ---
if [ -d "/tmp/detectron2" ]; then
    export PYTHONPATH="/tmp/detectron2:$PYTHONPATH"
fi

# Use /workspace for temp files
export TMPDIR="$WORKSPACE/tmp"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# =============================================================================
# Step 1: Start PartCATSeg server (background)
# =============================================================================
echo "[1/2] Starting PartCATSeg server on port $CATSEG_PORT ..."

# Kill any existing server on that port
pkill -f "inference_server.py" 2>/dev/null || true
sleep 1

conda activate partcatseg

cd "$REPO_DIR/part-catseg"

# Start in background, log to file
nohup python inference_server.py \
    --port $CATSEG_PORT \
    --device cuda \
    > "$WORKSPACE/catseg_server.log" 2>&1 &

CATSEG_PID=$!
echo "  ✓ PartCATSeg server started (PID: $CATSEG_PID)"
echo "  Log: $WORKSPACE/catseg_server.log"

conda deactivate

# =============================================================================
# Step 2: Wait for CATSeg server to be ready
# =============================================================================
echo ""
echo "  Waiting for PartCATSeg server to load model ..."
echo "  (This takes 30-60 seconds on first run)"

MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$CATSEG_PORT/health" > /dev/null 2>&1; then
        echo "  ✓ PartCATSeg server is ready!"
        break
    fi
    sleep 3
    WAITED=$((WAITED + 3))
    echo "    ... waiting ($WAITED/${MAX_WAIT}s)"
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

conda activate sam3env

cd "$REPO_DIR"

echo "=============================================="
echo " Access the UI at:"
echo "   http://localhost:$PIPELINE_PORT"
echo "   https://\$(hostname)-${PIPELINE_PORT}.proxy.runpod.net"
echo "=============================================="
echo ""

python -m part_sam_pipeline.app \
    --checkpoint "$CHECKPOINT" \
    --catseg-url "http://localhost:$CATSEG_PORT" \
    --port $PIPELINE_PORT \
    --device cuda \
    $SHARE_FLAG
