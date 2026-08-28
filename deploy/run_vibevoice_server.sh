#!/bin/bash
# run_vibevoice_server.sh — launch the VibeVoice OpenAI-compatible server (port 8000)
#
# Usage:  ./run_vibevoice_server.sh
# Env:    DEVICE, MODEL_PATH, HOST, PORT, INFERENCE_STEPS,
#         HSA_OVERRIDE_GFX_VERSION, PYTORCH_NO_HIP_MEMORY_CACHING (AMD only)
set -euo pipefail
cd "$(dirname "$0")"

# --- device / GPU knobs ---
export DEVICE="${DEVICE:-cuda}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"   # AMD RDNA2 only
export PYTORCH_NO_HIP_MEMORY_CACHING="${PYTORCH_NO_HIP_MEMORY_CACHING:-1}"  # AMD only

# --- model / server ---
MODEL_PATH="${MODEL_PATH:-models/vibevoice-1.5b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"

echo "VibeVoice server on :${PORT} | device=${DEVICE} | steps=${INFERENCE_STEPS}"
exec .venv/bin/python -m demo.gradio_demo \
     --model_path "$MODEL_PATH" \
     --device "$DEVICE" \
     --host "$HOST" --port "$PORT" \
     --inference_steps "$INFERENCE_STEPS"