#!/usr/bin/env bash
# Native vLLM launch script (for running outside Docker)
# Usage: bash vllm/start.sh [model] [quantization]

set -euo pipefail

MODEL="${1:-${BULK_MODEL:-Qwen/Qwen3.5-0.8B}}"
QUANTIZATION="${2:-${BULK_QUANTIZATION:-fp8}}"
MAX_MODEL_LEN="${BULK_MAX_MODEL_LEN:-262144}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.95}"
KV_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GPU="${GPU_DEVICE:-GPU-3ad3e2fe}"

export NVIDIA_VISIBLE_DEVICES="$GPU"

echo "Starting vLLM with model: $MODEL"
echo "  Quantization: $QUANTIZATION"
echo "  Max model len: $MAX_MODEL_LEN"
echo "  GPU utilization: $GPU_UTIL"
echo "  KV cache dtype: $KV_DTYPE"
echo "  GPU device: $GPU"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization "$QUANTIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000
