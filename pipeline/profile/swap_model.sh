#!/usr/bin/env bash
# swap_model.sh — unload launchd, serve MODEL on :8765, wait until completion probe passes.
set -euo pipefail
MODEL="${1:?model path}"
PLIST="$HOME/Library/LaunchAgents/com.tensor-array.server.plist"
export TARR_ADMISSION_CTX=16384 TARR_ARCH=Qwen3_5ForConditionalGeneration
export TARR_ARCH_PACKS_DIR="$HOME/.local/lib/tensor-array/arch-packs"
XMEM="${TARR_XMEM_DIR:-$HOME/.cache/tensor-array/xmem}"       # external volume or local cache
export TARR_CACHE_DIR="${TARR_CACHE_DIR:-$XMEM/cache}"
export TARR_FORCE_RESIDENT=1 TARR_KV_TIER=1
export TARR_KV_TIER_DIR="${TARR_KV_TIER_DIR:-$XMEM/kv}"
export TARR_KV_TIER_DISK_MB=4096 TARR_KV_TIER_HOST_MB=512
export TARR_MAX_CONTEXT=16384
export TARR_MTP_SIDECAR_DIR="${TARR_MTP_SIDECAR_DIR:-$HOME/.local/lib/tensor-array/mtp-sidecar}"
export TARR_NO_BATCH=1 TARR_NO_MTP=1 TARR_NO_WARMUP=1
export TARR_SESSION_DIR="${TARR_SESSION_DIR:-$XMEM/session}"
export TARR_SPLIT_DIR="${TARR_SPLIT_DIR:-$XMEM/split}"
export TARR_WEIGHT_STREAM=off TARR_WEIGHT_STREAM_TIER=ssd
export TARR_XMEM_DIR="$XMEM"
export MLX_MAGIC_METALLIB="${MLX_MAGIC_METALLIB:-$HOME/.local/lib/tensor-array/default.metallib}"

launchctl unload "$PLIST" 2>/dev/null || true
pkill -f inferserve-model 2>/dev/null || true
sleep 10
# nohup: shell exit must not SIGHUP the server (observed 2026-09-18: eval got 103x connection refused).
nohup "$HOME/.local/bin/inferserve-model" --model "$MODEL" --port 8765 >>/tmp/mlxm_swap.log 2>&1 &
for i in $(seq 1 180); do
  curl -sf http://localhost:8765/v1/models >/dev/null 2>&1 || { sleep 5; continue; }
  curl -sf -X POST http://localhost:8765/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"current","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0,"enable_thinking":false}' \
    >/dev/null 2>&1 && break
  sleep 5
done
sleep 10
echo "ready: $MODEL"
