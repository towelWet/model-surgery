#!/usr/bin/env bash
# gpu-around-poll.sh — write gpu-around.flag; hot-flip :8766 env on change.
# metal = share (Insta playing is fine). cpu = proven contention only.
# Never a second 27B. wired_gb=16 leaves display headroom on 32GB unified.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
FLAG="${TARR_GPU_AROUND_FLAG:-$HOME/.config/tensor-array/gpu-around.flag}"
mkdir -p "$(dirname "$FLAG")"
prev=$(head -1 "$FLAG" 2>/dev/null || true)
while true; do
  want=$("$DIR/gpu-around-detect.sh" || echo metal)
  tmp="$FLAG.tmp.$$"
  printf '%s\nwired_gb=16\n' "$want" >"$tmp"
  mv "$tmp" "$FLAG"
  if [ "$want" != "$prev" ]; then
    printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$want" >>"$DIR/gpu-around.log"
    bash "$DIR/gpu-around-hotenv.sh" "$want" || true
  fi
  prev=$want
  sleep "${GPU_AROUND_SLEEP:-2}"
done
