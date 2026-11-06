#!/usr/bin/env bash
# gpu-around-detect.sh — stdout metal|cpu. Default metal (share with Insta/video).
# pgrep Insta + %CPU is NOT contention. Video decode uses the media engine.
# cpu ONLY on proven GPU/display contention (no sudo): recoveryCount rise
# (GPU hang / Metal timeout) or live Bonsai log GPU-wait / command-buffer timeout.
set -eu
want=metal
st="${TMPDIR:-/tmp}/gpu-around-recovery.prev"
rec=$(ioreg -r -c IOAccelerator -d 1 2>/dev/null \
  | awk -F= '/recoveryCount/{gsub(/[^0-9]/,"",$2); print $2+0; found=1} END{if(!found) print 0}')
rec=${rec:-0}
prev=$(cat "$st" 2>/dev/null || true)
prev=${prev:-$rec}
prev=${prev:-0}
printf '%s\n' "$rec" >"$st"
if [ "$rec" -gt "$prev" ]; then want=cpu; fi
log=/tmp/bonsai-8766-evalbatch.log
if [ -f "$log" ] && grep -qiE 'metal timeout|MTLCommandBufferError|GPU hang|command buffer timed out|gpu wait' \
  <<<"$(tail -n 40 "$log" 2>/dev/null || true)"; then
  want=cpu
fi
echo "$want"
