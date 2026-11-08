#!/usr/bin/env bash
# Idle tok/s poll: POST 8-token thinking-off Count to $PORT, time, log JSONL.
# Usage: bash profile/perf-monitor.sh [port] [interval_minutes] [tok/s_threshold]
set -euo pipefail
PORT=${1:-8765}
N="${2:-10}"
THRESH="${3:-${THRESH:-1.0}}"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/perf-log.jsonl"
trap 'exit 0' INT
BODY='{"model":"current","messages":[{"role":"user","content":"Count 1 to 5, digits only."}],"max_tokens":8,"temperature":0,"enable_thinking":false}'
while true; do
  T0=$(python3 "$DIR/monitor/clock.py")
  RESP=$(curl -sS --max-time 180 -H 'Content-Type: application/json' -d "$BODY" \
    http://127.0.0.1:${PORT}/v1/chat/completions || echo '{}')
  T1=$(python3 "$DIR/monitor/clock.py")
  LOG="$LOG" THRESH="$THRESH" T0="$T0" T1="$T1" RESP="$RESP" python3 "$DIR/monitor/record.py"
  sleep $((N * 60))
done
