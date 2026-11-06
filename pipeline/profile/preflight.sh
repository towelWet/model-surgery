#!/usr/bin/env bash
# Preflight — checks only; starts nothing.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
set -u
f=0
DIR="$(cd "$(dirname "$0")" && pwd)"                            # profile dir
SRV="${TENSOR_ARRAY_SERVER:-$(command -v inferserve-model 2>/dev/null || echo "")}" # server binary
GOD="${GOD_BIN:-$(command -v god 2>/dev/null || echo "")}"      # god binary (optional)
CFG="${SURGERY_BASELINE_CONFIG:-}"                              # baseline model config.json
PORT="${TARR_EVAL_PORT:-8765}"                                  # eval server port
RUN="$DIR/benchmark/run.sh"
RPT="$DIR/benchmark/report.sh"
FB="$DIR/fast/bench.py"
pass() { echo "PASS $1"; }

if test -n "$SRV" && test -f "$SRV"; then pass "inferserve-model exists"; else f=1; echo "FAIL inferserve-model missing"; fi
if test -n "$CFG" && test -f "$CFG"; then pass "baseline config.json"; else echo "SKIP baseline config (SURGERY_BASELINE_CONFIG not set)"; fi
if test -n "$GOD" && test -f "$GOD"; then pass "god binary exists"; else echo "SKIP god binary (GOD_BIN not set)"; fi
if test -x "$RUN"; then pass "benchmark/run.sh executable"; else f=1; echo "FAIL benchmark/run.sh not executable"; fi
if test -x "$RPT"; then pass "benchmark/report.sh executable"; else f=1; echo "FAIL benchmark/report.sh not executable"; fi
if test -x "$FB"; then pass "fast/bench.py executable"; else f=1; echo "FAIL fast/bench.py not executable"; fi
if nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1; then pass ":$PORT listening"; else f=1; echo "FAIL :$PORT not listening"; fi
exit "$f"
