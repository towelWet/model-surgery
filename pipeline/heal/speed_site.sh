#!/usr/bin/env bash
# speed_site.sh — optimal env, 4B skeleton → Tau polish, timing report.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FOUR_B="${SURGERY_DRAFT_MODEL:?Set SURGERY_DRAFT_MODEL to small draft model path}"
TAU="${SURGERY_STUDENT_MODEL:?Set SURGERY_STUDENT_MODEL to main model path}"
S1="${ROOT}/db/speed_site_stage1.html"
S2="${ROOT}/db/speed_site_final.html"
export TARR_WEIGHT_STREAM=off TARR_FORCE_RESIDENT=1 TARR_NO_MTP=1 TARR_NO_WARMUP=1
export TARR_NO_BATCH=1 TARR_MAX_CONTEXT=16384
export MAX_TOKENS1="${MAX_TOKENS1:-1024}" MAX_TOKENS2="${MAX_TOKENS2:-3072}"

t0=$(date +%s)
echo "Stage 1: ${FOUR_B}"
"$ROOT/profile/swap_model.sh" "$FOUR_B"
python3 "$ROOT/heal/site_pipeline.py" --stage1-only --max-tokens1 "$MAX_TOKENS1" --out "$S1"
t1=$(date +%s)
echo "Stage 2: ${TAU}"
"$ROOT/profile/swap_model.sh" "$TAU"
python3 "$ROOT/heal/_speed_site_stage2.py" "$S1" "$S2"
t2=$(date +%s)
echo "timing_s: stage1=$((t1 - t0)) stage2=$((t2 - t1)) total=$((t2 - t0))"
