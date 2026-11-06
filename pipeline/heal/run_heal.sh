#!/usr/bin/env bash
# run_heal.sh — corpus → optional teacher-cache → tensor-array-train LoRA (tool_distill) → fuse.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Was: mlx_lm lora via train_lora_distill.py (banned; quarantined in db/rejected/mlx-lm-heal).
set -euo pipefail
ROOT="${MODEL_SURGERY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # pipeline/
STUDENT="${HEAL_STUDENT:?set HEAL_STUDENT}"
DEST="${HEAL_DEST:?set HEAL_DEST}"
DATA="${HEAL_DATA:-$ROOT/heal/data/distill_corpus.jsonl}"
ADAPTER="${HEAL_ADAPTER:-$ROOT/heal/adapters/$(basename "$DEST")}"
RESTORE="${HEAL_RESTORE:-}"                                # model to swap back to after caching
CACHE="${HEAL_TEACHER_CACHE:-$ROOT/heal/cache/kd_top32.npz}"
ITERS="${HEAL_ITERS:-800}"
[[ "$DEST" == *Tau* ]] || { echo "HEAL_DEST must include Tau"; exit 1; }
python3 "$ROOT/heal/gen/corpus.py" --prompts-only --out "$DATA"   # was heal/gen_corpus.py (moved)
if [[ "${HEAL_SKIP_CACHE:-0}" != 1 ]]; then
  python3 "$ROOT/heal/cache_teacher_logits.py" --data "$DATA" --out "$CACHE" \
    ${RESTORE:+--restore "$RESTORE"} ${HEAL_TEACHER:+--teacher "$HEAL_TEACHER" --serve}  # --out: train reads same file
fi
python3 "$ROOT/heal/tool_distill.py" --student "$STUDENT" --data "$DATA" \
  --adapter-path "$ADAPTER" --iters "$ITERS" --teacher-cache "$CACHE"
python3 "$ROOT/heal/fuse_adapter.py" --base "$STUDENT" --adapter "$ADAPTER" --out "$DEST"
echo "Done -> $DEST"
