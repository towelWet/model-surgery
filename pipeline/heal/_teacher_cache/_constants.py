# _constants.py — teacher-cache settings shared by harvest helpers; no logic.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Output format (NPZ + sidecar JSONL, consumed by train_lora_distill.py --teacher-cache):
#   offsets  int32 [N+1]     row boundaries into flat position arrays
#   idx      int32 [T, K]    top-K token ids per assistant position (K=32)
#   prob     float32 [T, K]  natural-log prob of each top-K id (from API logprobs)
#   label    int32 [T]       ground-truth token id at each position (teacher-forced)
#   weight   float32 [T]     per-position loss weight (body=1.0, last8 ramp, eos=2.0)
#   meta.jsonl (parallel):   {"row": i, "n_pos": P, "messages": [...]} per kept row
import os

TOP_K = 32                                                  # tensor-array top_logprobs per position
DEFAULT_XMEM = os.environ.get("SURGERY_CACHE_DIR", "heal/cache/teacher_logits")  # cwd-relative default
FALLBACK_CACHE = os.path.join(os.path.dirname(__file__), "cache")  # last resort: inside package
