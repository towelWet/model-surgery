# cache_teacher_logits.py — two-pass teacher top-32 logit cache for KD heal.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# MEMORY: only the teacher runs (tensor-array on :8765). Student is never loaded here.
# --serve swaps teacher in first; without it, teacher must already be resident on :8765.
# Cache root: $SURGERY_CACHE_DIR, else heal/cache/teacher_logits (cwd), else _teacher_cache/cache/.
#
# Usage:
#   python3 cache_teacher_logits.py --data heal/data/distill_corpus.jsonl
#   python3 cache_teacher_logits.py --data ... --teacher /path/to/27B --serve
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _teacher_cache import cache_args, cache_root, harvest, load_jsonl, swap_server, tokenizer_path


def main():
    args = cache_args()

    root = cache_root()
    os.makedirs(root, exist_ok=True)
    out_npz = args.out or os.path.join(root, "kd_top32.npz")
    out_meta = out_npz.replace(".npz", "_meta.jsonl")

    if args.serve:
        if not args.teacher:
            print("--serve requires --teacher PATH")
            sys.exit(1)
        swap_server(args.teacher)                          # teacher resident before harvest

    rows = load_jsonl(args.data)
    print(f"Corpus: {len(rows)} rows from {args.data}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path(args.teacher))  # local dir only

    harvest(rows, tok, out_npz, out_meta)

    if args.restore:
        swap_server(args.restore)                          # put student/production model back

if __name__ == "__main__":
    main()
