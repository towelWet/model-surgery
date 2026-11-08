# pipeline.py — two-stage site gen: structure draft, then polish (optional small draft model).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Stage 1: small model (swap server to 4B) OR same model with --stage1-only.
# Stage 2: polish on whatever tensor-array serves. Manual swap: profile/swap_model.sh PATH.
#
# Usage:
#   python3 pipeline.py [--stage1-only] [--max-tokens1 1024] [--max-tokens2 3072]
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile.site.gen._api import complete
from pipeline.profile.site.gen._validate import score_html
from pipeline.heal.site._msgs import stage1_messages, stage2_messages

FOUR_B = os.environ.get("SURGERY_DRAFT_MODEL", "")         # small model for stage-1 draft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-only", action="store_true")
    p.add_argument("--max-tokens1", type=int, default=1024)
    p.add_argument("--max-tokens2", type=int, default=3072)
    p.add_argument("--out", default="db/site_pipeline_last.html")
    args = p.parse_args()

    print("Stage 1: structure (serve 4B first for speed — see profile/site_speed.toml)")
    s1 = complete(stage1_messages(), max_tokens=args.max_tokens1)
    print(f"  stage1 {s1['completion_tokens']} tok @ {s1['tok_per_s']:.1f} tok/s")
    if args.stage1_only:
        html = s1["content"]
    else:
        print("Stage 2: polish on currently served model")
        s2 = complete(stage2_messages(s1["content"]), max_tokens=args.max_tokens2)
        print(f"  stage2 {s2['completion_tokens']} tok @ {s2['tok_per_s']:.1f} tok/s")
        html = s2["content"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)                                       # html page for manual eyeballing
    print(json.dumps({"quality": score_html(html), "four_b_hint": FOUR_B}, indent=2))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
