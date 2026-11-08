# gen/bench.py — end-to-end site task: wall time, tok/s, HTML sanity.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage:
#   python3 gen/bench.py [--max-tokens 4096] [--scaffold] [--out db/site_gen_last.json]
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile.site.gen._api import complete
from pipeline.profile.site.gen._task import merge_scaffold
from pipeline.profile.site._prompt import site_prompt
from pipeline.profile.site._result import site_result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--scaffold", action="store_true", help="prepend HTML skeleton")
    p.add_argument("--session-id", default="site-gen-bench")
    p.add_argument("--out", default="db/site_gen_last.json")
    p.add_argument("--port", type=int, default=int(os.environ.get("TARR_EVAL_PORT", "8765")))
    args = p.parse_args()

    system, user = site_prompt(args.scaffold)

    print(f"Site-gen bench (tensor-array :{args.port}, thinking off)...")
    run = complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=args.max_tokens,
        session_id=args.session_id,
        port=args.port,
    )
    html = merge_scaffold(run["content"]) if args.scaffold else run["content"]
    result = site_result(args, run, html)
    est_15k_min = 15000 / max(run["tok_per_s"], 0.1) / 60
    print(json.dumps(result, indent=2))
    print(f"  ~15k tokens at this rate: {est_15k_min:.1f} min")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
