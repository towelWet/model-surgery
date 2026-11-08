#!/usr/bin/env python3
# bench/core.py — decode probe: thinking-off Count 1-5, N runs, TTFT + decode tok/s, exit 1 below threshold.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile.fast._pass_fail import fmt_result
from pipeline.profile.fast._stream_run import one_run
from pipeline.profile.fast.bench.status import show                                  # direct sibling import, not dynamic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("TARR_EVAL_PORT", "8765")))
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--threshold", type=float, default=1.5)     # tok/s floor for PASS
    args = p.parse_args()
    show(args.port)
    rates, ttfts = [], []
    for i in range(1, args.n + 1):
        ttft, tps = one_run(args.port)
        ttfts.append(ttft); rates.append(tps)
        print(f"run {i} TTFT {ttft:.3f}s decode {tps:.2f} tok/s")
    mean_t = sum(ttfts) / max(len(ttfts), 1)
    mean = sum(rates) / max(len(rates), 1)
    print(f"mean TTFT {mean_t:.3f}s decode {mean:.2f} tok/s")
    print(fmt_result(mean, args.threshold))
    sys.exit(0 if mean >= args.threshold else 1)


if __name__ == "__main__":
    main()
