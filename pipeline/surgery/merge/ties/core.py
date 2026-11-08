# merge/ties/core.py — merge models of one base: DARE, TIES, DARE-TIES, task arithmetic, linear soup, SLERP.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage:
#   python3 merge/ties/core.py <base> <finetune>... <dest> [--method dare|ties|dare_ties|task_arithmetic|linear|slerp|breadcrumbs[_ties]|della[_linear]|model_stock|nuslerp|karcher]
#                        [--density 0.5] [--lambda 1.0] [--weights 1.0 ...] [--seed 42] [--bf16-only]
#   Two paths + dest keeps the old DARE single-task-vector behaviour (--weight-b still accepted).
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.merge._ties import run_merge
from pipeline.surgery.merge._ties.multi_np import METHODS
from pipeline.surgery.checkpoint.destination import validate_destination
import mlx.core as mx

mx.set_default_device(mx.cpu)                             # host math; GPU shared with other apps hit ~2s watchdog on big dequant/save


def main():
    p = argparse.ArgumentParser(description="Merge models of one base: DARE, TIES, task arithmetic, soup, SLERP")
    p.add_argument("paths", nargs="+", help="base, one or more fine-tunes, then output dir")
    p.add_argument("--method", default="dare", choices=METHODS)
    p.add_argument("--density", type=float, default=0.5, help="DARE keep probability / TIES keep fraction")
    p.add_argument("--lambda", dest="lam", type=float, default=1.0, help="task-vector scale; SLERP t (0 = base, 1 = other)")
    p.add_argument("--weights", type=float, nargs="*", help="one weight per fine-tune (default 1.0 each)")
    p.add_argument("--weight-b", type=float, help="legacy: weight for single fine-tune")
    p.add_argument("--gamma", type=float, default=0.01, help="breadcrumbs: top outlier fraction dropped")
    p.add_argument("--epsilon", type=float, default=0.1, help="DELLA: keep-probability spread around density")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16-only", action="store_true", help="skip requant; write bf16 weights")
    args = p.parse_args()
    if len(args.paths) < 3:
        p.error("need base, at least one fine-tune, and dest")
    *sources, dest = args.paths
    weights = args.weights or [args.weight_b if args.weight_b is not None else 1.0] * (len(sources) - 1)
    validate_destination(dest, sources=sources)             # before any writes
    spec = {"method": args.method, "density": args.density, "lambda": args.lam, "weights": weights,
            "gamma": args.gamma, "epsilon": args.epsilon}
    run_merge(sources, dest, spec, args.seed, args.bf16_only)


if __name__ == "__main__":
    main()
