# entry.py — CLI: evolutionary search of merge weights / density / lambda against held-out score.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <base> <ft>... --work DIR --method ties --score "cmd {model}" [--evals 8]
#   Score command prints a number (lower = better); combine domains inside it if needed.
#   Every candidate stays in DIR/cand-NN (disk: evals x model size); evolve.json logs all.
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.soup._score_cmd import score_cmd
from pipeline.surgery.evolve._genome_merge import genome_merge
from pipeline.surgery.evolve._evolve import evolve

mx.set_default_device(mx.cpu)                               # merge math on host; GPU left to scorer

p = argparse.ArgumentParser()
p.add_argument("base")
p.add_argument("fts", nargs="+")
p.add_argument("--work", required=True)
p.add_argument("--method", default="ties")
p.add_argument("--score", required=True)
p.add_argument("--evals", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
work = str(validate_destination(a.work, sources=[a.base, *a.fts]))
os.makedirs(work)


def evaluate(genome, i):
    out = os.path.join(work, f"cand-{i:02d}")
    genome_merge(a.base, a.fts, a.method, genome, out, a.seed)
    s = score_cmd(a.score, out)
    print(f"eval {i}: {s:.4f} {json.dumps(genome)}", flush=True)
    return s


start = {"weights": [1.0] * len(a.fts), "density": [0.5], "lambda": [1.0]}
best, score, log = evolve(start, evaluate, a.evals, a.seed)
json.dump({"method": a.method, "best": best, "score": score, "log": log}, open(os.path.join(work, "evolve.json"), "w"), indent=1)
print(f"best {score:.4f}: {json.dumps(best)}")
