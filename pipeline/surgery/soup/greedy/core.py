# greedy/core.py — CLI: greedy Model Soup over fine-tunes of one base.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 greedy/core.py <base> <cand>... --work DIR --score "cmd {model}"
#   Trial soups are written under --work (new dir); best soup path is printed, not moved.
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from functools import partial
from pipeline.surgery.soup._merge_linear import merge_linear
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.soup.greedy._core import greedy_soup
from pipeline.surgery.soup._score_cmd import score_cmd

mx.set_default_device(mx.cpu)                               # merge math on host; GPU left to scorer

p = argparse.ArgumentParser()
p.add_argument("base")                                     # layout source for merges
p.add_argument("cands", nargs="+")
p.add_argument("--work", required=True)
p.add_argument("--score", required=True)
a = p.parse_args()
work = str(validate_destination(a.work, sources=[a.base, *a.cands]))
os.makedirs(work)
soup, best, best_dir, log = greedy_soup(a.cands, work, partial(merge_linear, a.base),
                                        partial(score_cmd, a.score))
json.dump({"soup": soup, "best": best, "best_dir": best_dir, "log": log}, open(os.path.join(work, "soup.json"), "w"), indent=1)
print(f"soup of {len(soup)}: score {best:.4f} at {best_dir}")
