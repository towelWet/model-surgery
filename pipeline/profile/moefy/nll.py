# nll.py — mean next-token NLL of served model on text pieces calibration never saw.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# capture.py uses pieces 0..N-1; this scores pieces N..N+M-1 of same deterministic list,
# so dense and converted checkpoints are compared on identical, unseen text.
# Runtime route /v1/layer_ablation with no zero_layers = plain teacher-forced NLL.
#
# Usage: python3 nll.py [--skip 64] [--count 32] [--max-tokens 256]
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile._api import _post
from pipeline.profile.moefy._pieces import pieces

p = argparse.ArgumentParser()
p.add_argument("--skip", type=int, default=64)             # pieces used by capture.py
p.add_argument("--count", type=int, default=32)            # unseen pieces to score
p.add_argument("--max-tokens", type=int, default=256)
a = p.parse_args()

texts = pieces(a.skip + a.count)[a.skip:]
reply = _post("layer_ablation", {"prompts": texts, "max_tokens": a.max_tokens})
print(f"heldout pieces={reply['prompts']} mean_nll={reply['baseline_nll']:.4f}")
