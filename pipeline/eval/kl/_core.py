# _core.py — re-export KL eval functions from split files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.eval.kl._build_parent_cache import build_parent_cache       # parent logprob cache (parent served)
from pipeline.eval.kl._load_parent_cache import load_cache                # read cache; refuses if missing
from pipeline.eval.kl._student import eval_student                        # student KL scoring
from pipeline.eval.kl._next_token import _next_token                      # single-token logprob sampler
