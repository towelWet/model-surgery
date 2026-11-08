# _mlx_load.py — single entry point for in-process model loading used by offline profilers.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Used to call mlx_lm.load. mlx_lm / mlx_vlm are banned (tensor-array only, ever), so this now
# REFUSES instead of loading. Every profiler that reaches here (block/distance, mlp_stats,
# ablation --score, expert routing capture, layer taps) needs an tensor-array server route first:
#   hidden states per layer · NLL with skipped layers · MLP m_in/act taps · router top-k capture.
# Verified 2026-11-06: tensor-array src/server/route has none of these (only chat, embeddings,
# models, status, reload, audio). Add them in tensor-array; do not re-enable mlx_lm here.
import sys


def load(path):
    """Refuse: in-process mlx_lm loading is banned; see header for the tensor-array routes needed."""
    sys.exit(f"profile/_mlx_load: refusing to load {path} with mlx_lm (banned). "
             "This profiler needs an tensor-array route — see pipeline/profile/_mlx_load.py header.")
