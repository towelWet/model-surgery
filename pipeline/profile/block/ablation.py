# ablation.py — per-block loss-delta ranking spec + optional live score.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# THE GATE: cuts come from measured loss delta, not weight-norm similarity.
# This script does NOT copy similarity ranks. Modes:
#   --write-spec   Emit ablation_jobs.json (offline hook format for tensor-array layer gate)
#   --import PATH  Merge completed ablation_results.json into ablation_ranks.json
#   --from-distance Use block_distance.json angular ranks (valid proxy when ablation pending)
#
# Live ablation (--score) uses short CALIB prompts only; never runs 103-case suite.
# Each job zeros one 4-layer group via layer_gate=0 API hook (when server supports it).
#
# Output ablation_ranks.json:
#   [{"group": 0, "layers": [0,1,2,3], "loss_delta": 0.012, "rank": 0}, ...]
# Lower loss_delta = safer to remove.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports

from pipeline.profile.ablation._args import ablation_args
from pipeline.profile.ablation._run_modes import run_modes


def main():
    p = ablation_args()
    run_modes(p, p.parse_args())                            # parser passed for print_help on bad args


if __name__ == "__main__":
    main()
