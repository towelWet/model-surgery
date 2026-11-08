# _scorers.py — re-export ablation scoring functions from split files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.profile.ablation._layer_ablation import layer_ablation                    # single-setting scorer
from pipeline.profile.ablation._score_groups import score_groups                        # full group ranker
