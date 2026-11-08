# _rank.py — re-export ablation rank functions from split files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.profile.ablation._import_results import import_results               # import completed ablation JSON
from pipeline.profile.ablation._from_distance import from_distance                  # proxy from block_distance
