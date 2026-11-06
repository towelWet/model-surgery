# _ablation_paths.py — ablation paths and re-exports; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os

SPEC_PATH = os.path.join(os.path.dirname(__file__), "ablation_jobs.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "ablation_results.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "ablation_ranks.json")

from _n_layers import n_layers_from_argv                 # re-export for importers
from _write_spec import write_spec                       # re-export for importers
