# _merge_linear.py — uniform linear merge of models onto base's tensor layout.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.surgery.merge._ties import run_merge


def merge_linear(base, models, out):
    spec = {"method": "linear", "density": 1.0, "lambda": 1.0, "weights": [1.0] * len(models)}
    run_merge([base, *models], out, spec, 0, False)          # seed unused: linear has no randomness
