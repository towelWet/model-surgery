# _write_report.py — write one suite's scores as flat TOML (gate + history input).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Hand-rolled TOML (numbers/strings only) so no tomli_w dependency is needed.
import datetime
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.eval.evidence.fingerprint import fingerprint
from pipeline.eval.evidence.format import format_scalar


def write_report(path, suite, model, metrics, by_category):
    """metrics: {name: float|int}; by_category: {cat: {name: value}}."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [f'suite = {format_scalar(suite)}',
             f'model = {format_scalar(model)}',                           # caller label, not verified identity
             f'when  = "{datetime.datetime.now().isoformat(timespec="seconds")}"', ""]
    lines += ["diagnostic_only = true", 'grader = "legacy-diagnostic"',
              f'evaluation_fingerprint = "{fingerprint()}"']
    lines += [f"{k} = {format_scalar(v)}" for k, v in metrics.items()]
    for cat, vals in by_category.items():                                # one table per case category
        lines += ["", f"[category.{format_scalar(cat)}]"]
        lines += [f"{k} = {format_scalar(v)}" for k, v in vals.items()]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
