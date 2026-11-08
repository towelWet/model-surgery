# _score_cmd.py — run user scoring command on a model dir; last number on stdout = score.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Any evaluator works (heldout-loss script, nll.py behind a server, ...): "{model}" in the
# template is replaced by model path. Lower must mean better.
import re
import shlex
import subprocess


def score_cmd(template, model_dir):
    out = subprocess.run(template.replace("{model}", shlex.quote(model_dir)), shell=True,
                         capture_output=True, text=True)
    nums = re.findall(r"-?\d+\.\d+", out.stdout)
    if out.returncode or not nums:
        raise RuntimeError(f"score command failed for {model_dir}: {out.stderr[-300:] or out.stdout[-300:]}")
    return float(nums[-1])
