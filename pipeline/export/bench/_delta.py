# _delta.py — print metric deltas of this bench run vs a saved baseline JSON.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os

KEYS = ("tok_per_s", "tool_selection_acc", "arg_validity")      # metrics worth comparing


def print_delta(result, baseline_path):
    """No-op when baseline path is empty or missing."""
    if not baseline_path or not os.path.exists(baseline_path):
        return
    with open(baseline_path) as f:
        base = json.load(f)
    print("\nDelta vs baseline:")
    for k in KEYS:
        if result.get(k) is None:
            continue                                            # metric absent this run
        d = result[k] - base.get(k, 0)
        print(f"  {k}: {'+' if d >= 0 else ''}{d:.3f}")
