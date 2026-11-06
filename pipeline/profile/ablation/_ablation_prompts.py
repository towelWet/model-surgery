# _ablation_prompts.py — short CALIBRATION prompts for ablation scoring; data only, no logic.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Live ablation (--score) uses short CALIB prompts only; never runs 103-case suite.
# Each job zeros one 4-layer group via layer_gate=0 API hook (when server supports it).

CALIBRATION = [
    "The quick brown fox jumps",
    "def merge_sorted(a, b):",
    "import json\nwith open('/tmp/x'",
    "SELECT id FROM orders WHERE",
    "The capital of France is",
]
