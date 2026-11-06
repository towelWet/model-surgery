# usage.py — measure per-expert routing frequency via API probe.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Requires MoE model running on :8765 with router-logit logging enabled.
# Output: profile/expert_usage.json for surgery/expert_prune.py.
# NOTE: placeholder until tensor-array exposes /v1/internals/router_logits.
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROBES = [                                 # diverse calibration prompts
    "Read the file at /etc/hosts",
    "List all running processes",
    "Write a function to sort a list in Python",
    "What is the speed of light?",
    "def fibonacci(n): return",
]

N_LAYERS  = 48                            # update for different depth
N_EXPERTS = 64                            # update to match moefy --experts value

print("NOTE: requires tensor-array /v1/internals/router_logits endpoint (not yet live).")
print("Writing uniform-usage placeholder.")

# Placeholder: uniform score per expert per layer.
# Replace body of this loop with real routing data when endpoint is available.
usage = {
    str(li): sorted(
        [{"expert": e, "score": 1.0 / N_EXPERTS} for e in range(N_EXPERTS)],
        key=lambda x: x["score"],         # ascending: lowest-scored first = prune candidates
    )
    for li in range(N_LAYERS)
}

out_path = os.path.join(os.path.dirname(__file__), "expert_usage.json")
with open(out_path, "w") as f:
    json.dump(usage, f, indent=2)
print(f"Saved: {out_path}")
