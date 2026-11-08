# usage.py — retired placeholder; refuses to write expert usage.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# DISABLED 2026-11-07. The old body wrote a UNIFORM score for every expert to
# profile/expert/expert_usage.json — the same path routing_profile.py writes real data to.
# prune.py accepts any valid permutation, so running this silently replaced measured
# routing with fake data and pruned experts 0..N-1 by index. Never restore that body.
# Real per-expert usage: routing_profile.py (runtime HOPE stats; prune_order least-used first).
import sys

sys.exit("usage.py is retired: it only produced fake uniform usage. "
         "Run expert/routing_profile.py <model_dir> --stats <stats> for measured routing.")
