# _moe_config.py — write dest config.json describing the new MoE layout (shared + E experts, top-K).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def write_moe_config(dest, cfg, tc, E, K, S, H):
    raise NotImplementedError("MoE config requires a verified architecture adapter and trained routers")
