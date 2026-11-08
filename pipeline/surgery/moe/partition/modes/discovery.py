# ../../moe/partition/modes/discovery.py — find partition modes via the generic cli plugin loader.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Modes are modes/**/manifest.py carrying MANIFEST = {"name", "help"?, "run"}:
# run(mass, prof, p) -> (shared_ids, experts[E, expert_width]); no central name list.
import os

from pipeline.cli.plugins.discovery import load_plugins


def discover_modes():
    modes = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    return load_plugins(modes, "MANIFEST", "**/manifest.py")   # manifests only, not impl files
