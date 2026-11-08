# gguf.py — `ms gguf <verb>`: GGUF editing verbs from gguf/commands/**/manifest.py.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from pipeline.cli.plugins.discovery import load_plugins      # same plugin seam as `ms weights`


def run(argv):
    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "gguf", "commands")
    verbs = load_plugins(folder, "MANIFEST", pattern="**/manifest.py")
    if not argv or argv[0] not in verbs:
        print("ms gguf <verb> [args]\n")
        for name, v in verbs.items():
            print(f"  {name:<14} {v.get('help', '')}")
        return 0 if not argv else 2                          # unknown verb = usage error
    return verbs[argv[0]]["run"](argv[1:]) or 0


COMMAND = {"name": "gguf", "help": "edit GGUF files: inspect, prune-layers, prune-experts", "run": run}
