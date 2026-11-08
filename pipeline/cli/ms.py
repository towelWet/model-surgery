#!/usr/bin/env python3
# ms.py — model-surgery CLI: `ms <command> [args]`; commands are discovered from cli/commands/.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Usage: python3 pipeline/cli/ms.py            (lists commands)
#        python3 pipeline/cli/ms.py doctor     (repo health checks)
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.cli._load_plugins import load_plugins

cmds = load_plugins(os.path.join(HERE, "commands"), "COMMAND")
if len(sys.argv) < 2 or sys.argv[1] not in cmds:
    print("ms <command> [args]\n")
    for name, c in cmds.items():
        print(f"  {name:<10} {c['help']}")
    sys.exit(0 if len(sys.argv) < 2 else 2)                 # unknown command = usage error
sys.exit(cmds[sys.argv[1]]["run"](sys.argv[2:]) or 0)
