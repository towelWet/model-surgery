# manifest.py — moeify split: default goal. Forwards to `ms weights moe-split` (one implementation).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))     # project root for pipeline.* imports
from pipeline.cli.plugins.execution import exec_fresh


def run(argv):
    verb = Path(__file__).resolve().parents[4] / "storage/commands/moe/split/manifest.py"
    return exec_fresh(verb, "ms_moeify_split").MANIFEST["run"](argv)   # no copy of split logic here


MANIFEST = {"name": "split", "help": "MoE -> split-file MoE (= ms weights moe-split)", "run": run}
