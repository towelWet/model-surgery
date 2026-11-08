# manifest.py — moeify convert: dispatch dense -> MoE conversion to the converter named by the arch's pack.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Args after verb pass through verbatim to <converter>/entry.py (BF16 QUANT CALIB DEST [--experts --topk ...]).
import json
import subprocess
import sys
from pathlib import Path
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))))             # verb/commands/moefy/surgery/pipeline -> root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.moefy._target_packs import target_packs


def run(argv):
    if len(argv) < 4:
        print("usage: ms moeify convert BF16 QUANT CALIB DEST [converter flags]"); return 2
    arch = (json.loads((Path(argv[0]) / "config.json").read_text()).get("architectures") or ["?"])[0]
    pack = target_packs().get(arch)
    if pack is None:
        print(f"{arch}: no target pack (ms moeify plan {argv[0]})"); return 1
    entry = os.path.join(ROOT, pack["converter"], "entry.py")   # converter = repo-relative dir (TOML)
    print(f"{arch} -> {pack['converter']}/entry.py ({pack['pack']})")
    return subprocess.call([sys.executable, entry, *argv])  # converter owns its own arg parsing


MANIFEST = {"name": "convert", "help": "dense -> MoE via arch target pack's converter", "run": run}
