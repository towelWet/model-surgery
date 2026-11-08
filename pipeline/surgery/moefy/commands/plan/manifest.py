# manifest.py — moeify plan: read config.json, report which route applies (already MoE / pack / unsupported).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Read-only: never writes. First step of every /moeify run.
import json
import sys
from pathlib import Path
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))))             # verb/commands/moefy/surgery/pipeline -> root
sys.path.insert(0, ROOT)  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.moefy._target_packs import target_packs
from pipeline.storage.experts.expert_count import expert_count
from pipeline.storage.experts.load.layout import load_layout
from pipeline.storage.experts.load.strategies import load_strategies


def run(argv):
    if len(argv) != 1:
        print("usage: ms moeify plan MODEL_DIR"); return 2
    cfg = json.loads((Path(argv[0]) / "config.json").read_text())
    arch = (cfg.get("architectures") or ["?"])[0]
    try:
        e = expert_count(argv[0], load_layout())                 # MoE already? -> straight to split
        print(f"{arch}: already MoE, {e} routed experts")
        print(f"next: ms moeify split {argv[0]} DEST --by {{{','.join(load_strategies())}}} --dry-run")
        return 0
    except ValueError:
        pass                                                     # dense: look for a target pack
    pack = target_packs().get(arch)
    if pack is None:
        print(f"{arch}: no target pack; add pipeline/surgery/moefy/targets/<name>.toml"); return 1
    print(f"{arch}: dense, pack {pack['pack']} -> {pack.get('moe_arch', '?')} via {pack['converter']}/")
    print(f"status: {pack.get('status', 'unknown')}")
    print("next: profile capture -> ms moeify convert BF16 QUANT CALIB DEST --experts E --topk E (exact baseline)")
    return 0


MANIFEST = {"name": "plan", "help": "detect route for MODEL_DIR (read-only)", "run": run}
