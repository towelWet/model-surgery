# manifest.py — group mode: quant-group-safe expert partition (keeps group_size runs intact).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # project root for pipeline.* imports
from pipeline.surgery.moe_partition.modes.group.partition import group_mode


def run(mass, prof, p):
    return group_mode(mass, prof, p.experts, p.expert_width, p.shared_width, p.group_size, p.width)


MANIFEST = {"name": "group", "help": "partition keeping quantization groups whole", "run": run}
