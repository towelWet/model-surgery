# manifest.py — neuron mode: free neuron-level partition (best fidelity, needs requant).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # project root for pipeline.* imports
from pipeline.surgery.moe_partition.modes.neuron.clustering import neuron_mode


def run(mass, prof, p):
    return neuron_mode(mass, prof, p.experts, p.expert_width, p.shared_width, p.width)


MANIFEST = {"name": "neuron", "help": "free per-neuron partition; forces down_proj requant", "run": run}
