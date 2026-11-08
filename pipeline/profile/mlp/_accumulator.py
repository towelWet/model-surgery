# mlp/_accumulator.py — per-layer running sums for MoE conversion stats (see mlp/stats.py header).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# One class = one unit: on_layer is the capture_mlp callback, save writes the .npy files.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.profile.accumulator.initialise import initialise
from pipeline.profile.accumulator.on_layer import on_layer
from pipeline.profile.accumulator.save import save


class MlpAccumulator:
    __init__ = initialise
    on_layer = on_layer
    save = save
