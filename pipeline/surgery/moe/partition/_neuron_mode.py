# ../moe/partition/_neuron_mode.py — compatibility shim; implementation lives in modes/neuron/clustering.py.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.surgery.moe.partition.modes.neuron.clustering import neuron_mode  # noqa: F401  re-export
