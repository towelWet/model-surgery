# layer/_inputs.py — captured inputs for all 7 linears of one decoder layer (dense-model capture).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Simplification vs reference GPTQ/AWQ: inputs come from DENSE model capture, not re-fed through
# already-quantized earlier layers; down_proj input recomputed from dense gate/up.
from pipeline.surgery.moefy._load_calib import load_calib
from pipeline.surgery.moefy._neuron_acts import neuron_acts


def layer_inputs(files, li, wg, wu):
    xa, xo, xm = load_calib(files, li, "attn"), load_calib(files, li, "o"), load_calib(files, li)
    return {"q": xa, "k": xa, "v": xa, "o": xo, "gate": xm, "up": xm, "down": neuron_acts(xm, wg, wu)}
