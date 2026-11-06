# _slice_mlp.py — cut one dense-MLP tensor down to first new_interm neurons; others pass through.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# down_proj input dim is cut PROPORTIONALLY to its own width, so it works for bf16 weights,
# packed uint32 weights at any bit-width (mixed recipes too) and scales/biases alike.
# Old code derived packed cols from a global "bits" (default 3): wrong for bf16 and mixed-bit models.

GATE_UP = (".mlp.gate_proj.", ".mlp.up_proj.")             # dense MLP only; MoE switch_mlp/shared_expert untouched
DOWN = ".mlp.down_proj."


def slice_mlp(key, arr, old_interm, new_interm):
    if any(m in key for m in GATE_UP):
        return arr[:new_interm]                             # output-neuron dim0: keep first rows
    if DOWN in key:
        cols = arr.shape[1] * new_interm                    # same fraction of whatever this tensor stores
        if cols % old_interm != 0:
            raise ValueError(f"{key}: width {arr.shape[1]} can't cut to {new_interm}/{old_interm}")
        return arr[:, : cols // old_interm]                 # input-neuron dim1
    return arr                                              # attention, norms, embeddings unchanged
