# stack/_config.py — config for stacked model: new layer count + per-layer lists re-sliced.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Any list whose length equals a source's layer count (layer_types, window sizes, ...) is
# per-layer metadata: it is rebuilt from the source entry of every stacked layer.


def stack_config(cfgs, slices):
    out = dict(cfgs[0])
    n0 = cfgs[0]["num_hidden_layers"]
    for key, val in cfgs[0].items():
        if isinstance(val, list) and len(val) == n0:
            out[key] = [cfgs[s][key][l] for s, l in slices]
    out["num_hidden_layers"] = len(slices)
    return out
