# probe/_hidden.py — pooled residual stream per layer via runtime /v1/internals/hidden_states.
# Values are residual dimensions (hidden_size), not MLP intermediate neurons. Server probe gate
# (TARR_PROBES) must be on; archs without probe hooks answer 501.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.profile._api import _post


def probe_hidden(prompt, layer=None):
    """Mean-pooled residual entering each layer (key N = last layer output) for one prompt."""
    body = {"input": prompt, "pool": "mean"}
    if layer is not None:
        body["layer"] = layer                              # restrict to single layer
    return _post("internals/hidden_states", body)     # {"activations": {"<layer>": [H]}}
