# _probe_hidden.py — hidden state probe via tensor-array API.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _api import _post


def probe_hidden(prompt, layer=None):
    """Query tensor-array hidden_states endpoint for one prompt."""
    body = {"model": "current", "input": prompt}
    if layer is not None:
        body["layer"] = layer                              # restrict to single layer
    return _post("hidden_states", body)
