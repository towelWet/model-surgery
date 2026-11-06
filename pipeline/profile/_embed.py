# _embed.py — single-prompt embedding via tensor-array API.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np

from _api import _post


def embed(prompt):
    """Single-prompt embedding; returns float32 numpy array [hidden_size]."""
    resp = _post("embeddings", {
        "model": "current",
        "input": prompt,
    })
    return np.array(
        resp["data"][0]["embedding"],
        dtype=np.float32,
    )
