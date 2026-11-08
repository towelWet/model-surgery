# _embed.py — single-prompt embedding via tensor-array API.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np

from pipeline.profile._api import _post
from pipeline.runtime.settings import runtime_settings


def embed(prompt):
    """Single-prompt embedding; returns float32 numpy array [hidden_size]."""
    resp = _post("embeddings", {
        "model": runtime_settings()["http"]["model"],
        "input": prompt,
    })
    return np.array(
        resp["data"][0]["embedding"],
        dtype=np.float32,
    )
