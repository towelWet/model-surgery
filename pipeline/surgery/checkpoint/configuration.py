"""Read checkpoint configuration without inventing quantization defaults."""
import json
from pathlib import Path


def read_configuration(source):
    cfg = json.loads((Path(source) / "config.json").read_text())
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint config must be an object")
    return cfg
