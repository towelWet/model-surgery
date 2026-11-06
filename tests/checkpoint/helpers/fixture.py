"""Write tiny synthetic safetensors fixtures, never model evaluation evidence."""
import json
import struct
from pathlib import Path


def write_checkpoint(folder, tensors=None, config=None, data=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if tensors is None:
        tensors = {"net.layers.2.ffn.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
    payload = json.dumps(tensors).encode()
    size = max(entry["data_offsets"][1] for entry in tensors.values())
    (folder / "weights.safetensors").write_bytes(struct.pack("<Q", len(payload)) + payload + (bytes(size) if data is None else data))
    (folder / "config.json").write_text(json.dumps(config or {"model_type": "fixture", "num_hidden_layers": 1}))
    return folder
