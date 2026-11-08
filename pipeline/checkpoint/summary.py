"""Read checkpoint geometry and storage from metadata; never load tensor payloads."""
import collections
import json
import re
from pathlib import Path
from pipeline.checkpoint.tensors import tensor_index


def summarize(folder):
    folder = Path(folder).resolve(strict=True)
    config = json.loads((folder / "config.json").read_text())
    text = config.get("text_config", config)
    tensors = tensor_index(folder)
    shards = {loc[0] for loc in tensors.values()}
    layers = sorted({int(m[1]) for name in tensors for m in re.finditer(r"(?:^|\.)layers\.(\d+)\.", name)})
    counts = collections.Counter(loc[1]["dtype"] for loc in tensors.values())
    return {
        "model_type": text.get("model_type", config.get("model_type", "unspecified")),
        "architectures": ", ".join(config.get("architectures", [])),
        "configured_layers": text.get("num_hidden_layers", 0),
        "observed_layers": len(layers),
        "layer_ids": ", ".join(map(str, layers)),
        "experts": text.get("num_experts", text.get("num_local_experts", 0)),
        "tensors": len(tensors), "shards": len(shards),
        "storage_bytes": sum(path.stat().st_size for path in shards),
        "dtypes": ", ".join(f"{key}:{value}" for key, value in sorted(counts.items())),
    }
