"""Preserve the entire ancillary file tree; write the external loader's config last."""
import json
import shutil
from pathlib import Path


def copy_checkpoint_assets(source, dest, assets, config=None):
    for rel in sorted(assets - {"model.safetensors.index.json", "config.json"}):
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / rel, path)
    if config is not None:
        (dest / "config.json").write_text(json.dumps(config, indent=2))
    elif "config.json" in assets:
        shutil.copyfile(Path(source) / "config.json", dest / "config.json")
