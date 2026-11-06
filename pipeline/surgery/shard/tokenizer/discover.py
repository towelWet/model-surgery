"""Discover tokenizer assets from format manifests and checkpoint references."""
import json
import tomllib
from pathlib import Path
from .references import referenced_assets


def discover_assets(source, formats=None):
    source = Path(source).resolve()
    formats = Path(formats) if formats else Path(__file__).parent / "formats"
    assets = set()
    for manifest in sorted(formats.rglob("*.toml")):
        spec = tomllib.loads(manifest.read_text())
        for pattern in spec["patterns"]:
            if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ValueError(f"Unsafe asset pattern in {manifest.name}")
            assets.update(p.relative_to(source) for p in source.glob(pattern) if p.is_file())
    for path in sorted(source.glob("tokenizer*config.json")):
        assets.update(referenced_assets(json.loads(path.read_text())))
    for relative in sorted(assets):
        path = source / relative
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Tokenizer asset escapes checkpoint: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing referenced tokenizer asset: {relative}")
    return sorted(assets)
