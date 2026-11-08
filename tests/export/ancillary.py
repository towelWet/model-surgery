"""copy_ancillary copies flat configs/code; shards, weight index, tokenizer files excluded."""
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

IO_DIR = Path(__file__).resolve().parents[2] / "pipeline" / "export" / "_convert_io"
pkg = types.ModuleType("_cio")                                   # stub package: skips __init__ (mlx-free)
pkg.__path__ = [str(IO_DIR)]
sys.modules.setdefault("_cio", pkg)
spec = importlib.util.spec_from_file_location("_cio.copy_ancillary", IO_DIR / "copy_ancillary.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
copy_ancillary = mod.copy_ancillary


def test_copy_ancillary_dynamic_discovery_excludes_owned_files():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        dest = Path(tmp) / "dest"
        src.mkdir()
        dest.mkdir()
        for name in ("config.json", "generation_config.json",
                     "modeling_custom.py", "processor_config.json"):
            (src / name).write_text("{}")                        # ancillary: must be copied
        (src / "w.safetensors").write_bytes(b"\x00" * 16)        # weights: caller-owned
        (src / "model.safetensors.index.json").write_text("{}")  # weight map: caller-owned
        (src / "tokenizer.json").write_text("{}")                # tokenizer asset: copy_tokenizer owns
        copy_ancillary(str(src), str(dest))
        copied = {p.name for p in dest.iterdir()}
        assert copied == {"config.json", "generation_config.json",
                          "modeling_custom.py", "processor_config.json"}
