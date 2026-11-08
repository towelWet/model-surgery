"""Join writes manifest+complete markers; interrupted join resumes; completed join refused."""
import tempfile
from pathlib import Path
from pipeline.storage.reassemble import reassemble_checkpoint
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_join_markers_resume_and_completed_refusal():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [("a", "F32", [2], b"\x01" * 8)])
        (src / "config.json").write_text("{}")
        store = Path(tmp) / "store-Tau"
        split_checkpoint(src, store)
        dest = Path(tmp) / "out-Tau"
        reassemble_checkpoint(store, dest)
        assert (dest / "complete.toml").exists() and (dest / "manifest.toml").exists()
        assert (dest / "w.safetensors").read_bytes() == (src / "w.safetensors").read_bytes()
        try:
            reassemble_checkpoint(store, dest, resume=True)   # completed output refuses
        except ValueError:
            pass
        else:
            raise AssertionError("resumed a completed join")
        dest2 = Path(tmp) / "out2-Tau"
        dest2.mkdir()
        (dest2 / "transaction.toml").write_text(
            f'operation = "join"\nschema_version = 1\n'
            f'source_fingerprint = "{__import__("tomllib").loads((store / "manifest.toml").read_text())["source_fingerprint"]}"\n')
        reassemble_checkpoint(store, dest2, resume=True)      # interrupted join resumes
        assert (dest2 / "complete.toml").exists()
        assert (dest2 / "w.safetensors").read_bytes() == (src / "w.safetensors").read_bytes()
