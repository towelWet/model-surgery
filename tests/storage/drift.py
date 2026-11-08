"""Source bytes mutated mid-split are caught by the commit-time stability rehash."""
import tempfile
from pathlib import Path
from unittest.mock import patch
from pipeline.storage.partition import shards as shard_plan
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st
from tests.storage.helpers.drifting import drifting_region
from tests.storage.helpers.mutate import STATE


def test_source_drift_during_copy_refuses_without_complete():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        shard = write_st(src / "w.safetensors", [("a", "F32", [1], b"\x01" * 4)])
        (src / "config.json").write_text("{}")
        store = Path(tmp) / "store-Tau"
        STATE["shard"], STATE["real"] = shard, shard_plan.region_sha256

        with patch.object(shard_plan, "region_sha256", drifting_region):
            try:
                split_checkpoint(src, store)
            except ValueError as e:
                assert "changed" in str(e)                  # stability rehash refusal
            else:
                raise AssertionError("split accepted source drift")
        assert not (store / "manifest.toml").exists()
        assert not (store / "complete.toml").exists()
