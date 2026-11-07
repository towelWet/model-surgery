"""Moefy refuses unvalidated routing before destination creation, including valid geometry."""
import json
import sys
import tempfile
from pathlib import Path


def test_moefy_untrained_router_creates_nothing():
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "pipeline/surgery"))       # surgery/ bare-import dir
    sys.path.insert(0, str(root / "pipeline/surgery/moefy")) # moefy/ siblings
    from _moefy import moefy
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        # Valid geometry must still refuse: a dimensionally valid partition is not a learned MoE.
        (src / "config.json").write_text(json.dumps(
            {"hidden_size": 32, "intermediate_size": 128, "num_hidden_layers": 1}))
        dest = Path(tmp) / "out-Tau"
        try:
            moefy(str(src), str(dest), 3, 1)
        except NotImplementedError as error:
            assert "trained routers" in str(error)
        else:
            raise AssertionError("accepted impossible partition")
        assert not dest.exists()                           # validation left no output
