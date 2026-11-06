"""Impossible moefy partition geometry must fail before destination creation."""
import json
import sys
import tempfile
from pathlib import Path


def test_moefy_invalid_geometry_creates_nothing():
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "pipeline/surgery"))       # surgery/ bare-import dir
    sys.path.insert(0, str(root / "pipeline/surgery/moefy")) # moefy/ siblings
    from _moefy import moefy
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        # W=100 H=30 E=3: remainder 70 not divisible by 3 -> expert_width raises pre-mkdir
        (src / "config.json").write_text(json.dumps(
            {"hidden_size": 30, "intermediate_size": 100, "num_hidden_layers": 1}))
        dest = Path(tmp) / "out-Tau"
        try:
            moefy(str(src), str(dest), 3, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted impossible partition")
        assert not dest.exists()                           # validation left no output
