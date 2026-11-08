"""Moefy rejects expert widths that break quantization groups before creating any output."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import tempfile
from pathlib import Path


def test_moefy_bad_geometry_creates_nothing():
    from pipeline.surgery.moefy._moefy import moefy
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {n: Path(tmp) / n for n in ("src", "quant", "calib")}
        for d in dirs.values():
            d.mkdir()
        # 3072 / 32 = 96 neurons per expert: not a multiple of group 64, so down_proj cannot quantize.
        (dirs["quant"] / "config.json").write_text(json.dumps({"intermediate_size": 3072, "num_hidden_layers": 1,
                                                               "quantization": {"group_size": 64, "bits": 8}}))
        dest = Path(tmp) / "out"
        try:
            moefy(str(dirs["src"]), str(dirs["quant"]), str(dirs["calib"]), str(dest),
                  {"experts": 32, "topk": 16, "logit_std": 0.002})
        except ValueError as error:
            assert "multiple of 64" in str(error)
        else:
            raise AssertionError("accepted expert width 96 with group 64")
        assert not dest.exists()                             # validation left no output
