"""Sample decoding must preserve BF16 values and reject out-of-bounds requests."""
import tempfile
from pathlib import Path
import numpy as np
from pipeline.profile.verify._st_rows import rows


def test_bfloat_rows_preserve_special_values_and_bounds():
    bits = np.array([0x0001, 0x8000, 0x7f80, 0x7fc0], dtype="<u2")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data"
        path.write_bytes(bits.tobytes())
        loc = (path, {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}, 0)
        values = rows(loc, 0, 2).ravel()
        assert values[0] > 0 and np.signbit(values[1])
        assert np.isinf(values[2]) and np.isnan(values[3])
        try:
            rows(loc, 1, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("sample beyond tensor accepted")
