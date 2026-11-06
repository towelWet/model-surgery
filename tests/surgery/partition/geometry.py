# geometry.py — both accepted geometries produce exact-cover partitions and unsplit groups.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))           # project root for tests.*
from tests.surgery.partition.helpers.statistics import make_stats
from tests.surgery.partition.helpers.invocation import invoke

CASES = [dict(w=4, E=2, ew=2, sw=0, G=2), dict(w=6, E=2, ew=2, sw=2, G=2)]


def test_geometry_both_cases():
    for c in CASES:
        with tempfile.TemporaryDirectory() as t:
            make_stats(t, [2, 7], c["w"])
            r = invoke(t, "--experts", str(c["E"]), "--expert-width", str(c["ew"]),
                       "--shared-width", str(c["sw"]), "--group-size", str(c["G"]))
            assert r.returncode == 0, r.stderr
            for mode in ("group", "neuron"):
                for d in ("L02", "L07"):
                    z = np.load(Path(t) / "partition" / mode / f"{d}.npz")
                    assert z["shared"].shape == (c["sw"],) and z["experts"].shape == (c["E"], c["ew"])
                    ids = np.sort(np.concatenate([z["shared"], z["experts"].ravel()]))
                    assert (ids == np.arange(c["w"])).all(), f"{mode}/{d}: coverage wrong"
                    if mode == "group":                      # whole quant groups stay together
                        for e in z["experts"]:
                            mates = {int(x) + 1 if x % c["G"] == 0 else int(x) - 1 for x in e}
                            assert mates <= set(e.tolist()), f"{d}: group split in {e}"
