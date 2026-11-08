# inputs.py — invalid policy/malformed arrays/preexisting outputs fail before any write.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
from tests.surgery.partition.helpers.statistics import make_stats
from tests.surgery.partition.helpers.invocation import invoke

FL = ["--experts", "2", "--expert-width", "2", "--shared-width", "0", "--group-size", "2"]


def test_invalid_inputs_write_nothing():
    with tempfile.TemporaryDirectory() as t:
        make_stats(t, [2], 4)
        r = invoke(t, "--experts", "3", "--expert-width", "2",         # 3*2+0 != W=4
                   "--shared-width", "0", "--group-size", "2")
        assert r.returncode != 0 and not (Path(t) / "partition").exists()
    with tempfile.TemporaryDirectory() as t:
        make_stats(t, [2], 4, sample_w=5)                              # sample width mismatch
        r = invoke(t, *FL)
        assert r.returncode != 0 and not (Path(t) / "partition").exists()
    with tempfile.TemporaryDirectory() as t:
        make_stats(t, [2], 4)
        out = Path(t) / "partition" / "group" / "L02.npz"              # preexisting output: refuse
        out.parent.mkdir(parents=True)
        np.savez(out, shared=np.array([9]), experts=np.array([[9]]))
        before = out.read_bytes()
        r = invoke(t, *FL)
        assert r.returncode != 0 and "already exists" in r.stderr + r.stdout
        assert out.read_bytes() == before                              # untouched, not overwritten
