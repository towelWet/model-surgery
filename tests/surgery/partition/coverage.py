# coverage.py — a mode whose ids don't exactly cover range(W) is rejected; nothing written.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import importlib.util, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))           # project root for tests.*
from tests.surgery.partition.helpers.statistics import make_stats

ROOT = Path(__file__).resolve().parents[3]
LAYER = ROOT / "pipeline/surgery/moe_partition/output/layer.py"
BAD = {"run": lambda m, s, p: (np.array([]), np.array([[2, 2], [3, 1]]))}   # dup ids, holes
SNIP = ("import importlib.util;spec=importlib.util.spec_from_file_location('m',r'%s');"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "import sys;from pathlib import Path;sys.path.insert(0,r'%s');"
        "from types import SimpleNamespace;import numpy as np;"
        "p=SimpleNamespace(experts=2,expert_width=2,shared_width=0,group_size=2,width=4);"
        "bad={'run':lambda a,b,p:(np.array([]),np.array([[2,2],[3,1]]))};"
        "m.run_layer(r'%s','L02','bogus',bad,p)")                       # raises ValueError under -O


def test_bad_mode_coverage_rejected():
    spec = importlib.util.spec_from_file_location("ms_layer", LAYER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = SimpleNamespace(experts=2, expert_width=2, shared_width=0, group_size=2, width=4)
    with tempfile.TemporaryDirectory() as t:
        make_stats(t, [2], 4)
        try:
            mod.run_layer(t, "L02", "bogus", BAD, p)
        except ValueError:
            pass
        else:
            raise AssertionError("bad coverage accepted")
        assert not (Path(t) / "partition").exists()                    # rejected before any output
        r = subprocess.run([sys.executable, "-O", "-c", SNIP % (LAYER, ROOT, t)],
                           capture_output=True, text=True)
        assert r.returncode != 0, "python -O must not disable validation"
        assert not (Path(t) / "partition").exists()
