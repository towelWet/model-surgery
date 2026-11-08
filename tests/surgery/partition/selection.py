# selection.py — only dirs count; duplicate numeric ids and absent requested ids are errors.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
from tests.surgery.partition.helpers.statistics import make_stats
from tests.surgery.partition.helpers.invocation import invoke

FL = ["--experts", "2", "--expert-width", "2", "--shared-width", "0", "--group-size", "2"]


def test_selection_rejects_files_dups_and_missing():
    with tempfile.TemporaryDirectory() as t:
        make_stats(t, [2, 7], 4)
        (Path(t) / "L09").write_text("not a dir")                      # file, not a layer dir
        r = invoke(t, *FL, "--layers", "9")
        assert r.returncode != 0 and "absent" in r.stderr + r.stdout
        r = invoke(t, *FL, "--layers", "7")
        assert r.returncode == 0, r.stderr
        assert not (Path(t) / "partition" / "group" / "L02.npz").exists()   # only L07 ran
    with tempfile.TemporaryDirectory() as t2:
        (Path(t2) / "L02").mkdir(); (Path(t2) / "L2").mkdir()          # same numeric id 2
        r = invoke(t2, *FL)
        assert r.returncode != 0 and "duplicate layer id 2" in r.stderr + r.stdout
