# reload.py — repeated load_plugins calls in one process see create/edit/remove.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
from tests.cli.plugins.helpers.loader import loader
from tests.cli.plugins.helpers.source import write_plugin

MAN = 'MAN = {"name": "%s", "run": lambda argv: 0}\n'


def test_loader_sees_create_edit_remove():
    with tempfile.TemporaryDirectory() as t:
        lp = loader()
        a = write_plugin(t, "a.py", MAN % "a")
        assert set(lp(t, "MAN")) == {"a"}                              # create seen
        a.write_text(MAN % "a2")
        assert set(lp(t, "MAN")) == {"a2"}                             # edit seen (fresh exec, no pyc)
        write_plugin(t, "b.py", MAN % "b")
        assert set(lp(t, "MAN")) == {"a2", "b"}                        # second create seen
        with tempfile.TemporaryDirectory() as hold:                    # outside discovery root t
            subprocess.run(["mv", str(a), str(Path(hold) / "a.py")], check=True)  # own tmp file only
            assert set(lp(t, "MAN")) == {"b"}                          # remove seen
