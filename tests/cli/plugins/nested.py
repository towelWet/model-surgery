# nested.py — recursive discovery finds deep plugins; _-files and non-manifest files skipped.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))           # project root for tests.*
from tests.cli.plugins.helpers.loader import loader
from tests.cli.plugins.helpers.source import write_plugin

MAN = 'MAN = {"name": "%s", "run": lambda argv: 0}\n'


def test_recursive_discovery_skips_private_and_plain():
    with tempfile.TemporaryDirectory() as t:
        write_plugin(t, "top.py", MAN % "top")
        write_plugin(t, "a/b/deep.py", MAN % "deep")
        write_plugin(t, "_private.py", MAN % "priv")                   # helper: must be ignored
        write_plugin(t, "plain.py", "x = 1\n")                         # no manifest attr: ignored
        got = loader()(t, "MAN")
        assert set(got) == {"top", "deep"}, got
