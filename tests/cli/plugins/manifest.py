# manifest.py — malformed manifests raise: non-dict attr, missing name, missing callable run.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
from tests.cli.plugins.helpers.loader import loader
from tests.cli.plugins.helpers.source import write_plugin

CASES = [
    ('MAN = 5\n', TypeError),                                          # manifest must be a dict
    ('MAN = {"run": lambda a: 0}\n', ValueError),                      # name required
    ('MAN = {"name": "x"}\n', TypeError),                              # run must be callable
]


def test_malformed_manifests_rejected():
    for i, (src, exc) in enumerate(CASES):
        with tempfile.TemporaryDirectory() as t:                       # own dir per case, no deletion needed
            write_plugin(t, "bad.py", src)
            try:
                loader()(t, "MAN")
            except exc:
                pass
            else:
                raise AssertionError(f"case {i} not rejected with {exc.__name__}")
