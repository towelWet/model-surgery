# duplicates.py — two plugins claiming the same manifest name raise ValueError.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))           # project root for tests.*
from tests.cli.plugins.helpers.loader import loader
from tests.cli.plugins.helpers.source import write_plugin

DUP = 'MAN = {"name": "dup", "run": lambda argv: 0}\n'


def test_duplicate_names_raise():
    with tempfile.TemporaryDirectory() as t:
        write_plugin(t, "a.py", DUP)
        write_plugin(t, "nested/b.py", DUP)                            # duplicates caught across dirs too
        try:
            loader()(t, "MAN")
        except ValueError as e:
            assert "dup" in str(e)
        else:
            raise AssertionError("duplicate manifest names not rejected")
