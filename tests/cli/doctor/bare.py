"""doctor `bare` check: flags a bare sibling import and a non-root sys.path dir; passes absolute style."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import tempfile
from pathlib import Path

from pipeline.cli.doctor.checks.bare import run


def test_bare_check_flags_sibling_imports_and_dir_hacks_only():
    with tempfile.TemporaryDirectory() as tmp:
        pipe = Path(tmp) / "pipeline" / "x"
        pipe.mkdir(parents=True)
        (pipe / "_helper.py").write_text("def helper():\n    return 1\n")
        (pipe / "bad.py").write_text(                          # both forbidden shapes, one per line
            "import os, sys\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            "from _helper import helper\n")
        (pipe / "good.py").write_text(                         # root bootstrap + absolute import: allowed
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n"
            "from pipeline.x._helper import helper\n")
        problems = run(str(Path(tmp) / "pipeline"))
    assert len(problems) == 2, problems
    assert all(p.startswith("x/bad.py:") for p in problems)
    assert any("bare import of sibling '_helper'" in p for p in problems)
    assert any("sys.path gets repo dir pipeline/x" in p for p in problems)
