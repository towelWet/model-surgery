# _schema.py — re-export TOOL_SCHEMA definitions (shared by calib, heal, eval, export bench).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Eval CASES moved to eval/tool/cases/*.py (load via eval/_load_cases.py).
# This file used to import numbered shards _cases_1/2/3 and only worked when
# caller had profile/schema/ on sys.path; now it adds that dir itself.
import os
import sys

_SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")
if _SCHEMA_DIR not in sys.path:
    sys.path.insert(0, _SCHEMA_DIR)                   # _tools.py + _distractor.py live there

from _tools import (                                  # noqa: E402  (path set above)
    TOOL_SCHEMA,
    TOOL_SCHEMA_READ_BASH,
    TOOL_SCHEMA_WRITE_ONLY,
)
from _distractor import TOOL_SCHEMA_DISTRACTOR       # noqa: E402
