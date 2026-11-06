# _eval_paths.py — put every dir an eval suite imports from on sys.path.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Why: pipeline modules import each other by bare name (`from _chat import chat`).
# Before this file each eval guessed its own sys.path and three of four suites
# could not import (tool/code/rev broke after profile/ splits). One resolver,
# derived from __file__, so moving pipeline/ never breaks evals again.
import os
import sys


def add_eval_paths(suite_file):
    """Insert suite dir, eval/, profile/, profile/schema/ (in that priority)."""
    suite = os.path.dirname(os.path.abspath(suite_file))       # eval/<suite>/
    eval_dir = os.path.dirname(suite)                          # eval/  (shared grade/loop/loader)
    pipeline = os.path.dirname(eval_dir)                       # pipeline/
    project = os.path.dirname(pipeline)
    if project not in sys.path:
        sys.path.insert(0, project)
    profile = os.path.join(pipeline, "profile")                # _chat, _api, _eval_case, _validators
    schema = os.path.join(profile, "schema")                   # TOOL_SCHEMA definitions (shared w/ calib+heal)
    for p in (schema, profile, eval_dir, suite):               # reversed: suite ends up first
        if p not in sys.path:
            sys.path.insert(0, p)
    return suite
