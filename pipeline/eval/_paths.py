# _paths.py — put every dir an eval suite imports from on sys.path.
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
    """Put the repo root on sys.path (imports are absolute pipeline.*); return the suite dir."""
    suite = os.path.dirname(os.path.abspath(suite_file))       # eval/<suite>/
    eval_dir = os.path.dirname(suite)                          # eval/  (shared grade/loop/loader)
    pipeline = os.path.dirname(eval_dir)                       # pipeline/
    project = os.path.dirname(pipeline)
    if project not in sys.path:
        sys.path.insert(0, project)
    # Was: suite, eval/, profile/, profile/schema/ inserted for bare imports. Absolute imports since
    # 2026-11-07 (py-absolutize) made them dead; the root above is all any suite needs.
    return suite
