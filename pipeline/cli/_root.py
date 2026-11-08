# _root.py — absolute path of pipeline/ (cli/ lives directly inside it).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os


def pipeline_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
