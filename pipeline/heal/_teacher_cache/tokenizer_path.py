# tokenizer_path.py — local dir to load teacher tokenizer from (checkpoint dirs carry it).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys


def tokenizer_path(teacher):
    for cand in (teacher, os.environ.get("TARR_MODEL"),        # explicit, then resident server model
                 os.environ.get("SURGERY_TEACHER_MODEL"), os.environ.get("SURGERY_PARENT_MODEL")):
        if cand and os.path.isdir(cand):
            return cand
    # Old code fell through with a non-dir string -> from_pretrained could hit the HF hub. Refuse instead.
    print("Need --teacher or TARR_MODEL / SURGERY_TEACHER_MODEL dir for tokenizer")
    sys.exit(1)
