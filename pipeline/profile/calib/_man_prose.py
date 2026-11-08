# _man_prose.py — macOS man pages rendered to plain text as prose calibration text.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import subprocess

MANPAGES = ["ls", "grep", "find", "rsync", "ssh", "tar", "curl", "zsh"]


def _prose():
    out = []
    for m in MANPAGES:
        raw = subprocess.run(["man", "-P", "cat", m], capture_output=True, text=True).stdout
        out.append(subprocess.run(["col", "-b"], input=raw, capture_output=True, text=True).stdout)  # strip overstrike
    return [t for t in out if t.strip()]      # page missing on this machine -> dropped
