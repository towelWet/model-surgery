# _load_lines.py — file I/O helper for neuron_probe.py.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def load_lines(path):
    """Read non-blank lines from a text file."""
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]
