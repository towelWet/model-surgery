# _emit.py — buffered weight emitter for MoE partition loop.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Buffered emitter factory; the callback owns its own file.
from functools import partial
from emitter.append import emit


def make_emitter(out):
    return partial(emit, out, 20)
