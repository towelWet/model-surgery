# cache_root.py — pick teacher-cache output dir: env override, then default, then package fallback.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os

from ._constants import DEFAULT_XMEM, FALLBACK_CACHE


def cache_root():
    xmem = os.environ.get("SURGERY_CACHE_DIR", "")        # prefer env override
    if xmem and os.path.isdir(xmem):
        return xmem
    if os.path.isdir(DEFAULT_XMEM):                        # module-level default
        return DEFAULT_XMEM
    os.makedirs(FALLBACK_CACHE, exist_ok=True)
    return FALLBACK_CACHE
