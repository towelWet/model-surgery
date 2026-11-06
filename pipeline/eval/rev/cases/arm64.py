# arm64.py — AArch64 decode; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    {
        "id": "arm64_ret",
        "prompt": "AArch64 instruction `d65f03c0` — what is it?",
        "need": ["ret"],
        "max_tokens": 120,
    },
    {
        "id": "arm64_bl",
        "prompt": "What does ARM64 `bl` do vs `b`?",
        "need": ["branch", "link"],
        "max_tokens": 160,
    },
]
