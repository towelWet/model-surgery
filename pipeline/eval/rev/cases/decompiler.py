# decompiler.py — decompiler IR (Ghidra pcode/SSA); data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    {
        "id": "ghidra_pcode",
        "prompt": "In Ghidra decompiler, what is SSA vs pcode briefly?",
        "need": ["ssa", "pcode"],
        "max_tokens": 200,
    },
]
