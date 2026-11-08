# rev/_topics.py — seed topics for code/rev corpus; data only.
PY_TASKS = [
    ("implement {algo} in Python with type hints", "def {algo}", "stdlib patterns"),
    ("refactor this snippet to use {pat}", "# uses {pat}", "clean code"),
    ("write pytest for {algo}", "def test_{algo}", "testing"),
]
WEB_TASKS = [
    ("build {comp} with Tailwind", "<div", "frontend"),
    ("vanilla JS {feat}", "function ", "dom"),
]
REV_TASKS = [
    ("explain x86 opcode {hex}", "instruction", "disasm"),
    ("ELF section {name} purpose", ".", "binary"),
    ("find bug in decompiled {fn}", "buffer", "review"),
]
ALGOS = ["bisect_left", "toposort", "dijkstra", "union_find", "knapsack"]
PATS = ["dataclass", "contextlib", "functools.lru_cache", "asyncio"]
