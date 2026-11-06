# run.py — stdlib test runner: import every tests.* module, run every test_* callable in ONE process.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Discovery is recursive rglob (no hardcoded module list): drop a file under tests/ and it runs.
# Usage: python3 tests/run.py            (exit 1 on any failure)
import importlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                                          # root: tests.* and pipeline.* imports


def main():
    ran = failed = 0
    for p in sorted((ROOT / "tests").rglob("*.py")):
        if p.name in ("__init__.py", "run.py") or "helpers" in p.parts:
            continue                                                   # packages and helpers carry no tests
        mod = importlib.import_module("tests." + ".".join(p.relative_to(ROOT / "tests").with_suffix("").parts))
        for name in sorted(dir(mod)):
            if not name.startswith("test_") or not callable(getattr(mod, name)):
                continue
            ran += 1
            try:
                getattr(mod, name)()
                print(f"PASS {mod.__name__}.{name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {mod.__name__}.{name}: {type(e).__name__}: {e}")
    print(f"{ran} tests, {failed} failed")
    return 1 if failed or not ran else 0


if __name__ == "__main__":
    sys.exit(main())
