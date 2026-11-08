# _load_cases.py — gather CASES from every <suite>/cases/*.py file.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Hot-loadable: drop a new cases/<category>.py exporting CASES and it joins the
# suite with zero edits. Files starting with "_" are skipped (helpers/drafts).
import glob
import importlib.util
import os
from pipeline.eval.inspection.cases import check_cases


def load_cases(suite_dir, only=None):
    """Return (cases, by_category); `only` = set of category names to keep."""
    cases, by_cat = [], {}
    for path in sorted(glob.glob(os.path.join(suite_dir, "cases", "*.py"))):  # sorted = stable order
        cat = os.path.splitext(os.path.basename(path))[0]                     # file name = category
        if cat.startswith("_") or (only and cat not in only):
            continue
        spec = importlib.util.spec_from_file_location(f"cases_{cat}", path)   # unique module name per file
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        by_cat[cat] = list(mod.CASES)                                         # every cases file exports CASES
        cases.extend(by_cat[cat])
    errors = check_cases(cases)
    if errors:
        raise ValueError("invalid eval cases: " + "; ".join(errors))
    return cases, by_cat
