# loader.py — spec-load the compat shim; returns load_plugins for tests.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]         # tests/cli/plugins/helpers/loader.py -> root


def loader():
    sys.path.insert(0, str(ROOT))                  # shim's qualified import needs project root
    spec = importlib.util.spec_from_file_location("ms_shim", ROOT / "pipeline/cli/_load_plugins.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_plugins
