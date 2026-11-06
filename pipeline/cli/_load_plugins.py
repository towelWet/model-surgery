# _load_plugins.py — compatibility shim; implementation lives in pipeline/cli/plugins/.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.cli.plugins.discovery import load_plugins  # noqa: F401  re-export for legacy callers
