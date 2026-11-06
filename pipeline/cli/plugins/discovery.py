# discovery.py — recursively discover plugin sources and collect their manifest dicts.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Hot-loadable: any source matching `pattern` under folder that carries the
# requested attr is a plugin. Fresh source exec per call — repeated calls inside
# one process see file creates/edits/removes (request-boundary reload; a CLI has
# no need for a resident watcher). Duplicate manifest names raise.
# (Implementation moved here from cli/_load_plugins.py, which now re-exports.)
import pathlib

from pipeline.cli.plugins.execution import exec_fresh
from pipeline.cli.plugins.manifest import validate


def load_plugins(folder, attr, pattern="**/*.py"):
    """Return {manifest['name']: manifest} sorted by path; attr e.g. COMMAND/CHECK/MANIFEST."""
    found, src = {}, {}
    for path in sorted(pathlib.Path(folder).rglob(pattern)):
        if path.name.startswith("_"):
            continue                                        # private helpers are not plugins
        man = validate(getattr(exec_fresh(path, f"ms_{attr.lower()}_{path.stem}"), attr, None), path, attr)
        if man is None:
            continue
        if man["name"] in found:
            raise ValueError(f"duplicate plugin name {man['name']!r}: {src[man['name']]} vs {path}")
        found[man["name"]], src[man["name"]] = man, path
    return found
