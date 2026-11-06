# _load_reports.py — {suite: report dict} from every <suite>.toml in a reports dir.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import pathlib
import tomllib


def load_reports(folder):
    return {p.stem: tomllib.loads(p.read_text()) for p in sorted(pathlib.Path(folder).glob("*.toml"))
            if p.stem != "verdict"}                        # verdict.toml is gate output, not a report
