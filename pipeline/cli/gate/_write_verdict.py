# _write_verdict.py — record gate decision + every rule line in <cand>/verdict.toml.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import datetime
import json
import os


def write_verdict(cand_dir, passed, rows, base_dir):
    lines = [f"passed = {str(passed).lower()}",
             f"when   = \"{datetime.datetime.now().isoformat(timespec='seconds')}\"",
             f"base   = {json.dumps(os.path.abspath(base_dir))}",   # json.dumps = valid TOML basic string
             "rules  = ["]
    lines += [f"  {json.dumps(status + ' ' + text)}," for status, text in rows]
    with open(os.path.join(cand_dir, "verdict.toml"), "w") as f:
        f.write("\n".join(lines + ["]"]) + "\n")
