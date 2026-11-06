# _stdlib_code.py — Python standard-library source as code-shaped calibration text.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os, sysconfig

STDLIB = ["json/decoder.py", "argparse.py", "asyncio/tasks.py", "pathlib/_local.py",
          "subprocess.py", "http/client.py", "shutil.py", "tarfile.py"]   # ships with every interpreter


def _code():
    root = sysconfig.get_paths()["stdlib"]
    return [open(os.path.join(root, f), errors="ignore").read()
            for f in STDLIB if os.path.exists(os.path.join(root, f))]   # skip files absent in this Python
