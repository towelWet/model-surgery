"""Local publication gate using the installed publisher's exact file selection and private rules."""
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import sys
import tomllib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.cli.doctor.target_scan import scan_target_literals


def run(argv):
    parser = argparse.ArgumentParser(prog="ms privacy")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    binary = shutil.which("x-viewer")
    if not binary:
        raise FileNotFoundError("publication gate requires the installed x-viewer CLI")
    sys.path.insert(0, str(Path(binary).resolve().parent / "lib"))
    root = Path(__file__).resolve().parents[3]
    manifest = importlib.import_module("github.manifest").read_manifest(root)
    files = importlib.import_module("github.files").publish_files(root, manifest["excludes"])
    blocking, allowed = importlib.import_module("github.audit.audit").audit(root, files, ())
    sys.path.insert(0, str(root))
    from pipeline.runtime.private_toml import private_toml_path
    local = private_toml_path()  # private values live outside the tree
    private = tomllib.loads(local.read_text()) if local.is_file() else {}
    hashes = set(private.get("training", {}).get("forbidden_password_sha256", []))
    literals = private.get("publication", {}).get("forbidden_literals", [])
    blocking += scan_target_literals(root, files, hashes, literals)
    oversized = [rel for rel in files if (root / rel).stat().st_size > 2_000_000]
    fingerprint = hashlib.sha256(b"".join(rel.encode() + b"\0" + hashlib.sha256(
        (root / rel).read_bytes()).digest() for rel in files)).hexdigest()
    print(f"Publication files: {len(files)}; blocking: {len(blocking)}; allowed: {len(allowed)}")
    for item in blocking:
        print(f"{item['cat']}: {item['rel']}:{item['line']}")
    if args.report:
        text = f'files = {len(files)}\nblocking = {len(blocking)}\nallowed = {len(allowed)}\n'
        text += f'scanner_window_exceeded = {len(oversized)}\nsource_sha256 = "{fingerprint}"\n'
        args.report.write_text(text)
    return int(bool(blocking or oversized))


COMMAND = {"name": "privacy", "help": "read-only local publication audit; never uploads", "run": run}
