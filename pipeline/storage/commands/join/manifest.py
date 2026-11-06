# manifest.py — weights join: verified modular store -> original checkpoint bytes.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # project root for pipeline.* imports
from pipeline.storage.reassemble import reassemble_checkpoint


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms weights join")
    p.add_argument("store")
    p.add_argument("destination")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    result = reassemble_checkpoint(a.store, a.destination, resume=a.resume)
    print(f"joined: {result['shards']} shards, {result['tensors']} tensors -> {a.destination}")
    return 0


MANIFEST = {"name": "join", "help": "modular store -> original checkpoint (lossless bytes)", "run": run}
