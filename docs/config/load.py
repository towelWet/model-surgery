import tomllib
from pathlib import Path


def load(name):
    return tomllib.loads((Path(__file__).resolve().parents[1] / name).read_text())
