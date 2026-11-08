"""Load dtype name -> byte width table from dtypes.toml in this folder."""
import tomllib
from pathlib import Path


def dtype_bytes():
    path = Path(__file__).resolve().parent / "dtypes.toml"
    return dict(tomllib.loads(path.read_text()))
