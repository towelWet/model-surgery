"""Discover split strategies from strategies/*.py (STRATEGY dicts) via model-surgery's plugin seam.
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
from pathlib import Path
from pipeline.cli.plugins.discovery import load_plugins


def load_strategies():
    """Return {name: STRATEGY}; adding a strategy = dropping one file, zero edits here."""
    return load_plugins(Path(__file__).resolve().parents[1] / "strategies", "STRATEGY")
