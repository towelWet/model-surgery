"""Reports must round-trip escaped labels, preserve precision, and declare limitations."""
import importlib.util
import tempfile
import tomllib
from pathlib import Path


def test_report_roundtrip_and_precision():
    path = Path(__file__).resolve().parents[2] / "pipeline/eval/_write_report.py"
    spec = importlib.util.spec_from_file_location("test_report_writer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "report.toml"
        label = 'model "quoted"\nline\\end'
        module.write_report(output, "test", label, {"acc": 1 / 3, "total": 3}, {"a.b": {"total": 3}})
        report = tomllib.loads(output.read_text())
        assert report["model"] == label
        assert report["acc"] == 1 / 3
        assert report["category"]["a.b"]["total"] == 3
        assert report["diagnostic_only"] is True
        assert len(report["evaluation_fingerprint"]) == 64
