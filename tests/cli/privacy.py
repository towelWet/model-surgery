"""Private brand strings and target digests block publication without appearing in findings."""
import hashlib
import tempfile
from pathlib import Path
from pipeline.cli.doctor.target_scan import scan_target_literals


def test_publication_guard_redacts_configured_private_literals():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "comment.py").write_text("# private-backend example-secret\n")
        findings = scan_target_literals(root, ["comment.py"],
            {hashlib.sha256(b"example-secret").hexdigest()}, ["private-backend"])
        assert {item["id"] for item in findings} == {"privacy:private-brand", "privacy:target-literal"}
        assert all("example-secret" not in str(item) and "private-backend" not in str(item) for item in findings)
