"""Find private target literals by configured digest without printing or embedding the literals."""
import hashlib
import re


def scan_target_literals(root, files, hashes, literals=()):
    findings = []
    for rel in files:
        data = (root / rel).read_bytes()
        for literal in literals:
            offset = data.lower().find(literal.encode().lower())
            if offset >= 0:
                findings.append({"cat": "privacy", "id": "privacy:private-brand", "rel": rel,
                                 "line": data[:offset].count(b"\n") + 1})
        for token in re.finditer(rb"[A-Za-z0-9_!@#-]+", data):
            if hashlib.sha256(token.group()).hexdigest() in hashes:
                findings.append({"cat": "privacy", "id": "privacy:target-literal", "rel": rel,
                                 "line": data[:token.start()].count(b"\n") + 1})
    return findings
