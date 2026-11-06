"""Write transaction.toml exclusively: operation + schema version + source fingerprint."""
from ..manifest.escape import toml_str
from ..files.exclusive import write_xb


def write_transaction(store, operation, fingerprint):
    text = (f"operation = {toml_str(operation)}\n"
            f"schema_version = 1\n"
            f"source_fingerprint = {toml_str(fingerprint)}\n")
    return write_xb(store / "transaction.toml", text.encode())
