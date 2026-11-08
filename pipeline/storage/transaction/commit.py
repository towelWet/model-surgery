"""Write complete.toml LAST containing the manifest sha256."""
from ..hashes.digest import file_sha256
from ..manifest.escape import toml_str
from ..files.exclusive import write_xb


def write_complete(store):
    digest = file_sha256(store / "manifest.toml")
    return write_xb(store / "complete.toml", f"manifest_sha256 = {toml_str(digest)}\n".encode())
