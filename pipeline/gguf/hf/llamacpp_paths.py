# llamacpp_paths.py — llama.cpp source dir (gguf-py, convert_hf_to_gguf.py) and binary dir.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Paths are workstation-private: $CLAUDE_DIR/private/model-surgery.toml [runtime.llamacpp] or env LLAMA_CPP_SOURCE / LLAMA_CPP_BIN.
import os
import sys
from pathlib import Path


def llamacpp_paths():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root for pipeline.*
    from pipeline.runtime.settings import runtime_settings
    cfg = runtime_settings().get("llamacpp", {})
    src = os.environ.get("LLAMA_CPP_SOURCE") or cfg.get("source", "")
    bin_dir = os.environ.get("LLAMA_CPP_BIN") or cfg.get("bin", "")
    if not (src and Path(src, "gguf-py").is_dir() and Path(src, "convert_hf_to_gguf.py").is_file()):
        raise FileNotFoundError("llama.cpp source not configured: set [runtime.llamacpp] source or LLAMA_CPP_SOURCE")
    if str(Path(src, "gguf-py")) not in sys.path:
        sys.path.insert(0, str(Path(src, "gguf-py")))        # import gguf from same checkout as converter
    return Path(src), Path(bin_dir) if bin_dir else None
