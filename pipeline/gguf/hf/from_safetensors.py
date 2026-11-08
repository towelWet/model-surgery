# from_safetensors.py — HF safetensors -> GGUF via llama.cpp's convert_hf_to_gguf.py, then llama-quantize.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Official converter + quantizer only: tokenizer, metadata and k-quant packing stay exactly
# what llama.cpp produces for any HF upload. Intermediate f16 GGUF is kept beside output.
import os
import subprocess
import sys
from .llamacpp_paths import llamacpp_paths
from .aligned_dir import aligned_dir


def from_safetensors(hf_dir, out_gguf, qtype="Q4_K_M"):
    src, bin_dir = llamacpp_paths()
    if os.path.exists(out_gguf):
        raise FileExistsError(out_gguf)
    f16 = out_gguf[:-5] + ".f16.gguf" if qtype.upper() != "F16" else out_gguf
    env = {**os.environ, "PYTHONPATH": str(src / "gguf-py")}
    hf_dir = aligned_dir(hf_dir, out_gguf[:-5] + ".aligned")  # MLX-written shards lack 8-byte header pad
    subprocess.run([sys.executable, str(src / "convert_hf_to_gguf.py"), hf_dir, "--outfile", f16,
                    "--outtype", "f16"], check=True, env=env)
    if f16 != out_gguf:
        if not bin_dir or not (bin_dir / "llama-quantize").exists():
            raise FileNotFoundError("llama-quantize not built: set [runtime.llamacpp] bin")
        subprocess.run([str(bin_dir / "llama-quantize"), f16, out_gguf, qtype], check=True)
    return out_gguf
