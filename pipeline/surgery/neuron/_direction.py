# _direction.py — compute behavioral direction from prompt contrast.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx
import numpy as np


def compute_direction(args, embed_fn):
    """Probe target vs baseline embeddings, return edit spec dict."""
    if not (args.target and args.baseline and args.layer is not None and args.columns):
        raise ValueError("direction requires --target --baseline --layer --columns")
    target = [l.strip() for l in open(args.target) if l.strip()]
    baseline = [l.strip() for l in open(args.baseline) if l.strip()]
    print(f"Probing: {len(target)} target, {len(baseline)} baseline...")
    t_vecs = np.stack([embed_fn(pr) for pr in target])     # [N, H] target activations
    b_vecs = np.stack([embed_fn(pr) for pr in baseline])    # [M, H] baseline activations
    d = t_vecs.mean(0) - b_vecs.mean(0)                    # mean difference = direction
    d_hat = d / (np.linalg.norm(d) + 1e-9)                 # unit direction
    print(f"Direction norm: {np.linalg.norm(d):.4f}")
    cols = [int(c) for c in args.columns.split(",")]
    return {
        "layer": args.layer,
        "proj": args.proj,
        "columns": cols,
        "direction": mx.array(d_hat, dtype=mx.float32),
        "alpha": args.alpha,
    }
