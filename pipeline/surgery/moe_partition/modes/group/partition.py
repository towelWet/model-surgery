# partition.py — group-level expert partition (keeps group_size-neuron quant groups intact).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery._balanced import balanced_kmeans  # qualified; root on sys.path


def group_mode(mass, prof, E, S_EXP, SHARED, G, W):
    """Partition into experts keeping quantization groups whole."""
    gmass = mass.reshape(-1, G).sum(1)                                      # [W//G] group masses
    shared_g = np.argsort(-gmass)[: SHARED // G]                            # hottest shared groups
    rest = np.setdiff1d(np.arange(W // G), shared_g)                        # remaining groups
    gprof = prof.reshape(prof.shape[0], -1, G).sum(2).T[rest]               # [rest, N]
    lab = balanced_kmeans(gprof, E, S_EXP // G)                             # cluster groups
    to_neurons = lambda gs: (np.asarray(gs)[:, None] * G + np.arange(G)).ravel()
    shared = to_neurons(np.sort(shared_g))
    experts = np.stack([to_neurons(np.sort(rest[lab == e])) for e in range(E)])
    return shared, experts
