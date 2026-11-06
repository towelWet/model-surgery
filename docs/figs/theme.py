# theme.py — shared matplotlib setup for every Model Surgery figure.
# Why: one surface colour, one ink colour, one font across five figures; without
# this each figure drifts and the document stops reading as one system.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

import matplotlib                                          # imported before pyplot to fix the backend
matplotlib.use("Agg")                                      # headless PNG writer; no window, no focus steal
import matplotlib.pyplot as plt                            # figure/axes API used by every fig_*.py

SURFACE = "#fcfcfb"                                        # validated light chart surface
INK     = "#0b0b0b"                                        # text-primary
MUTED   = "#52514e"                                        # text-secondary: axis labels, captions
BLUE    = "#2a78d6"                                        # categorical slot 1 — the kept/default mark
ORANGE  = "#eb6834"                                        # categorical slot 2 — the cut/highlighted mark
FAINT   = "#cde2fb"                                        # blue ramp step 100: recessive fills
RULE    = "#d9d8d4"                                        # hairline grid/box edges, kept below the data

plt.rcParams.update({                                      # applied globally on import — figures stay thin
    "font.family": "Helvetica Neue",                       # matches the document body font
    "font.size": 10,                                       # figure text one step under 11pt body copy
    "text.color": INK,
    "axes.labelcolor": MUTED,                              # labels recede; the marks carry the message
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8,         # recessive axes per the mark spec
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,                          # PNG must not land on transparent/white mismatch
})


def save(fig, path):                                       # one exit point so DPI and padding never drift
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.18)   # 220dpi prints clean in Word
    plt.close(fig)                                         # release the figure; five builds in one process
    return path                                            # caller prints/collects the written path
