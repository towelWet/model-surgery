# fig_distance.py — Figure 5: the three candidate 16-layer cuts, ranked.
# Why: this is the evidence behind the gate — the tidy-looking contiguous cut is
# the worst one. Lower angular distance = more redundant = safer to remove.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

from theme import plt, save, BLUE, ORANGE, MUTED, INK

# ROWS holds historical values measured before profiling was disabled; the
# instrumented profiler that produced them is currently blocked — do not re-derive.
ROWS = [                                                    # (spec, summed angular distance, outcome note)
    ("4-15, 40-43", 0.336, "chosen — non-contiguous"),
    ("28-43",       0.459, "best contiguous"),
    ("24-39",       0.480, "blind cut — tool accuracy fell to 20%"),
]


def draw(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.1))
    ys = range(len(ROWS))
    colors = [BLUE, BLUE, ORANGE]                           # orange marks the failure, not a third category
    ax.barh(list(ys), [r[1] for r in ROWS], height=0.5, color=colors)
    for y, (_, dist, note) in zip(ys, ROWS):
        ax.text(dist + 0.012, y, f"{dist:.3f}   ·   {note}", # direct label; no number on a hidden axis
                va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(list(ys), [r[0] for r in ROWS], fontsize=9)
    ax.invert_yaxis()                                       # best candidate on top
    ax.set_xlim(0, 0.62); ax.set_xticks([0, 0.2, 0.4, 0.6])  # headroom for the trailing labels
    ax.set_xlabel("summed angular distance — lower is more redundant")
    ax.xaxis.grid(True, color="#ecebe8", linewidth=0.8)
    ax.set_axisbelow(True)                                  # grid stays behind the bars
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)                  # baseline only
    return save(fig, path)
