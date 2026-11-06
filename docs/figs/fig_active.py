# fig_active.py — Figure 4: active parameters inside total parameters, per model.
# Why: the whole economic argument is active-vs-total; a faint total track with a
# solid active bar shows the ratio directly instead of asking for mental division.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

from theme import plt, save, BLUE, FAINT, MUTED, INK

ROWS = [                                                    # (model, active B, total B) — ordered worst last
    ("Qwen3-30B-A3B",   3.0, 30.0),
    ("Whittle-27B-A3B", 3.0, 27.0),
    ("Whittle-MoE-27B", 17.8, 27.0),
    ("Dense 27B",       27.0, 27.0),
]


def draw(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ys = range(len(ROWS))                                   # one row per model, top row drawn last
    ax.barh(list(ys), [r[2] for r in ROWS], height=0.52, color=FAINT)        # total = recessive track
    ax.barh(list(ys), [r[1] for r in ROWS], height=0.52, color=BLUE)         # active = the real cost
    for y, (_, act, tot) in zip(ys, ROWS):
        ax.text(tot + 0.6, y, f"{act:g}B of {tot:g}B   ·   {act/tot:.2f}",   # direct label: value + ratio
                va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(list(ys), [r[0] for r in ROWS], fontsize=9)
    ax.invert_yaxis()                                       # first row at the top, reading order
    ax.set_xlim(0, 40); ax.set_xticks([0, 10, 20, 30])       # headroom on the right for the direct labels
    ax.set_xlabel("parameters (billions)")
    ax.xaxis.grid(True, color="#ecebe8", linewidth=0.8)      # grid below the data, barely visible
    ax.set_axisbelow(True)                                  # matplotlib draws grid over bars otherwise
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)                  # keep only the baseline the bars sit on
    ax.text(0, -0.95, "solid = active per token      faint = total stored",
            fontsize=8.5, color=MUTED)                      # legend line: two marks, named in text
    return save(fig, path)
