# _layer.py — render one layer's heatmap row + top-N table.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def render_layer(li, layer, top_n=20):
    """Return HTML fragment for one layer: heatmap row + top-N neurons table."""
    parts = [f'<h2>Layer {li}</h2>']
    acts = layer.get("mean_abs", [])                         # mean |activation| per neuron
    if not acts:
        return '<p class="info">No data</p>'
    mx = max(acts) if acts else 1.0
    parts.append('<div class="row">')
    for val in acts[:512]:                                   # cap at 512 cells for perf
        ratio = val / mx if mx else 0
        r = int(200 * ratio + 40)                            # red channel scales with activation
        g = int(180 * (1 - ratio) + 40)                      # green fades as activation grows
        parts.append(f'<div class="cell" style="background:rgb({r},{g},60)" '
                     f'title="n={acts.index(val)} v={val:.4f}"></div>')
    parts.append('</div>')

    ranked = sorted(enumerate(acts), key=lambda x: -x[1])[:top_n]
    parts.append('<table><tr><th>Neuron</th><th>Mean |act|</th></tr>')
    for idx, val in ranked:
        parts.append(f'<tr><td>{idx}</td><td>{val:.6f}</td></tr>')
    parts.append('</table>')
    return "\n".join(parts)
