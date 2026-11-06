# _render.py — assemble full HTML heatmap from layer fragments.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _template import HTML_HEAD, HTML_TAIL, LEGEND   # page skeleton
from _layer import render_layer                      # per-layer renderer


def render_heatmap(layers_data, prompts, top_n=20):
    """Build complete HTML string with per-layer neuron heatmaps."""
    parts = [HTML_HEAD]
    parts.append(f'<p class="info">Prompts: {len(prompts)}, '
                 f'Layers: {len(layers_data)}</p>')
    parts.append(LEGEND)                                     # color gradient legend

    for li, layer in enumerate(layers_data):
        parts.append(render_layer(li, layer, top_n))         # delegate each layer

    parts.append(HTML_TAIL)
    return "\n".join(parts)
