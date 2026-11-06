# _template.py — HTML template constants for heatmap page.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Neuron Explorer</title>
<style>
body{font-family:Inter,system-ui;background:#fff;margin:2em;color:#222}
h1{font-size:1.4em}h2{font-size:1.1em;margin-top:1.5em}
.row{display:flex;gap:1px;margin:2px 0}
.cell{width:4px;height:18px;border-radius:1px}
.legend{display:flex;gap:.5em;font-size:.8em;margin:1em 0}
.info{font-size:.85em;color:#555;margin:.3em 0}
</style></head><body>
<h1>Neuron Explorer — Activation Heatmap</h1>
"""

HTML_TAIL = "</body></html>"

LEGEND = ('<div class="legend"><span>Low</span>'
          '<span style="background:linear-gradient(90deg,'
          '#eee,#f90,#e22);width:80px;height:14px;'
          'display:inline-block;border-radius:2px"></span>'
          '<span>High</span></div>')
