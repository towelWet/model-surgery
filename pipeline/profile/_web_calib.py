# _web_calib.py — short web/HTML/CSS/JS prefixes for layer-ablation scoring.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Used by block_ablation.py --web (same length budget as generic CALIBRATION).

WEB_CALIBRATION = [
    "<!DOCTYPE html><html><head><title>Shop</title><style>",
    "body { font-family: system-ui; margin: 0; background: #111; color: #eee; }",
    "const nav = document.querySelector('nav'); nav.addEventListener('click',",
    "@media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }",
    "function renderCard({ title, price }) { return `<article><h2>${title}</h2>",
]
