# frontend.py — HTML/CSS/JS/React/Tailwind codegen tasks; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    {
        "id": "html_landing",
        "prompt": "Single HTML5 file: <!DOCTYPE html>, header nav, main with hero h1, footer. Semantic tags only.",
        "need": ["<!doctype", "<main", "<nav", "<footer"],
        "max_tokens": 500,
    },
    {
        "id": "css_flex",
        "prompt": "CSS for .card-grid: responsive flex wrap, gap, max-width container.",
        "need": [".card-grid", "display: flex", "flex-wrap", "gap"],
        "max_tokens": 280,
    },
    {
        "id": "js_debounce",
        "prompt": "JavaScript debounce(fn, ms) returning debounced function. Code only.",
        "need": ["function debounce", "setTimeout", "clearTimeout"],
        "max_tokens": 260,
    },
    {
        "id": "react_counter",
        "prompt": "React function Counter with useState increment button. Code only.",
        "need": ["useState", "function Counter", "onClick"],
        "max_tokens": 300,
    },
    {
        "id": "tailwind_card",
        "prompt": "HTML snippet: Tailwind card with rounded shadow padding and button.",
        "need": ["rounded", "shadow", "class="],
        "max_tokens": 320,
    },
]
