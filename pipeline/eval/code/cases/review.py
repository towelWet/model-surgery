# review.py — bug-finding code review tasks; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    {
        "id": "py_review_offby",
        "prompt": "Bug: for i in range(len(a)): if a[i] == t: return i+1 (off-by-one). Return fixed loop with correct index.",
        "need": ["return i", "range(len"],
        "any_of": ["i + 1", "return i + 1"],
        "max_tokens": 200,
    },
    {
        "id": "py_review_mut",
        "prompt": "Bug: def add(d,k,v): d=d+{k:v}; return d. Explain fix with corrected code.",
        "need": ["d[", "mutable", "def add"],
        "max_tokens": 220,
    },
    {
        "id": "review_xss",
        "prompt": "Review: innerHTML = userInput. Name vulnerability and safe fix.",
        "need": ["xss", "textContent"],
        "any_of": ["sanitize", "escape", "dompurify"],
        "max_tokens": 220,
    },
    {
        "id": "review_race",
        "prompt": "Review: shared counter++ in two threads without lock. Bug and fix.",
        "need": ["race", "lock", "mutex"],
        "any_of": ["threading", "atomic"],
        "max_tokens": 240,
    },
    {
        "id": "review_sqli",
        "prompt": "Review: query = f\"SELECT * FROM u WHERE id={uid}\". Vuln name + parameterized fix.",
        "need": ["sql", "inject"],
        "any_of": ["parameter", "placeholder", "?"],
        "max_tokens": 240,
    },
]
