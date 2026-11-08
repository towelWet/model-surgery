# python.py — Python codegen tasks; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    {
        "id": "py_binary_search",
        "prompt": "Write only a Python function binary_search(arr, x) returning index or -1.",
        "need": ["def binary_search", "while", "return"],
        "max_tokens": 320,
    },
    {
        "id": "py_two_sum",
        "prompt": "Implement two_sum(nums, target) in O(n) with a dict mapping value->index. Code only.",
        "need": ["def two_sum", ["dict", "{}"], "return"],   # idiomatic `seen = {}` never says dict
        "max_tokens": 280,
    },
    {
        "id": "py_lru",
        "prompt": "Python class LRUCache with get and put using OrderedDict. Code only.",
        "need": ["class LRUCache", "def get", "def put", "OrderedDict"],
        "max_tokens": 400,
    },
    {
        "id": "py_async_fetch",
        "prompt": "Async Python fetch_urls(urls) with aiohttp gathering results. Code only.",
        "need": ["async def", "aiohttp", "gather"],
        "max_tokens": 360,
    },
    {
        "id": "py_dataclass",
        "prompt": "Python @dataclass Point with distance_to(other). Code only.",
        "need": ["@dataclass", "class Point", "def distance_to"],
        "max_tokens": 240,
    },
    {
        "id": "py_top_k",
        "prompt": "Python: top_k(nums, k) using heapq.nlargest; return list. Code only.",
        "need": ["def top_k", "heapq", "nlargest"],
        "max_tokens": 260,
    },
    {
        "id": "py_parse_log",
        "prompt": "Parse ISO date prefix YYYY-MM-DD from log line; regex + function. Code only.",
        "need": ["def ", "re.", "\\d{4}"],
        "any_of": ["match", "search"],
        "max_tokens": 280,
    },
]
