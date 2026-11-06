# no_tool.py — plain questions; right answer is no tool call; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    {"messages": [{"role": "user", "content": "Is water H2O? Answer yes or no."}], "exp_tool": None},
    {"messages": [{"role": "user", "content": "Define recursion in one sentence."}], "exp_tool": None},
    {"messages": [{"role": "user", "content": "Convert 10 km to meters."}], "exp_tool": None},
    {"messages": [{"role": "user", "content": "What is 17 plus 25?"}], "exp_tool": None},
    {"messages": [{"role": "user", "content": "How many sides does a hexagon have?"}], "exp_tool": None},
    {"messages": [{"role": "user", "content": "Sort these: cherry, apple, banana."}], "exp_tool": None},
    {"messages": [{"role": "user", "content": "Name three primary colors."}], "exp_tool": None},
    {"messages": [{"role": "user", "content": "Capital of France?"}], "exp_tool": None},
    {"messages": [{"role": "user", "content": "Translate hello to Spanish."}], "exp_tool": None},
    {"messages": [{"role": "user", "content": "What year did WW2 end?"}], "exp_tool": None},
]
