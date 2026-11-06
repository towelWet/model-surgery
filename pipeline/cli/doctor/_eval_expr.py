# _eval_expr.py — evaluate an AST expression with only os + given names (no builtins); None on failure.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast
import os


def eval_expr(node, ns):
    try:
        return eval(compile(ast.Expression(node), "<doctor>", "eval"), {"os": os, "__builtins__": {}, **ns})
    except Exception:
        return None                                         # dynamic / non-path expression
