# _suite_args.py — CLI flags every eval suite shares.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import argparse


def suite_args(description):
    """--only cat[,cat]  --out report.toml  --model label  --list (print categories, no run)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--only", default="", help="comma list of case categories (cases/<name>.py)")
    p.add_argument("--out", default="", help="write TOML scores here (input for promotion gate)")
    p.add_argument("--model", default="served", help="label recorded in report (server decides real model)")
    p.add_argument("--list", action="store_true", help="list categories + case counts, then exit")
    a = p.parse_args()
    a.only = set(filter(None, a.only.split(",")))     # empty string -> empty set -> all categories
    return a
