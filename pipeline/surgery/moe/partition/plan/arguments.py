# ../../moe/partition/plan/arguments.py — argparse interface for moe_partition; geometry is explicit caller policy.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import argparse


def parse_args():
    """Model-independent CLI: stats dir + the four geometry ints the caller must state."""
    a = argparse.ArgumentParser(prog="moe_partition")
    a.add_argument("stats_dir", help="dir holding L<digits>/{neuron.npy,sample.npy}")
    a.add_argument("--experts",      type=int, required=True, help="expert count E")
    a.add_argument("--expert-width", type=int, required=True, help="neurons per expert")
    a.add_argument("--shared-width", type=int, required=True, help="always-on shared neurons (>=0)")
    a.add_argument("--group-size",   type=int, required=True, help="quant group size G (group mode)")
    a.add_argument("--layers", default=None, help="spec like 0-55,2,7; default = every L<digits> dir")
    return a.parse_args()
