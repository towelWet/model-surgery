# parse_eval100.py — extract metrics from eval100 output files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import re, sys, glob

def parse(path):
    text = open(path).read()
    sel = re.search(r"tool_selection_acc : (\d+)/(\d+) = ([\d.]+)%", text)
    arg = re.search(r"arg_validity\s+: (\d+)/(\d+) = ([\d.]+)%", text)
    kl  = re.search(r"mean_kl_parent_student : ([\d.]+)", text)
    return {
        "tool_sel":  float(sel.group(3))/100 if sel else None,
        "arg_valid": float(arg.group(3))/100 if arg else None,
        "kl":        float(kl.group(1)) if kl else None,
    }

if __name__ == "__main__":
    for p in sorted(glob.glob("/tmp/eval100_*_r*.txt")):
        m = parse(p)
        print(p, m)
