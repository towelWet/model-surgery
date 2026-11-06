# _run_modes.py — dispatch profile/block/ablation.py flags to spec / import / distance / score modes.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys

from _rank import from_distance, import_results
from _ablation_paths import write_spec
from _web_prompts import web_prompts


def run_modes(p, args):
    if args.do_import:
        import_results()
        return
    if not args.model_dir:
        p.print_help()
        sys.exit(1)
    if args.write_spec:
        write_spec(args.model_dir)
    elif args.from_distance:
        from_distance(args.model_dir)
    elif args.score_api:
        from _live import score_live
        score_live(args.model_dir)
    elif args.score:
        from _ablation_scorers import score_groups                     # local NLL (loads model in-process)
        score_groups(args.model_dir, prompts=web_prompts() if args.web else None)
    else:
        write_spec(args.model_dir)
        from_distance(args.model_dir)

