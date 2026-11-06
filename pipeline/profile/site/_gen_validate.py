# _gen_validate.py — cheap HTML sanity checks for site_gen_bench (not a linter).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _html_tree import _Tree                    # tag recorder (split out)
from _html_unfence import strip_fences          # fence remover (split out)


def score_html(text):
    raw = strip_fences(text)
    low = raw.lower()
    has_doctype = low.startswith("<!doctype") or "<!doctype" in low[:120]
    has_html = "<html" in low and "</html>" in low
    has_style = "<style" in low
    parser = _Tree()
    try:
        parser.feed(raw[:500_000])              # cap input: runaway generations
    except Exception as exc:
        parser.errors.append(str(exc))
    return {
        "chars": len(raw),
        "has_doctype": has_doctype,
        "has_html_root": has_html,
        "has_style": has_style,
        "parse_errors": len(parser.errors),
        "ok": has_doctype and has_html and has_style and not parser.errors,
    }
