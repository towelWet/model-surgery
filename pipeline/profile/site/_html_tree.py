# _html_tree.py — minimal HTMLParser that records tag order + parse errors.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# One class = one unit; its callbacks are HTMLParser hooks, not separate concerns.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from html.parser import HTMLParser
from pipeline.profile.site.markup.initialise import initialise
from pipeline.profile.site.markup.handle_starttag import handle_starttag
from pipeline.profile.site.markup.handle_endtag import handle_endtag
from pipeline.profile.site.markup.error import error


class _Tree(HTMLParser):
    __init__ = initialise
    handle_starttag = handle_starttag
    handle_endtag = handle_endtag
    error = error
