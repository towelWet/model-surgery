# render.py — table rendering helper for the Model Surgery .docx.
# Why: table styling is fiddly (header bold, column fit) and would blow build.py
# past the 50-line cap; it lives here as one function with one job.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

from docx.shared import Pt                                 # cell font sizing


def table(doc, spec):                                      # spec = one [[table]] entry from tables.toml
    t = doc.add_table(rows=1, cols=len(spec["head"]))      # start with the header row only
    t.style = "Light Grid Accent 1"                        # built-in style: hairline rules, no fill blocks
    t.autofit = True                                       # let Word size columns to content width

    for cell, label in zip(t.rows[0].cells, spec["head"]): # populate header labels left to right
        run = cell.paragraphs[0].add_run(label)            # add_run so the bold applies to text, not cell
        run.bold = True                                    # header row must read as a header
        run.font.size = Pt(10)                             # one step down from body; tables carry short values

    for row in spec["rows"]:                               # body rows, already short strings by design
        cells = t.add_row().cells                          # add_row returns the new row; take its cells
        for cell, value in zip(cells, row):                # zip stops at the shorter side, so ragged rows are safe
            run = cell.paragraphs[0].add_run(str(value))   # str() guards against TOML ints slipping in
            run.font.size = Pt(10)                         # match the header size for an even grid

    doc.add_paragraph()                                    # spacer: Word jams the next heading against a table
    return t                                               # returned for callers that want further tweaks
