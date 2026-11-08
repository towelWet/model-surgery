import place
import render


def render_sections(doc, content, figs, by_anchor, base):
    for sec in content["section"]:                          # sections render in file order
        doc.add_heading(sec["head"], 1)                     # H1 per section
        for para in sec["body"]:                            # each body string is one paragraph
            doc.add_paragraph(para)                         # inherits the restyled Normal
        for spec in figs.get(sec["head"], []):              # figure first: the picture leads, the table backs it
            place.figure(doc, base, spec)
        for spec in by_anchor.get(sec["head"], []):         # zero or more tables attached to this head
            render.table(doc, spec)                         # draw it immediately after the prose

