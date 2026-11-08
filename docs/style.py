# style.py — document-wide typography for the Model Surgery .docx.
# Why: python-docx defaults (Calibri 11, tight headings, blue) read as a draft;
# this module sets one clean serif/sans pairing so build.py stays layout-only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

from docx.shared import Pt, RGBColor                      # Pt = points; RGBColor = heading ink
from docx.enum.text import WD_ALIGN_PARAGRAPH             # justification enum for body text

INK   = RGBColor(0x1A, 0x1A, 0x1A)                        # near-black; pure #000 looks harsh in print
ACCNT = RGBColor(0x2B, 0x4C, 0x7E)                        # muted navy for headings, not python-docx blue


def apply(doc):                                           # mutate the Document's named styles in place
    body = doc.styles["Normal"]                           # every unstyled paragraph inherits Normal
    body.font.name = "Helvetica Neue"                     # macOS-native sans; renders in Word and Pages
    body.font.size = Pt(11)                               # 11pt at 1.25 leading = comfortable column
    body.font.color.rgb = INK                             # override the theme's automatic colour
    body.paragraph_format.space_after = Pt(10)            # air between paragraphs instead of indents
    body.paragraph_format.line_spacing = 1.25             # loosened leading; default 1.0 reads cramped
    body.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT   # ragged right; justified sans gets rivers

    for name, size in (("Title", 30), ("Heading 1", 16), ("Heading 2", 13)):
        st = doc.styles[name]                             # Title/H1/H2 are the only levels this doc uses
        st.font.name = "Helvetica Neue"                   # keep one family; weight carries the hierarchy
        st.font.size = Pt(size)                           # explicit sizes; theme defaults are inconsistent
        st.font.bold = True                               # H1/H2 ship semi-bold, which vanishes on screen
        st.font.color.rgb = ACCNT if name != "Title" else INK   # navy heads, black title
        st.paragraph_format.space_before = Pt(18 if name == "Heading 1" else 12)  # separate from prior block
        st.paragraph_format.space_after = Pt(6)           # heading hugs the text it introduces
