# place.py — insert a rendered figure plus its caption into the document.
# Why: image sizing and caption styling are their own concern; keeping them here
# lets build.py stay a pure "read TOML, walk sections" loop.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

from docx.shared import Inches, Pt, RGBColor            # Inches: page-width sizing for images
from docx.enum.text import WD_ALIGN_PARAGRAPH           # centring for both image and caption

WIDTH   = Inches(6.0)                                   # US Letter minus 1.25in margins each side
CAPTION = RGBColor(0x52, 0x51, 0x4E)                    # text-secondary; a caption must not fight the body


def figure(doc, base, spec):                            # base = docs/ dir, spec = one [[figure]] entry
    img = doc.add_paragraph()                           # own paragraph so the picture can be centred
    img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    img.add_run().add_picture(str(base / spec["file"]), width=WIDTH)   # height follows the aspect ratio

    cap = doc.add_paragraph()                           # caption sits directly under its figure
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(2)           # hug the image above it
    cap.paragraph_format.space_after = Pt(14)           # then a clear break before the next prose block
    run = cap.add_run(spec["caption"])
    run.italic, run.font.size, run.font.color.rgb = True, Pt(9), CAPTION   # italic 9pt grey = caption voice
    return cap                                          # returned so callers can post-process if needed
