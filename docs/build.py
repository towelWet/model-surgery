#!/usr/bin/env python3
# build.py — assemble docs/Model Surgery.docx from content.toml + tables.toml.
# Why: the explainer is regenerated, never hand-edited in Word, so prose edits
# happen in TOML and the layout stays reproducible. Entry point for this folder.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

import tomllib, sys                                         # tomllib = stdlib TOML reader (3.11+)
from pathlib import Path                                    # path joins relative to this file
from docx import Document                                   # python-docx: the .docx writer
from docx.shared import Pt                                  # subtitle sizing

sys.path.insert(0, str(Path(__file__).parent))              # import siblings when run from any cwd
import style, render, place                                 # typography, table drawing, figure placement

HERE = Path(__file__).parent                                # docs/ — both TOMLs and the output live here
OUT  = HERE / "Model Surgery.docx"                          # final deliverable name, spaces intentional


from config.load import load
from sections.render import render_sections


def main():
    content, tables = load("content.toml"), load("tables.toml")
    figures = load("figures.toml")                          # figure anchors + captions
    doc = Document()                                        # blank document using the default template
    style.apply(doc)                                        # restyle Normal/Title/H1 before adding content

    doc.add_heading(content["title"], 0)                    # level 0 = Title style
    sub = doc.add_paragraph()                               # tagline under the title
    run = sub.add_run(content["subtitle"])                  # own run so italics/size apply to text only
    run.italic, run.font.size = True, Pt(13)                # italic 13pt reads as a deck, not body copy

    figs = {}                                               # head -> [figure spec], drawn before tables
    for spec in figures["figure"]:
        figs.setdefault(spec["after"], []).append(spec)

    by_anchor = {}                                          # head -> [table spec], tables follow their section
    for spec in tables["table"]:                            # tables.toml lists anchors out of document order
        by_anchor.setdefault(spec["after"], []).append(spec)

    render_sections(doc, content, figs, by_anchor, HERE)

    doc.save(OUT)                                           # write once, at the end
    print(OUT)                                              # echo path so the caller can open it


if __name__ == "__main__":
    main()                                                  # no args: the TOMLs are the only input
