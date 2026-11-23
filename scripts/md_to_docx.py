"""Minimal Markdown -> DOCX with embedded images/tables (python-docx). Handles headings, paragraphs, bullets, numbered items, tables, images, **bold**, `code`."""
import re, os
from docx import Document
from docx.shared import Pt, Cm

def runs(par, text):
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)", text):
        if part.startswith("**") and part.endswith("**"): par.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"): r = par.add_run(part[1:-1]); r.font.name = "Consolas"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2: par.add_run(part[1:-1]).italic = True
        else: par.add_run(part)

d = Document(); d.styles["Normal"].font.size = Pt(10.5)
lines = open("report/report.md", encoding="utf8").read().split("\n"); i = 0
while i < len(lines):
    l = lines[i]
    if l.startswith("#"):
        n = len(l) - len(l.lstrip("#")); d.add_heading(l.lstrip("# ").strip(), min(n, 3) - (1 if n == 1 else 0) if n > 1 else 0)
    elif l.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            if not re.match(r"^\|[\s\-|]+\|$", lines[i]): rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        t = d.add_table(rows=len(rows), cols=max(len(r) for r in rows)); t.style = "Table Grid"
        for a, r in enumerate(rows):
            for b, c in enumerate(r):
                cell = t.cell(a, b); cell.text = ""; runs(cell.paragraphs[0], c)
                for rr in cell.paragraphs[0].runs: rr.font.size = Pt(8); rr.bold = rr.bold or a == 0
        continue
    elif l.startswith("!["):
        m = re.match(r"!\[.*?\]\((.*?)\)", l); p = os.path.normpath(os.path.join("report", m.group(1)))
        if os.path.exists(p): d.add_picture(p, width=Cm(14))
    elif re.match(r"^\s*[\*-] ", l): runs(d.add_paragraph(style="List Bullet"), re.sub(r"^\s*[\*-] ", "", l))
    elif re.match(r"^\d+\. ", l): runs(d.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", l))
    elif l.strip() and not l.startswith("<!--"): runs(d.add_paragraph(), l)
    i += 1
d.save("report/report.docx"); print("saved")
