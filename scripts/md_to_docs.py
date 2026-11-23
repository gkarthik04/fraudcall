"""Convert report/report.md to report/report.docx and report.pdf with markdown-it + LibreOffice (images embedded)."""
import os, re, subprocess, shutil
from markdown_it import MarkdownIt
src = open("report/report.md", encoding="utf8").read()
html = MarkdownIt("commonmark").enable("table").render(src)
html = html.replace('src="../figures/', 'src="file:///' + os.path.abspath("figures").replace("\\", "/") + "/")
css = "<style>body{font-family:Calibri,Arial;font-size:11pt}table{border-collapse:collapse}td,th{border:1px solid #888;padding:3px 6px;font-size:9pt}img{width:15cm}</style>"
os.makedirs("tmp_phase0/doc", exist_ok=True)
open("tmp_phase0/doc/report.html", "w", encoding="utf8").write(f"<html><head><meta charset='utf-8'>{css}</head><body>{html}</body></html>")
so = "C:/Program Files/LibreOffice/program/soffice.exe"
subprocess.run([so, "--headless", "--convert-to", "docx:MS Word 2007 XML", "--outdir", "report", "tmp_phase0/doc/report.html"], check=False)
subprocess.run([so, "--headless", "--convert-to", "pdf", "--outdir", "report", "report/report.docx"], check=False)
print(os.listdir("report"))
