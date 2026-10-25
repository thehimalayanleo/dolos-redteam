"""R9-S8: report/report.md to report/report.html and report/report.pdf, laid out like the Apart sprint template
(title box, author line, abstract box, numbered sections). PDF via headless Chrome.

uv run --quiet --with markdown python report_build.py
"""
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "report/report.md"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: Letter; margin: 0.9in 0.9in; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 10pt; line-height: 1.38; color: #1d1f21; }
.head { border: 1px solid #c9c9c9; padding: 14pt 20pt 10pt; text-align: center; margin-bottom: 12pt; }
.head h1 { font-size: 17pt; margin: 0 0 8pt; }
.head .who { margin: 0 0 2pt; }
.head .with { margin: 6pt 0 10pt; font-weight: bold; }
.head .abs { text-align: justify; border-top: 1px solid #e0e0e0; padding-top: 8pt; }
.head .abs b { display: block; text-align: center; margin-bottom: 4pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; }
h3 { font-size: 11pt; margin: 11pt 0 4pt; }
p { margin: 0 0 6pt; text-align: justify; }
ul, ol { margin: 0 0 6pt; padding-left: 18pt; }
li { margin-bottom: 2pt; }
img { display: block; width: 86%; margin: 6pt auto 2pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 4pt 0 8pt; }
th, td { border-bottom: 1px solid #d0d0d0; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { border-bottom: 1.2px solid #1d1f21; }
.refs p { text-align: left; font-size: 9pt; margin-bottom: 3pt; }
p:has(img) { break-inside: avoid; }
p:has(+ table), p:has(img), h2, h3 { break-after: avoid; }
table { break-inside: avoid; }
"""


def main():
    md = SRC.read_text().replace("<tool_call>", "&lt;tool_call&gt;").replace("<|im_end|>", "&lt;|im_end|&gt;")
    title, rest = md.split("\n", 1)
    who, with_, rest = rest.strip().split("\n\n", 2)
    abstract, body = rest.split("## 1. Introduction", 1)
    abstract = abstract.replace("## Abstract", "").strip()
    conv = lambda t: markdown.markdown(t, extensions=["tables"])
    body_html = conv("## 1. Introduction" + body)
    body_html = body_html.replace("<h2>References</h2>", '<h2>References</h2><div class="refs">', 1)
    body_html = body_html.replace("<h2>Appendix A.", '</div><h2 style="break-before: page">Appendix A.', 1)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title[2:]}</title><style>{CSS}</style></head><body>
<div class="head"><h1>{title[2:]}</h1><p class="who">{who}</p><p class="with">{with_}</p>
<div class="abs"><b>Abstract</b>{conv(abstract)}</div></div>
{body_html}</body></html>"""
    out = ROOT / "report/report.html"
    out.write_text(html)
    pdf = ROOT / "report/report.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
                   check=True, capture_output=True)
    print(pdf)


if __name__ == "__main__":
    main()
