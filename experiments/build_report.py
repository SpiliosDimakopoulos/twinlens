"""Render paper/REPORT.md into a styled web page for the live site (and the PDF).

    python experiments/build_report.py

Writes ``web/public/report/index.html`` with the figures next to it. The PDF is printed from
that page by ``web/scripts/report-pdf.mjs``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "report"
REPO = "https://github.com/SpiliosDimakopoulos/twinlens"

CSS = """
:root { color-scheme: light; --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --line:#e1e0d9; --accent:#2a78d6; --code:#f3f2ee; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --bg:#141413; --ink:#f0efec;
  --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --accent:#6da7ec; --code:#1f1f1d; }
  img { background:#fcfcfb; border-radius:8px; } }
* { box-sizing: border-box; }
html { background: var(--bg); }
body { margin: 0 auto; max-width: 860px; padding: 40px 20px 80px; color: var(--ink);
  font: 16px/1.65 system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); }
nav { display:flex; gap:16px; flex-wrap:wrap; font-size:14px; margin-bottom:28px; }
nav a { color: var(--ink2); }
h1 { font-size: 30px; line-height: 1.2; letter-spacing: -0.01em; margin: 0 0 8px; }
h1 + h3 { font-weight: 500; color: var(--ink2); margin-top: 0; font-size: 18px; }
h2 { margin-top: 40px; font-size: 22px; }
h3 { margin-top: 30px; font-size: 18px; }
a { color: var(--accent); }
p, li { color: var(--ink); }
em { color: var(--ink2); }
img { max-width: 100%; height: auto; display: block; margin: 16px auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 16px 0;
  font-variant-numeric: tabular-nums; display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
th { color: var(--ink2); font-weight: 600; }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; font-size: 0.9em; }
pre { background: var(--code); padding: 12px 14px; border-radius: 8px; overflow-x: auto; font-size: 13px; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid var(--line); margin: 28px 0; }
.meta { color: var(--muted); font-size: 13px; }
@media print { html, body { background: #fff; } body { max-width: none; padding: 0 8mm; font-size: 11pt; } nav { display:none; }
  h2 { break-before: auto; } img, table, pre { break-inside: avoid; } a { color: inherit; } }
"""


def normalise_indent(md: str) -> str:
    """GitHub accepts 2-3 space list nesting; Python-Markdown needs 4. Re-indent list content
    (outside fenced/indented code) to multiples of 4 so both render the same structure."""
    out, in_fence = [], False
    for line in md.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        n = len(line) - len(line.lstrip(" "))
        if not in_fence and line.strip() and n in (2, 3, 5, 6, 7):
            line = " " * (4 if n < 4 else 8) + line.lstrip(" ")
        # Python-Markdown also needs a blank line between a paragraph and a top-level list.
        is_item = bool(re.match(r"(- |\d+\. )", line))
        prev = out[-1] if out else ""
        if (
            not in_fence
            and is_item
            and prev.strip()
            and not re.match(r"\s*(- |\d+\. )", prev)
            and not prev.startswith(" ")
            and not prev.startswith("|")
        ):
            out.append("")
        out.append(line)
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figs = OUT / "figures"
    figs.mkdir(exist_ok=True)
    for png in (ROOT / "experiments" / "figures").glob("*.png"):
        shutil.copy2(png, figs / png.name)

    md = (ROOT / "paper" / "REPORT.md").read_text(encoding="utf-8")
    md = md.replace("../experiments/figures/", "figures/")
    md = normalise_indent(md)
    # Repository-relative code references become links to GitHub.
    md = re.sub(
        r"`((?:tests|twinlens|experiments|server)/[\w/.]+\.py)`",
        lambda m: f"[`{m.group(1)}`]({REPO}/blob/main/{m.group(1)})",
        md,
    )
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    title = re.search(r"^# (.+)$", md, re.M).group(1)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TwinLens report</title>
<meta name="description" content="{title}">
<style>{CSS}</style></head>
<body>
<nav><a href="../">← Live demo</a><a href="TwinLens_report.pdf">PDF</a><a href="{REPO}">Code on GitHub</a></nav>
{body}
</body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} ({len(html) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
