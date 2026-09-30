#!/usr/bin/env python3
"""Build the HTML site, the full PDF, the sample PDF and the EPUB from the chapter markdown.

    python3 build/build.py            # everything
    python3 build/build.py site       # HTML site only
    python3 build/build.py pdf        # full PDF + sample PDF
    python3 build/build.py epub       # EPUB only

Needs: quarto, pandoc, Google Chrome/Chromium (headless) and PyMuPDF (pip install pymupdf).
Sources are copied to a temporary directory for rendering, so nothing but the
outputs is written into the book folder.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

BUILD = Path(__file__).resolve().parent
BOOK = BUILD.parent  # html/NN/ml2

TITLE = "The Mathematical Intuition Behind Deep Learning"
SUBTITLE = "From the Dot Product to Multivariate Calculus and the Jacobian, with Python"
AUTHOR = "Alex Punnen"
PDF_FULL = BOOK / "pdf" / "The-Mathematical-Intuition-Behind-Deep-Learning.pdf"
PDF_SAMPLE = BOOK / "pdf" / "The-Mathematical-Intuition-Behind-Deep-Learning-Sample.pdf"
EPUB = BOOK / "epub" / "The-Mathematical-Intuition-Behind-Deep-Learning.epub"
COVER = BOOK / "images" / "cover.png"
SAMPLE_CHAPTERS = 2  # the sample has the front matter plus chapters 1..SAMPLE_CHAPTERS

MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"


def chapters():
    """index.md followed by the numbered chapter files, in order."""
    return ["index.md"] + sorted(p.name for p in BOOK.glob("[0-9]_*.md"))


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def find_chrome():
    for name in (os.environ.get("CHROME"), "google-chrome", "chromium", "chromium-browser"):
        if name and shutil.which(name):
            return shutil.which(name)
    sys.exit("Chrome/Chromium not found; set CHROME=/path/to/chrome")


def print_to_pdf(html, pdf):
    run([find_chrome(), "--headless=new", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", "--virtual-time-budget=60000",
         "--run-all-compositor-stages-before-draw",
         f"--print-to-pdf={pdf}", html.as_uri()],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not Path(pdf).exists():
        sys.exit(f"Chrome did not produce {pdf}")


def copy_sources(dst):
    for f in chapters():
        shutil.copy(BOOK / f, dst / f)
    shutil.copytree(BOOK / "images", dst / "images",
                    ignore=shutil.ignore_patterns("*.md"))


# --------------------------------------------------------------------------
# HTML site (Quarto website)
# --------------------------------------------------------------------------

def build_site(tmp):
    src = tmp / "site"
    src.mkdir()
    copy_sources(src)
    shutil.copy(BUILD / "_quarto.yml", src / "_quarto.yml")
    run(["quarto", "render"], cwd=src)

    out = src / "_site"
    for f in out.glob("*.html"):
        shutil.copy(f, BOOK / f.name)
    shutil.copy(out / "search.json", BOOK / "search.json")
    shutil.rmtree(BOOK / "site_libs", ignore_errors=True)
    shutil.copytree(out / "site_libs", BOOK / "site_libs")
    print("site  ->", BOOK)


# --------------------------------------------------------------------------
# PDF: one HTML page with every chapter, printed by Chrome, then the cover,
# links and page numbers are added with PyMuPDF.
# --------------------------------------------------------------------------

def book_html(work):
    parts = []
    for f in chapters():
        html = run(["pandoc", str(BOOK / f), "-f", "markdown", "-t", "html5", "--mathjax"],
                   capture_output=True, text=True).stdout
        parts.append(f'<section class="chapter" id="{f[:-3]}">\n{html}\n</section>')
    body = "".join(parts)
    # links between chapter files become jumps inside the PDF
    body = re.sub(r'href="((?:index|\d_[a-z_]+))\.md"', r'href="#\1"', body)

    css = (BUILD / "pdf.css").read_text()
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{TITLE}</title>
<style>{css}</style>
<script>window.MathJax = {{
  tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
  chtml: {{ scale: 0.95 }}
}};</script>
<script src="{MATHJAX}"></script>
</head><body>{body}</body></html>"""
    html_path = work / "book.html"
    html_path.write_text(page)
    return html_path


def chapter_start_pages(body):
    """0-based page index in `body` where each chapter starts, from the contents links."""
    starts = sorted({l["page"] for l in body[0].get_links()
                     if l.get("page") is not None and l["page"] > 0})
    return starts


def assemble(body, last_page, extra=None):
    """Cover + body pages [0, last_page] (+ extra pages), keeping links, adding page numbers."""
    out = fitz.open()
    w, h = body[0].rect.width, body[0].rect.height
    cover = out.new_page(width=w, height=h)
    cover.insert_image(cover.rect, filename=str(COVER), keep_proportion=True)

    out.insert_pdf(body, to_page=last_page, links=False)
    # Chrome writes named destinations, which insert_pdf drops; copy links across
    for i in range(last_page + 1):
        page = out[i + 1]
        for l in body[i].get_links():
            if l["kind"] == fitz.LINK_URI:
                page.insert_link({"kind": fitz.LINK_URI, "from": l["from"], "uri": l["uri"]})
            elif l.get("page") is not None and 0 <= l["page"] <= last_page:
                page.insert_link({"kind": fitz.LINK_GOTO, "from": l["from"],
                                  "page": l["page"] + 1, "to": l.get("to", fitz.Point(0, 0))})

    # page 1 is the first page of Chapter 1 (after the cover and the contents page)
    for i in range(2, len(out)):
        n = str(i - 1)
        tw = fitz.get_text_length(n, fontname="helv", fontsize=8)
        out[i].insert_text(((w - tw) / 2, h - 30), n, fontname="helv", fontsize=8,
                           color=(0.45, 0.45, 0.45))

    if extra is not None:
        out.insert_pdf(extra)  # unnumbered closing page(s), links kept

    out.set_metadata({"title": TITLE, "author": AUTHOR, "subject": SUBTITLE})
    return out


def build_pdf(tmp):
    work = tmp / "pdf" / "ml2"  # chapter 4 uses ../ml2/images paths
    work.mkdir(parents=True)
    shutil.copytree(BOOK / "images", work / "images")

    body_pdf = work / "body.pdf"
    print_to_pdf(book_html(work), body_pdf)
    body = fitz.open(body_pdf)

    full = assemble(body, len(body) - 1)
    full.save(PDF_FULL, garbage=4, deflate=True)
    print("pdf   ->", PDF_FULL, f"({len(full)} pages)")

    starts = chapter_start_pages(body)
    if len(starts) <= SAMPLE_CHAPTERS:
        sys.exit("could not find chapter start pages for the sample")
    end_pdf = work / "sample_end.pdf"
    print_to_pdf(BUILD / "sample_end.html", end_pdf)
    sample = assemble(body, starts[SAMPLE_CHAPTERS] - 1, extra=fitz.open(end_pdf))
    sample.save(PDF_SAMPLE, garbage=4, deflate=True)
    print("pdf   ->", PDF_SAMPLE, f"({len(sample)} pages)")


# --------------------------------------------------------------------------
# EPUB: pandoc, one section per chapter, MathML for the equations.
# --------------------------------------------------------------------------

def build_epub(tmp):
    work = tmp / "epub"
    work.mkdir()
    sources = []
    for f in chapters():
        text = (BOOK / f).read_text()
        # give each chapter's title an id and turn links between chapter files
        # into links inside the book
        text = re.sub(r"\A(\s*# .*?)[ \t]*$", rf"\1 {{#{f[:-3]}}}", text, count=1, flags=re.M)
        text = re.sub(r"\]\(((?:index|\d_[a-z_]+))\.md\)", r"](#\1)", text)
        (work / f).write_text(text)
        sources.append(str(work / f))

    EPUB.parent.mkdir(exist_ok=True)
    run(["pandoc", *sources, "-o", str(EPUB),
         "--mathml", "--split-level=1", "--toc-depth=2",
         "--resource-path", str(BOOK),
         "--epub-cover-image", str(COVER),
         "--css", str(BUILD / "epub.css"),
         "--metadata", f"title={TITLE}",
         "--metadata", f"subtitle={SUBTITLE}",
         "--metadata", f"author={AUTHOR}",
         "--metadata", "lang=en",
         "--metadata", "rights=© All Rights Reserved"],
        cwd=BOOK)  # chapter 4 uses ../ml2/images paths
    print("epub  ->", EPUB)


def main():
    targets = sys.argv[1:] or ["site", "pdf", "epub"]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        if "site" in targets:
            build_site(tmp)
        if "pdf" in targets:
            build_pdf(tmp)
        if "epub" in targets:
            build_epub(tmp)


if __name__ == "__main__":
    main()
