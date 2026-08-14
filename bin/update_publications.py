#!/usr/bin/env python
"""Sync _bibliography/papers.bib with the author's Google Scholar profile.

Existing entries are never modified or deleted — manual curation (preview
images, selected={true}, oral={true}, co-first-author stars, short venue
names) always wins. Publications found on Scholar that are not yet in the
bib file are appended, enriched with arXiv metadata when a confident title
match exists there.

Runs from .github/workflows/update-publications.yml; safe to run locally:
    python bin/update_publications.py
"""

import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import yaml
from scholarly import scholarly

BIB_FILE = "_bibliography/papers.bib"
SOCIALS_FILE = "_data/socials.yml"
PREVIEW_DIR = "assets/img/publication_preview"
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


def load_scholar_user_id() -> str:
    if not os.path.exists(SOCIALS_FILE):
        print(f"Configuration file {SOCIALS_FILE} not found.")
        sys.exit(1)
    with open(SOCIALS_FILE) as f:
        config = yaml.safe_load(f)
    scholar_user_id = config.get("scholar_userid")
    if not scholar_user_id:
        print(f"No 'scholar_userid' found in {SOCIALS_FILE}.")
        sys.exit(1)
    return scholar_user_id


def normalize_title(title: str) -> str:
    """Reduce a title to a comparable key: lowercase alphanumerics only.

    LaTeX math spans are dropped entirely because Scholar tends to lose
    superscripts: "MERGE$^3$: ..." must collide with "MERGE : ...".
    """
    title = unicodedata.normalize("NFKD", title)
    title = re.sub(r"\$[^$]*\$", "", title)
    title = re.sub(r"\\[a-zA-Z]+", "", title)
    return re.sub(r"[^a-z0-9]", "", title.lower())


def existing_titles(bib_text: str) -> set:
    """Collect normalized titles already present in papers.bib."""
    titles = set()
    for match in re.finditer(r'\btitle\s*=\s*(?:\{+(.*?)\}+|"(.*?)")\s*,?\s*$', bib_text, re.MULTILINE | re.IGNORECASE):
        raw = match.group(1) or match.group(2) or ""
        if raw:
            titles.add(normalize_title(raw))
    return titles


def arxiv_lookup(title: str) -> dict | None:
    """Search arXiv for an exact (normalized) title match.

    Best-effort enrichment only: the arXiv API rate-limits aggressively, so
    we space requests ≥3s apart (per their ToS) and give up after one retry.
    """
    query = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "max_results": 5})
    root = None
    for attempt in range(2):
        time.sleep(3.1)
        try:
            with urllib.request.urlopen(f"{ARXIV_API}?{query}", timeout=20) as resp:
                root = ET.fromstring(resp.read())
            break
        except Exception as e:
            print(f"  arXiv lookup attempt {attempt + 1} failed for '{title}': {e}")
    if root is None:
        return None
    wanted = normalize_title(title)
    for entry in root.iter(f"{ATOM}entry"):
        entry_title = entry.findtext(f"{ATOM}title", default="")
        if normalize_title(entry_title) != wanted:
            continue
        arxiv_url = entry.findtext(f"{ATOM}id", default="").strip()
        eprint = re.sub(r"v\d+$", "", arxiv_url.rsplit("/", 1)[-1])
        authors = [a.findtext(f"{ATOM}name", default="").strip() for a in entry.iter(f"{ATOM}author")]
        category = entry.find("{http://arxiv.org/schemas/atom}primary_category")
        return {
            "eprint": eprint,
            "url": f"https://arxiv.org/abs/{eprint}",
            "authors": [a for a in authors if a],
            "primary_class": category.get("term") if category is not None else "",
        }
    return None


def shrink_preview(dest: str) -> None:
    """Downscale a preview image to a web-friendly width (needs Pillow)."""
    try:
        from PIL import Image
    except ImportError:
        return
    try:
        img = Image.open(dest)
        if img.width > 800:
            img.thumbnail((800, 8000))
            img.save(dest)
    except Exception as e:
        print(f"  could not downscale {dest}: {e}")


def save_preview(data: bytes, eprint: str, ext: str = ".png") -> str:
    name = f"auto-{eprint.replace('.', '-')}{ext}"
    dest = os.path.join(PREVIEW_DIR, name)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    shrink_preview(dest)
    return name


_PDF_CACHE: dict = {}


def _open_pdf(eprint: str):
    """Download and open the arXiv PDF (cached); returns a pymupdf doc or None."""
    try:
        import pymupdf
    except ImportError:
        print("  PyMuPDF not installed — skipping PDF preview")
        return None
    if eprint not in _PDF_CACHE:
        try:
            req = urllib.request.Request(f"https://arxiv.org/pdf/{eprint}", headers={"User-Agent": "personal-site-build"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                _PDF_CACHE[eprint] = resp.read()
        except Exception as e:
            print(f"  could not download PDF for {eprint}: {e}")
            _PDF_CACHE[eprint] = None
    if _PDF_CACHE[eprint] is None:
        return None
    return pymupdf.open(stream=_PDF_CACHE[eprint], filetype="pdf")


def pdf_figure1_region(eprint: str) -> str | None:
    """Extract Figure 1 from the arXiv PDF (needs PyMuPDF).

    Finds the "Figure 1" caption block, collects the drawings/images sitting
    above it in the same column, and renders that clipped region. Returns
    None when no caption is found.
    """
    import pymupdf  # guaranteed present when _open_pdf succeeds

    doc = _open_pdf(eprint)
    if doc is None:
        return None
    for page in doc.pages(0, min(8, doc.page_count)):
        caption = next(
            (
                pymupdf.Rect(b[:4])
                for b in page.get_text("blocks")
                if b[6] == 0 and re.match(r"\s*(?:Figure|Fig\.?)\s*1\s*[:.)]", b[4])
            ),
            None,
        )
        if caption is None:
            continue
        pieces = []
        try:
            pieces += page.cluster_drawings()
        except Exception:
            pass
        try:
            pieces += [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
        except Exception:
            pass
        figure = None
        for rect in pieces:
            # Allow the artwork's bounding box to overlap the caption a little
            # (matplotlib exports often have padding); we clamp the crop below.
            near_above = rect.y0 < caption.y0 - 20 and rect.y1 <= caption.y0 + 60 and caption.y0 - rect.y0 < 500
            same_column = rect.x1 > caption.x0 - 20 and rect.x0 < caption.x1 + 20
            if near_above and same_column and rect.width > 60 and rect.height > 40:
                figure = rect if figure is None else figure | rect
        if figure is None:  # caption found but nothing detectable above it
            figure = pymupdf.Rect(caption.x0, max(caption.y0 - 380, 20), caption.x1, caption.y0 - 2)
        figure = figure + (-6, -6, 6, 6)
        # Axis/tick labels are text (invisible to cluster_drawings): cover the
        # caption's full width and run right down to it.
        figure.x0 = min(figure.x0, caption.x0)
        figure.x1 = max(figure.x1, caption.x1)
        figure.y1 = caption.y0 - 2
        figure = figure & page.rect
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=figure)
        name = save_preview(pix.tobytes("png"), eprint)
        print(f"  preview: extracted Figure 1 from PDF as {name}")
        return name
    return None


def pdf_page1_render(eprint: str) -> str | None:
    """Last-resort preview: render the first page of the PDF."""
    import pymupdf

    doc = _open_pdf(eprint)
    if doc is None:
        return None
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
    name = save_preview(pix.tobytes("png"), eprint)
    print(f"  preview: rendered PDF page 1 as {name}")
    return name


def find_figure1_src(html: str) -> str | None:
    """Locate Figure 1's raster image in arXiv's LaTeXML HTML.

    The caption tag's text may be wrapped in nested spans, so we strip tags
    before checking for "Figure 1". Returns None when Figure 1 doesn't exist
    or is vector-only (an SVG <object>, typical for TikZ) — callers should
    then extract the figure from the PDF instead.
    """
    for tag in re.finditer(r'class="ltx_tag ltx_tag_figure">', html):
        visible = re.sub(r"<[^>]+>", "", html[tag.end() : tag.end() + 300])
        if not re.match(r"\s*(?:Figure|Fig\.?)\s*1(?!\d)", visible):
            continue
        fig_start = html.rfind("<figure", 0, tag.start())
        if fig_start == -1:
            return None
        fig_end = html.find("</figure>", tag.start())
        # Images usually precede the caption; search after it too for \caption-first figures.
        for segment in (html[fig_start : tag.start()], html[tag.start() : fig_end if fig_end != -1 else None]):
            for img_tag in re.findall(r"<img[^>]+>", segment):
                if "ltx_graphics" in img_tag:
                    src_match = re.search(r'src="([^"]+)"', img_tag)
                    if src_match:
                        return src_match.group(1)
        return None
    return None


def download_html_image(src: str, base_url: str, eprint: str, label: str) -> str | None:
    # The final page URL may or may not carry the version suffix, and the
    # src may or may not repeat the versioned directory — try both bases.
    candidates = [
        urllib.parse.urljoin(base_url, src),
        urllib.parse.urljoin("https://arxiv.org/html/", src),
        urllib.parse.urljoin(base_url.rstrip("/") + "/", src),
    ]
    ext = os.path.splitext(urllib.parse.urlparse(src).path)[1] or ".png"
    for img_url in dict.fromkeys(candidates):
        try:
            req = urllib.request.Request(img_url, headers={"User-Agent": "personal-site-build"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except Exception:
            continue
        name = save_preview(data, eprint, ext)
        print(f"  preview: {label} saved as {name}")
        return name
    print(f"  could not download {label} for {eprint} (tried {len(set(candidates))} URLs)")
    return None


def fetch_first_figure(eprint: str) -> str | None:
    """Download Figure 1 of the paper.

    Order of attempts: the raster image inside the figure captioned
    "Figure 1" in arXiv's HTML rendering; Figure 1 extracted from the PDF
    (covers vector/TikZ figures the HTML serves only as SVG); the first
    raster figure in the HTML; a render of PDF page 1.
    """
    time.sleep(3.1)
    html, base_url = "", ""
    try:
        req = urllib.request.Request(f"https://arxiv.org/html/{eprint}", headers={"User-Agent": "personal-site-build"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            base_url = resp.geturl()
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  no arXiv HTML version for {eprint}: {e}")

    src = find_figure1_src(html)
    if src and (name := download_html_image(src, base_url, eprint, "Figure 1")):
        return name

    if name := pdf_figure1_region(eprint):
        return name

    # Old submissions without a tagged Figure 1: first raster figure in the HTML.
    for img_tag in re.findall(r"<img[^>]+>", html):
        if "ltx_graphics" in img_tag and (src_match := re.search(r'src="([^"]+)"', img_tag)):
            if name := download_html_image(src_match.group(1), base_url, eprint, "first figure"):
                return name
            break

    return pdf_page1_render(eprint)


def fetch_abstract(eprint: str) -> str | None:
    """Fetch a paper's abstract from its arXiv abs page (og:description)."""
    from html import unescape

    time.sleep(1.0)
    try:
        req = urllib.request.Request(f"https://arxiv.org/abs/{eprint}", headers={"User-Agent": "personal-site-build"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  could not fetch abstract for {eprint}: {e}")
        return None
    match = re.search(r'<meta property="og:description" content="(.*?)"\s*/?>', page, re.S)
    if not match:
        return None
    abstract = unescape(match.group(1))
    abstract = re.sub(r"\s+", " ", abstract).strip()
    # Braces would break the BibTeX field delimiters.
    return abstract.replace("{", "(").replace("}", ")")


def bibtex_key(title: str, year: str, authors: str) -> str:
    first_author = re.split(r"\s+and\s+", authors)[0] if authors else "unknown"
    surname = first_author.strip().split()[-1].lower() if first_author.strip() else "unknown"
    slug = normalize_title(title)[:30]
    return re.sub(r"[^a-z0-9]", "", f"{surname}{year}{slug}")


def entry_from_scholar(pub: dict) -> str | None:
    """Build a BibTeX entry string for one filled scholarly publication."""
    bib = pub.get("bib", {})
    title = bib.get("title")
    if not title:
        return None
    year = str(bib.get("pub_year", "")) or "n.d."
    venue = bib.get("journal") or bib.get("booktitle") or bib.get("conference") or bib.get("citation") or ""
    venue = re.sub(r",?\s*\d{4}\s*$", "", venue).strip()  # drop trailing year Scholar appends
    authors = bib.get("author", "")

    # Scholar's pub_url often points straight at arXiv — that beats the API.
    url_match = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})", pub.get("pub_url") or "")
    eprint = url_match.group(1) if url_match else None
    primary_class = ""

    arxiv = arxiv_lookup(title)
    if arxiv:
        eprint = arxiv["eprint"]
        primary_class = arxiv["primary_class"]
        if arxiv["authors"]:  # arXiv names keep their accents; Scholar's often don't
            authors = " and ".join(arxiv["authors"])

    # "arXiv preprint arXiv:2607.0" is Scholar's (truncated) way of saying
    # there is no real venue yet — those papers are under submission.
    if re.match(r"arxiv preprint", venue, re.IGNORECASE):
        venue = ""
    # Scholar venue strings can be verbose and truncated ("ECAI 2023: 26th
    # European Conference …"); keep the short name before the colon and drop
    # any dangling ellipsis.
    venue = re.sub(r"[,\s]*(…|\.\.\.)\s*$", "", venue.split(":")[0]).strip()

    if not venue:
        entry_type, venue_field, venue = "article", "journal", "Under Sub."
    elif bib.get("journal"):
        entry_type, venue_field = "article", "journal"
    else:
        entry_type, venue_field = "inproceedings", "booktitle"

    preview = fetch_first_figure(eprint) if eprint else None

    key = bibtex_key(title, year, authors)
    lines = [f"@{entry_type}{{{key},", f"  title        = {{{title}}},", f"  author       = {{{authors}}},"]
    lines.append(f"  {venue_field:<13}= {{{venue}}},")
    lines.append(f"  year         = {{{year}}},")
    if preview:
        lines.append(f"  preview      = {{{preview}}},")
    if eprint and (abstract := fetch_abstract(eprint)):
        lines.append(f"  abstract     = {{{abstract}}},")
    if eprint:
        lines.append(f"  eprint       = {{{eprint}}},")
        lines.append("  archivePrefix= {arXiv},")
        if primary_class:
            lines.append(f"  primaryClass = {{{primary_class}}},")
        lines.append(f"  url          = {{https://arxiv.org/abs/{eprint}}},")
    elif pub.get("pub_url"):
        lines.append(f"  url          = {{{pub['pub_url']}}},")
    lines.append("}")
    return "\n".join(lines)


def assign_scholar_ranks(publications: list) -> bool:
    """Write scholar_rank = {NNN} into every entry of papers.bib, mirroring
    the Google Scholar profile order (most cited first; zero-padded so the
    lexical sort in jekyll-scholar matches numeric order).

    This is the one field the sync maintains inside existing entries — all
    other manual curation is never touched. Entries not found on Scholar
    sink to rank 900. Returns True if the file changed.
    """
    rank_of = {}
    for i, pub in enumerate(publications, start=1):
        key = normalize_title(pub.get("bib", {}).get("title", ""))
        if key and key not in rank_of:
            rank_of[key] = i

    with open(BIB_FILE) as f:
        lines = f.readlines()

    out, i = [], 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("@"):
            out.append(lines[i])
            i += 1
            continue
        entry = [lines[i]]
        i += 1
        while i < len(lines):
            entry.append(lines[i])
            i += 1
            if entry[-1].strip() == "}":
                break
        text = "".join(entry)
        title_match = re.search(r'\btitle\s*=\s*(?:\{+(.*?)\}+|"(.*?)")', text, re.S | re.I)
        title = (title_match.group(1) or title_match.group(2) or "") if title_match else ""
        rank_line = f"  scholar_rank = {{{rank_of.get(normalize_title(title), 900):03d}}},\n"

        replaced = False
        for j, line in enumerate(entry):
            if re.match(r"\s*scholar_rank\s*=", line):
                entry[j] = rank_line
                replaced = True
                break
        if not replaced:
            close = len(entry) - 1
            while close > 0 and entry[close].strip() != "}":
                close -= 1
            prev = close - 1
            while prev > 0 and not entry[prev].strip():
                prev -= 1
            body = entry[prev].rstrip("\n").rstrip()
            if not body.endswith(","):  # BibTeX needs a comma before the new field
                entry[prev] = body + ",\n"
            entry.insert(close, rank_line)
        out.extend(entry)

    new_text = "".join(out)
    old_text = "".join(lines)
    if new_text != old_text:
        with open(BIB_FILE, "w") as f:
            f.write(new_text)
        return True
    return False


def backfill_abstracts() -> int:
    """Add an abstract field to existing arXiv entries that lack one.

    Only ever ADDS a missing field — hand-written abstracts are never touched.
    Returns the number of abstracts added.
    """
    with open(BIB_FILE) as f:
        lines = f.readlines()
    out, i, added = [], 0, 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("@"):
            out.append(lines[i])
            i += 1
            continue
        entry = [lines[i]]
        i += 1
        while i < len(lines):
            entry.append(lines[i])
            i += 1
            if entry[-1].strip() == "}":
                break
        text = "".join(entry)
        eprint_match = re.search(r'^\s*eprint\s*=\s*[{"]([^}"]+)[}"]', text, re.M)
        if eprint_match and not re.search(r"^\s*abstract\s*=", text, re.M):
            abstract = fetch_abstract(eprint_match.group(1).strip())
            if abstract:
                close = len(entry) - 1
                while close > 0 and entry[close].strip() != "}":
                    close -= 1
                prev = close - 1
                while prev > 0 and not entry[prev].strip():
                    prev -= 1
                body = entry[prev].rstrip("\n").rstrip()
                if not body.endswith(","):
                    entry[prev] = body + ",\n"
                entry.insert(close, f"  abstract     = {{{abstract}}},\n")
                added += 1
        out.extend(entry)
    if added:
        with open(BIB_FILE, "w") as f:
            f.write("".join(out))
    return added


def main() -> None:
    scholar_user_id = load_scholar_user_id()
    print(f"Fetching publications for Google Scholar ID: {scholar_user_id}")

    with open(BIB_FILE) as f:
        bib_text = f.read()
    known = existing_titles(bib_text)
    print(f"{len(known)} publications already in {BIB_FILE}")

    scholarly.set_timeout(15)
    scholarly.set_retries(3)
    # sortby="year" mirrors the Scholar profile's chronological view (newest first)
    author = scholarly.fill(scholarly.search_author_id(scholar_user_id), sections=["publications"], sortby="year")
    publications = author.get("publications", [])
    print(f"{len(publications)} publications on Google Scholar")

    new_entries = []
    for pub in publications:
        title = pub.get("bib", {}).get("title", "")
        if not title or normalize_title(title) in known:
            continue
        print(f"New publication: {title}")
        try:
            filled = scholarly.fill(pub)
        except Exception as e:
            print(f"  Could not fetch details, using summary data: {e}")
            filled = pub
        entry = entry_from_scholar(filled)
        if entry:
            known.add(normalize_title(title))
            new_entries.append(entry)
        time.sleep(2)  # be polite to Scholar between detail fetches

    if new_entries:
        with open(BIB_FILE, "a") as f:
            f.write("\n" + "\n\n".join(new_entries) + "\n")
        print(f"Appended {len(new_entries)} new publication(s) to {BIB_FILE}")
        print("Tip: add preview={...}, selected={true} or a short booktitle by editing the new entries.")
    else:
        print("No new publications to add.")

    if assign_scholar_ranks(publications):
        print("Updated scholar_rank ordering to match the Scholar profile.")
    else:
        print("scholar_rank ordering already up to date.")

    added_abstracts = backfill_abstracts()
    if added_abstracts:
        print(f"Backfilled {added_abstracts} abstract(s) from arXiv.")


if __name__ == "__main__":
    main()
