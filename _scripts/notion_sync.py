#!/usr/bin/env python3
"""Sync Notion pages into Jekyll blog posts.

Two modes:
  python _scripts/notion_sync.py                 # sync the Blog Posts database
  python _scripts/notion_sync.py --page <url|id> # convert a single page

Environment:
  NOTION_TOKEN        internal integration token (required)
  NOTION_DATABASE_ID  database to sync (required unless --page is given)

Database pages with Status = "Publish" become posts in _posts/; images are
downloaded to assets/img/posts/<slug>/. Each generated post records its
notion_id and notion_last_edited in front matter, so unchanged pages are
skipped and edited pages are regenerated (the old file is replaced even if
the title or date changed). Posts you write by hand are never touched.

Setup (one time):
  1. Create an internal integration at https://www.notion.so/my-integrations
     (read content capability is enough) and copy the secret.
  2. In Notion, open the Blog Posts database → ••• → Connections → add the
     integration.
  3. For the GitHub Action: add NOTION_TOKEN and NOTION_DATABASE_ID as
     repository secrets (Settings → Secrets and variables → Actions).
"""

import argparse
import json
import os
import re
import shutil
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent
POSTS_DIR = WORKSPACE_ROOT / "_posts"
ASSETS_DIR = WORKSPACE_ROOT / "assets" / "img" / "posts"
API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
CALLOUT_COLORS = {"red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
                  "orange": "yellow", "purple": "purple", "pink": "purple",
                  "gray": "gray", "brown": "gray", "default": "gray"}


def api_request(path: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s_]+", "-", text).strip("-") or "untitled"


def rich_text_to_md(rich: list) -> str:
    out = []
    for seg in rich:
        if seg.get("type") == "equation":
            expr = seg["equation"]["expression"]
            if not re.search(r"[A-Za-z0-9\\]", expr):
                out.append(expr)  # decorative glyph, not math
            else:
                # raw pipes would make kramdown parse the line as a table
                expr = expr.replace("|", " \\vert ")
                out.append(f"$${expr}$$")
            continue
        text = seg.get("plain_text", "")
        ann = seg.get("annotations", {})
        # Notion often includes leading/trailing spaces inside styled segments;
        # markdown delimiters must hug the text, so keep the spaces outside.
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()) :]
        core = text.strip()
        if core and ann.get("code"):
            text = f"{lead}`{core}`{trail}"
        elif core:
            if ann.get("bold"):
                core = f"**{core}**"
            if ann.get("italic"):
                core = f"_{core}_"
            if ann.get("strikethrough"):
                core = f"~~{core}~~"
            text = f"{lead}{core}{trail}"
        if seg.get("href"):
            text = f"[{text}]({seg['href']})"
        out.append(text)
    return "".join(out)


def fetch_children(block_id: str) -> list:
    blocks, cursor = [], None
    while True:
        query = f"?start_cursor={cursor}" if cursor else ""
        data = api_request(f"/blocks/{block_id}/children{query}")
        blocks.extend(data.get("results", []))
        if not data.get("has_more"):
            return blocks
        cursor = data["next_cursor"]


def download_image(url: str, slug: str, block_id: str) -> str:
    """Save a (possibly expiring) Notion image URL locally; return site path."""
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".png"
    name = f"notion-{block_id[:8]}{ext}"
    dest = ASSETS_DIR / slug / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            dest.write_bytes(resp.read())
    except Exception as e:
        print(f"    warning: could not download image {url[:80]}: {e}")
        return url
    return f"/assets/img/posts/{slug}/{name}"


def blocks_to_md(blocks: list, slug: str, depth: int = 0) -> list:
    lines = []
    indent = "    " * depth

    def add(text: str = ""):
        for line in text.split("\n"):
            lines.append(f"{indent}{line}" if line else "")

    list_types = ("bulleted_list_item", "numbered_list_item", "to_do")
    prev_type = None
    for block in blocks:
        btype = block.get("type")
        if prev_type in list_types and btype not in list_types:
            add()  # close the list so kramdown doesn't absorb the next block
        prev_type = btype
        content = block.get(btype, {})
        rich = rich_text_to_md(content.get("rich_text", []))

        if btype == "paragraph":
            add(rich)
            add()
        elif btype in ("heading_1", "heading_2", "heading_3"):
            level = int(btype[-1]) + 1  # post title is the only h1
            add(f"{'#' * level} {rich}")
            add()
        elif btype == "bulleted_list_item":
            add(f"- {rich}")
        elif btype == "numbered_list_item":
            add(f"1. {rich}")
        elif btype == "to_do":
            add(f"- [{'x' if content.get('checked') else ' '}] {rich}")
        elif btype == "quote":
            add(f"> {rich}")
            add()
        elif btype == "toggle":
            add('<details class="notion-toggle" markdown="1">')
            add(f'<summary markdown="span">{rich}</summary>')
            add()
            if block.get("has_children"):
                lines.extend(blocks_to_md(fetch_children(block["id"]), slug, depth))
            add("</details>")
            add()
        elif btype == "callout":
            icon = (content.get("icon") or {}).get("emoji", "💡")
            hue = CALLOUT_COLORS.get(content.get("color", "").replace("_background", ""), "gray")
            add(f'<div class="notion-callout notion-callout--{hue}" markdown="1">')
            add(f'<div class="notion-callout-icon">{icon}</div>')
            add('<div class="notion-callout-body" markdown="1">')
            add()
            add(rich)
            add()
            if block.get("has_children"):
                lines.extend(blocks_to_md(fetch_children(block["id"]), slug, depth))
            add("</div>")
            add("</div>")
            add()
        elif btype == "column_list":
            add('<div class="notion-columns" markdown="1">')
            for col in fetch_children(block["id"]):
                ratio = (col.get("column") or {}).get("width_ratio") or 1
                add(f'<div class="notion-column" style="flex: {round(float(ratio) * 100)} 1 0%" markdown="1">')
                add()
                lines.extend(blocks_to_md(fetch_children(col["id"]), slug, depth))
                add("</div>")
            add("</div>")
            add()
        elif btype == "code":
            lang = content.get("language", "")
            lang = "" if lang == "plain text" else lang
            add(f"```{lang}\n{''.join(s.get('plain_text', '') for s in content.get('rich_text', []))}\n```")
            add()
        elif btype == "equation":
            expr = content.get("expression", "")
            if not re.search(r"[A-Za-z0-9\\]", expr):
                add(f'<div class="notion-divider">{expr}</div>')  # decorative divider
            else:
                add(f"$$\n{expr}\n$$")
            add()
        elif btype == "divider":
            add("---")
            add()
        elif btype == "image":
            url = content.get(content.get("type", "file"), {}).get("url", "")
            caption = rich_text_to_md(content.get("caption", []))
            if url:
                add(f"![{caption or slug}]({download_image(url, slug, block['id'])})")
                add()
        elif btype in ("video", "embed", "bookmark", "link_preview"):
            url = content.get("url") or content.get(content.get("type", ""), {}).get("url", "")
            if url:
                add(f"<{url}>")
                add()
        elif btype == "table":
            rows = fetch_children(block["id"])
            for i, row in enumerate(rows):
                cells = [rich_text_to_md(cell) for cell in row.get("table_row", {}).get("cells", [])]
                add("| " + " | ".join(cells) + " |")
                if i == 0:
                    add("|" + "---|" * len(cells))
            add()
            continue  # children already consumed
        elif btype == "child_page":
            add(f"_(sub-page “{content.get('title', '')}” not included)_")
            add()

        if block.get("has_children") and btype not in ("table", "child_page", "toggle", "callout", "column_list"):
            child_depth = depth + 1 if btype in ("bulleted_list_item", "numbered_list_item", "to_do") else depth
            lines.extend(blocks_to_md(fetch_children(block["id"]), slug, child_depth))

    return lines


def page_meta(page: dict) -> dict:
    props = page.get("properties", {})

    def prop(name, ptype):
        return props.get(name, {}).get(ptype)

    title_rich = next((v["title"] for v in props.values() if v.get("type") == "title"), [])
    title = "".join(s.get("plain_text", "") for s in title_rich).strip() or "Untitled"
    date_prop = prop("Date", "date")
    date = (date_prop or {}).get("start") or page.get("created_time", "")[:10]
    tags = [t["name"] for t in (prop("Tags", "multi_select") or [])]
    desc_rich = prop("Description", "rich_text") or []
    description = "".join(s.get("plain_text", "") for s in desc_rich).strip()
    return {
        "id": page["id"],
        "title": title,
        "date": date[:10],
        "tags": tags,
        "description": description,
        "last_edited": page.get("last_edited_time", ""),
    }


def existing_post_for(notion_id: str) -> tuple[Path, str] | None:
    for post in POSTS_DIR.glob("*.md"):
        text = post.read_text(encoding="utf-8")
        match = re.search(r"^notion_id:\s*(\S+)\s*$", text, re.MULTILINE)
        if match and match.group(1) == notion_id:
            edited = re.search(r"^notion_last_edited:\s*(\S+)\s*$", text, re.MULTILINE)
            return post, (edited.group(1) if edited else "")
    return None


def convert_page(page: dict) -> bool:
    meta = page_meta(page)
    existing = existing_post_for(meta["id"])
    if existing and existing[1] == meta["last_edited"]:
        print(f"  up to date: {meta['title']}")
        return False

    slug = slugify(meta["title"])
    print(f"  converting: {meta['title']} -> {meta['date']}-{slug}.md")
    body = "\n".join(blocks_to_md(fetch_children(meta["id"]), slug)).strip() + "\n"
    if not meta["description"]:
        first_par = next((line for line in body.split("\n") if line.strip() and not line.startswith(("#", "!", ">", "```"))), "")
        meta["description"] = re.sub(r"[*_`#\[\]$]", "", first_par)[:160]

    front = [
        "---",
        "layout: post",
        f'title: "{meta["title"].replace(chr(34), chr(39))}"',
        f"date: {meta['date']}",
        f'description: "{meta["description"].replace(chr(34), chr(39))}"',
        f"tags: {' '.join(meta['tags']) if meta['tags'] else 'notes'}",
        "categories: blog",
        "related_posts: false",
        "toc:",
        "  sidebar: left",
        f"notion_id: {meta['id']}",
        f"notion_last_edited: {meta['last_edited']}",
        "---",
        "",
    ]
    if existing:
        existing[0].unlink()  # title/date may have changed; regenerate cleanly
    POSTS_DIR.mkdir(exist_ok=True)
    out = POSTS_DIR / f"{meta['date']}-{slug}.md"
    out.write_text("\n".join(front) + body, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", help="URL or ID of a single Notion page to convert")
    args = parser.parse_args()

    if not os.environ.get("NOTION_TOKEN"):
        print("NOTION_TOKEN is not set. See the setup notes at the top of this script.")
        sys.exit(1)

    if args.page:
        page_id = re.sub(r".*?([0-9a-f]{32}).*", r"\1", args.page.replace("-", ""))
        page_id = "-".join([page_id[:8], page_id[8:12], page_id[12:16], page_id[16:20], page_id[20:]])
        convert_page(api_request(f"/pages/{page_id}"))
        return

    database_id = os.environ.get("NOTION_DATABASE_ID")
    if not database_id:
        print("Set NOTION_DATABASE_ID or pass --page <url>.")
        sys.exit(1)

    print(f"Querying Notion database {database_id} for Status = Publish...")
    try:
        api_request(f"/databases/{database_id}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("Got 404: the integration cannot see this database. In Notion, open the")
            print("Blog Posts database: ••• → Connections → search your integration by name → add it.")
            sys.exit(1)
        raise
    converted, cursor = 0, None
    published_ids = set()
    while True:
        payload = {"filter": {"property": "Status", "select": {"equals": "Publish"}}}
        if cursor:
            payload["start_cursor"] = cursor
        data = api_request(f"/databases/{database_id}/query", payload)
        for page in data.get("results", []):
            published_ids.add(page["id"])
            converted += convert_page(page)
        if not data.get("has_more"):
            break
        cursor = data["next_cursor"]

    # Posts whose Notion page is no longer Published (status changed or page
    # deleted) are removed, along with their images. Hand-written posts have
    # no notion_id and are never touched.
    removed = 0
    for post in POSTS_DIR.glob("*.md"):
        match = re.search(r"^notion_id:\s*(\S+)\s*$", post.read_text(encoding="utf-8"), re.MULTILINE)
        if match and match.group(1) not in published_ids:
            print(f"  removing: {post.name} (page no longer published in Notion)")
            post.unlink()
            slug_assets = ASSETS_DIR / post.stem[11:]  # strip the YYYY-MM-DD- prefix
            if slug_assets.is_dir():
                shutil.rmtree(slug_assets)
            removed += 1
    print(f"Done: {converted} post(s) created or updated, {removed} removed.")


if __name__ == "__main__":
    main()
