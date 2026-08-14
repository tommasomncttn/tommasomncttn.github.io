#!/usr/bin/env python3
"""
Convert Notion exported markdown folders to Jekyll blog posts.
Reads folders from _notion_draft/ and creates blog posts in _posts/.
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

# Configuration
WORKSPACE_ROOT = Path(__file__).parent.parent
NOTION_DRAFT_DIR = WORKSPACE_ROOT / "_notion_draft"
POSTS_DIR = WORKSPACE_ROOT / "_posts"
ASSETS_DIR = WORKSPACE_ROOT / "assets" / "img" / "posts"


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def extract_metadata_from_content(content: str) -> dict:
    """Extract metadata from Notion markdown content."""
    metadata = {
        "title": "",
        "categories": "",
        "image": "",
        "description": "",
    }

    # Extract title from first # heading
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()

    # Extract Categories line
    cat_match = re.search(r"^Categories:\s*(.+)$", content, re.MULTILINE)
    if cat_match:
        metadata["categories"] = cat_match.group(1).strip().lower()

    # Extract image line
    img_match = re.search(r"^image:\s*(.+)$", content, re.MULTILINE)
    if img_match:
        metadata["image"] = img_match.group(1).strip()

    return metadata


def clean_notion_content(content: str, post_slug: str) -> str:
    """Clean up Notion markdown content for Jekyll."""
    lines = content.split("\n")
    cleaned_lines = []
    skip_next_empty = False
    in_metadata_section = True

    for i, line in enumerate(lines):
        # Skip the first title (it's in front matter)
        if in_metadata_section:
            if line.startswith("# "):
                skip_next_empty = True
                continue
            if line.startswith("Categories:") or line.startswith("image:"):
                skip_next_empty = True
                continue
            if line.strip() == "" and skip_next_empty:
                skip_next_empty = False
                continue
            # Stop skipping after we hit real content
            if line.strip() and not line.startswith("#"):
                in_metadata_section = False

        # Convert Notion image references to Jekyll assets
        # Pattern: ![image.png](FolderName/image.png)
        line = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)/([^/)]+\.(?:png|jpg|jpeg|gif|webp))\)",
            rf"![{post_slug}-\3](/assets/img/posts/{post_slug}/\3)",
            line,
        )

        # Also handle direct image references
        line = re.sub(
            r"!\[([^\]]*)\]\(([^/)]+\.(?:png|jpg|jpeg|gif|webp))\)",
            rf"![\1](/assets/img/posts/{post_slug}/\2)",
            line,
        )

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def find_notion_markdown(folder_path: Path) -> tuple[Path | None, str]:
    """Find the main markdown file in a Notion export folder."""
    for f in folder_path.iterdir():
        if f.suffix == ".md":
            return f, f.read_text(encoding="utf-8")
    return None, ""


def copy_images(source_folder: Path, dest_folder: Path):
    """Copy images from Notion export to assets folder."""
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Look for images in subfolders (Notion structure)
    for item in source_folder.rglob("*"):
        if item.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            dest_path = dest_folder / item.name
            shutil.copy2(item, dest_path)
            print(f"  Copied image: {item.name}")


def convert_notion_folder(folder_path: Path) -> bool:
    """Convert a single Notion folder to a blog post."""
    print(f"\nProcessing: {folder_path.name}")

    # Find markdown file
    md_file, content = find_notion_markdown(folder_path)
    if not md_file:
        print(f"  No markdown file found, skipping...")
        return False

    # Extract metadata
    metadata = extract_metadata_from_content(content)
    if not metadata["title"]:
        metadata["title"] = folder_path.name

    # Generate slug and filename
    slug = slugify(metadata["title"])
    date_str = datetime.now().strftime("%Y-%m-%d")
    post_filename = f"{date_str}-{slug}.md"

    print(f"  Title: {metadata['title']}")
    print(f"  Slug: {slug}")
    print(f"  Output: {post_filename}")

    # Clean content
    cleaned_content = clean_notion_content(content, slug)

    # Build front matter
    front_matter = f"""---
layout: post
title: "{metadata['title']}"
date: {date_str}
description: A deep dive into {metadata['title'].lower()}
tags: {metadata['categories'] if metadata['categories'] else 'tutorial'}
categories: {metadata['categories'] if metadata['categories'] else 'blog'}
giscus_comments: true
toc:
  sidebar: left
---

"""

    # Write post file
    post_path = POSTS_DIR / post_filename
    with open(post_path, "w", encoding="utf-8") as f:
        f.write(front_matter + cleaned_content)
    print(f"  Created post: {post_path}")

    # Copy images
    image_dest = ASSETS_DIR / slug
    copy_images(folder_path, image_dest)

    return True


def main():
    """Main entry point."""
    print("=" * 50)
    print("Notion to Jekyll Blog Post Converter")
    print("=" * 50)

    if not NOTION_DRAFT_DIR.exists():
        print(f"Error: {NOTION_DRAFT_DIR} does not exist")
        return

    # Process each folder in _notion_draft
    converted = 0
    for item in NOTION_DRAFT_DIR.iterdir():
        if item.is_dir():
            if convert_notion_folder(item):
                converted += 1

    print("\n" + "=" * 50)
    print(f"Converted {converted} Notion page(s) to blog posts")
    print("=" * 50)


if __name__ == "__main__":
    main()
