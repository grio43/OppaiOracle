#!/usr/bin/env python3
"""Interactive duplicate cluster viewer - generates HTML to visually compare duplicates."""

import json
import webbrowser
import random
from pathlib import Path
import base64
from html import escape

CLUSTERS_FILE = Path("logs/dedup_hashes/dedup_clusters.json")
OUTPUT_HTML = Path("duplicate_review.html")

def image_to_data_uri(path: str) -> str:
    """Convert image to base64 data URI for embedding in HTML."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = Path(path).suffix.lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext.lstrip("."), "jpeg")
        return f"data:image/{mime};base64,{data}"
    except Exception as e:
        return ""

def generate_html(clusters: list, title: str = "Duplicate Cluster Review") -> str:
    """Generate HTML to view clusters."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{escape(title)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #00d9ff; }}
        .cluster {{ background: #16213e; border-radius: 10px; padding: 20px; margin: 20px 0; }}
        .cluster-header {{ font-size: 1.2em; color: #00d9ff; margin-bottom: 15px; }}
        .images {{ display: flex; flex-wrap: wrap; gap: 15px; align-items: flex-start; }}
        .image-card {{ background: #0f3460; border-radius: 8px; padding: 10px; max-width: 300px; }}
        .image-card.keep {{ border: 3px solid #00ff88; }}
        .image-card.delete {{ border: 3px solid #ff4444; opacity: 0.8; }}
        .image-card img {{ max-width: 280px; max-height: 280px; border-radius: 5px; }}
        .image-card .label {{ font-weight: bold; padding: 5px 0; }}
        .image-card .label.keep {{ color: #00ff88; }}
        .image-card .label.delete {{ color: #ff4444; }}
        .image-card .info {{ font-size: 0.85em; color: #aaa; word-break: break-all; }}
        .stats {{ background: #0f3460; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .more-indicator {{ color: #888; font-style: italic; padding: 10px; }}
    </style>
</head>
<body>
    <h1>{escape(title)}</h1>
"""

    for cluster in clusters:
        keep = cluster["keep"]
        deletes = cluster["delete"]

        html += f"""
    <div class="cluster">
        <div class="cluster-header">Cluster #{cluster['id']} - {cluster['size']} images ({len(deletes)} to delete)</div>
        <div class="images">
            <div class="image-card keep">
                <div class="label keep">✓ KEEP</div>
                <img src="{image_to_data_uri(keep['path'])}" alt="keep">
                <div class="info">
                    <div>Tags: {keep['tags']}</div>
                    <div>Resolution: {keep['resolution']:,}</div>
                    <div>Size: {keep['size_bytes']:,} bytes</div>
                    <div title="{escape(keep['path'])}">{escape(Path(keep['path']).name)}</div>
                </div>
            </div>
"""
        # Show first 5 duplicates
        for i, dup in enumerate(deletes[:5]):
            html += f"""
            <div class="image-card delete">
                <div class="label delete">✗ DELETE</div>
                <img src="{image_to_data_uri(dup['path'])}" alt="delete">
                <div class="info">
                    <div>Tags: {dup['tags']}</div>
                    <div>Resolution: {dup['resolution']:,}</div>
                    <div>Size: {dup['size_bytes']:,} bytes</div>
                    <div title="{escape(dup['path'])}">{escape(Path(dup['path']).name)}</div>
                </div>
            </div>
"""
        if len(deletes) > 5:
            html += f'<div class="more-indicator">... and {len(deletes) - 5} more duplicates</div>'

        html += """
        </div>
    </div>
"""

    html += """
</body>
</html>
"""
    return html


def main():
    print("Loading clusters...")
    with open(CLUSTERS_FILE) as f:
        data = json.load(f)

    clusters = data["clusters"]
    summary = data["summary"]

    print(f"\nSummary:")
    print(f"  Total images scanned: {summary['total_images']:,}")
    print(f"  Unique clusters: {summary['unique_clusters']:,}")
    print(f"  Images to keep: {summary['images_to_keep']:,}")
    print(f"  Images to delete: {summary['images_to_delete']:,}")

    # Sort by cluster size (largest first)
    clusters_by_size = sorted(clusters, key=lambda x: x["size"], reverse=True)

    print(f"\nTop 10 largest clusters:")
    for c in clusters_by_size[:10]:
        print(f"  Cluster #{c['id']}: {c['size']} images")

    # Select clusters to show
    print("\nGenerating review HTML...")

    # Show: top 5 largest + 10 random medium-sized + 5 random small
    selected = []
    selected.extend(clusters_by_size[:5])  # 5 largest

    medium = [c for c in clusters if 3 <= c["size"] <= 10]
    if medium:
        selected.extend(random.sample(medium, min(10, len(medium))))

    small = [c for c in clusters if c["size"] == 2]
    if small:
        selected.extend(random.sample(small, min(5, len(small))))

    html = generate_html(selected, f"Duplicate Review - {len(selected)} sample clusters")

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nGenerated: {OUTPUT_HTML.absolute()}")
    print("Opening in browser...")
    webbrowser.open(OUTPUT_HTML.absolute().as_uri())


if __name__ == "__main__":
    main()
