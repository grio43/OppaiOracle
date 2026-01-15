#!/usr/bin/env python3
"""
Interactive Duplicate Cluster Reviewer

Launches a local web server for reviewing and managing duplicate image clusters
with live JSON updates.

Usage:
    python review_duplicates_interactive.py [--port PORT] [--no-browser]
"""

import argparse
import webbrowser
import threading
import time
from pathlib import Path

import uvicorn

from .duplicate_reviewer.server import create_app


DEFAULT_CLUSTERS_FILE = Path("logs/dedup_hashes/dedup_clusters.json")
DEFAULT_PORT = 8765


def open_browser_delayed(url: str, delay: float = 1.5):
    """Open browser after a short delay to let server start."""
    time.sleep(delay)
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive duplicate cluster reviewer with live JSON updates"
    )
    parser.add_argument(
        "--clusters", "-c",
        type=Path,
        default=DEFAULT_CLUSTERS_FILE,
        help=f"Path to clusters JSON file (default: {DEFAULT_CLUSTERS_FILE})"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run server on (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    # Validate clusters file
    if not args.clusters.exists():
        print(f"Error: Clusters file not found: {args.clusters}")
        print()
        print("Run the duplicate detection first:")
        print("  python find_near_dupes_cluster.py --phase 2 --threshold 0.92")
        return 1

    # Create the app
    print(f"Loading clusters from: {args.clusters}")
    try:
        app = create_app(args.clusters)
    except Exception as e:
        print(f"Error loading clusters: {e}")
        return 1

    url = f"http://{args.host}:{args.port}"
    print()
    print("=" * 60)
    print("  Duplicate Cluster Reviewer")
    print("=" * 60)
    print(f"  Server: {url}")
    print()
    print("  Actions:")
    print("    - Click any image to view full size")
    print("    - 'Set as Keep' to change which image is kept")
    print("    - 'Exclude' to prevent an image from being deleted")
    print("    - 'Exclude Cluster' to skip the entire cluster")
    print("    - 'Restore Defaults' to reset a cluster")
    print()
    print("  All changes are saved immediately to the JSON file.")
    print()
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)
    print()

    # Open browser in background thread
    if not args.no_browser:
        threading.Thread(
            target=open_browser_delayed,
            args=(url,),
            daemon=True
        ).start()

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning"
    )

    return 0


if __name__ == "__main__":
    exit(main())
