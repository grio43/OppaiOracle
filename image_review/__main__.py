"""Entry point for image_review module.

Usage:
    python -m image_review [--tensorboard-dir DIR] [--port PORT] [--host HOST]

Examples:
    python -m image_review
    python -m image_review --tensorboard-dir ./tensorboard --port 8080
"""

from .server import main

if __name__ == "__main__":
    main()
