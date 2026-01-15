"""FastAPI server for interactive duplicate review."""

import os
import hashlib
from pathlib import Path
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from PIL import Image

from .cluster_manager import ClusterManager

# Thumbnail cache directory
THUMBNAIL_DIR = Path("logs/dedup_hashes/thumbnails")
THUMBNAIL_SIZE = (300, 300)


class SetKeepRequest(BaseModel):
    image_path: str


class ExcludeImageRequest(BaseModel):
    exclude: bool = True


class ExcludeClusterRequest(BaseModel):
    exclude: bool = True


def create_app(clusters_json: Path) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="Duplicate Cluster Reviewer")

    # Initialize cluster manager
    manager = ClusterManager(clusters_json)
    if not manager.load():
        raise RuntimeError(f"Could not load clusters from {clusters_json}")

    # Create thumbnail directory
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main HTML page."""
        html_path = static_dir / "index.html"
        return html_path.read_text(encoding='utf-8')

    @app.get("/api/stats")
    async def get_stats():
        """Get summary statistics."""
        stats = manager.get_stats()
        return {
            "total_clusters": stats.total_clusters,
            "pending": stats.pending,
            "reviewed": stats.reviewed,
            "excluded": stats.excluded,
            "total_images": stats.total_images,
            "images_to_keep": stats.images_to_keep,
            "images_to_delete": stats.images_to_delete,
            "images_excluded": stats.images_excluded
        }

    @app.get("/api/clusters")
    async def get_clusters(
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
        status: Optional[str] = Query(None),
        size_min: Optional[int] = Query(None),
        size_max: Optional[int] = Query(None),
        search: Optional[str] = Query(None)
    ):
        """Get paginated list of clusters with optional filtering."""
        clusters = manager.get_clusters(
            offset=offset,
            limit=limit,
            status_filter=status,
            size_min=size_min,
            size_max=size_max,
            search=search
        )
        total = manager.get_total_count(
            status_filter=status,
            size_min=size_min,
            size_max=size_max,
            search=search
        )
        return {
            "clusters": clusters,
            "total": total,
            "offset": offset,
            "limit": limit
        }

    @app.get("/api/clusters/{cluster_id}")
    async def get_cluster(cluster_id: int):
        """Get a single cluster by ID."""
        cluster = manager.get_cluster(cluster_id)
        if cluster is None:
            raise HTTPException(status_code=404, detail="Cluster not found")
        return cluster

    @app.post("/api/clusters/{cluster_id}/set-keep")
    async def set_keep(cluster_id: int, request: SetKeepRequest):
        """Change which image is marked as keep."""
        success = manager.set_keep(cluster_id, request.image_path)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to set keep image")
        return {"success": True, "cluster": manager.get_cluster(cluster_id)}

    @app.post("/api/clusters/{cluster_id}/exclude-image/{image_idx}")
    async def exclude_image(cluster_id: int, image_idx: int, request: ExcludeImageRequest):
        """Exclude or include a single image from deletion."""
        success = manager.exclude_image(cluster_id, image_idx, request.exclude)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to exclude image")
        return {"success": True, "cluster": manager.get_cluster(cluster_id)}

    @app.post("/api/clusters/{cluster_id}/exclude")
    async def exclude_cluster(cluster_id: int, request: ExcludeClusterRequest):
        """Exclude or include an entire cluster."""
        success = manager.exclude_cluster(cluster_id, request.exclude)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to exclude cluster")
        return {"success": True, "cluster": manager.get_cluster(cluster_id)}

    @app.post("/api/clusters/{cluster_id}/restore")
    async def restore_cluster(cluster_id: int):
        """Restore cluster to algorithm defaults."""
        success = manager.restore_cluster(cluster_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to restore cluster")
        return {"success": True, "cluster": manager.get_cluster(cluster_id)}

    @app.get("/api/image")
    async def get_image(path: str = Query(...)):
        """Serve an image file."""
        image_path = Path(path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        # Security: ensure path is within expected directories
        try:
            image_path = image_path.resolve()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid path")

        return FileResponse(
            image_path,
            media_type=_get_media_type(image_path.suffix)
        )

    @app.get("/api/thumbnail")
    async def get_thumbnail(path: str = Query(...)):
        """Serve or generate a thumbnail for an image."""
        image_path = Path(path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        # Generate cache key from path
        path_hash = hashlib.md5(str(image_path).encode()).hexdigest()
        thumb_path = THUMBNAIL_DIR / f"{path_hash}.jpg"

        # Generate thumbnail if not cached
        if not thumb_path.exists():
            try:
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary (for PNG with alpha, etc.)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')

                    # Create thumbnail maintaining aspect ratio
                    img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

                    # Save thumbnail
                    img.save(thumb_path, "JPEG", quality=85)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to generate thumbnail: {e}")

        return FileResponse(thumb_path, media_type="image/jpeg")

    @app.get("/api/deletions")
    async def get_deletions():
        """Get the current list of files to be deleted."""
        paths = manager.get_deletion_list()
        return {"count": len(paths), "paths": paths}

    @app.post("/api/export-deletions")
    async def export_deletions(output_path: str = Query("near_duplicate_deletions.txt")):
        """Export deletion list to a file."""
        paths = manager.get_deletion_list()
        output = Path(output_path)
        output.write_text('\n'.join(paths), encoding='utf-8')
        return {"success": True, "count": len(paths), "path": str(output.absolute())}

    return app


def _get_media_type(suffix: str) -> str:
    """Get MIME type for image suffix."""
    types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    return types.get(suffix.lower(), 'image/jpeg')
