"""Manages duplicate cluster JSON data with atomic save operations."""

import json
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import threading


@dataclass
class ClusterStats:
    """Summary statistics for clusters."""
    total_clusters: int = 0
    pending: int = 0
    reviewed: int = 0
    excluded: int = 0
    total_images: int = 0
    images_to_keep: int = 0
    images_to_delete: int = 0
    images_excluded: int = 0


class ClusterManager:
    """Thread-safe manager for duplicate cluster data with live JSON updates."""

    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        self._data: Optional[Dict] = None
        self._lock = threading.Lock()
        self._cluster_index: Dict[int, int] = {}  # cluster_id -> list index

    def load(self) -> bool:
        """Load clusters from JSON file. Returns True if successful."""
        if not self.json_path.exists():
            return False

        with self._lock:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)

            # Migrate schema if needed
            self._migrate_schema()

            # Build index for fast lookup
            self._build_index()

        return True

    def _migrate_schema(self):
        """Add new fields if missing (backward compatibility)."""
        if 'review_state' not in self._data:
            self._data['review_state'] = {
                'last_modified': datetime.now().isoformat(),
                'reviewed_count': 0,
                'excluded_clusters': 0
            }

        for cluster in self._data.get('clusters', []):
            # Add status if missing
            if 'status' not in cluster:
                cluster['status'] = 'pending'

            # Add manual flag to keep if missing
            if 'manual' not in cluster.get('keep', {}):
                cluster['keep']['manual'] = False

            # Store original keep for restore functionality
            if '_original_keep_path' not in cluster:
                cluster['_original_keep_path'] = cluster['keep']['path']

            # Add excluded flag to delete items if missing
            for item in cluster.get('delete', []):
                if 'excluded' not in item:
                    item['excluded'] = False

    def _build_index(self):
        """Build cluster_id -> index mapping for fast lookups."""
        self._cluster_index = {}
        for idx, cluster in enumerate(self._data.get('clusters', [])):
            self._cluster_index[cluster['id']] = idx

    def save(self):
        """Atomically save data to JSON file."""
        with self._lock:
            if self._data is None:
                return

            # Update timestamp
            self._data['review_state']['last_modified'] = datetime.now().isoformat()

            # Atomic write: write to temp file then rename
            tmp_path = self.json_path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, indent=2)
                os.replace(tmp_path, self.json_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

    def get_stats(self) -> ClusterStats:
        """Get summary statistics."""
        with self._lock:
            stats = ClusterStats()
            if self._data is None:
                return stats

            clusters = self._data.get('clusters', [])
            stats.total_clusters = len(clusters)

            for cluster in clusters:
                status = cluster.get('status', 'pending')
                if status == 'pending':
                    stats.pending += 1
                elif status == 'reviewed':
                    stats.reviewed += 1
                elif status == 'excluded':
                    stats.excluded += 1

                # Count images
                stats.images_to_keep += 1  # Always one keep per cluster
                delete_list = cluster.get('delete', [])
                for item in delete_list:
                    if item.get('excluded', False):
                        stats.images_excluded += 1
                    else:
                        stats.images_to_delete += 1

            stats.total_images = stats.images_to_keep + stats.images_to_delete + stats.images_excluded
            return stats

    def get_clusters(
        self,
        offset: int = 0,
        limit: int = 50,
        status_filter: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        search: Optional[str] = None
    ) -> List[Dict]:
        """Get clusters with filtering and pagination."""
        with self._lock:
            if self._data is None:
                return []

            clusters = self._data.get('clusters', [])

            # Apply filters
            filtered = []
            for cluster in clusters:
                # Status filter
                if status_filter and cluster.get('status', 'pending') != status_filter:
                    continue

                # Size filter
                size = cluster.get('size', 0)
                if size_min is not None and size < size_min:
                    continue
                if size_max is not None and size > size_max:
                    continue

                # Search filter (search in paths)
                if search:
                    search_lower = search.lower()
                    found = False
                    if search_lower in cluster['keep']['path'].lower():
                        found = True
                    else:
                        for item in cluster.get('delete', []):
                            if search_lower in item['path'].lower():
                                found = True
                                break
                    if not found:
                        continue

                filtered.append(cluster)

            # Pagination
            return filtered[offset:offset + limit]

    def get_cluster(self, cluster_id: int) -> Optional[Dict]:
        """Get a single cluster by ID."""
        with self._lock:
            if self._data is None:
                return None

            idx = self._cluster_index.get(cluster_id)
            if idx is None:
                return None

            return self._data['clusters'][idx]

    def set_keep(self, cluster_id: int, image_path: str) -> bool:
        """Change which image is marked as keep in a cluster."""
        with self._lock:
            if self._data is None:
                return False

            idx = self._cluster_index.get(cluster_id)
            if idx is None:
                return False

            cluster = self._data['clusters'][idx]
            current_keep = cluster['keep']

            # If already the keep image, nothing to do
            if current_keep['path'] == image_path:
                return True

            # Find the new keep in delete list
            new_keep_idx = None
            for i, item in enumerate(cluster['delete']):
                if item['path'] == image_path:
                    new_keep_idx = i
                    break

            if new_keep_idx is None:
                return False

            # Swap: current keep goes to delete, selected goes to keep
            new_keep_data = cluster['delete'][new_keep_idx].copy()
            new_keep_data['manual'] = True
            if 'excluded' in new_keep_data:
                del new_keep_data['excluded']

            old_keep_data = {
                'path': current_keep['path'],
                'resolution': current_keep['resolution'],
                'tags': current_keep['tags'],
                'size_bytes': current_keep['size_bytes'],
                'excluded': False
            }

            cluster['keep'] = new_keep_data
            cluster['delete'][new_keep_idx] = old_keep_data
            cluster['status'] = 'reviewed'

        self.save()
        return True

    def exclude_image(self, cluster_id: int, image_idx: int, exclude: bool = True) -> bool:
        """Mark an image as excluded from deletion."""
        with self._lock:
            if self._data is None:
                return False

            idx = self._cluster_index.get(cluster_id)
            if idx is None:
                return False

            cluster = self._data['clusters'][idx]
            delete_list = cluster.get('delete', [])

            if image_idx < 0 or image_idx >= len(delete_list):
                return False

            delete_list[image_idx]['excluded'] = exclude

            if cluster['status'] == 'pending':
                cluster['status'] = 'reviewed'

        self.save()
        return True

    def exclude_cluster(self, cluster_id: int, exclude: bool = True) -> bool:
        """Exclude or include an entire cluster."""
        with self._lock:
            if self._data is None:
                return False

            idx = self._cluster_index.get(cluster_id)
            if idx is None:
                return False

            cluster = self._data['clusters'][idx]
            cluster['status'] = 'excluded' if exclude else 'reviewed'

            # Mark all delete items as excluded too
            for item in cluster.get('delete', []):
                item['excluded'] = exclude

        self.save()
        return True

    def restore_cluster(self, cluster_id: int) -> bool:
        """Restore a cluster to its original algorithm-selected state."""
        with self._lock:
            if self._data is None:
                return False

            idx = self._cluster_index.get(cluster_id)
            if idx is None:
                return False

            cluster = self._data['clusters'][idx]
            original_keep_path = cluster.get('_original_keep_path')

            # Reset all exclusions
            for item in cluster.get('delete', []):
                item['excluded'] = False

            # If keep was changed, restore original
            if original_keep_path and cluster['keep']['path'] != original_keep_path:
                # Find original in delete list
                for i, item in enumerate(cluster['delete']):
                    if item['path'] == original_keep_path:
                        # Swap back
                        new_keep_data = item.copy()
                        new_keep_data['manual'] = False
                        if 'excluded' in new_keep_data:
                            del new_keep_data['excluded']

                        old_keep_data = {
                            'path': cluster['keep']['path'],
                            'resolution': cluster['keep']['resolution'],
                            'tags': cluster['keep']['tags'],
                            'size_bytes': cluster['keep']['size_bytes'],
                            'excluded': False
                        }

                        cluster['keep'] = new_keep_data
                        cluster['delete'][i] = old_keep_data
                        break

            cluster['keep']['manual'] = False
            cluster['status'] = 'pending'

        self.save()
        return True

    def get_deletion_list(self) -> List[str]:
        """Get list of all paths that should be deleted (excluding excluded items)."""
        with self._lock:
            if self._data is None:
                return []

            paths = []
            for cluster in self._data.get('clusters', []):
                if cluster.get('status') == 'excluded':
                    continue

                for item in cluster.get('delete', []):
                    if not item.get('excluded', False):
                        paths.append(item['path'])

            return paths

    def get_total_count(
        self,
        status_filter: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        search: Optional[str] = None
    ) -> int:
        """Get total count of clusters matching filters."""
        with self._lock:
            if self._data is None:
                return 0

            count = 0
            for cluster in self._data.get('clusters', []):
                if status_filter and cluster.get('status', 'pending') != status_filter:
                    continue

                size = cluster.get('size', 0)
                if size_min is not None and size < size_min:
                    continue
                if size_max is not None and size > size_max:
                    continue

                if search:
                    search_lower = search.lower()
                    found = False
                    if search_lower in cluster['keep']['path'].lower():
                        found = True
                    else:
                        for item in cluster.get('delete', []):
                            if search_lower in item['path'].lower():
                                found = True
                                break
                    if not found:
                        continue

                count += 1

            return count
