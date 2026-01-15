// Duplicate Cluster Reviewer Application

class DuplicateReviewer {
    constructor() {
        this.clusters = [];
        this.total = 0;
        this.offset = 0;
        this.limit = 50;
        this.loading = false;
        this.filters = {
            status: null,
            sizeMin: null,
            sizeMax: null,
            search: ''
        };

        this.init();
    }

    async init() {
        this.bindEvents();
        await this.loadStats();
        await this.loadClusters();
    }

    bindEvents() {
        // Filter buttons - status
        document.querySelectorAll('.filter-btn[data-status]').forEach(btn => {
            btn.addEventListener('click', () => this.setStatusFilter(btn.dataset.status));
        });

        // Filter buttons - size
        document.querySelectorAll('.filter-btn[data-size]').forEach(btn => {
            btn.addEventListener('click', () => this.setSizeFilter(btn.dataset.size));
        });

        // Search input
        const searchInput = document.getElementById('search-input');
        let searchTimeout;
        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.filters.search = searchInput.value;
                this.resetAndLoad();
            }, 300);
        });

        // Jump to cluster
        document.getElementById('jump-btn').addEventListener('click', () => this.jumpToCluster());
        document.getElementById('jump-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.jumpToCluster();
        });

        // Load more button
        document.getElementById('load-more-btn').addEventListener('click', () => this.loadMore());

        // Modal close
        document.getElementById('modal-overlay').addEventListener('click', (e) => {
            if (e.target.id === 'modal-overlay' || e.target.classList.contains('modal-close')) {
                this.closeModal();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.closeModal();
        });

        // Infinite scroll
        window.addEventListener('scroll', () => {
            if (this.loading) return;
            const scrollBottom = window.innerHeight + window.scrollY;
            const docHeight = document.documentElement.scrollHeight;
            if (scrollBottom > docHeight - 500) {
                this.loadMore();
            }
        });
    }

    async loadStats() {
        try {
            const response = await fetch('/api/stats');
            const stats = await response.json();

            document.getElementById('stat-total').textContent = stats.total_clusters.toLocaleString();
            document.getElementById('stat-pending').textContent = stats.pending.toLocaleString();
            document.getElementById('stat-reviewed').textContent = stats.reviewed.toLocaleString();
            document.getElementById('stat-excluded').textContent = stats.excluded.toLocaleString();
            document.getElementById('stat-to-delete').textContent = stats.images_to_delete.toLocaleString();
            document.getElementById('stat-excluded-images').textContent = stats.images_excluded.toLocaleString();
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    async loadClusters(append = false) {
        if (this.loading) return;
        this.loading = true;

        if (!append) {
            this.showLoading();
        }

        try {
            const params = new URLSearchParams({
                offset: this.offset,
                limit: this.limit
            });

            if (this.filters.status) params.append('status', this.filters.status);
            if (this.filters.sizeMin) params.append('size_min', this.filters.sizeMin);
            if (this.filters.sizeMax) params.append('size_max', this.filters.sizeMax);
            if (this.filters.search) params.append('search', this.filters.search);

            const response = await fetch(`/api/clusters?${params}`);
            const data = await response.json();

            this.total = data.total;

            if (append) {
                this.clusters.push(...data.clusters);
            } else {
                this.clusters = data.clusters;
            }

            this.render();
            this.updateLoadMoreButton();
        } catch (error) {
            console.error('Failed to load clusters:', error);
            this.showError('Failed to load clusters');
        } finally {
            this.loading = false;
        }
    }

    setStatusFilter(status) {
        // Toggle filter
        this.filters.status = this.filters.status === status ? null : status;

        // Update button states
        document.querySelectorAll('.filter-btn[data-status]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.status === this.filters.status);
        });

        this.resetAndLoad();
    }

    setSizeFilter(size) {
        const sizeRanges = {
            '2': [2, 2],
            '3-5': [3, 5],
            '6-10': [6, 10],
            '10+': [10, null]
        };

        // Toggle filter
        const isActive = this.filters.sizeMin === sizeRanges[size]?.[0];
        if (isActive) {
            this.filters.sizeMin = null;
            this.filters.sizeMax = null;
        } else {
            [this.filters.sizeMin, this.filters.sizeMax] = sizeRanges[size] || [null, null];
        }

        // Update button states
        document.querySelectorAll('.filter-btn[data-size]').forEach(btn => {
            const range = sizeRanges[btn.dataset.size];
            btn.classList.toggle('active', range && this.filters.sizeMin === range[0]);
        });

        this.resetAndLoad();
    }

    resetAndLoad() {
        this.offset = 0;
        this.clusters = [];
        this.loadClusters();
    }

    async loadMore() {
        if (this.loading || this.offset + this.limit >= this.total) return;
        this.offset += this.limit;
        await this.loadClusters(true);
    }

    async jumpToCluster() {
        const input = document.getElementById('jump-input');
        const clusterId = parseInt(input.value, 10);

        if (isNaN(clusterId)) return;

        try {
            const response = await fetch(`/api/clusters/${clusterId}`);
            if (response.ok) {
                const cluster = await response.json();
                // Clear filters and show just this cluster
                this.filters = { status: null, sizeMin: null, sizeMax: null, search: '' };
                this.clusters = [cluster];
                this.total = 1;
                this.offset = 0;
                this.render();

                // Clear filter button states
                document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
                document.getElementById('search-input').value = '';
            } else {
                alert(`Cluster #${clusterId} not found`);
            }
        } catch (error) {
            console.error('Failed to jump to cluster:', error);
        }

        input.value = '';
    }

    showLoading() {
        document.getElementById('clusters-container').innerHTML = `
            <div class="loading">
                <div class="loading-spinner"></div>
                <p>Loading clusters...</p>
            </div>
        `;
    }

    showError(message) {
        document.getElementById('clusters-container').innerHTML = `
            <div class="empty-state">
                <h3>Error</h3>
                <p>${message}</p>
            </div>
        `;
    }

    updateLoadMoreButton() {
        const btn = document.getElementById('load-more-btn');
        const container = document.getElementById('load-more-container');
        const hasMore = this.offset + this.limit < this.total;

        container.style.display = hasMore ? 'block' : 'none';
        btn.disabled = this.loading;
        btn.textContent = this.loading ? 'Loading...' : `Load More (${this.clusters.length} of ${this.total})`;
    }

    render() {
        const container = document.getElementById('clusters-container');
        document.getElementById('showing-count').textContent =
            `Showing ${this.clusters.length} of ${this.total} clusters`;

        if (this.clusters.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No clusters found</h3>
                    <p>Try adjusting your filters</p>
                </div>
            `;
            return;
        }

        container.innerHTML = this.clusters.map(cluster => this.renderCluster(cluster)).join('');

        // Bind cluster event handlers
        this.bindClusterEvents();
    }

    renderCluster(cluster) {
        const keepImage = cluster.keep;
        const deleteImages = cluster.delete || [];
        const isExcluded = cluster.status === 'excluded';

        return `
            <div class="cluster-card ${isExcluded ? 'excluded' : ''}" data-cluster-id="${cluster.id}">
                <div class="cluster-header">
                    <span class="cluster-title">
                        Cluster #${cluster.id} - ${cluster.size} images
                        (${deleteImages.filter(d => !d.excluded).length} to delete)
                    </span>
                    <span class="cluster-status ${cluster.status}">${cluster.status.toUpperCase()}</span>
                </div>

                <div class="cluster-actions">
                    <button class="cluster-btn exclude" data-action="exclude-cluster">
                        ${isExcluded ? 'Include Cluster' : 'Exclude Cluster'}
                    </button>
                    <button class="cluster-btn restore" data-action="restore">
                        Restore Defaults
                    </button>
                </div>

                <div class="images-grid">
                    ${this.renderImageCard(keepImage, 'keep', cluster.id, -1)}
                    ${deleteImages.map((img, idx) => this.renderImageCard(img, 'delete', cluster.id, idx)).join('')}
                </div>
            </div>
        `;
    }

    renderImageCard(image, type, clusterId, imageIdx) {
        const isExcluded = image.excluded === true;
        const labelClass = isExcluded ? 'excluded' : type;
        const cardClass = isExcluded ? 'excluded' : type;
        const labelText = type === 'keep' ? 'KEEP' : (isExcluded ? 'EXCLUDED' : 'DELETE');
        const labelIcon = type === 'keep' ? '&#x2713;' : (isExcluded ? '&#x2717;' : '&#x2717;');

        const filename = image.path.split(/[/\\]/).pop();
        const resolution = Math.sqrt(image.resolution).toFixed(0);

        const actions = type === 'keep' ? '' : `
            <div class="image-actions">
                <button class="image-btn set-keep" data-action="set-keep" data-path="${this.escapeHtml(image.path)}">
                    Set as Keep
                </button>
                <button class="image-btn ${isExcluded ? 'include' : 'exclude'}"
                        data-action="exclude-image"
                        data-idx="${imageIdx}"
                        data-excluded="${isExcluded}">
                    ${isExcluded ? 'Include' : 'Exclude'}
                </button>
            </div>
        `;

        return `
            <div class="image-card ${cardClass}" data-cluster-id="${clusterId}" data-image-idx="${imageIdx}">
                <div class="image-label ${labelClass}">${labelIcon} ${labelText}</div>
                <img src="/api/thumbnail?path=${encodeURIComponent(image.path)}"
                     alt="${type}"
                     loading="lazy"
                     data-full-path="${this.escapeHtml(image.path)}"
                     onclick="app.openModal(this.dataset.fullPath)">
                <div class="image-info">
                    <div>Tags: ${image.tags}</div>
                    <div>Resolution: ~${resolution}x${resolution}</div>
                    <div>Size: ${this.formatBytes(image.size_bytes)}</div>
                    <div class="image-path" title="${this.escapeHtml(image.path)}">${this.escapeHtml(filename)}</div>
                </div>
                ${actions}
            </div>
        `;
    }

    bindClusterEvents() {
        // Cluster-level actions
        document.querySelectorAll('.cluster-btn[data-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const card = e.target.closest('.cluster-card');
                const clusterId = parseInt(card.dataset.clusterId, 10);
                const action = btn.dataset.action;

                if (action === 'exclude-cluster') {
                    const isExcluded = card.classList.contains('excluded');
                    this.excludeCluster(clusterId, !isExcluded);
                } else if (action === 'restore') {
                    this.restoreCluster(clusterId);
                }
            });
        });

        // Image-level actions
        document.querySelectorAll('.image-btn[data-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const card = e.target.closest('.cluster-card');
                const clusterId = parseInt(card.dataset.clusterId, 10);
                const action = btn.dataset.action;

                if (action === 'set-keep') {
                    this.setKeep(clusterId, btn.dataset.path);
                } else if (action === 'exclude-image') {
                    const idx = parseInt(btn.dataset.idx, 10);
                    const wasExcluded = btn.dataset.excluded === 'true';
                    this.excludeImage(clusterId, idx, !wasExcluded);
                }
            });
        });
    }

    async setKeep(clusterId, imagePath) {
        try {
            const response = await fetch(`/api/clusters/${clusterId}/set-keep`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_path: imagePath })
            });

            if (response.ok) {
                const data = await response.json();
                this.updateClusterInList(data.cluster);
                this.showSaveIndicator();
                await this.loadStats();
            }
        } catch (error) {
            console.error('Failed to set keep:', error);
        }
    }

    async excludeImage(clusterId, imageIdx, exclude) {
        try {
            const response = await fetch(`/api/clusters/${clusterId}/exclude-image/${imageIdx}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ exclude })
            });

            if (response.ok) {
                const data = await response.json();
                this.updateClusterInList(data.cluster);
                this.showSaveIndicator();
                await this.loadStats();
            }
        } catch (error) {
            console.error('Failed to exclude image:', error);
        }
    }

    async excludeCluster(clusterId, exclude) {
        try {
            const response = await fetch(`/api/clusters/${clusterId}/exclude`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ exclude })
            });

            if (response.ok) {
                const data = await response.json();
                this.updateClusterInList(data.cluster);
                this.showSaveIndicator();
                await this.loadStats();
            }
        } catch (error) {
            console.error('Failed to exclude cluster:', error);
        }
    }

    async restoreCluster(clusterId) {
        try {
            const response = await fetch(`/api/clusters/${clusterId}/restore`, {
                method: 'POST'
            });

            if (response.ok) {
                const data = await response.json();
                this.updateClusterInList(data.cluster);
                this.showSaveIndicator();
                await this.loadStats();
            }
        } catch (error) {
            console.error('Failed to restore cluster:', error);
        }
    }

    updateClusterInList(cluster) {
        const idx = this.clusters.findIndex(c => c.id === cluster.id);
        if (idx !== -1) {
            this.clusters[idx] = cluster;

            // Re-render just this cluster
            const card = document.querySelector(`.cluster-card[data-cluster-id="${cluster.id}"]`);
            if (card) {
                const temp = document.createElement('div');
                temp.innerHTML = this.renderCluster(cluster);
                card.replaceWith(temp.firstElementChild);
                this.bindClusterEvents();
            }
        }
    }

    showSaveIndicator() {
        const indicator = document.getElementById('save-indicator');
        indicator.classList.add('visible');
        setTimeout(() => indicator.classList.remove('visible'), 2000);
    }

    openModal(imagePath) {
        const modal = document.getElementById('modal-overlay');
        const img = document.getElementById('modal-image');
        img.src = `/api/image?path=${encodeURIComponent(imagePath)}`;
        modal.classList.add('visible');
    }

    closeModal() {
        const modal = document.getElementById('modal-overlay');
        modal.classList.remove('visible');
        document.getElementById('modal-image').src = '';
    }

    formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new DuplicateReviewer();
});
