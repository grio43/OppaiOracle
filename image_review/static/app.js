// Image Review Application

class ImageReviewApp {
    constructor() {
        this.currentRun = null;
        this.currentSample = null;
        this.allSamples = [];
        this.currentIndex = 0;  // Index within allSamples

        this.initElements();
        this.bindEvents();
        this.loadRuns();
    }

    initElements() {
        // Selectors
        this.runSelect = document.getElementById('run-select');
        this.stepSelect = document.getElementById('step-select');

        // Navigation
        this.btnPrev = document.getElementById('btn-prev');
        this.btnNext = document.getElementById('btn-next');
        this.currentNum = document.getElementById('current-num');
        this.totalNum = document.getElementById('total-num');
        this.jumpInput = document.getElementById('jump-input');
        this.btnJump = document.getElementById('btn-jump');

        // Display areas
        this.loadingState = document.getElementById('loading-state');
        this.reviewContainer = document.getElementById('review-container');
        this.sampleImage = document.getElementById('sample-image');
        this.imageContainer = document.getElementById('image-container');
        this.expectedTags = document.getElementById('expected-tags');
        this.predictions = document.getElementById('predictions');
        this.infoStep = document.getElementById('info-step');
        this.infoIdx = document.getElementById('info-idx');
        this.runInfo = document.getElementById('run-info');

        // Stats
        this.statTp = document.getElementById('stat-tp');
        this.statFp = document.getElementById('stat-fp');
        this.statFn = document.getElementById('stat-fn');

        // Rating
        this.ratingDisplay = document.getElementById('rating-display');
        this.ratingPredicted = document.getElementById('rating-predicted');
        this.ratingActual = document.getElementById('rating-actual');
        this.ratingStatus = document.getElementById('rating-status');

        // Modal
        this.modal = document.getElementById('image-modal');
        this.modalImage = document.getElementById('modal-image');
        this.modalClose = document.getElementById('modal-close');

        // Mobile elements
        this.hamburgerBtn = document.getElementById('hamburger-btn');
        this.sidebar = document.getElementById('sidebar');
        this.sidebarOverlay = document.getElementById('sidebar-overlay');
        this.mobileBtnPrev = document.getElementById('mobile-btn-prev');
        this.mobileBtnNext = document.getElementById('mobile-btn-next');
        this.mobileCurrentNum = document.getElementById('mobile-current-num');
        this.mobileTotalNum = document.getElementById('mobile-total-num');
        this.swipeHint = document.getElementById('swipe-hint');

        // Touch state
        this.touchStartX = 0;
        this.touchStartY = 0;
        this.touchEndX = 0;
        this.isSwiping = false;
    }

    bindEvents() {
        this.runSelect.addEventListener('change', () => this.onRunChange());
        this.stepSelect.addEventListener('change', () => this.onStepChange());

        this.btnPrev.addEventListener('click', () => this.navigate('prev'));
        this.btnNext.addEventListener('click', () => this.navigate('next'));
        this.btnJump.addEventListener('click', () => this.jumpToSample());
        this.jumpInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.jumpToSample();
        });

        this.imageContainer.addEventListener('click', () => this.showModal());
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal || e.target === this.modalClose) {
                this.hideModal();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Mobile: Hamburger menu
        if (this.hamburgerBtn) {
            this.hamburgerBtn.addEventListener('click', () => this.toggleSidebar());
        }
        if (this.sidebarOverlay) {
            this.sidebarOverlay.addEventListener('click', () => this.closeSidebar());
        }

        // Mobile: Navigation buttons
        if (this.mobileBtnPrev) {
            this.mobileBtnPrev.addEventListener('click', () => this.navigate('prev'));
        }
        if (this.mobileBtnNext) {
            this.mobileBtnNext.addEventListener('click', () => this.navigate('next'));
        }

        // Touch/Swipe gestures
        this.bindTouchEvents();
    }

    bindTouchEvents() {
        const mainContent = document.querySelector('.main-content');
        if (!mainContent) return;

        mainContent.addEventListener('touchstart', (e) => {
            this.touchStartX = e.changedTouches[0].screenX;
            this.touchStartY = e.changedTouches[0].screenY;
            this.isSwiping = false;
        }, { passive: true });

        mainContent.addEventListener('touchmove', (e) => {
            const deltaX = Math.abs(e.changedTouches[0].screenX - this.touchStartX);
            const deltaY = Math.abs(e.changedTouches[0].screenY - this.touchStartY);
            // Consider horizontal swipe if X movement is greater than Y
            if (deltaX > deltaY && deltaX > 30) {
                this.isSwiping = true;
            }
        }, { passive: true });

        mainContent.addEventListener('touchend', (e) => {
            this.touchEndX = e.changedTouches[0].screenX;
            this.handleSwipe();
        }, { passive: true });

        // Modal swipe support
        this.modal.addEventListener('touchstart', (e) => {
            this.touchStartX = e.changedTouches[0].screenX;
        }, { passive: true });

        this.modal.addEventListener('touchend', (e) => {
            this.touchEndX = e.changedTouches[0].screenX;
            this.handleSwipe();
        }, { passive: true });
    }

    handleSwipe() {
        const swipeThreshold = 50;
        const diff = this.touchStartX - this.touchEndX;

        if (Math.abs(diff) < swipeThreshold) return;
        if (!this.isSwiping && !this.modal.classList.contains('visible')) return;

        if (diff > 0) {
            // Swipe left -> next
            if (!this.btnNext.disabled) {
                this.navigate('next');
                if (this.modal.classList.contains('visible')) {
                    setTimeout(() => {
                        this.modalImage.src = this.sampleImage.src;
                    }, 100);
                }
                this.hideSwipeHint();
            }
        } else {
            // Swipe right -> prev
            if (!this.btnPrev.disabled) {
                this.navigate('prev');
                if (this.modal.classList.contains('visible')) {
                    setTimeout(() => {
                        this.modalImage.src = this.sampleImage.src;
                    }, 100);
                }
                this.hideSwipeHint();
            }
        }
    }

    toggleSidebar() {
        if (this.sidebar) {
            this.sidebar.classList.toggle('open');
        }
        if (this.sidebarOverlay) {
            this.sidebarOverlay.classList.toggle('visible');
        }
    }

    closeSidebar() {
        if (this.sidebar) {
            this.sidebar.classList.remove('open');
        }
        if (this.sidebarOverlay) {
            this.sidebarOverlay.classList.remove('visible');
        }
    }

    hideSwipeHint() {
        if (this.swipeHint && !this.swipeHintHidden) {
            this.swipeHint.style.opacity = '0';
            setTimeout(() => {
                this.swipeHint.style.display = 'none';
            }, 300);
            this.swipeHintHidden = true;
        }
    }

    async loadRuns() {
        try {
            const response = await fetch('/api/runs');
            const data = await response.json();

            this.runSelect.innerHTML = '<option value="">Select a run...</option>';
            data.runs.forEach(run => {
                const option = document.createElement('option');
                option.value = run.name;
                option.textContent = run.name;
                this.runSelect.appendChild(option);
            });

            if (data.runs.length === 0) {
                this.runInfo.textContent = 'No TensorBoard runs found';
            }
        } catch (error) {
            console.error('Failed to load runs:', error);
            this.runInfo.textContent = 'Error loading runs';
        }
    }

    async onRunChange() {
        const runName = this.runSelect.value;
        if (!runName) {
            this.showLoading();
            return;
        }

        // Close sidebar on mobile after selection
        this.closeSidebar();

        this.runInfo.textContent = 'Loading...';

        try {
            const response = await fetch(`/api/runs/${encodeURIComponent(runName)}/load`, {
                method: 'POST'
            });
            const data = await response.json();

            if (!data.success) {
                throw new Error('Failed to load run');
            }

            this.currentRun = runName;
            this.runInfo.textContent = `${data.total_samples} samples`;

            // Populate steps
            this.stepSelect.innerHTML = '<option value="">All steps</option>';
            data.steps.forEach(step => {
                const option = document.createElement('option');
                option.value = step.step;
                option.textContent = `Step ${step.step} (${step.sample_count} samples)`;
                this.stepSelect.appendChild(option);
            });

            this.totalNum.textContent = data.total_samples;
            if (this.mobileTotalNum) {
                this.mobileTotalNum.textContent = data.total_samples;
            }

            // Load samples
            if (data.total_samples > 0) {
                await this.loadSamples();
            } else {
                this.showLoading('No samples found in this run');
            }
        } catch (error) {
            console.error('Failed to load run:', error);
            this.runInfo.textContent = 'Error loading run';
        }
    }

    async onStepChange() {
        await this.loadSamples();
    }

    async loadSamples() {
        const step = this.stepSelect.value || null;

        try {
            const url = step
                ? `/api/samples?step=${step}&limit=500`
                : '/api/samples?limit=500';

            const response = await fetch(url);
            const data = await response.json();

            this.allSamples = data.samples;
            this.currentIndex = 0;
            this.totalNum.textContent = this.allSamples.length;
            if (this.mobileTotalNum) {
                this.mobileTotalNum.textContent = this.allSamples.length;
            }

            if (this.allSamples.length > 0) {
                await this.loadSampleByIndex(0);
                this.hideLoading();
            } else {
                this.showLoading('No samples found');
            }
        } catch (error) {
            console.error('Failed to load samples:', error);
        }
    }

    async loadSampleByIndex(index) {
        if (index < 0 || index >= this.allSamples.length) return;

        const sampleInfo = this.allSamples[index];
        this.currentIndex = index;

        try {
            const response = await fetch(`/api/samples/${sampleInfo.step}/${sampleInfo.sample_index}`);
            const data = await response.json();

            this.currentSample = data.sample;
            this.renderSample();
            this.updateNavigation();
        } catch (error) {
            console.error('Failed to load sample:', error);
        }
    }

    renderSample() {
        const sample = this.currentSample;

        // Show loading state
        this.imageContainer.classList.add('loading');

        // Update image with load handler
        const newSrc = `/api/samples/${sample.step}/${sample.sample_index}/image`;
        this.sampleImage.onload = () => {
            this.imageContainer.classList.remove('loading');
        };
        this.sampleImage.onerror = () => {
            this.imageContainer.classList.remove('loading');
        };
        this.sampleImage.src = newSrc;

        // Update info
        this.infoStep.textContent = sample.step;
        this.infoIdx.textContent = sample.sample_index;

        // Update stats
        this.statTp.textContent = sample.tp_count;
        this.statFp.textContent = sample.fp_count;
        this.statFn.textContent = sample.fn_count;

        // Update rating display
        if (sample.rating) {
            this.ratingDisplay.style.display = 'flex';

            // Predicted rating
            const predConfPct = Math.round(sample.rating.predicted_confidence * 100);
            this.ratingPredicted.textContent = `${sample.rating.predicted} (${predConfPct}%)`;
            this.ratingPredicted.className = `rating-predicted rating-${sample.rating.predicted}`;

            // Actual rating
            this.ratingActual.textContent = sample.rating.actual;
            this.ratingActual.className = `rating-actual rating-${sample.rating.actual}`;

            // Status
            if (sample.rating.is_correct) {
                this.ratingStatus.textContent = 'CORRECT';
                this.ratingStatus.className = 'rating-status correct';
            } else {
                this.ratingStatus.textContent = 'WRONG';
                this.ratingStatus.className = 'rating-status wrong';
            }
        } else {
            this.ratingDisplay.style.display = 'none';
        }

        // Render expected tags
        if (sample.ground_truth_tags.length > 0) {
            this.expectedTags.innerHTML = sample.ground_truth_tags
                .map(tag => `<div class="tag-item">${this.escapeHtml(tag)}</div>`)
                .join('');
        } else {
            this.expectedTags.innerHTML = '<div class="empty-state">No ground truth tags</div>';
        }

        // Render predictions
        if (sample.predictions.length > 0) {
            this.predictions.innerHTML = sample.predictions
                .map(p => `
                    <div class="prediction-item ${p.status.toLowerCase()}">
                        <span class="tag-name">${this.escapeHtml(p.tag)}</span>
                        <span class="prob">${p.probability.toFixed(4)}</span>
                        <span class="status ${p.status.toLowerCase()}">${p.status}</span>
                    </div>
                `)
                .join('');
        } else {
            this.predictions.innerHTML = '<div class="empty-state">No predictions</div>';
        }
    }

    updateNavigation() {
        const currentDisplay = this.currentIndex + 1;
        const totalDisplay = this.allSamples.length;
        const atStart = this.currentIndex <= 0;
        const atEnd = this.currentIndex >= this.allSamples.length - 1;

        // Desktop navigation
        this.currentNum.textContent = currentDisplay;
        this.btnPrev.disabled = atStart;
        this.btnNext.disabled = atEnd;

        // Mobile navigation
        if (this.mobileCurrentNum) {
            this.mobileCurrentNum.textContent = currentDisplay;
        }
        if (this.mobileTotalNum) {
            this.mobileTotalNum.textContent = totalDisplay;
        }
        if (this.mobileBtnPrev) {
            this.mobileBtnPrev.disabled = atStart;
        }
        if (this.mobileBtnNext) {
            this.mobileBtnNext.disabled = atEnd;
        }
    }

    async navigate(direction) {
        if (this.allSamples.length === 0) return;

        let newIndex = this.currentIndex;
        if (direction === 'prev' && this.currentIndex > 0) {
            newIndex = this.currentIndex - 1;
        } else if (direction === 'next' && this.currentIndex < this.allSamples.length - 1) {
            newIndex = this.currentIndex + 1;
        }

        if (newIndex !== this.currentIndex) {
            await this.loadSampleByIndex(newIndex);
        }
    }

    async jumpToSample() {
        const input = this.jumpInput.value.trim();
        if (!input) return;

        const index = parseInt(input) - 1; // Convert to 0-based
        if (isNaN(index) || index < 0 || index >= this.allSamples.length) {
            // Show visual feedback for invalid input
            this.jumpInput.classList.add('invalid');
            setTimeout(() => this.jumpInput.classList.remove('invalid'), 300);
            return;
        }

        await this.loadSampleByIndex(index);
        this.jumpInput.value = '';
    }

    showModal() {
        if (this.currentSample) {
            this.modalImage.src = this.sampleImage.src;
            this.modal.classList.add('visible');
        }
    }

    hideModal() {
        this.modal.classList.remove('visible');
    }

    showLoading(message = 'Select a TensorBoard run to begin reviewing images') {
        this.loadingState.innerHTML = `<p>${message}</p>`;
        this.loadingState.style.display = 'flex';
        this.reviewContainer.style.display = 'none';
    }

    hideLoading() {
        this.loadingState.style.display = 'none';
        this.reviewContainer.style.display = 'grid';
    }

    handleKeyboard(e) {
        // Don't handle shortcuts when typing in input (except when modal is open)
        const modalOpen = this.modal.classList.contains('visible');
        if (!modalOpen && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;

        switch (e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                if (!this.btnPrev.disabled) {
                    this.navigate('prev');
                    // Update modal image if open
                    if (modalOpen) {
                        setTimeout(() => {
                            this.modalImage.src = this.sampleImage.src;
                        }, 100);
                    }
                }
                break;
            case 'ArrowRight':
                e.preventDefault();
                if (!this.btnNext.disabled) {
                    this.navigate('next');
                    // Update modal image if open
                    if (modalOpen) {
                        setTimeout(() => {
                            this.modalImage.src = this.sampleImage.src;
                        }, 100);
                    }
                }
                break;
            case 'g':
            case 'G':
                if (!modalOpen) {
                    e.preventDefault();
                    this.jumpInput.focus();
                    this.jumpInput.select();
                }
                break;
            case 'Escape':
                this.hideModal();
                break;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ImageReviewApp();
});
