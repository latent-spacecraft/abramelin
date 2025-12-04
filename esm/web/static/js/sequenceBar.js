/**
 * SequenceBar - Interactive sequence visualization with drag-to-mask
 */
class SequenceBar {
    constructor(containerId) {
        this.container = document.querySelector(containerId);
        this.sequence = '';
        this.plddt = [];
        this.isDragging = false;
        this.dragStartIndex = null;
        this.dragMode = null;  // 'add' or 'remove'
        this.tempSelection = new Set();

        this._setupEventListeners();
    }

    _setupEventListeners() {
        this.container.addEventListener('mousedown', this._onMouseDown.bind(this));
        this.container.addEventListener('mousemove', this._onMouseMove.bind(this));
        document.addEventListener('mouseup', this._onMouseUp.bind(this));

        // Prevent text selection during drag
        this.container.addEventListener('selectstart', (e) => {
            if (this.isDragging) e.preventDefault();
        });
    }

    /**
     * Render the sequence bar
     */
    render(sequence, plddt = []) {
        this.sequence = sequence;
        this.plddt = plddt;

        this.container.innerHTML = '';

        if (!sequence) {
            this.container.innerHTML = '<div class="empty-state-text">Enter a sequence to begin</div>';
            return;
        }

        const maskedIndices = window.maskSync.getMaskedIndices();

        sequence.split('').forEach((aa, idx) => {
            const residue = document.createElement('span');
            residue.className = 'residue';
            residue.dataset.index = idx;

            // Show amino acid letter or block for mask
            residue.textContent = aa === '_' ? '?' : aa;

            // Apply pLDDT coloring
            if (plddt[idx] !== undefined && aa !== '_') {
                residue.classList.add(this._plddtClass(plddt[idx]));
            }

            // Masked styling
            if (maskedIndices.includes(idx)) {
                residue.classList.add('masked');
            }

            this.container.appendChild(residue);
        });

        // Update stats
        this._updateStats();
    }

    /**
     * Get pLDDT class name
     */
    _plddtClass(confidence) {
        if (confidence >= 0.9) return 'plddt-very-high';
        if (confidence >= 0.7) return 'plddt-high';
        if (confidence >= 0.5) return 'plddt-medium';
        if (confidence >= 0.3) return 'plddt-low';
        return 'plddt-very-low';
    }

    /**
     * Update mask highlighting
     */
    updateMasks() {
        const maskedIndices = new Set(window.maskSync.getMaskedIndices());

        this.container.querySelectorAll('.residue').forEach(el => {
            const idx = parseInt(el.dataset.index);
            el.classList.toggle('masked', maskedIndices.has(idx));
        });

        this._updateStats();
    }

    /**
     * Update sequence statistics
     */
    _updateStats() {
        const lengthEl = document.getElementById('seq-length');
        const maskEl = document.getElementById('mask-count');

        if (lengthEl) lengthEl.textContent = this.sequence.length;
        if (maskEl) maskEl.textContent = window.maskSync.getMaskedIndices().length;
    }

    /**
     * Mouse down - start drag selection
     */
    _onMouseDown(e) {
        const residue = e.target.closest('.residue');
        if (!residue) return;

        this.isDragging = true;
        this.dragStartIndex = parseInt(residue.dataset.index);
        this.tempSelection.clear();

        // Determine mode: if clicking on masked, we remove; otherwise add
        const isMasked = window.maskSync.isMasked(this.dragStartIndex);
        this.dragMode = isMasked ? 'remove' : 'add';

        // Add to temp selection
        this.tempSelection.add(this.dragStartIndex);
        this._updateTempSelection();

        e.preventDefault();
    }

    /**
     * Mouse move - extend selection
     */
    _onMouseMove(e) {
        if (!this.isDragging) return;

        const residue = e.target.closest('.residue');
        if (!residue) return;

        const currentIndex = parseInt(residue.dataset.index);

        // Calculate range
        const start = Math.min(this.dragStartIndex, currentIndex);
        const end = Math.max(this.dragStartIndex, currentIndex);

        // Update temp selection
        this.tempSelection.clear();
        for (let i = start; i <= end; i++) {
            this.tempSelection.add(i);
        }

        this._updateTempSelection();
    }

    /**
     * Mouse up - finalize selection
     */
    _onMouseUp(e) {
        if (!this.isDragging) return;

        this.isDragging = false;

        // Apply selection to maskSync
        this.tempSelection.forEach(idx => {
            if (this.dragMode === 'add') {
                window.maskSync.add(idx);
            } else {
                window.maskSync.remove(idx);
            }
        });

        this.tempSelection.clear();
        this._clearTempSelection();
    }

    /**
     * Update visual temp selection
     */
    _updateTempSelection() {
        this.container.querySelectorAll('.residue').forEach(el => {
            const idx = parseInt(el.dataset.index);
            el.classList.toggle('selecting', this.tempSelection.has(idx));
        });
    }

    /**
     * Clear temp selection visuals
     */
    _clearTempSelection() {
        this.container.querySelectorAll('.residue.selecting').forEach(el => {
            el.classList.remove('selecting');
        });
    }

    /**
     * Get current sequence (with masks from user's original input preserved)
     */
    getSequence() {
        return this.sequence;
    }
}

// Global instance
window.sequenceBar = null;
