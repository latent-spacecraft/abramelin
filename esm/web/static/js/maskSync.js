/**
 * MaskSync - Central state manager for mask synchronization
 *
 * Both the 3D viewer and sequence bar subscribe to this for bidirectional sync.
 */
class MaskSync extends EventTarget {
    constructor() {
        super();
        this.maskedIndices = new Set();
        this.sequenceLength = 0;
        this.originalSequence = '';
    }

    /**
     * Set the current sequence
     */
    setSequence(sequence) {
        this.originalSequence = sequence;
        this.sequenceLength = sequence.length;

        // Initialize masks from underscore positions
        this.maskedIndices = new Set();
        for (let i = 0; i < sequence.length; i++) {
            if (sequence[i] === '_') {
                this.maskedIndices.add(i);
            }
        }
        this._emitChange();
    }

    /**
     * Toggle mask at a specific index
     */
    toggle(index) {
        if (index < 0 || index >= this.sequenceLength) return;

        if (this.maskedIndices.has(index)) {
            this.maskedIndices.delete(index);
        } else {
            this.maskedIndices.add(index);
        }
        this._emitChange();
    }

    /**
     * Add mask at index
     */
    add(index) {
        if (index < 0 || index >= this.sequenceLength) return;
        if (!this.maskedIndices.has(index)) {
            this.maskedIndices.add(index);
            this._emitChange();
        }
    }

    /**
     * Remove mask at index
     */
    remove(index) {
        if (this.maskedIndices.has(index)) {
            this.maskedIndices.delete(index);
            this._emitChange();
        }
    }

    /**
     * Set masks for a range of indices
     */
    setRange(start, end, additive = true) {
        if (!additive) {
            this.maskedIndices.clear();
        }
        const [minIdx, maxIdx] = [Math.min(start, end), Math.max(start, end)];
        for (let i = minIdx; i <= maxIdx; i++) {
            if (i >= 0 && i < this.sequenceLength) {
                this.maskedIndices.add(i);
            }
        }
        this._emitChange();
    }

    /**
     * Clear all masks
     */
    clear() {
        this.maskedIndices.clear();
        this._emitChange();
    }

    /**
     * Get sorted array of masked indices
     */
    getMaskedIndices() {
        return Array.from(this.maskedIndices).sort((a, b) => a - b);
    }

    /**
     * Check if index is masked
     */
    isMasked(index) {
        return this.maskedIndices.has(index);
    }

    /**
     * Get sequence with masks replaced by underscore
     */
    getSequenceWithMasks() {
        if (!this.originalSequence) return '';

        return this.originalSequence
            .split('')
            .map((aa, idx) => this.maskedIndices.has(idx) ? '_' : aa)
            .join('');
    }

    /**
     * Emit change event
     */
    _emitChange() {
        const indices = this.getMaskedIndices();
        this.dispatchEvent(new CustomEvent('change', {
            detail: {
                indices,
                count: indices.length,
                sequenceLength: this.sequenceLength
            }
        }));
    }
}

// Global instance
window.maskSync = new MaskSync();
