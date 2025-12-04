/**
 * Viewer3D - 3Dmol.js wrapper with residue clicking support
 */
class Viewer3D {
    constructor(containerId) {
        this.container = document.querySelector(containerId);
        this.viewer = null;
        this.currentPdb = null;
        this.plddt = null;
        this.maskedResidues = new Set();
        this.onResidueClick = null;

        this._init();
    }

    _init() {
        // Create 3Dmol viewer
        this.viewer = $3Dmol.createViewer(this.container, {
            backgroundColor: '#BBB8CC',
            antialias: true,
        });

        // Add empty state message
        this._showEmptyState();
    }

    _showEmptyState() {
        this.viewer.render();
    }

    /**
     * Load PDB structure
     */
    loadStructure(pdbString, plddt = null) {
        if (!pdbString) {
            console.warn('No PDB string provided');
            return;
        }

        this.currentPdb = pdbString;
        this.plddt = plddt;

        // Clear and add new model
        this.viewer.removeAllModels();
        this.viewer.removeAllLabels();
        this.viewer.addModel(pdbString, 'pdb');

        // Apply cartoon style
        this.viewer.setStyle({}, {
            cartoon: {
                color: 'gray',
                opacity: 1.0,
            }
        });

        // Apply pLDDT coloring if available
        if (plddt && plddt.length > 0) {
            this._applyPlddtColoring(plddt);
        }

        // Setup click handlers
        this._setupClickHandlers();

        // Highlight current masks
        this._highlightMasked();

        // Center and render
        this.viewer.zoomTo();
        this.viewer.render();
    }

    /**
     * Apply pLDDT confidence coloring
     */
    _applyPlddtColoring(plddt) {
        plddt.forEach((confidence, idx) => {
            const color = this._plddtToColor(confidence);
            this.viewer.setStyle(
                { resi: idx + 1 },  // 1-indexed for PDB
                { cartoon: { color: color } }
            );
        });
    }

    /**
     * Convert pLDDT score to color
     */
    _plddtToColor(confidence) {
        if (confidence >= 0.9) return '#0077ff';      // Very high - dark blue
        if (confidence >= 0.7) return '#44aaff';      // High - light blue
        if (confidence >= 0.5) return '#ffcc00';      // Medium - yellow
        if (confidence >= 0.3) return '#ff8800';      // Low - orange
        return '#ff4444';                              // Very low - red
    }

    /**
     * Setup residue click handlers
     */
    _setupClickHandlers() {
        this.viewer.setClickable({}, true, (atom, viewer, event, container) => {
            if (atom && atom.resi !== undefined) {
                const residueIndex = atom.resi - 1;  // Convert to 0-indexed

                // Emit click event
                if (this.onResidueClick) {
                    this.onResidueClick(residueIndex);
                }

                // Show temporary label
                this._showResidueLabel(atom);
            }
        });
    }

    /**
     * Show temporary label on clicked residue
     */
    _showResidueLabel(atom) {
        const label = this.viewer.addLabel(
            `${atom.resn} ${atom.resi}`,
            {
                position: atom,
                backgroundColor: '#9b59b6',
                fontColor: 'white',
                fontSize: 12,
                borderRadius: 4,
            }
        );

        this.viewer.render();

        // Remove label after 1.5 seconds
        setTimeout(() => {
            this.viewer.removeLabel(label);
            this.viewer.render();
        }, 1500);
    }

    /**
     * Highlight masked residues
     */
    _highlightMasked() {
        const maskedIndices = window.maskSync.getMaskedIndices();

        // Reset styles based on pLDDT or gray
        if (this.plddt && this.plddt.length > 0) {
            this._applyPlddtColoring(this.plddt);
        } else {
            this.viewer.setStyle({}, { cartoon: { color: 'gray' } });
        }

        // Apply purple highlight to masked residues
        maskedIndices.forEach(idx => {
            this.viewer.setStyle(
                { resi: idx + 1 },
                {
                    cartoon: {
                        color: '#9b59b6',
                        opacity: 0.9,
                    }
                }
            );
        });

        this.viewer.render();
    }

    /**
     * Update mask highlighting (called when masks change)
     */
    updateMasks() {
        if (this.currentPdb) {
            this._highlightMasked();
        }
    }

    /**
     * Zoom to specific residues
     */
    zoomToResidues(indices) {
        if (indices.length === 0) {
            this.viewer.zoomTo();
        } else {
            const resis = indices.map(i => i + 1);  // 1-indexed
            this.viewer.zoomTo({ resi: resis });
        }
        this.viewer.render();
    }

    /**
     * Get current view state for history
     */
    getViewState() {
        return this.viewer.getView();
    }

    /**
     * Restore view state
     */
    setViewState(state) {
        if (state) {
            this.viewer.setView(state);
        }
    }
}

// Global instance
window.viewer3d = null;
