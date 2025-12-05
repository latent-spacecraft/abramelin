/**
 * Protein Alchemy - Main Application Controller
 *
 * Features pLDDT-weighted ensemble animation for pseudo-molecular dynamics.
 */
class ProteinAlchemyApp {
    constructor() {
        this.sessionId = null;
        this.isGenerating = false;
        this.history = [];
        this.ensembleSize = 1;

        this._init();
    }

    _init() {
        // Initialize components
        window.viewer3d = new Viewer3D('#viewer-container');
        window.sequenceBar = new SequenceBar('#sequence-container');

        // Connect mask sync to components
        window.maskSync.addEventListener('change', (e) => {
            window.viewer3d.updateMasks();
            window.sequenceBar.updateMasks();
        });

        // Connect 3D viewer clicks to mask sync
        window.viewer3d.onResidueClick = (index) => {
            window.maskSync.toggle(index);
        };

        // Connect frame change callback for animation display
        window.viewer3d.onFrameChange = (frame, total) => {
            document.getElementById('anim-frame').textContent = `${frame + 1}/${total}`;
        };

        // Setup UI event handlers
        this._setupEventHandlers();
        this._setupAnimationControls();

        // Load demo sequence
        this._loadDemoSequence();

        // Update status
        this._setStatus('ready', 'Ready - Mask residues and press Enter to regenerate');

        // Keep models warm with periodic pings (every 2 minutes)
        this._startKeepAlive();
    }

    /**
     * Keep models warm with periodic status checks
     */
    _startKeepAlive() {
        // Ping immediately to warm up
        this._pingStatus();

        // Then every 2 minutes
        setInterval(() => {
            this._pingStatus();
        }, 2 * 60 * 1000);
    }

    async _pingStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            console.log('[KeepAlive] Models warm:', status);
        } catch (e) {
            console.warn('[KeepAlive] Ping failed:', e);
        }
    }

    _setupEventHandlers() {
        // Generate button
        document.getElementById('generate-btn').addEventListener('click', () => {
            this.generate();
        });

        // Clear masks button
        document.getElementById('clear-btn').addEventListener('click', () => {
            window.maskSync.clear();
        });

        // Load sequence button
        document.getElementById('load-btn').addEventListener('click', () => {
            this._showLoadDialog();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && !this.isGenerating) {
                e.preventDefault();
                this.generate();
            }
            if (e.key === 'Escape') {
                window.maskSync.clear();
            }
            // Space to toggle animation
            if (e.key === ' ' && window.viewer3d.ensemble) {
                e.preventDefault();
                this._toggleAnimation();
            }
        });
    }

    _setupAnimationControls() {
        // Play/Pause button
        document.getElementById('anim-play').addEventListener('click', () => {
            this._toggleAnimation();
        });

        // Animation mode selector
        document.getElementById('anim-mode').addEventListener('change', (e) => {
            window.viewer3d.setAnimationMode(e.target.value);
        });

        // Speed slider
        document.getElementById('anim-speed').addEventListener('input', (e) => {
            // Invert: higher slider value = slower animation
            const speed = 550 - parseInt(e.target.value);
            window.viewer3d.setAnimationSpeed(speed);
        });

        // Ensemble size input
        document.getElementById('ensemble-size').addEventListener('change', (e) => {
            this.ensembleSize = Math.max(1, Math.min(10, parseInt(e.target.value) || 1));
            e.target.value = this.ensembleSize;
        });

        // GIF Export button
        document.getElementById('export-gif-btn').addEventListener('click', () => {
            this._exportGif();
        });
    }

    /**
     * Export animation as GIF
     */
    _exportGif() {
        const btn = document.getElementById('export-gif-btn');
        const originalText = btn.textContent;

        btn.textContent = '⏳ 0%';
        btn.disabled = true;

        window.viewer3d.exportGif(
            // Progress callback
            (progress) => {
                btn.textContent = `⏳ ${Math.round(progress * 100)}%`;
            },
            // Complete callback
            (blobUrl) => {
                btn.textContent = originalText;
                btn.disabled = false;

                // Create download link
                const a = document.createElement('a');
                a.href = blobUrl;
                a.download = `protein_ensemble_${Date.now()}.gif`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);

                // Clean up blob URL after download
                setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);

                this._setStatus('ready', 'GIF exported successfully!');
            }
        );
    }

    _toggleAnimation() {
        const isPlaying = window.viewer3d.toggleAnimation();
        const icon = document.getElementById('anim-play-icon');
        const controls = document.getElementById('animation-controls');

        icon.textContent = isPlaying ? '⏸' : '▶';
        controls.classList.toggle('playing', isPlaying);
    }

    _showAnimationControls(show) {
        const controls = document.getElementById('animation-controls');
        controls.classList.toggle('hidden', !show);
    }

    /**
     * Load demo sequence
     */
    _loadDemoSequence() {
        // GFP-like sequence with some masks
        const demoSequence = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK';

        // Add some masks in the middle
        const maskedSequence = demoSequence.substring(0, 80) +
            '________' +  // 8 masked positions
            demoSequence.substring(88);

        this._loadSequence(maskedSequence);
    }

    /**
     * Load a sequence
     */
    _loadSequence(sequence, plddt = null) {
        window.maskSync.setSequence(sequence);
        window.sequenceBar.render(sequence, plddt || []);

        this._setStatus('ready', `Loaded sequence: ${sequence.length} residues`);
    }

    /**
     * Trigger generation (sequence → structure ensemble)
     */
    async generate() {
        if (this.isGenerating) return;

        const sequence = window.maskSync.getSequenceWithMasks();
        const maskedCount = window.maskSync.getMaskedIndices().length;
        const ensembleSize = parseInt(document.getElementById('ensemble-size').value) || 1;

        // Stop any running animation
        window.viewer3d.stopAnimation();

        if (maskedCount === 0) {
            this._setStatus('loading', 'Re-folding structure...');
        }

        this.isGenerating = true;
        this._showLoading(true, maskedCount > 0 ? 'sequence' : 'structure');
        this._showAnimationControls(false);
        this._setStatus('loading', 'Starting generation...');

        try {
            // Start generation
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sequence: sequence,
                    num_steps: 8,
                    temperature: 1.0,
                    ensemble_size: ensembleSize,
                    session_id: this.sessionId,
                }),
            });

            const { session_id } = await response.json();
            this.sessionId = session_id;

            // Connect WebSocket for streaming
            window.wsClient.close();
            window.wsClient
                .on('start', (data) => {
                    this._setStatus('loading', 'Starting transmutation...');
                    if (data.ensemble_size > 1) {
                        document.getElementById('conf-progress').textContent =
                            `(Ensemble: 0/${data.ensemble_size})`;
                    }
                })
                .on('phase', (data) => {
                    const phaseLabel = data.phase === 'sequence' ? 'Sequence' : 'Structure';
                    document.getElementById('phase-label').textContent = phaseLabel;
                    document.getElementById('loading-phase').textContent = data.message;
                    this._setStatus('loading', data.message);

                    // Show conformation progress for ensemble
                    if (data.total_conformations > 1) {
                        document.getElementById('conf-progress').textContent =
                            `(${data.conformation}/${data.total_conformations})`;
                    }
                })
                .on('progress', (data) => {
                    document.getElementById('step-current').textContent = data.step;
                    document.getElementById('step-total').textContent = data.total_steps;

                    let statusMsg = `${data.phase === 'sequence' ? 'Sequence' : 'Structure'}: Step ${data.step}/${data.total_steps}`;
                    if (data.total_conformations > 1) {
                        statusMsg += ` (Conf ${data.conformation}/${data.total_conformations})`;
                    }
                    this._setStatus('loading', statusMsg);
                })
                .on('sequence_complete', (data) => {
                    window.maskSync.setSequence(data.sequence);
                    window.sequenceBar.render(data.sequence, []);
                    this._setStatus('loading', 'Sequence complete, folding structures...');
                })
                .on('conformation_ready', (data) => {
                    // Update progress as each conformation completes
                    this._setStatus('loading', `Conformation ${data.index + 1} ready (pLDDT: ${(data.avg_plddt * 100).toFixed(1)}%)`);
                })
                .on('function_complete', (data) => {
                    // Update status with function prediction count
                    const count = data.count || 0;
                    console.log('[function_complete] Received:', data);
                    console.log('[function_complete] Summary:', data.summary);
                    this._setStatus('loading', `Function prediction: ${count} annotations found`);

                    // Display function annotations immediately when received
                    const seqLen = window.maskSync.sequence?.length || 0;
                    if (data.annotations && data.annotations.length > 0) {
                        this._displayFunctionAnnotations(
                            data.annotations,
                            seqLen,
                            data.total_count || data.annotations.length,
                            data.summary || ''
                        );
                    }
                })
                .on('complete', (data) => {
                    this._onGenerationComplete(data);
                })
                .on('error', (data) => {
                    this._onGenerationError(data);
                });

            window.wsClient.connect(session_id);

        } catch (error) {
            this._onGenerationError({ message: error.message });
        }
    }

    /**
     * Handle generation complete
     */
    _onGenerationComplete(data) {
        this.isGenerating = false;
        this._showLoading(false);

        // Update sequence bar
        if (data.sequence) {
            window.maskSync.setSequence(data.sequence);
            window.sequenceBar.render(data.sequence, data.plddt || []);
        }

        // Handle ensemble vs single structure
        if (data.ensemble && data.ensemble.pdbs && data.ensemble.pdbs.length > 1) {
            // Load ensemble for animation
            window.viewer3d.loadEnsemble(data.ensemble);
            this._showAnimationControls(true);

            // Update frame display
            document.getElementById('anim-frame').textContent = `1/${data.ensemble.pdbs.length}`;

            // Auto-start animation
            setTimeout(() => {
                window.viewer3d.startAnimation('randomWalk');
                document.getElementById('anim-play-icon').textContent = '⏸';
                document.getElementById('animation-controls').classList.add('playing');
            }, 500);

            this._setStatus('ready', `Ensemble ready: ${data.ensemble.pdbs.length} conformations (Space to pause)`);
        } else if (data.pdb) {
            // Single structure
            window.viewer3d.loadStructure(data.pdb, data.plddt);
            this._showAnimationControls(false);
            this._setStatus('ready', 'Generation complete!');
        }

        // Display function annotations with LLM summary
        if (data.function_annotations && data.function_annotations.length > 0) {
            this._displayFunctionAnnotations(
                data.function_annotations,
                data.sequence.length,
                data.total_count || data.function_annotations.length,
                data.function_summary || ''
            );
        } else {
            this._displayFunctionAnnotations([], 0, 0, '');
        }

        // Add to history
        this._addToHistory({
            sequence: data.sequence,
            pdb: data.pdb,
            plddt: data.plddt,
            ensemble: data.ensemble,
            function_annotations: data.function_annotations || [],
            function_summary: data.function_summary || '',
            total_count: data.total_count || 0,
            timestamp: new Date().toISOString(),
        });
    }

    /**
     * Handle generation error
     */
    _onGenerationError(data) {
        this.isGenerating = false;
        this._showLoading(false);
        this._setStatus('error', `Error: ${data.message}`);
        console.error('Generation error:', data);
    }

    /**
     * Show/hide loading overlay
     */
    _showLoading(show, initialPhase = 'sequence') {
        const overlay = document.getElementById('loading-overlay');
        overlay.classList.toggle('hidden', !show);

        if (show) {
            document.getElementById('step-current').textContent = '0';
            document.getElementById('step-total').textContent = '8';
            document.getElementById('conf-progress').textContent = '';
            document.getElementById('phase-label').textContent =
                initialPhase === 'sequence' ? 'Sequence' : 'Structure';
            document.getElementById('loading-phase').textContent =
                initialPhase === 'sequence' ? 'Generating sequence...' : 'Folding structure...';
        }
    }

    /**
     * Set status bar
     */
    _setStatus(state, message) {
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');

        indicator.className = 'status-indicator';
        if (state === 'loading') indicator.classList.add('loading');

        text.textContent = message;
    }

    /**
     * Add to history panel
     */
    _addToHistory(item) {
        this.history.unshift(item);

        const panel = document.getElementById('history-panel');

        // Clear empty state
        if (this.history.length === 1) {
            panel.innerHTML = '';
        }

        // Create history item
        const el = document.createElement('div');
        el.className = 'history-item';

        const ensembleLabel = item.ensemble && item.ensemble.pdbs ?
            ` (${item.ensemble.pdbs.length} conf)` : '';

        el.innerHTML = `
            <div class="history-item-label">Generation #${this.history.length}${ensembleLabel}</div>
            <div class="history-item-seq">${item.sequence.substring(0, 30)}...</div>
        `;

        // Click to restore
        el.addEventListener('click', () => {
            this._restoreFromHistory(item);

            // Update active state
            panel.querySelectorAll('.history-item').forEach(h => h.classList.remove('active'));
            el.classList.add('active');
        });

        // Add to top
        panel.insertBefore(el, panel.firstChild);
    }

    /**
     * Restore from history
     */
    _restoreFromHistory(item) {
        window.maskSync.setSequence(item.sequence);
        window.sequenceBar.render(item.sequence, item.plddt || []);

        // Handle ensemble vs single
        if (item.ensemble && item.ensemble.pdbs && item.ensemble.pdbs.length > 1) {
            window.viewer3d.loadEnsemble(item.ensemble);
            this._showAnimationControls(true);
            document.getElementById('anim-frame').textContent = `1/${item.ensemble.pdbs.length}`;
        } else if (item.pdb) {
            window.viewer3d.loadStructure(item.pdb, item.plddt);
            this._showAnimationControls(false);
        }

        // Restore function annotations with summary
        if (item.function_annotations && item.function_annotations.length > 0) {
            this._displayFunctionAnnotations(
                item.function_annotations,
                item.sequence.length,
                item.total_count || item.function_annotations.length,
                item.function_summary || ''
            );
        } else {
            this._displayFunctionAnnotations([], 0, 0, '');
        }

        this._setStatus('ready', 'Restored from history');
    }

    /**
     * Display function annotations as tags with LLM summary
     */
    _displayFunctionAnnotations(annotations, sequenceLength, totalCount = 0, summary = '') {
        const panel = document.getElementById('function-panel');
        const countEl = document.getElementById('function-count');

        if (!panel) return;  // Panel not in DOM yet

        // Update count (show total if available)
        if (countEl) {
            countEl.textContent = totalCount || annotations.length;
        }

        // Clear panel
        panel.innerHTML = '';

        if (!annotations || annotations.length === 0) {
            panel.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-text">
                        No function annotations predicted
                    </div>
                </div>
            `;
            return;
        }

        // Add LLM summary at the top if available
        if (summary) {
            const summaryEl = document.createElement('div');
            summaryEl.className = 'function-summary';
            summaryEl.innerHTML = `<span class="summary-icon">✨</span> ${summary}`;
            panel.appendChild(summaryEl);
        }

        // Create tags container
        const tagsContainer = document.createElement('div');
        tagsContainer.className = 'function-tags';

        // Render each annotation as a tag
        annotations.forEach(ann => {
            const tag = this._createFunctionTag(ann, sequenceLength);
            tagsContainer.appendChild(tag);
        });

        panel.appendChild(tagsContainer);

        // Add total count note if there are more
        if (totalCount > annotations.length) {
            const moreNote = document.createElement('div');
            moreNote.className = 'function-more-note';
            moreNote.textContent = `Showing top ${annotations.length} of ${totalCount} annotations`;
            panel.appendChild(moreNote);
        }
    }

    /**
     * Create a function tag element
     */
    _createFunctionTag(annotation, sequenceLength) {
        const tag = document.createElement('div');
        tag.className = 'function-tag';

        // Calculate coverage percentage
        const coverage = annotation.total_residues
            ? ((annotation.total_residues / sequenceLength) * 100).toFixed(0)
            : '?';

        // Determine tag color based on label type
        const isIPR = annotation.label.includes('IPR');
        if (isIPR) {
            tag.classList.add('tag-domain');
        } else {
            tag.classList.add('tag-keyword');
        }

        // Format label nicely
        let displayLabel = annotation.label;
        if (displayLabel.length > 25) {
            displayLabel = displayLabel.substring(0, 22) + '...';
        }

        tag.innerHTML = `
            <span class="tag-label">${displayLabel}</span>
            <span class="tag-count">×${annotation.count || 1}</span>
            <span class="tag-coverage">${coverage}%</span>
        `;

        tag.title = `${annotation.label}\n${annotation.count || 1} region(s), ${coverage}% coverage\nClick to highlight`;

        // Click to highlight all regions
        tag.addEventListener('click', () => {
            if (annotation.regions) {
                // Highlight all regions for this label
                annotation.regions.forEach(([start, end]) => {
                    this._highlightRange(start - 1, end - 1);
                });
            } else if (annotation.start && annotation.end) {
                this._highlightRange(annotation.start - 1, annotation.end - 1);
            }
        });

        return tag;
    }

    /**
     * Highlight a range in the sequence bar
     */
    _highlightRange(start, end) {
        // Flash the range in the sequence bar
        const residues = document.querySelectorAll('.residue');
        for (let i = start; i <= end && i < residues.length; i++) {
            residues[i].classList.add('highlighted');
            setTimeout(() => residues[i].classList.remove('highlighted'), 1500);
        }
    }

    /**
     * Show load dialog
     */
    _showLoadDialog() {
        const sequence = prompt(
            'Enter a protein sequence (use _ for masked positions):',
            'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH'
        );

        if (sequence) {
            this._loadSequence(sequence.toUpperCase());
        }
    }
}

// Initialize app on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ProteinAlchemyApp();
});
