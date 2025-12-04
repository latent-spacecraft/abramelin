/**
 * Protein Alchemy - Main Application Controller
 */
class ProteinAlchemyApp {
    constructor() {
        this.sessionId = null;
        this.isGenerating = false;
        this.history = [];

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

        // Setup UI event handlers
        this._setupEventHandlers();

        // Load demo sequence
        this._loadDemoSequence();

        // Update status
        this._setStatus('ready', 'Ready - Mask residues and press Enter to regenerate');
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
        });
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
     * Trigger generation (always runs sequence then structure)
     */
    async generate() {
        if (this.isGenerating) return;

        const sequence = window.maskSync.getSequenceWithMasks();
        const maskedCount = window.maskSync.getMaskedIndices().length;

        // Allow generation even without masks (just refold structure)
        if (maskedCount === 0) {
            this._setStatus('loading', 'Re-folding structure...');
        }

        this.isGenerating = true;
        this._showLoading(true, maskedCount > 0 ? 'sequence' : 'structure');
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
                })
                .on('phase', (data) => {
                    // Update phase label
                    const phaseLabel = data.phase === 'sequence' ? 'Sequence' : 'Structure';
                    document.getElementById('phase-label').textContent = phaseLabel;
                    document.getElementById('loading-phase').textContent = data.message;
                    this._setStatus('loading', data.message);
                })
                .on('progress', (data) => {
                    document.getElementById('step-current').textContent = data.step;
                    document.getElementById('step-total').textContent = data.total_steps;
                    const phase = data.phase === 'sequence' ? 'Sequence' : 'Structure';
                    this._setStatus('loading', `${phase}: Step ${data.step}/${data.total_steps}`);
                })
                .on('sequence_complete', (data) => {
                    // Update sequence bar with new sequence (before structure)
                    window.maskSync.setSequence(data.sequence);
                    window.sequenceBar.render(data.sequence, []);
                    this._setStatus('loading', 'Sequence complete, folding structure...');
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

        // Update 3D viewer
        if (data.pdb) {
            window.viewer3d.loadStructure(data.pdb, data.plddt);
        }

        // Add to history
        this._addToHistory({
            sequence: data.sequence,
            pdb: data.pdb,
            plddt: data.plddt,
            timestamp: new Date().toISOString(),
        });

        this._setStatus('ready', 'Generation complete!');
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
        el.innerHTML = `
            <div class="history-item-label">Generation #${this.history.length}</div>
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

        if (item.pdb) {
            window.viewer3d.loadStructure(item.pdb, item.plddt);
        }

        this._setStatus('ready', 'Restored from history');
    }

    /**
     * Show load dialog (simple prompt for now)
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
