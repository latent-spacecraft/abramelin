/**
 * Viewer3D - 3Dmol.js wrapper with residue clicking and ensemble animation
 *
 * Supports pLDDT-weighted random walk animation through conformational ensemble.
 */
class Viewer3D {
    constructor(containerId) {
        this.container = document.querySelector(containerId);
        this.viewer = null;
        this.currentPdb = null;
        this.plddt = null;
        this.maskedResidues = new Set();
        this.onResidueClick = null;

        // Ensemble animation state
        this.ensemble = null;           // { pdbs: [], plddts: [], avg_plddts: [] }
        this.isAnimating = false;
        this.animationInterval = null;
        this.currentFrame = 0;
        this.animationSpeed = 150;      // ms between transitions
        this.animationMode = 'weighted'; // 'linear', 'backAndForth', 'weighted'
        this.transitionMatrix = null;   // For Markov chain animation

        // Smooth morphing state
        this.keyframeIndices = [];      // Which frames in the trajectory are original keyframes
        this.interpolatedFrameCount = 0; // Total frames after interpolation
        this.framesPerTransition = 12;  // Interpolated frames between keyframes (~400ms at 30fps)
        this.frameInterval = 33;        // ~30fps
        this.animationFrameId = null;

        this._init();
    }

    _init() {
        // Create 3Dmol viewer
        this.viewer = $3Dmol.createViewer(this.container, {
            backgroundColor: '#0d0d1a',
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

    // =========================================================================
    // Ensemble Animation
    // =========================================================================

    /**
     * Load ensemble of structures for animation with interpolated frames
     */
    loadEnsemble(ensemble) {
        if (!ensemble || !ensemble.pdbs || ensemble.pdbs.length === 0) {
            console.warn('No ensemble data provided');
            return;
        }

        this.stopAnimation();
        this.ensemble = ensemble;
        this.currentFrame = 0;

        // Build transition matrix for weighted random walk
        this._buildTransitionMatrix();

        // Parse all PDBs to extract atom coordinates
        const keyframes = ensemble.pdbs.map(pdb => this._parsePdbCoords(pdb));
        console.log(`Parsed ${keyframes.length} keyframes, ${keyframes[0]?.length || 0} atoms each`);

        // Align all keyframes to the first one (Kabsch algorithm)
        // This pins the center of mass and removes rigid body rotation
        const alignedKeyframes = this._alignKeyframes(keyframes);
        console.log('Aligned all keyframes to reference');

        // Generate interpolated trajectory with smooth transitions
        const { trajectory, keyframeIndices } = this._generateInterpolatedTrajectory(alignedKeyframes, ensemble.pdbs);
        this.keyframeIndices = keyframeIndices;
        this.interpolatedFrameCount = keyframeIndices.length > 0 ?
            keyframeIndices[keyframeIndices.length - 1] + 1 : keyframes.length;

        console.log(`Generated trajectory with ${this.interpolatedFrameCount} total frames`);
        console.log(`Keyframe indices:`, keyframeIndices);

        // Clear and load interpolated trajectory
        this.viewer.removeAllModels();
        this.viewer.removeAllLabels();
        this.viewer.addModelsAsFrames(trajectory, 'pdb');

        // Style all frames
        this.viewer.setStyle({}, {
            cartoon: { color: 'spectrum' }
        });

        // Apply pLDDT coloring
        if (ensemble.plddts && ensemble.plddts[0]) {
            this._applyPlddtColoringToFrame(0, ensemble.plddts[0]);
        }

        this.viewer.setFrame(0);
        this.viewer.zoomTo();
        this.viewer.render();

        console.log(`Loaded ensemble with ${ensemble.pdbs.length} conformations, ${this.interpolatedFrameCount} interpolated frames`);
    }

    /**
     * Parse PDB string to extract atom coordinates
     */
    _parsePdbCoords(pdbString) {
        const atoms = [];
        const lines = pdbString.split('\n');

        for (const line of lines) {
            if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
                atoms.push({
                    line: line,  // Keep full line for reconstruction
                    x: parseFloat(line.substring(30, 38)),
                    y: parseFloat(line.substring(38, 46)),
                    z: parseFloat(line.substring(46, 54)),
                });
            }
        }
        return atoms;
    }

    // =========================================================================
    // Kabsch Alignment - Superimpose structures to remove rigid body motion
    // =========================================================================

    /**
     * Align all keyframes to the first one using Kabsch algorithm
     */
    _alignKeyframes(keyframes) {
        if (keyframes.length < 2) return keyframes;

        const reference = keyframes[0];
        const aligned = [reference];  // First frame is the reference

        for (let i = 1; i < keyframes.length; i++) {
            const alignedFrame = this._kabschAlign(keyframes[i], reference);
            aligned.push(alignedFrame);
        }

        return aligned;
    }

    /**
     * Align mobile structure to target using Kabsch algorithm
     * Returns new atom array with aligned coordinates
     */
    _kabschAlign(mobile, target) {
        const n = Math.min(mobile.length, target.length);

        // 1. Compute centroids
        const centroidM = this._computeCentroid(mobile);
        const centroidT = this._computeCentroid(target);

        // 2. Center both structures
        const centeredM = mobile.map(a => ({
            ...a,
            x: a.x - centroidM.x,
            y: a.y - centroidM.y,
            z: a.z - centroidM.z,
        }));

        const centeredT = target.map(a => ({
            x: a.x - centroidT.x,
            y: a.y - centroidT.y,
            z: a.z - centroidT.z,
        }));

        // 3. Compute covariance matrix H = M^T * T
        const H = [[0,0,0], [0,0,0], [0,0,0]];
        for (let i = 0; i < n; i++) {
            const m = centeredM[i];
            const t = centeredT[i];
            H[0][0] += m.x * t.x; H[0][1] += m.x * t.y; H[0][2] += m.x * t.z;
            H[1][0] += m.y * t.x; H[1][1] += m.y * t.y; H[1][2] += m.y * t.z;
            H[2][0] += m.z * t.x; H[2][1] += m.z * t.y; H[2][2] += m.z * t.z;
        }

        // 4. SVD of H to get rotation matrix R = V * U^T
        const { U, V } = this._svd3x3(H);

        // 5. Compute rotation matrix R = V * U^T
        const R = this._matMul3x3(V, this._transpose3x3(U));

        // 6. Handle reflection (ensure proper rotation, det(R) = 1)
        const det = this._det3x3(R);
        if (det < 0) {
            // Flip sign of last column of V and recompute
            V[0][2] *= -1; V[1][2] *= -1; V[2][2] *= -1;
            const Rcorrected = this._matMul3x3(V, this._transpose3x3(U));
            return this._applyTransform(centeredM, Rcorrected, centroidT);
        }

        // 7. Apply rotation and translate to target centroid
        return this._applyTransform(centeredM, R, centroidT);
    }

    /**
     * Compute centroid of atom array
     */
    _computeCentroid(atoms) {
        const n = atoms.length;
        let cx = 0, cy = 0, cz = 0;
        for (const a of atoms) {
            cx += a.x; cy += a.y; cz += a.z;
        }
        return { x: cx / n, y: cy / n, z: cz / n };
    }

    /**
     * Apply rotation matrix and translation to atoms
     */
    _applyTransform(atoms, R, translation) {
        return atoms.map(a => {
            const rx = R[0][0] * a.x + R[0][1] * a.y + R[0][2] * a.z;
            const ry = R[1][0] * a.x + R[1][1] * a.y + R[1][2] * a.z;
            const rz = R[2][0] * a.x + R[2][1] * a.y + R[2][2] * a.z;
            return {
                ...a,
                x: rx + translation.x,
                y: ry + translation.y,
                z: rz + translation.z,
            };
        });
    }

    /**
     * 3x3 matrix multiplication
     */
    _matMul3x3(A, B) {
        const C = [[0,0,0], [0,0,0], [0,0,0]];
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j];
            }
        }
        return C;
    }

    /**
     * 3x3 matrix transpose
     */
    _transpose3x3(M) {
        return [
            [M[0][0], M[1][0], M[2][0]],
            [M[0][1], M[1][1], M[2][1]],
            [M[0][2], M[1][2], M[2][2]],
        ];
    }

    /**
     * 3x3 matrix determinant
     */
    _det3x3(M) {
        return M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[2][1])
             - M[0][1] * (M[1][0]*M[2][2] - M[1][2]*M[2][0])
             + M[0][2] * (M[1][0]*M[2][1] - M[1][1]*M[2][0]);
    }

    /**
     * Simple 3x3 SVD using Jacobi iterations
     * Returns { U, S, V } where A = U * diag(S) * V^T
     */
    _svd3x3(A) {
        // Use iterative Jacobi method for 3x3 SVD
        // First compute A^T * A
        const AtA = this._matMul3x3(this._transpose3x3(A), A);

        // Eigendecomposition of A^T*A using Jacobi
        const { eigenvalues, eigenvectors } = this._jacobi3x3(AtA);

        // V = eigenvectors of A^T*A (sorted by eigenvalue)
        const V = eigenvectors;

        // Singular values are sqrt of eigenvalues
        const S = eigenvalues.map(e => Math.sqrt(Math.max(0, e)));

        // U = A * V * S^-1 for each column
        const U = [[0,0,0], [0,0,0], [0,0,0]];
        for (let j = 0; j < 3; j++) {
            if (S[j] > 1e-10) {
                const Avj = [
                    A[0][0]*V[0][j] + A[0][1]*V[1][j] + A[0][2]*V[2][j],
                    A[1][0]*V[0][j] + A[1][1]*V[1][j] + A[1][2]*V[2][j],
                    A[2][0]*V[0][j] + A[2][1]*V[1][j] + A[2][2]*V[2][j],
                ];
                U[0][j] = Avj[0] / S[j];
                U[1][j] = Avj[1] / S[j];
                U[2][j] = Avj[2] / S[j];
            } else {
                // Null singular value - set to unit vector
                U[0][j] = j === 0 ? 1 : 0;
                U[1][j] = j === 1 ? 1 : 0;
                U[2][j] = j === 2 ? 1 : 0;
            }
        }

        return { U, S, V };
    }

    /**
     * Jacobi eigendecomposition for symmetric 3x3 matrix
     */
    _jacobi3x3(A) {
        // Copy matrix
        const M = A.map(row => [...row]);
        const V = [[1,0,0], [0,1,0], [0,0,1]];  // Eigenvector matrix

        const maxIter = 50;
        for (let iter = 0; iter < maxIter; iter++) {
            // Find largest off-diagonal element
            let maxVal = 0, p = 0, q = 1;
            for (let i = 0; i < 3; i++) {
                for (let j = i + 1; j < 3; j++) {
                    if (Math.abs(M[i][j]) > maxVal) {
                        maxVal = Math.abs(M[i][j]);
                        p = i; q = j;
                    }
                }
            }

            if (maxVal < 1e-12) break;  // Converged

            // Compute Jacobi rotation
            const theta = (M[q][q] - M[p][p]) / (2 * M[p][q]);
            const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta*theta + 1));
            const c = 1 / Math.sqrt(t*t + 1);
            const s = t * c;

            // Apply rotation to M
            const Mpp = M[p][p], Mqq = M[q][q], Mpq = M[p][q];
            M[p][p] = c*c*Mpp - 2*s*c*Mpq + s*s*Mqq;
            M[q][q] = s*s*Mpp + 2*s*c*Mpq + c*c*Mqq;
            M[p][q] = M[q][p] = 0;

            for (let k = 0; k < 3; k++) {
                if (k !== p && k !== q) {
                    const Mkp = M[k][p], Mkq = M[k][q];
                    M[k][p] = M[p][k] = c*Mkp - s*Mkq;
                    M[k][q] = M[q][k] = s*Mkp + c*Mkq;
                }
            }

            // Update eigenvectors
            for (let k = 0; k < 3; k++) {
                const Vkp = V[k][p], Vkq = V[k][q];
                V[k][p] = c*Vkp - s*Vkq;
                V[k][q] = s*Vkp + c*Vkq;
            }
        }

        // Extract eigenvalues (diagonal of M) and sort
        const eigenvalues = [M[0][0], M[1][1], M[2][2]];
        const indices = [0, 1, 2].sort((a, b) => eigenvalues[b] - eigenvalues[a]);

        const sortedEigenvalues = indices.map(i => eigenvalues[i]);
        const sortedEigenvectors = [
            [V[0][indices[0]], V[0][indices[1]], V[0][indices[2]]],
            [V[1][indices[0]], V[1][indices[1]], V[1][indices[2]]],
            [V[2][indices[0]], V[2][indices[1]], V[2][indices[2]]],
        ];

        return { eigenvalues: sortedEigenvalues, eigenvectors: sortedEigenvectors };
    }

    /**
     * Generate trajectory with FULL transition matrix (all keyframe pairs)
     * This ensures smooth morphing for any transition, not just adjacent ones.
     */
    _generateInterpolatedTrajectory(keyframes, originalPdbs) {
        const n = keyframes.length;
        const stepsPerTransition = this.framesPerTransition;
        let allFrames = [];
        let frameIndex = 0;

        // First, add all keyframes
        const keyframeIndices = [];
        for (let i = 0; i < n; i++) {
            keyframeIndices.push(frameIndex);
            allFrames.push(this._coordsToModelPdb(keyframes[i], frameIndex + 1));
            frameIndex++;
        }

        // Build transition lookup: transitionFrames[from][to] = { start, length }
        this.transitionFrames = {};
        for (let i = 0; i < n; i++) {
            this.transitionFrames[i] = {};
        }

        // Generate interpolated frames for ALL pairs (i → j where i ≠ j)
        for (let from = 0; from < n; from++) {
            for (let to = 0; to < n; to++) {
                if (from === to) continue;

                const startIdx = frameIndex;

                // Generate interpolation frames
                for (let step = 1; step <= stepsPerTransition; step++) {
                    const t = step / (stepsPerTransition + 1);
                    // Ease-in-out for smooth motion
                    const tEased = t < 0.5
                        ? 2 * t * t
                        : 1 - Math.pow(-2 * t + 2, 2) / 2;

                    const interpolated = this._interpolateCoords(keyframes[from], keyframes[to], tEased);
                    allFrames.push(this._coordsToModelPdb(interpolated, frameIndex + 1));
                    frameIndex++;
                }

                this.transitionFrames[from][to] = {
                    start: startIdx,
                    length: stepsPerTransition
                };
            }
        }

        console.log(`Generated ${n} keyframes + ${n * (n-1) * stepsPerTransition} transition frames = ${frameIndex} total`);
        console.log('Transition matrix:', this.transitionFrames);

        // Combine into multi-model PDB
        const trajectory = allFrames.join('\n');
        return { trajectory, keyframeIndices };
    }

    /**
     * Interpolate between two coordinate arrays
     */
    _interpolateCoords(from, to, t) {
        return from.map((atom, i) => ({
            line: atom.line,
            x: atom.x + t * (to[i].x - atom.x),
            y: atom.y + t * (to[i].y - atom.y),
            z: atom.z + t * (to[i].z - atom.z),
        }));
    }

    /**
     * Convert coordinates back to PDB MODEL format
     */
    _coordsToModelPdb(atoms, modelNum) {
        let pdb = `MODEL     ${modelNum}\n`;

        for (const atom of atoms) {
            // Reconstruct ATOM line with new coordinates
            const line = atom.line;
            const newLine = line.substring(0, 30) +
                atom.x.toFixed(3).padStart(8) +
                atom.y.toFixed(3).padStart(8) +
                atom.z.toFixed(3).padStart(8) +
                line.substring(54);
            pdb += newLine + '\n';
        }

        pdb += 'ENDMDL\n';
        return pdb;
    }

    /**
     * Convert array of PDB strings to multi-model trajectory format
     */
    _ensembleToTrajectory(pdbs) {
        let combined = '';
        pdbs.forEach((pdb, i) => {
            combined += `MODEL     ${i + 1}\n`;
            // Strip existing MODEL/ENDMDL if present
            const clean = pdb.split('\n')
                .filter(l => !l.startsWith('MODEL') && !l.startsWith('ENDMDL') && !l.startsWith('END'))
                .join('\n');
            combined += clean + '\n';
            combined += 'ENDMDL\n';
        });
        return combined;
    }

    /**
     * Build transition matrix for pLDDT-weighted random walk
     * P(i→j) ∝ pLDDT_j (favor high-confidence states)
     */
    _buildTransitionMatrix() {
        if (!this.ensemble || !this.ensemble.avg_plddts) return;

        const n = this.ensemble.avg_plddts.length;
        const weights = this.ensemble.avg_plddts.map(p => Math.pow(p, 2)); // Square for sharper discrimination
        const totalWeight = weights.reduce((a, b) => a + b, 0);

        // Normalize to probability distribution
        this.transitionMatrix = weights.map(w => w / totalWeight);

        // Also compute residence times (proportional to confidence)
        this.residenceTimes = weights.map(w => Math.floor((w / totalWeight) * 10) + 1);

        console.log('Transition weights:', this.transitionMatrix.map(w => w.toFixed(3)));
        console.log('Residence times:', this.residenceTimes);
    }

    /**
     * Apply pLDDT coloring to a specific frame
     */
    _applyPlddtColoringToFrame(frameIdx, plddt) {
        // 3Dmol.js applies styles to all frames, but we can use this for info display
        plddt.forEach((confidence, residueIdx) => {
            const color = this._plddtToColor(confidence);
            this.viewer.setStyle(
                { resi: residueIdx + 1 },
                { cartoon: { color: color } }
            );
        });
    }

    /**
     * Start animation through interpolated frames
     */
    startAnimation(mode = null) {
        if (!this.ensemble || this.keyframeIndices.length < 2) {
            console.warn('Need at least 2 keyframes for animation');
            return;
        }

        if (mode) this.animationMode = mode;

        this.isAnimating = true;
        this.currentKeyframe = 0;           // Which keyframe we're currently at
        this.targetKeyframe = 0;            // Which keyframe we're heading towards
        this.transitionProgress = 0;        // Current position in transition (0 = at keyframe)
        this.dwellCounter = 0;              // For residence time on keyframes
        this.inTransition = false;          // Are we currently transitioning?

        // Show first keyframe
        this.viewer.setFrame(this.keyframeIndices[0]);
        this.viewer.render();

        // Start animation loop
        this._animationLoop();

        console.log(`Started ${this.animationMode} animation (${this.keyframeIndices.length} keyframes, full transition matrix)`);
    }

    /**
     * Main animation loop - plays through transitions smoothly
     */
    _animationLoop() {
        if (!this.isAnimating) return;

        const numKeyframes = this.keyframeIndices.length;

        if (!this.inTransition) {
            // We're dwelling on a keyframe
            const residenceTime = this.residenceTimes?.[this.currentKeyframe] || 1;
            const dwellFrames = Math.floor(residenceTime * 5);  // Scale residence time

            if (this.dwellCounter < dwellFrames) {
                this.dwellCounter++;
                this.animationInterval = setTimeout(() => this._animationLoop(), this.frameInterval);
                return;
            }

            // Done dwelling - pick next keyframe and start transition
            this.dwellCounter = 0;
            this._pickNextKeyframe();

            if (this.targetKeyframe === this.currentKeyframe) {
                // Same keyframe selected, just keep dwelling
                this.animationInterval = setTimeout(() => this._animationLoop(), this.frameInterval);
                return;
            }

            // Start transition
            this.inTransition = true;
            this.transitionProgress = 0;
        }

        // We're in a transition - advance through transition frames
        const transition = this.transitionFrames[this.currentKeyframe]?.[this.targetKeyframe];

        if (!transition) {
            console.warn(`No transition found: ${this.currentKeyframe} → ${this.targetKeyframe}`);
            this.inTransition = false;
            this.currentKeyframe = this.targetKeyframe;
            this.animationInterval = setTimeout(() => this._animationLoop(), this.frameInterval);
            return;
        }

        if (this.transitionProgress < transition.length) {
            // Show next transition frame
            const frameIdx = transition.start + this.transitionProgress;
            this.viewer.setFrame(frameIdx);
            this.viewer.render();
            this.transitionProgress++;

            // Emit frame change
            if (this.onFrameChange) {
                // Show interpolated position between keyframes
                const progress = this.transitionProgress / transition.length;
                const displayFrame = progress < 0.5 ? this.currentKeyframe : this.targetKeyframe;
                this.onFrameChange(displayFrame, numKeyframes);
            }
        } else {
            // Transition complete - land on target keyframe
            this.viewer.setFrame(this.keyframeIndices[this.targetKeyframe]);
            this.viewer.render();

            this.currentKeyframe = this.targetKeyframe;
            this.inTransition = false;
            this.transitionProgress = 0;

            if (this.onFrameChange) {
                this.onFrameChange(this.currentKeyframe, numKeyframes);
            }
        }

        // Schedule next frame
        this.animationInterval = setTimeout(() => this._animationLoop(), this.frameInterval);
    }

    /**
     * Pick the next target keyframe based on animation mode
     */
    _pickNextKeyframe() {
        const n = this.keyframeIndices.length;

        switch (this.animationMode) {
            case 'linear':
                this.targetKeyframe = (this.currentKeyframe + 1) % n;
                break;

            case 'backAndForth':
                if (this._direction === undefined) this._direction = 1;
                let next = this.currentKeyframe + this._direction;
                if (next >= n || next < 0) {
                    this._direction *= -1;
                    next = this.currentKeyframe + this._direction;
                }
                this.targetKeyframe = next;
                break;

            case 'weighted':
                // Weighted random choice (favor high-confidence states)
                this.targetKeyframe = this._weightedRandomChoice();
                break;
        }
    }

    /**
     * Stop animation
     */
    stopAnimation() {
        this.isAnimating = false;

        if (this.animationInterval) {
            clearTimeout(this.animationInterval);
            this.animationInterval = null;
        }

        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    /**
     * Toggle animation play/pause
     */
    toggleAnimation() {
        if (this.isAnimating) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
        return this.isAnimating;
    }

    /**
     * Weighted random choice based on transition matrix
     */
    _weightedRandomChoice() {
        if (!this.transitionMatrix) {
            return Math.floor(Math.random() * this.ensemble.pdbs.length);
        }

        const r = Math.random();
        let cumulative = 0;
        for (let i = 0; i < this.transitionMatrix.length; i++) {
            cumulative += this.transitionMatrix[i];
            if (r < cumulative) return i;
        }
        return this.transitionMatrix.length - 1;
    }

    /**
     * Set animation speed (frame interval in ms)
     */
    setAnimationSpeed(ms) {
        this.animationSpeed = ms;
        // Map slider to frame interval: lower = faster
        // Slider range 50-500, map to frameInterval 16-66ms (60fps to 15fps)
        this.frameInterval = Math.max(16, Math.floor(ms / 7.5));
    }

    /**
     * Set animation mode
     */
    setAnimationMode(mode) {
        this.animationMode = mode;
        if (this.isAnimating) {
            this.stopAnimation();
            this.startAnimation();
        }
    }

    /**
     * Go to specific frame
     */
    goToFrame(frameIdx) {
        if (!this.ensemble) return;

        this.currentFrame = Math.max(0, Math.min(frameIdx, this.ensemble.pdbs.length - 1));
        this.viewer.setFrame(this.currentFrame);

        if (this.ensemble.plddts && this.ensemble.plddts[this.currentFrame]) {
            this._applyPlddtColoringToFrame(this.currentFrame, this.ensemble.plddts[this.currentFrame]);
        }

        this.viewer.render();
    }

    /**
     * Get ensemble info
     */
    getEnsembleInfo() {
        if (!this.ensemble) return null;
        return {
            size: this.ensemble.pdbs.length,
            currentFrame: this.currentFrame,
            isAnimating: this.isAnimating,
            mode: this.animationMode,
            avgPlddts: this.ensemble.avg_plddts,
        };
    }
}

// Global instance
window.viewer3d = null;
