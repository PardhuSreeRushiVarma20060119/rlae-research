# 🧠 Source Code (src)

The core implementation of the RLAE and SVAR research framework.

## Architecture Overview
The source code is organized by experimental phase and utility type:

### 🔬 Experimental Phases
- **`exp1_reset/`**: Baseline establishment and Reset Invariance verification.
- **`exp2_rlae/`**: Behavioral thinning via magnitude-based pruning.
- **`exp3_svar/`**: Structural stability analysis via adversarial perturbations.
- **`exp4_stress/`**: High-frequency autonomous stress testing.

### 🛠️ Infrastructure
- **`utils/`**: Shared utilities for model loading, OOM protection, and metadata metrics.
- **`analysis/`**: Post-run drift calculations and statistical summaries.
- **`verification/`**: The unified `robustness_suite.py` for final system validation.
