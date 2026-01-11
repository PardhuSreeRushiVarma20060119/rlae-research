# 🧠 Source Code (src)

The core implementation of the RLAE and SVAR research framework.

## Architecture Overview

The source code is organized by experimental phase and utility type:

### 🔬 Experimental Phases

- **`exp1_reset/`**: Baseline establishment and Reset Invariance (M1 Locked).
- **`exp2_rlae/`**: Behavioral thinning via magnitude-based pruning.
- **`exp3_svar/`**: Structural stability analysis via adversarial perturbations.
- **`exp4_stress/`**: High-frequency autonomous stress testing.
- **`exp5_comparison/`**: The core **Comparative Proof (SEC1, SEC2, SEC3)** showing identity scars vs. reversibility.

### 🛠️ Infrastructure

- **`utils/`**: Shared utilities for model loading, OOM protection, and metadata metrics.
- **`analysis/`**: Post-run drift calculations and statistical summaries.
- **`verification/`**: The unified `robustness_suite.py` for final system validation.

> [!NOTE]
> All source logic is optimized for **Deterministic Research (Seed 1337)**.
