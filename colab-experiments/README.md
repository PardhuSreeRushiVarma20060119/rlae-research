# 🧪 Colab Experiments (Staged Implementation)

This directory contains standalone, staged notebooks designed for interactive research, validation, and multi-run analysis of the RLAE & SVAR framework.

## 🚀 Multi-Test Analysis & Validation (M-Series)

We utilize outcome-level verification and grounding controls to prove the robustness of the RLAE & SVAR framework.

### **1. M1: Repeatability (Outcome-Class Protocol)**

- **Objective:** Proves outcomes are structurally invariant across independent seeds and runtimes.
- **Evidence:** See `logs/Sprint-3/exp5_comparison_results.json`.
- **Method:** `fixed_seed(1337)` applied to all stochastic generators.

### **2. M2: No-Op Control (Grounding)**

- **Objective:** Grounds metrics (KL, RF) to define "Numerical Zero" for behavioral attachment.
- **Evidence:** See `logs/Sprint-4/exp5_comparison_results.json`.
- **Method:** Execution with `--control` flag to bypass weight mutation (C5) and training (C4).

### **3. M3: Mutation Intensity Sweep**

- **Objective:** Proves irreversibility increases monotonically with mutation intensity (`0.001` → `0.05`).
- **Flow:** C4 (new intensity) → C5 → C8.
- **Method:** `m3_sweep.py` running automated intensity loop.

---

## 📂 Repository Contents

### **1. M-Series (Canonical Verification)**

Located in `M-Series/`, these notebooks represent the finalized proofs for the paper.

- **`PaperOne_CoreMExperiment.ipynb`**: Foundational Baseline Weight Mutation & RLAE Method runs.
- **`PaperOne_M1Experiment.ipynb`**: **M1 (Repeatability)** verification. Proves structural invariance via `fixed_seed(1337)`.
- **`PaperOne_M2Experiment.ipynb`**: **M2 (Grounding)** verification. Proves the "Identity Zero" baseline using the `--control` no-op execution.
- **`PaperOne_M3Experiment.ipynb`**: **M3 (Mutation Sweep)** verification. Maps the stability envelope of weight mutation scars.

### **2. StageExp (Developmental Validation)**

Located in `StageExp/`, these notebooks track the incremental development of the RLAE protocols.

- **`Stage1_Experiments.ipynb`**: Initial SFT & RL alignment environment mounting.
- **`Stage2_ILS_Experiment_Test1.ipynb`**: Diagnostic sensitivity testing and noise envelope detection.
- **`Stage2_ILS_Experiment_Test2.ipynb`**: Identity integrity verification and clean unmounting proofs.

## 🔬 Key Theorems Proven

1. **Structural Invariance:** The Base Model remains structurally consistent before/after training.
2. **Modular Efficiency:** Complex behavioral shifts achieved via **0.05% parameter modification**.
3. **Metric Grounding:** The M2 control run anchors all subsequent KL/RF measurements.

---

> [!IMPORTANT]
> **Cloud Setup:** Always unzip `REVA4-Research-Lab-Cloud.zip` to ensure you are using the latest standardized training and comparison logic.
