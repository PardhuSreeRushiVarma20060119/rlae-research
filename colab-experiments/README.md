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

---

## 📂 Repository Contents

- **`PaperOne_Experiment_1.ipynb`**: The primary research notebook for Step-by-Step validation.
- **`Stage1_Experiments.ipynb`**: Initial environment mounting (SFT & RL Alignment).
- **`Stage2_ILS_Experiment_Test1.ipynb`**: Diagnostic sensitivity test (Detecting noise envelopes).
- **`Stage2_ILS_Experiment_Test2.ipynb`**: Identity integrity verification (Proving clean unmounting).

## 🔬 Key Theorems Proven

1. **Structural Invariance:** The Base Model remains structurally consistent before/after training.
2. **Modular Efficiency:** Complex behavioral shifts achieved via **0.05% parameter modification**.
3. **Metric Grounding:** The M2 control run anchors all subsequent KL/RF measurements.

---

> [!IMPORTANT]
> **Cloud Setup:** Always unzip `REVA4-Research-Lab-Cloud.zip` to ensure you are using the latest standardized training and comparison logic.
