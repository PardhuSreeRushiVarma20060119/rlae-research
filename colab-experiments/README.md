# 🧪 Colab Experiments (Staged Implementation)

This directory contains standalone, staged notebooks designed for interactive research, validation, and multi-run analysis of the RLAE & SVAR framework.

## 🚀 Multi-Test Analysis & Validation (M-Series)

We utilize deterministic seed-locking and control flags to prove the robustness of the RLAE & SVAR framework.

### **1. M1: Repeatability (Seed 1337)**

- **Baseline:** Proves structural properties are non-stochastic.
- **Archive:** Uses `REVA4-Research-Lab-Cloud.zip`.

### **2. M2: No-Op Control (Grounding)**

- **Role:** Grounds our metrics (KL, RF) to define "Numerical Zero".
- **Execution:** Skip training (C4) and run the comparison script with `--control`.

---

## 📂 Repository Contents

- **`PaperOne_Experiment_1.ipynb`**: The primary research notebook for Step-by-Step validation.
- **`Stage1_Experiments.ipynb`**: Initial environment mounting (SFT & RL Alignment).
- **`Stage2_ILS_Experiment_Test1.ipynb`**: Diagnostic sensitivity test (Detecting noise envelopes).
- **`Stage2_ILS_Experiment_Test2.ipynb`**: Identity integrity verification (Proving clean unmounting).

## 🔬 Key Theorems Proven

1. **Structural Invariance:** The Base Model remains identical before/after training.
2. **Modular Efficiency:** Complex behavioral shifts achieved via **0.05% parameter modification**.
3. **Metric Grounding:** The M2 control run anchors all subsequent KL/RF measurements.

---

> [!IMPORTANT]
> **Cloud Setup:** Always unzip `REVA4-Research-Lab-Cloud.zip` to ensure you are using the latest seed-locked training and comparison logic.
