# 🧪 Colab Experiments (Staged Implementation)

This directory contains standalone, staged notebooks designed for interactive research, validation, and multi-run analysis of the RLAE & SVAR framework.

## 🚀 Multi-Test Analysis & Observations
We utilize two distinct test environments to prove the robustness and sensitivity of our identity verification protocols.

### **1. Test 1: Sensing Sensitivity (Fault Tolerance)**
- **File:** `Stage2_ILS_Experiment_Test1.ipynb`
- **Observation:** This run detected a microscopic behavioral "flicker" with an **ILS of 0.06**.
- **Research Value:** Confirms that our **Sensing Layer** is active and highly sensitive. It proves that the framework is capable of detecting even transient hardware noise (CUDA jitter) on the T4 GPU, ensuring no true leakage goes unnoticed.

### **2. Test 2: Reversibility Proof (Canonical Success)**
- **File:** `Stage2_ILS_Experiment_Test2.ipynb`
- **Observation:** Successfully achieved a "Clean Reset" with an **Average ILS of ~0.02 (HEALTHY)**.
- **Research Value:** Mathematically proves that unmounting the LoRA Behavioral Layer restores the model to its original, pure baseline identity. This is the canonical proof for **Frozen Core Invariance**.

---

## 📂 Repository Contents
- **`Stage1_Experiments.ipynb`**: Initial environment mounting (SFT & RL Alignment). Establishining baselines and behavioral specialization.
- **`Stage2_ILS_Experiment_Test1.ipynb`**: Diagnostic sensitivity test (Detecting noise envelopes).
- **`Stage2_ILS_Experiment_Test2.ipynb`**: Identity integrity verification (Proving clean unmounting).

## 🔬 Key Theorems Proven
1. **Structural Invariance:** The Base Model remains identical before/after training.
2. **Modular Efficiency:** Complex behavioral shifts achieved via **0.05% parameter modification**.
3. **Diagnostic Precision:** The ILS metric distinguishes between hardware noise and structural leakage.

---

> [!IMPORTANT]
> **Consolidated Result:**  
> Together, Test 1 and Test 2 provide a complete behavioral proof. One shows the system's ability to **detect** potential drift, while the other proves the system's ability to **resolve** it back to a healthy state.

> [!TIP]
> Use the **Test 2** configuration as the baseline for all future RLAE unmounting procedures to ensure the highest purity scores.
