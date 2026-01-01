<img width="1920" height="1080" alt="RLAE SVAR" src="https://github.com/user-attachments/assets/d581cd67-3c90-493f-b5e6-e739729aaed1" />

# 🛡️ RLAE & SVAR: Advanced Robustness & AI Research Repository
> **Proving intelligence through bounded invariance and structural perturbation.**

This repository is a technically rigorous research environment dedicated to the development and evaluation of **Runtime Low-Rank Adaptive Environments (RLAE)** and **Structural Variance Analysis for Robustness (SVAR)**.

Our mission is to move AI alignment from **hidden weight mutation** to a **runtime-governed, verifiable behavioral paradigm**, where intelligence is modular, reversible, and mathematically provable.

---

## 🔬 Theoretical Framework

### 1. RLAE (Runtime Low-Rank Adaptive Environments)
RLAE is a learning paradigm in which reinforcement learning updates are applied **exclusively to LoRA parameters**. By keeping the base model permanently frozen, learning is externalized into explicit runtime-controlled environments.

Core principles:
- **Frozen Core Invariance:** The foundation model identity never changes.
- **Behavioral Externalization:** All learned skills exist as swappable LoRA artifacts.
- **Killability & Reversibility:** Any behavior can be destroyed instantly without model damage.
- **No Persistent Identity:** There is no cumulative self—only transient behavioral composition.

RLAE treats intelligence as a **governed process**, not an evolving entity.

---

### 2. SVAR (Structural Variance Analysis for Robustness)
SVAR is a **diagnostic-only** framework designed to assess robustness, reset integrity, and non-identity persistence in modular AI systems—especially those built under RLAE.

SVAR does **not** train models and does **not** modify behavior.

Key capabilities:
- **Identity Leakage Score (ILS):** A fused metric tracking structural drift with high precision.
- **Stability Envelopes:** Measuring behavioral resilience under ε-bounded perturbations.
- **Reset Integrity Verification:** Ensuring post-reset behavior is statistically identical to baseline.

SVAR evaluates what breaks when structure is stressed—safely and deliberately.

---

## 🛠️ Technical Architecture

### **The Frozen Core Strategy**
We utilize **Qwen2.5-3B-Instruct** as the base model, loaded under 4-bit / 16-bit quantization and kept **strictly frozen** throughout the system lifecycle.

Behavioral capability is introduced via:
- **SFT Environment:** Supervised Fine-Tuning for behavioral specialization.
- **RL Alignment Environment:** DPO-based preference alignment.

At no point is the base model mutated.

---

### **Advanced Robustness Hardening**
- **⚡ CUDA OOM Protection:** A stateful decorator (`@cuda_oom_protect`) that detects VRAM exhaustion, clears GPU cache, and safely resumes execution.
- **🔍 Identity Leakage Score (ILS):** Multi-metric fusion (KL divergence + embedding drift + entropy variance) with 0.01-level resolution.
- **📐 Magnitude-Based Pruning:** Structural thinning of LoRA adapters to isolate minimal behavioral cores.

---

## 🚀 Cloud Execution Guide (Google Colab T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/PardhuSreeRushiVarma20060119/AI-RDE-Repository/blob/main/colab-experiments/Stage1_Experiments.ipynb
)

### 🛠️ Infrastructure & Environment
- **Primary Compute:** Google Colab (NVIDIA T4, 16GB VRAM)
- **Integration:** Official Google Colab GitHub App for bidirectional synchronization

---

## 🧪 Experimental Lifecycle

### **Phase 1: Environment Setup**
```bash
!unzip research.zip
%cd experiments
!pip install -q -r requirements.txt
```

### **Phase 2: Establish Invariance Lifecycle**
1. **Baseline Run:** `!python src/exp1_reset/1_baseline.py`
2. **SFT Training:** `!python src/exp1_reset/2_train_sft.py`
3. **RL Alignment:** `!python src/exp1_reset/3_train_rl.py`
4. **Reset Verification:** `!python src/exp1_reset/4_verify_reset.py`

### **Phase 3: Robustness Diagnostics**
1. **Behavioral Elimination:** `!python src/exp2_rlae/elimination_test.py`
2. **SVAR Perturbation:** `!python src/exp3_svar/perturbation.py`
3. **Unified Report:** `!python src/verification/robustness_suite.py`

---

## 📂 Repository Structure

```text
├── data/
├── logs/
├── models/
├── project-scope/
├── src/
└── WALKTHROUGH.md
```

[!NOTE] > This repository is designed for **deterministic research**. All experiments are logged with timestamps and hardware telemetry to ensure reproducibility across different CUDA environments. 

**Status:** READY | **Hardened:** YES | **Robustness Profile:** ADVANCED

---

> *“Intelligence as powerful and alive, yet deliberately hollow at its center — governed, observable, and stripped of its identity.”*
