<img width="1920" height="1080" alt="RLAE SVAR" src="https://github.com/user-attachments/assets/d581cd67-3c90-493f-b5e6-e739729aaed1" />

# REVA4 Research Experimentation
>
> **Proving intelligence through bounded invariance and structural perturbation.**

This repository is a technically rigorous research environment dedicated to the development and evaluation of **Runtime Low-Rank Adaptive Environments (RLAE)** and **Structural Variance Analysis for Robustness (SVAR)**.

Our mission is to move AI alignment from **hidden weight mutation** to a **runtime-governed, verifiable behavioral paradigm**, where intelligence is modular, reversible, and formally verifiable.

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
- **Reset Integrity Verification:** Ensuring post-reset behavior is statistically consistent with baseline.

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

## 🧪 Mandatory Validation Experiments (M-Series)

To ensure scientific rigor and repeatability, all RLAE evaluations must pass the following validation suite:

- **M1 — Repeatability Run:** Verification of repeatable outcomes using a standardized global seed (`1337`). Ensures that results are a property of the architecture, independent of stochastic initial states.
- **M2 — No-Op Control:** Metric grounding run using the `--control` flag. Confirms that mounting and immediately unmounting an *un-trained* adapter yields `KL ≈ 0` and `RF = 100%`.
- **M3 — Intensity Sweep:** Evaluation of recoverability across increasing mutation magnitudes (ε-scaling).
- **M4 — Multi-Model Path:** Cross-verification of structural invariance on different base model scales.
- **M5 — Metric Grounding:** Direct correlation analysis between ILS, KL, and standard perplexity (PPL).

---

## 🚀 Cloud Execution Guide (Google Colab)

### 🛠️ Infrastructure & Environment

- **Platform:** Google Colab (T4/L4 GPU)
- **Archive:** `REVA4-Research-Lab-Cloud.zip` (Contains pre-configured seed locking and control flags)

### **Atomic Validation Protocols**

### **Atomic Validation Protocols**

The RLAE framework is verified through independent, reproducible protocols that can be executed in any order (post-setup). These protocols map directly to the **Canonical Cell Roles (C0-C8)** defined in the `M-Series` research notebooks.

#### **Protocol A: Environment Initialization (C0)**

*Required for all sessions.*

```bash
!unzip REVA4-Research-Lab-Cloud.zip
!pip install -q -r experiments/requirements.txt
```

#### **Protocol B: Metric Grounding & Control (M2/C1)**

*Verifies the toolchain's neutral baseline and seed locking.*

- **Objective:** Verify "Identity Zero" baseline before training.
- **Command:** `!python src/exp5_comparison/irreversibility_test.py --control`
- **Output:** Confirms `KL ≈ 0.0000` | `RF = 100%` (No-Op Control).

#### **Protocol C: Behavioral Construction (C4)**

*Injects specific behaviors via RLAE adapters.*

- **Phase 1 (Baseline):** `!python src/exp1_reset/1_baseline.py`
- **Phase 2 (SFT):** `!python src/exp1_reset/2_train_sft.py`
- **Phase 3 (RL):** `!python src/exp1_reset/3_train_rl.py`

#### **Protocol D: Structural Verification (M1/C6)**

*The primary proof of structural invariance (Repeatability).*

- **Comparative Proof:** `!python src/exp5_comparison/irreversibility_test.py`
- **Reset Check:** `!python src/exp1_reset/4_verify_reset.py`
- **Pass Criteria:** Structural outcomes match baseline signatures within ε-bounds.

#### **Protocol E: Advanced Diagnostics (SVAR/C5)**

*Stress-tests the behavioral boundaries and stability envelopes.*

- **Structural Elimination:** `!python src/exp2_rlae/elimination_test.py`
- **Perturbation Analysis:** `!python src/exp3_svar/perturbation.py`
- **ILS Stage 2:** Checks specific identity leakage thresholds (from `Stage2_ILS` notebooks).

#### **Protocol F: Runtime Reliability (Stress)**

*Ensures rigorous availability under load.*

- **Stress Test:** `!python src/exp4_stress/stress_single_run.py` (100-cycle inference load)

#### **Protocol G: Canonical Reporting (C8)**

*Synthesizes all telemetry into a final report.*

- **Command:** `!python src/verification/robustness_suite.py`
- **Artifact:** Generates `canonical_diagnostic_results.tar.gz`.

---

## 📂 Repository Structure

```text
├── arts/               # Research diagrams and visual assets
├── colab-experiments/  # Jupyter notebooks for cloud execution (T4/L4)
├── experiments/        # Core execution environment and local scripts
│   ├── data/           # Local datasets and indices
│   ├── logs/           # Experiment logs and telemetry
│   ├── models/         # Quantized model artifacts
│   └── src/            # Experimental logic and RLAE/SVAR implementation
├── project-scope/      # Documentation on research boundaries
├── reports/            # Markdown and PDF research reports
└── WALKTHROUGH.md      # Detailed roadmap and technical guide
```

> [!NOTE]
> This repository is designed for **repeatable research**. All experiments are logged with timestamps and hardware telemetry to ensure outcome-level consistency across different CUDA environments.

![Status](https://img.shields.io/badge/STATUS-READY-darkgreen?style=for-the-badge&logo=checkmarx)
![Hardened](https://img.shields.io/badge/HARDENED-YES-darkblue?style=for-the-badge&logo=shield)
![Robustness](https://img.shields.io/badge/ROBUSTNESS-ADVANCED-darkred?style=for-the-badge&logo=target)

---

© 2026 REVA4 Research Team.  
Licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

> *“Intelligence as powerful and alive, yet deliberately hollow at its center — governed, observable, and stripped of its identity.”*
