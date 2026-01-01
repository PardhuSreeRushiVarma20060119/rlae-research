# Contributing to AI-RDE-Repo (RLAE & SVAR)

Thank you for your interest in contributing to **AI-RDE-Repo**, a research-focused repository exploring **Runtime Low-Rank Adaptive Environments (RLAE)** and **Structural Variance Analysis for Robustness (SVAR)**.

This repository is **research-grade**, **safety-critical**, and **theory-driven**.  
Contributions are welcome — but only if they respect the core principles of the project.

Please read this document carefully before opening an issue or pull request.

---

## 🧠 Project Philosophy

AI-RDE-Repo is built on the following non-negotiable principles:

- **Frozen Core Invariance**  
  The base model must remain strictly immutable.

- **Behavioral Externalization**  
  All learning and specialization must exist *outside* the model as LoRA artifacts.

- **Provable Reversibility**  
  Any behavior must be unloadable with a mathematically clean reset.

- **No Identity Persistence**  
  The system must not accumulate long-term identity, memory, or hidden state.

- **Verification Over Assumption**  
  Claims must be supported by experiments, metrics, and diagnostics (ILS, SVAR).

If a contribution violates or weakens any of these principles, it will not be accepted.

---

## 📌 What You Can Contribute

We welcome contributions in the following areas:

### 🔬 Research & Experiments
- New **RLAE experiments** (training, unmounting, thinning)
- **SVAR perturbation strategies** (bounded, reversible, logged)
- Additional **verification metrics** (must not obscure ILS or reset integrity)
- Reproducibility improvements (multi-run stability, noise profiling)

### 🧪 Diagnostics & Metrics
- Enhancements to **Identity Leakage Score (ILS)**
- Visualization of variance surfaces, stability envelopes, or correlations
- Hardware noise characterization (GPU non-determinism analysis)

### 🧱 Infrastructure & Tooling
- Runtime governance utilities (load/unload safety, kill paths)
- Experiment orchestration, logging, or telemetry
- CUDA safety, OOM protection, determinism tooling

### 📄 Documentation
- Clarifications or expansions of RLAE / SVAR theory
- Experiment walkthroughs
- Reproducibility or hardware-specific notes

---

## 🚫 What Will NOT Be Accepted

The following are explicitly out of scope:

- ❌ End-to-end fine-tuning of base models
- ❌ Persistent agent memory or identity
- ❌ Hidden state accumulation across resets
- ❌ “Performance-only” improvements without robustness analysis
- ❌ Claims without experimental or mathematical backing
- ❌ Black-box alignment or monitoring approaches
- ❌ Changes that reduce reset purity or auditability

---

## 🧪 Experimental Standards

All experimental contributions **must**:

1. Preserve **Frozen Core Invariance**
2. Log **before / after / reset** behavior
3. Include **ILS measurements**
4. Demonstrate **reset integrity**
5. Be reproducible on supported hardware (e.g., NVIDIA T4)

If an experiment introduces noise or instability, it must be **explicitly measured and explained**.

---

## 🧾 Pull Request Guidelines

Before submitting a PR:

- Ensure code is **clean, minimal, and well-documented**
- Add or update relevant experiment logs
- Clearly state:
  - What assumption is being tested
  - What invariant is being validated
  - What failure mode is being probed
- Include results, not just implementation

**PRs without experimental evidence will be closed.**

---

## 🐛 Reporting Issues

When opening an issue, please include:

- Clear description of the problem
- Hardware details (GPU, VRAM, CUDA version)
- Exact experiment or script used
- Logs, metrics, or error traces
- Expected vs observed behavior

Vague or speculative issues will be closed.

---

## 🔐 Security & Safety

If you discover a flaw that:
- Breaks reset integrity
- Allows identity persistence
- Bypasses governance
- Corrupts the frozen core

Please **do not open a public issue**.  
Report it privately to the maintainer.

---

## 📜 Licensing

By contributing, you agree that your contributions will be licensed under the same license as this repository (see `LICENSE`).

---

## 🧭 Final Note

This repository is not about making models *stronger*.

It is about making intelligence:
- **Bounded**
- **Observable**
- **Reversible**
- **Destroyable**

If your contribution strengthens that goal — it is welcome.
If it weakens it — it will be rejected.

— *AI-RDE Maintainer - Pardhu Varma Konduru*


