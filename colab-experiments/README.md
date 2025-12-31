# 🧪 Colab Experiments (Staged Implementation)

This directory contains standalone, staged notebooks designed for execution in Google Colab. These notebooks are optimized for the free T4 GPU tier and provide a more interactive way to explore specific research phases.

## Notebooks
- **`Stage1_Experiments.ipynb`**: Focuses on the foundational RLAE lifecycle. This includes establishining baselines, Supervised Fine-Tuning (SFT), and the initial Reinforcement Learning (RL) alignment steps.
- **`Stage2_ILS_Experiment.ipynb`**: A specialized notebook for advanced Identity Leakage Score (ILS) analysis. It performs deeper structural tests on the "Frozen Core" to detect behavioral drift with higher precision.

## Relationship to Central Repository
While the `src/` scripts in the main `experiments/` directory provide the "Canonical Implementation" for automated pipelines, these notebooks are intended for:
1. **Interactive Research**: Testing new reward functions or instruction templates.
2. **Visual Verification**: Generating inline plots for behavior distribution and log-probability shifts.
3. **Debugging**: Isolating issues in the TRL (Transformer Reinforcement Learning) stack.

---

> [!TIP]
> Use these notebooks if you want a step-by-step interactive experience with visible outputs for every transformation layer.
