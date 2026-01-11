# 🧪 Experiment 1: Reset Invariance

This phase focuses on establishing the "Identity Baseline" and verifying that the model's core remains invariant after behavioral transformations.

## Scripts

- **Deterministic Training (M1 Ready)**: All scripts use `seed=1337` to ensure reproducible structural properties.
- **`1_baseline.py`**: Records the "Identity Zero" patterns.
- **`2_train_sft.py`**: Mounts SFT environment (Supports `seed=1337`).
- **`3_train_rl.py`**: Mounts RL alignment environment (Supports `seed=1337`).
- **`4_verify_reset.py`**: Calculates ILS to verify identity restoration.
