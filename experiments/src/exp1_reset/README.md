# 🧪 Experiment 1: Reset Invariance

This phase focuses on establishing the "Identity Baseline" and verifying that the model's core remains invariant after behavioral transformations.

## Scripts
- **`1_baseline.py`**: Runs inference on the base model to record original response patterns and embeddings.
- **`2_train_sft.py`**: Mounts the SFT (Supervised Fine-Tuning) behavioral environment.
- **`3_train_rl.py`**: Mounts the RL (Direct Preference Optimization) alignment environment.
- **`4_verify_reset.py`**: Unmounts all adapters and calculates the **Identity Leakage Score (ILS)** to prove reset success.
