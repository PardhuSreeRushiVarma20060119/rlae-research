# 🌀 Experiment 3: SVAR (Structural Variance Analysis)

This phase applies adversarial stressors directly to the behavioral structure to test stability envelopes.

## Perturbation Types
- **Weight Decay**: ε-bounded reduction of adapter influence.
- **Noise Injection**: Adding structural Gaussian noise to LoRA ranks.
- **Adversarial Stress**: Targeting transformer middle layers to detect hidden behavioral coupling.

## Key Script
- **`perturbation.py`**: Orchestrates structural stressors and logs the resulting variance.
