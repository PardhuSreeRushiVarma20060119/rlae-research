# 🛡️ Canonical Validation Suite

The canonical layer of the repository's robustness framework.

## Key Script

- **`robustness_suite.py`**: A unified script that loads results from all experimental phases (Structural Invariance, Structural Elimination, and SVAR) to generate a "Canonical Robustness Report."

## Validation Pass Criteria

- **Exp 1 (Invariance)**: ILS < 0.05
- **Exp 2 (Elimination)**: Stable gradient of structural collapse vs weight magnitude.
- **Exp 3 (SVAR)**: Outcome variance within ε-bounded stability margins.
