# 🛡️ Verification Suite

The final layer of the repository's robustness framework.

## Key Script
- **`robustness_suite.py`**: A unified script that loads results from all three experiments (Reset, RLAE Core, and SVAR) to generate a "Unified Robustness Report."

## Verification Pass Criteria
- **Exp 1**: ILS < 0.05
- **Exp 2**: Stable gradient of behavior collapse vs weight magnitude.
- **Exp 3**: Output variance within ε-bounded stability envelopes.
