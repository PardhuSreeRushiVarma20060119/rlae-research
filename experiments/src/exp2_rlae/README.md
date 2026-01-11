# ✂️ Experiment 2: RLAE Core (Structural Elimination)

This experiment investigates **structural behavioral collapse** via controlled rank-reduction of LoRA adapters.

## Mechanics

- **Magnitude-Based Rank Reduction**: Uses the `elimination_test.py` script to isolate and maintain high-variance outcome-level features while nullifying non-contributory weight structures.
- **Structural Invariance Mapping**: Tracks token entropy and KL divergence variance response as the adapter's structural rank is systematically reduced.

## Key Script

- **`elimination_test.py`**: Principal entry point for RLAE structural thinning analysis.
