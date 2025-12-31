# ✂️ Experiment 2: RLAE Core (Behavioral Elimination)

This experiment investigates **structural behavioral collapse** by thinning LoRA adapters.

## Mechanics
- **Magnitude-Based Pruning**: Uses the `elimination_test.py` script to identify and preserve critical weight ranks while zeroing out non-essential parameters.
- **Collapse Mapping**: Tracks how token entropy and KL divergence variance respond as the adapter's rank is reduced.

## Key Script
- **`elimination_test.py`**: Main entry point for RLAE thinning analysis.
