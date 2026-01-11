# 📂 Data Directory

This directory contains the foundational datasets used for model training, alignment, and canonical validation.

## Files

- **`fixed_prompts.json`**: A controlled selection of 10 prompts used to establish the baseline identity and validate structural invariance. These are structured to probe specific reasoning and behavioral traits.
- **`training_data.json`**: The dataset for Supervised Fine-Tuning (SFT). It contains instruction-response pairs designed to shift the model's behavior towards a structured, "concept-category-summary" outcome-level format.

## Usage in Research

These datasets are loaded by:

- `1_baseline.py`
- `2_train_sft.py`
- `exp5_comparison/irreversibility_test.py` (M1/M2 runs)
- `4_verify_reset.py`
