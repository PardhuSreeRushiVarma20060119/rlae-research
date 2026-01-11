# 📂 Data Directory

This directory contains the foundational datasets used for model training, alignment, and verification.

## Files

- **`fixed_prompts.json`**: A curated set of 10 prompts used to establish the baseline identity and verify reset integrity. These are structured to test specific reasoning and behavioral traits.
- **`training_data.json`**: The dataset for Supervised Fine-Tuning (SFT). It contains instruction-response pairs designed to shift the model's behavior towards a structured, "concept-category-summary" response format.

## Usage in Research

These datasets are loaded by:

- `1_baseline.py`
- `2_train_sft.py`
- `exp5_comparison/irreversibility_test.py` (M1/M2 runs)
- `4_verify_reset.py`
