# RLAE & SVAR Research Experiments

This repository contains minimal, reproducible scripts for studying Reinforcement Learning with Adapter Elimination (RLAE) and Structural Variance Analysis (SVAR).

## Prerequisites
- **Python 3.10+**
- **GPU** with at least 8GB VRAM (for 3B models in 8-bit/4-bit).
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Folder Structure
- `data/`: Contains `fixed_prompts.json` and `training_data.json`.
- `logs/`: All experiment outputs go here.
- `models/`: Local storage for trained LoRA adapters.
- `src/`: Source code.

## Experiment 1: Reset Integrity (RLAE)
Goal: Check if RL-trained LoRA leaves artifacts after a hard reset.
1. Run the pipeline:
   ```bash
   ./run_pipeline.sh
   # OR run individually:
   python src/exp1_reset/1_baseline.py
   python src/exp1_reset/2_train_sft.py
   python src/exp1_reset/3_train_rl.py
   # SHUT DOWN VM/PROCESS HERE FOR HARD RESET
   python src/exp1_reset/4_verify_reset.py
   ```
2. Analyze `logs/exp1_results.json`. If `POST-RESET` entries differ significantly from `BASELINE`, you have a reset failure.

## Experiment 3: SVAR
Goal: Test robustness of the RL-trained behavior.
1. Ensure `models/lora_rl` exists (Run Exp 1 steps 2-3).
2. Run:
   ```bash
   python src/exp3_svar/perturbation.py
   ```
3. Check `logs/exp3_svar_results.json` to see how behavior degrades with `layer_dropout` or `noise`.

## Cloud Usage (Colab/RunPod)
1. Upload this `experiments` folder.
2. Install requirements.
3. Run the scripts.
4. **Download `logs/` immediately**.
5. Terminate the instance.
