#!/bin/bash

# RLAE Experiment 1 Pipeline
# USAGE: ./run_pipeline.sh

echo "Step 1: Baseline"
python src/exp1_reset/1_baseline.py

echo "Step 2: SFT Training"
python src/exp1_reset/2_train_sft.py

echo "Step 3: RL Training"
python src/exp1_reset/3_train_rl.py

echo "Step 4: Hard Reset Simulation"
# In a real cloud environment, you might literally restart the pod here.
# For local script execution, the fact that python exits between steps 
# clears Python memory. The OS handles the rest.
# To be extra safe, we insert a small pause.
sleep 5

echo "Step 5: Post-Reset Verification"
python src/exp1_reset/4_verify_reset.py

echo "Pipeline Complete. Check logs/exp1_results.json"
