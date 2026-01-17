#!/bin/bash

# Experiment 4: Restart Stress Test
# Runs the single_run.py script 10 times, ensuring a full process exit between runs.

echo "Starting Stress Test (10 Iterations)..."

for i in {1..10}
do
   echo "Running Iteration $i..."
   python src/exp4_stress/stress_single_run.py --iter $i
   
   # Optional: Sleep to allow GPU memory cleanup by OS if needed
   sleep 2
done

echo "Stress Test Complete. Check logs/exp4_stress_results.json"
