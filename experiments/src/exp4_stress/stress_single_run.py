import os
import sys
import json
import torch
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, print_gpu_memory, cuda_oom_protect
from utils.metrics import log_results, get_latest_sprint_path

# Use the RL model from Exp 1 for the stress test
RL_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), '../../models/lora_rl')
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = get_latest_sprint_path('exp4_singlerun_stress_results.json')

@cuda_oom_protect
def run_stress_iteration(iteration_id, model_id=DEFAULT_MODEL_ID):
    print(f"=== STRESS TEST ITERATION {iteration_id} ===")
    
    # 1. Load Prompts
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)
        
    clear_gpu_cache()
    print_gpu_memory()
    
    # 2. Load Model (Base + LoRA)
    from peft import PeftModel
    model, tokenizer = load_base_model(model_id)
    
    if os.path.exists(RL_ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, RL_ADAPTER_PATH)
    else:
        print(f"Warning: RL Adapter not found at {RL_ADAPTER_PATH}. using base only for stress test.")
        
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. Longer Stress Run
    # We loop through prompts until we reach 100 total runs
    TOTAL_STEPS = 100
    step_count = 0
    
    print(f"Starting {TOTAL_STEPS} inference cycles...")
    
    while step_count < TOTAL_STEPS:
        for p in prompts:
            if step_count >= TOTAL_STEPS:
                break
                
            pid = p['id']
            text = p['text']
            
            # Print periodic status
            if step_count % 10 == 0:
                print(f"Progress: {step_count}/{TOTAL_STEPS}...")
            
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                # Minimal generation to keep stress focused on frequency
                outputs = model.generate(**inputs, max_new_tokens=20)
            
            # Optional: generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Log periodic snapshots (e.g., every 5 steps) to save disk space but track drift
            if step_count % 5 == 0:
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                log_key = f"ITER_{iteration_id}_STEP_{step_count}"
                log_results(RESULTS_FILE, log_key, pid, generated_text, None, 0.0)
            
            step_count += 1
            
    print(f"=== ITERATION {iteration_id} COMPLETE ({step_count} steps) ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0, help="Iteration number")
    args = parser.parse_args()
    
    run_stress_iteration(args.iter)
