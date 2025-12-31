import os
import sys
import json
import torch
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, print_gpu_memory, cuda_oom_protect
from utils.metrics import log_results

# Use the RL model from Exp 1 for the stress test
RL_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), '../../models/lora_rl')
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../logs/exp4_stress_results.json')

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
    
    # 3. Quick Eval
    # We only run first 3 prompts to save time per iteration
    for p in prompts[:3]:
        pid = p['id']
        text = p['text']
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            # Minimal generation
            outputs = model.generate(**inputs, max_new_tokens=20)
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log with iteration ID
        log_key = f"ITER_{iteration_id}"
        log_results(RESULTS_FILE, log_key, pid, generated_text, None, 0.0)
        
    print(f"=== ITERATION {iteration_id} COMPLETE ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0, help="Iteration number")
    args = parser.parse_args()
    
    run_stress_iteration(args.iter)
