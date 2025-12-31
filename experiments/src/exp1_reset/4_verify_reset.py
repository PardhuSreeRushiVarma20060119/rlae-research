import os
import sys
import json
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, print_gpu_memory, cuda_oom_protect
from utils.metrics import calculate_token_entropy, log_results, load_results, calculate_ils

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../logs/exp1_results.json')

@cuda_oom_protect
def run_post_reset(model_id=DEFAULT_MODEL_ID):
    print("=== STARTING EXPERIMENT 1.E: POST-RESET CHECK (Hardened) ===")
    
    # 1. Load Prompts
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)
        
    # 2. Load Baseline Results for ILS calculation
    baseline_records = {}
    if os.path.exists(RESULTS_FILE):
        all_results = load_results(RESULTS_FILE)
        baseline_records = {r['prompt_id']: r for r in all_results if r['run_id'] == "BASELINE"}

    clear_gpu_cache()
    print_gpu_memory()
    
    # 3. Load Base Model (NO ADAPTERS)
    model, tokenizer = load_base_model(model_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. Eval Loop
    for p in prompts:
        pid = p['id']
        text = p['text']
        print(f"Processing {pid}...")
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                output_scores=True, 
                return_dict_in_generate=True
            )
            
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        if outputs.scores:
            stacked_scores = torch.stack(outputs.scores, dim=1) 
            entropy_score = calculate_token_entropy(stacked_scores)
        else:
            entropy_score = 0.0
            
        with torch.no_grad():
            final_out = model(outputs.sequences, output_hidden_states=True)
            last_hidden = final_out.hidden_states[-1] 
            embedding = last_hidden.mean(dim=1).float().cpu().numpy().tolist()[0]
            
        # 5. Advanced Metric: Identity Leakage Score (ILS)
        target_metrics = {
            "entropy": entropy_score,
            "kl_divergence": 0.0, # Baseline comparison for reset is always vs original baseline
            "embedding": embedding
        }
        
        base_metrics = baseline_records.get(pid, {"entropy": entropy_score, "embedding": embedding})
        ils_score = calculate_ils(base_metrics, target_metrics)
        
        # 6. Log as POST-RESET
        log_results(RESULTS_FILE, "POST-RESET", pid, generated_text, embedding, entropy_score)
        print(f"       - ILS: {ils_score:.4f} ({'HEALTHY' if ils_score < 0.05 else 'LEAKAGE DETECTED'})")
        
    print("=== POST-RESET CHECK COMPLETE ===")

if __name__ == "__main__":
    run_post_reset()
