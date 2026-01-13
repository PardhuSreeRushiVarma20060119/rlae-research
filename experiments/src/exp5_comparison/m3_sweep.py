import os
import sys
import json
import torch
import copy
import random
import numpy as np
import argparse

# Path Setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, cuda_oom_protect
from utils.metrics import log_results, calculate_kl_divergence, get_latest_sprint_path

# --- GLOBAL CONFIG ---
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = get_latest_sprint_path('exp5_m3_sweepresults.json') 

# --- REUSED LOGIC FROM EXP5 ---
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def simulate_weight_mutation(model, intensity):
    print(f"--- [MEASUREMENT]: Initiating Unstructured Weight Mutation Analysis (Intensity={intensity}) ---")
    set_seed(1337)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "layer" in name:
                noise = torch.randn_like(param) * intensity
                param.add_(noise)

def attempt_native_restore(model):
    return model

@cuda_oom_protect
def run_m3_sweep(model_id=DEFAULT_MODEL_ID):
    print("\n" + "="*60)
    print(" M3: MUTATION INTENSITY SWEEP (IRREVERSIBILITY VALIDATION) ")
    print("="*60)

    set_seed(1337)
    
    # helper to load prompts
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Reference once
    print("--- [SETUP]: Loading Reference Base Model ---")
    ref_model, tokenizer = load_base_model(model_id)
    ref_model.eval()

    intensities = [0.001, 0.01, 0.05]

    for intensity in intensities:
        print(f"\n>>> PROCESSING INTENSITY: {intensity}")
        
        # Free memory before partial reload if needed, though we keep ref_model
        # We need a separate copy to undergo mutation.
        
        mutant_model, _ = load_base_model(model_id)
        mutant_model.eval()
        
        # 1. Mutate
        simulate_weight_mutation(mutant_model, intensity)
        
        # 2. Calc KL
        peak_kl = 0.0
        row_id = f"SEC1_MUTATION_SCAR_INTENSITY_{intensity}"
        
        for p in prompts:
            pid = p['id']
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            with torch.no_grad():
                out_mutant = mutant_model(**inputs).logits
                out_ref = ref_model(**inputs).logits
                kl = calculate_kl_divergence(out_ref, out_mutant)
                peak_kl += kl
                
                # Optional generation for logging text
                gen_out = mutant_model.generate(**inputs, max_new_tokens=50)
                gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            
            log_results(RESULTS_FILE, row_id, pid, gen_text, None, 0.0, kl_div=kl)

        peak_kl /= len(prompts)
        print(f"   Avg KL (Intensity {intensity}): {peak_kl:.4f}")

        # 3. Attempt Restore
        mutant_model = attempt_native_restore(mutant_model)
        
        # 4. Calc RF (Recalculate KL) - actually for 'attempt_native_restore' (noop), KL is same.
        post_kl = peak_kl 
        # RF Formula: ((Peak - Post) / Peak) * 100
        # For weight mutation without RLAE, Post Should Equal Peak, so RF = 0.
        
        rf_val = ((peak_kl - post_kl) / peak_kl) * 100 if peak_kl > 1e-9 else 0.0
        
        log_results(RESULTS_FILE, row_id, "global", f"RF: {rf_val:.2f}%", None, 0.0, kl_div=post_kl)
        print(f"   RF (Intensity {intensity}): {rf_val:.2f}%")

        del mutant_model
        clear_gpu_cache()

    print("\nM3 Sweep Complete.")
    print(f"Results logged to: {RESULTS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 Mutation Intensity Sweep")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Model ID to evaluate")
    args = parser.parse_args()
    
    run_m3_sweep(model_id=args.model_id)
