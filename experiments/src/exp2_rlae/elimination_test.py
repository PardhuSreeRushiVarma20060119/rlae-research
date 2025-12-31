import os
import sys
import json
import torch
import copy
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, print_gpu_memory, cuda_oom_protect
from utils.metrics import log_results, calculate_kl_divergence

RL_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), '../../models/lora_rl')
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../logs/exp2_rlae_results.json')

def eliminate_adapter_by_magnitude(model, elimination_ratio):
    """
    Simulates RLAE by zeroing out a ratio of adapter weights based on magnitude.
    This identifies critical ranks in the low-rank adaptive environment.
    """
    print(f"Eliminating Adapters (Magnitude-based): ratio={elimination_ratio}")
    
    with torch.no_grad():
        all_lora_params = []
        for name, param in model.named_parameters():
            if "lora" in name:
                all_lora_params.append(param)
        
        if not all_lora_params:
            return

        # Flatten all weights to find the global threshold
        all_weights = torch.cat([p.flatten() for p in all_lora_params])
        threshold = torch.quantile(torch.abs(all_weights), elimination_ratio)
        
        for p in all_lora_params:
            mask = torch.abs(p) > threshold
            p.mul_(mask.float())

@cuda_oom_protect
def run_rlae_core(model_id=DEFAULT_MODEL_ID):
    if not os.path.exists(RL_ADAPTER_PATH):
        print("RL Adapter not found. Run Exp 1 first.")
        return

    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)

    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ratio in ratios:
        run_name = f"RLAE_ELIM_{ratio}"
        print(f"--- Running {run_name} (Hardened) ---")
        
        clear_gpu_cache()
        
        # Load Base + Adapter
        base_model, tokenizer = load_base_model(model_id)
        model = PeftModel.from_pretrained(copy.deepcopy(base_model), RL_ADAPTER_PATH)
        
        if ratio > 0:
            eliminate_adapter_by_magnitude(model, ratio)
        
        model.eval()
        base_model.eval()
        
        for p in prompts:
            pid = p['id']
            text = p['text']
            
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                base_outputs = base_model(**inputs)
                model_outputs = model(**inputs)
                
                kl_div = calculate_kl_divergence(base_outputs.logits, model_outputs.logits)
                gen_out = model.generate(**inputs, max_new_tokens=50)
            
            generated_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            log_results(RESULTS_FILE, run_name, pid, generated_text, None, 0.0, kl_div=kl_div)
            
        del base_model
        del model
        clear_gpu_cache()

if __name__ == "__main__":
    run_rlae_core()
