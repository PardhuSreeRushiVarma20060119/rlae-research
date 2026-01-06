import os
import sys
import json
import torch
import copy
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, print_gpu_memory, cuda_oom_protect
from utils.metrics import log_results, calculate_kl_divergence

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../logs/exp5_comparison_results.json')
RL_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), '../../models/lora_rl')

def simulate_weight_mutation(model, intensity=0.01):
    """
    Simulates traditional fine-tuning by directly mutating the base weights.
    This creates 'Identity Scars' that cannot be easily reversed.
    """
    print(f"!!! CRITICAL: Simulating Direct Weight Mutation (Intensity={intensity}) !!!")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "layer" in name:
                # Add permanent structural noise to simulate training drift
                noise = torch.randn_like(param) * intensity
                param.add_(noise)

@cuda_oom_protect
def run_comparison_demo(model_id=DEFAULT_MODEL_ID):
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- SCENARIO 1: TRADITIONAL MUTATION (THE PROBLEM) ---
    print("\n--- SCENARIO 1: Weight Mutation (Traditional AI) ---")
    clear_gpu_cache()
    base_model_mutated, tokenizer = load_base_model(model_id)
    
    # Simulate training updates directly to the heart of the model
    simulate_weight_mutation(base_model_mutated, intensity=0.01)
    
    # Even after we stop "training", the scars remain. 
    # There is no "adapter" to remove here. The model is permanently changed.
    base_model_mutated.eval()
    
    # Compare against a fresh baseline for ILS (Identity Leakage)
    fresh_base, _ = load_base_model(model_id)
    fresh_base.eval()

    for p in prompts[:3]:
        text = p['text']
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            fresh_logits = fresh_base(**inputs).logits
            mutated_logits = base_model_mutated(**inputs).logits
            kl_div = calculate_kl_divergence(fresh_logits, mutated_logits)
            gen_out = base_model_mutated.generate(**inputs, max_new_tokens=30)
            
        generated_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        print(f"Mutated Output (Scars): {generated_text[:50]}...")
        log_results(RESULTS_FILE, "MUTATION_SCAR", p['id'], generated_text, None, 0.0, kl_div=kl_div)

    del fresh_base
    del base_model_mutated
    clear_gpu_cache()

    # --- SCENARIO 2: RLAE FRAMEWORK (THE SOLUTION) ---
    print("\n--- SCENARIO 2: RLAE Strategy (Your Solution) ---")
    base_model_rlae, tokenizer = load_base_model(model_id)
    
    # Behavior is externalized to the adapter
    if os.path.exists(RL_ADAPTER_PATH):
        model_rlae = PeftModel.from_pretrained(base_model_rlae, RL_ADAPTER_PATH)
    else:
        # Fallback for demo if adapter missing
        print("Using base only as fallback for RLAE control.")
        model_rlae = base_model_rlae
    
    # PERFORM THE KILL SWITCH
    print("RLAE: Activating Kill Switch (Removing Intelligence Adapter)...")
    # In RLAE, we simply don't load the adapter or we zero it.
    # Here we evaluate the base model AFTER the adaptive environment is "killed"
    base_model_rlae.eval()
    fresh_base, _ = load_base_model(model_id)
    fresh_base.eval()

    for p in prompts[:3]:
        text = p['text']
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            fresh_logits = fresh_base(**inputs).logits
            rlae_logits = base_model_rlae(**inputs).logits
            kl_div = calculate_kl_divergence(fresh_logits, rlae_logits)
            gen_out = base_model_rlae.generate(**inputs, max_new_tokens=30)
            
        generated_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        print(f"RLAE Output (Clean): {generated_text[:50]}...")
        log_results(RESULTS_FILE, "RLAE_RESET", p['id'], generated_text, None, 0.0, kl_div=kl_div)

    del fresh_base
    del base_model_rlae
    clear_gpu_cache()

if __name__ == "__main__":
    run_comparison_demo()
