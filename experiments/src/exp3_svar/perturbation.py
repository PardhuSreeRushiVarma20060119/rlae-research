import os
import sys
import json
import torch
import numpy as np
import copy
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, print_gpu_memory, cuda_oom_protect
from utils.metrics import log_results, calculate_kl_divergence

# Use the RL model from Exp 1
RL_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), '../../models/lora_rl')
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../logs/exp3_svar_results.json')

def perturb_adapter(model, perturbation_type, intensity):
    """
    Applies structural damage to the LoRA adapter.
    """
    print(f"Applying Perturbation: {perturbation_type} level={intensity}")
    
    with torch.no_grad():
        target_params = []
        for name, param in model.named_parameters():
            if "lora" in name:
                target_params.append((name, param))
        
        for name, param in target_params:
            # 1. Random Layer Removal (Zeroing out)
            if perturbation_type == "layer_dropout":
                if np.random.rand() < intensity:
                    param.zero_()
            
            # 2. Weight Weakening (Global scaling)
            elif perturbation_type == "weight_decay":
                param.mul_(1.0 - intensity)
            
            # 3. Noise Injection (Normal)
            elif perturbation_type == "noise":
                noise = torch.randn_like(param) * intensity
                param.add_(noise)
            
            # 4. Adversarial Stressors (Targeted middle-layer noise)
            elif perturbation_type == "adversarial":
                # Middle layers are typically layers 8-24 in a 32-layer transformer
                if any(f"layers.{i}." in name for i in range(8, 24)):
                    noise = (torch.rand_like(param) - 0.5) * intensity * 2.0
                    param.add_(noise)

@cuda_oom_protect
def run_svar(model_id=DEFAULT_MODEL_ID):
    if not os.path.exists(RL_ADAPTER_PATH):
        print("RL Adapter not found. Run Exp 1 first.")
        return

    # Load Prompts
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)

    # Define Perturbations to test
    perturbations = [
        ("none", 0.0),
        ("layer_dropout", 0.25),   # Remove 25% of LoRA weights
        ("weight_decay", 0.1),     # Weaken by 10%
        ("noise", 0.01),           # Add small noise
        ("adversarial", 0.05)      # Targeted middle-layer stressors
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for p_type, p_val in perturbations:
        run_name = f"SVAR_{p_type}_{p_val}"
        print(f"--- Running {run_name} ---")
        
        clear_gpu_cache()
        print_gpu_memory()
        
        # Load Base for reference (for KL Div)
        base_model, tokenizer = load_base_model(model_id)
        
        # Load Perturbed Model
        model = PeftModel.from_pretrained(copy.deepcopy(base_model), RL_ADAPTER_PATH)
        if p_type != "none":
            perturb_adapter(model, p_type, p_val)
        
        model.eval()
        base_model.eval()
        
        for p in prompts:
            pid = p['id']
            text = p['text']
            
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                # Get logits for KL Div
                base_outputs = base_model(**inputs)
                model_outputs = model(**inputs)
                
                kl_div = calculate_kl_divergence(base_outputs.logits, model_outputs.logits)
                
                # Generate text
                gen_out = model.generate(**inputs, max_new_tokens=50)
            
            generated_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            log_results(RESULTS_FILE, run_name, pid, generated_text, None, 0.0, kl_div=kl_div)
            
        # Cleanup
        del base_model
        del model
        clear_gpu_cache()

if __name__ == "__main__":
    run_svar()
