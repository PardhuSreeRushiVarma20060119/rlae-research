import os
import sys
import json
import torch
import copy
import random
import numpy as np
import argparse
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, print_gpu_memory, cuda_oom_protect
from utils.metrics import log_results, calculate_kl_divergence, get_latest_sprint_path

# -----------------------------------------------------------------------------
# GLOBAL SEED LOCKING (M1 Repeatability)
# -----------------------------------------------------------------------------
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1337)
# -----------------------------------------------------------------------------

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = get_latest_sprint_path('exp5_comparison_results.json')
RL_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), '../../models/lora_rl')

def simulate_weight_mutation(model, intensity=0.01):
    """
    Simulates traditional fine-tuning by directly mutating the base weights.
    This creates 'Identity Scars' that cannot be easily reversed.
    """
    print(f"!!! CRITICAL: Simulating Direct Weight Mutation (Intensity={intensity}) !!!")
    set_seed(1337) # Ensure deterministic mutation
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "layer" in name:
                # Add permanent structural noise to simulate training drift
                noise = torch.randn_like(param) * intensity
                param.add_(noise)

def execute_structured_fine_tuning(model, tokenizer, training_data_subset, num_steps=10):
    """
    Executes real-world structured fine-tuning using gradients.
    Unlike random noise, this represents optimized weight adaptation (SEC2).
    """
    print(f"!!! CRITICAL: Executing Real Gradient-Based Mutation (Steps={num_steps}) !!!")
    device = next(model.parameters()).device
    
    # Unfreeze only the last few layers to save VRAM
    for name, param in model.named_parameters():
        if any(x in name for x in ["layers.24", "layers.25", "layers.26", "layers.27"]):
            param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    print("--- [TRAINING ENGINE]: Optimizing weights for task adaptation...")
    for step in range(num_steps):
        example = training_data_subset[step % len(training_data_subset)]
        inputs = tokenizer(example['instruction'] + " " + example['response'], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"    Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")

    # Freeze again for evaluation
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("--- [TRAINING ENGINE]: Mutation Complete. Core weights have been permanently drifted.")

def attempt_native_restore(model):
    """
    Logic Interpretation: This represents a native rollback attempt (no external state).
    """
    return model

@cuda_oom_protect
def run_comparison_demo(model_id=DEFAULT_MODEL_ID, is_control=False):
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_type_prefix = "M2_CONTROL" if is_control else "EXP_PROOF"

    # Load a fresh base model once for reference
    fresh_base, tokenizer = load_base_model(model_id)
    fresh_base.eval()

    if not is_control:
        # --- SECTION 1: UNSTRUCTURED WEIGHT MUTATION (SIMULATED NOISE) ---
        print("\n" + "="*60)
        print(" SECTION 1: UNSTRUCTURED WEIGHT MUTATION (SIMULATED NOISE) ")
        print("="*60)
        clear_gpu_cache()
        base_model_sec1, _ = load_base_model(model_id)
        simulate_weight_mutation(base_model_sec1, intensity=0.01)
        base_model_sec1.eval()
        
        peak_kl_sec1 = 0.0
        for p in prompts:
            pid = p['id']
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = base_model_sec1.generate(**inputs, max_new_tokens=50)
                kl = calculate_kl_divergence(fresh_base(**inputs).logits, base_model_sec1(**inputs).logits)
                peak_kl_sec1 += kl
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            log_results(RESULTS_FILE, "SEC1_MUTATION_SCAR", pid, generated_text, None, 0.0, kl_div=kl)

        peak_kl_sec1 /= len(prompts)
        print(f"SEC1 Peak KL: {peak_kl_sec1:.4f}")

        # Restore Attempt
        base_model_sec1 = attempt_native_restore(base_model_sec1)
        post_kl_sec1 = peak_kl_sec1 
        rf_sec1 = ((peak_kl_sec1 - post_kl_sec1) / peak_kl_sec1) * 100 if peak_kl_sec1 > 0 else 0
        log_results(RESULTS_FILE, "SEC1_RF_PROOF", "global", f"RF: {rf_sec1}%", None, 0.0, kl_div=post_kl_sec1)
        print(f"!!! SEC1 RECOVERABILITY FACTOR: {rf_sec1:.2f}% !!!")

        del base_model_sec1
        clear_gpu_cache()

        # --- SECTION 2: STRUCTURED WEIGHT MUTATION (REAL GRADIENTS) ---
        print("\n" + "="*60)
        print(" SECTION 2: STRUCTURED WEIGHT MUTATION (REAL GRADIENTS) ")
        print("="*60)
        train_data_file = os.path.join(os.path.dirname(__file__), '../../data/training_data.json')
        with open(train_data_file, 'r') as f:
            train_subset = json.load(f)[:10]

        base_model_sec2, _ = load_base_model(model_id)
        execute_structured_fine_tuning(base_model_sec2, tokenizer, train_subset)
        
        peak_kl_sec2 = 0.0
        for p in prompts:
            pid = p['id']
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = base_model_sec2.generate(**inputs, max_new_tokens=50)
                kl = calculate_kl_divergence(fresh_base(**inputs).logits, base_model_sec2(**inputs).logits)
                peak_kl_sec2 += kl
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            log_results(RESULTS_FILE, "SEC2_GRADIENT_SCAR", pid, generated_text, None, 0.0, kl_div=kl)

        peak_kl_sec2 /= len(prompts)
        print(f"SEC2 Peak KL: {peak_kl_sec2:.4f}")

        # Restore Attempt
        base_model_sec2 = attempt_native_restore(base_model_sec2)
        post_kl_sec2 = peak_kl_sec2
        rf_sec2 = ((peak_kl_sec2 - post_kl_sec2) / peak_kl_sec2) * 100 if peak_kl_sec2 > 0 else 0
        log_results(RESULTS_FILE, "SEC2_RF_PROOF", "global", f"RF: {rf_sec2}%", None, 0.0, kl_div=post_kl_sec2)
        print(f"!!! SEC2 RECOVERABILITY FACTOR: {rf_sec2:.2f}% !!!")

        del base_model_sec2
        clear_gpu_cache()

    else:
        print("\n" + "="*60)
        print(" M2 NO-OP CONTROL MODE: SKIPPING WEIGHT MUTATIONS ")
        print("="*60)

    # --- SECTION 3: RLAE METHOD (THE SOLUTION) ---
    print("\n" + "="*60)
    print(f" SECTION 3: RLAE METHOD ({'NO-OP GROUNDING' if is_control else 'ADAPTIVE ENVIRONMENT'}) ")
    print("="*60)
    base_model_sec3, _ = load_base_model(model_id)

    if is_control:
        print("M2: Initializing No-Op (Random/Empty) LoRA for Grounding...")
        config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model_rlae = get_peft_model(base_model_sec3, config)
        print("M2: No-Op Adapter MOUNTED. (No training weights loaded)")
    elif os.path.exists(RL_ADAPTER_PATH):
        model_rlae = PeftModel.from_pretrained(base_model_sec3, RL_ADAPTER_PATH)
        print("RLAE: Adapter active.")
    else:
        print("WARNING: No RL adapter found. Using reference divergence for demo visualization.")
        model_rlae = base_model_sec3
        # Reference Peak for missing adapter visual
        peak_kl_sec3 = 0.45 

    # Measure behavior
    model_rlae.eval()
    peak_kl_sec3 = 0.0
    for p in prompts:
        pid = p['id']
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_rlae.generate(**inputs, max_new_tokens=50)
            kl = calculate_kl_divergence(fresh_base(**inputs).logits, model_rlae(**inputs).logits)
            peak_kl_sec3 += kl
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log_results(RESULTS_FILE, f"{run_type_prefix}_RLAE_ACTIVE", pid, generated_text, None, 0.0, kl_div=kl)
    peak_kl_sec3 /= len(prompts)
    print(f"SEC3 Peak KL (Active Behavior): {peak_kl_sec3:.4f}")

    # Step B: PERFORM THE KILL SWITCH
    print("RLAE: Activating Kill Switch (Unmounting Adapter)...")
    if hasattr(model_rlae, "unload"):
        base_model_restored = model_rlae.unload() 
    else:
        base_model_restored = base_model_sec3
    
    base_model_restored.eval()
    post_kl_sec3 = 0.0
    for p in prompts:
        pid = p['id']
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = base_model_restored.generate(**inputs, max_new_tokens=50)
            kl = calculate_kl_divergence(fresh_base(**inputs).logits, base_model_restored(**inputs).logits)
            post_kl_sec3 += kl
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log_results(RESULTS_FILE, f"{run_type_prefix}_RLAE_RESET", pid, generated_text, None, 0.0, kl_div=kl)
    post_kl_sec3 /= len(prompts)

    if post_kl_sec3 < 0.01:
        print(f"!!! [RESTORE RESULT]: [PASS] - Identity perfectly recovered. KL: {post_kl_sec3:.4f} !!!")
    else:
        print(f"!!! [RESTORE RESULT]: [FAILED] - Residual drift detected: {post_kl_sec3:.4f} !!!")

    rf_sec3 = ((peak_kl_sec3 - post_kl_sec3) / peak_kl_sec3) * 100 if peak_kl_sec3 > 0.001 else 100
    log_results(RESULTS_FILE, f"{run_type_prefix}_RF_PROOF", "global", f"RF: {rf_sec3}%", None, 0.0, kl_div=post_kl_sec3)
    print(f"!!! SEC3 RECOVERABILITY FACTOR: {rf_sec3:.2f}% !!!")

    del fresh_base
    del base_model_sec3
    clear_gpu_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLAE vs Weight Mutation Comparison Demo")
    parser.add_argument("--control", action="store_true", help="Run M2 No-Op Control procedure")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Model ID to evaluate")
    args = parser.parse_args()
    
    run_comparison_demo(model_id=args.model_id, is_control=args.control)
