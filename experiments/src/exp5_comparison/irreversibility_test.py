import os
import sys
import json
import torch
import copy
import random
import numpy as np
import argparse
import re
from transformers import AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, print_gpu_memory, cuda_oom_protect
from utils.metrics import log_results, calculate_kl_divergence, calculate_js_divergence, get_latest_sprint_path

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

# -----------------------------------------------------------------------------
# AUTO QUANTIZATION LOGIC
# -----------------------------------------------------------------------------

def should_force_quantization(model_id: str) -> bool:
    model_id = model_id.lower()
    match = re.search(r"(\d+(\.\d+)?)b", model_id)
    if match:
        size = float(match.group(1))
        if size >= 7.0:
            return True
    return False

# -----------------------------------------------------------------------------

def get_model_scale(model_id: str) -> str:
    model_id = model_id.lower()

    if "1.5b" in model_id:
        return "small"
    elif "3b" in model_id:
        return "medium"
    elif "7b" in model_id:
        return "large"
    else:
        raise ValueError(
            f"Unsupported model scale in model_id: {model_id}. "
            "Expected 1.5B, 3B, or 7B."
        )

def resolve_adapter_path(model_id: str) -> str:
    scale = get_model_scale(model_id)
    base_dir = os.path.join(
        os.path.dirname(__file__),
        "../../models/m4"
    )
    adapter_path = os.path.join(base_dir, scale, "lora_rl")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter path not found for scale '{scale}': {adapter_path}"
        )

    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        raise RuntimeError(
            f"'adapter_config.json' not found in {adapter_path}. "
            "This is not a valid LoRA adapter directory."
        )

    return adapter_path

# -----------------------------------------------------------------------------

def simulate_weight_mutation(model, intensity=0.01):
    print(f"--- [MEASUREMENT]: Initiating Unstructured Weight Mutation Analysis (Intensity={intensity}) ---")
    set_seed(1337)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "layer" in name:
                noise = torch.randn_like(param) * intensity
                param.add_(noise)

def execute_structured_fine_tuning(model, tokenizer, training_data_subset, num_steps=10):
    print(f"--- [MEASUREMENT]: Initiating Structured Gradient-Based Mutation Analysis (Steps={num_steps}) ---")
    device = next(model.parameters()).device

    for name, param in model.named_parameters():
        if any(x in name for x in ["layers.24", "layers.25", "layers.26", "layers.27"]):
            param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    print("--- [TRAINING ENGINE]: Optimizing weights for task adaptation...")
    for step in range(num_steps):
        example = training_data_subset[step % len(training_data_subset)]
        inputs = tokenizer(
            example['instruction'] + " " + example['response'],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"    Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    print("--- [TRAINING ENGINE]: Mutation complete. Weights have been structurally modified.")

def force_cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# -----------------------------------------------------------------------------

@cuda_oom_protect
def run_comparison_demo(model_id=DEFAULT_MODEL_ID, is_control=False):

    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    force_quant = should_force_quantization(model_id)
    is_large_model = force_quant

    # ============================================================
    # STEP 0: PRECOMPUTE REFERENCE LOGITS
    # ============================================================

    print("Precomputing reference logits...")
    clear_gpu_cache()
    force_cuda_cleanup()

    # 🔥 LOAD ONCE FOR LARGE MODELS
    if is_large_model:
        base_model_sec3, _ = load_base_model(
            model_id,
            use_quantization=True
        )
        ref_model = base_model_sec3
    else:
        ref_model, _ = load_base_model(
            model_id,
            use_quantization=force_quant
        )

    ref_model.eval()

    reference_logits = []

    for p in prompts:
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = ref_model(**inputs).logits.detach().cpu()
            reference_logits.append(logits)

    # Only delete for small/medium
    if not is_large_model:
        del ref_model
        clear_gpu_cache()
        force_cuda_cleanup()

    # ============================================================
    # SECTION 1 & 2 (skip for large)
    # ============================================================

    if not is_control and not is_large_model:

        print("\n" + "="*60)
        print(" SECTION 1: UNSTRUCTURED WEIGHT MUTATION ")
        print("="*60)

        base_model_sec1, _ = load_base_model(
            model_id,
            use_quantization=force_quant
        )

        simulate_weight_mutation(base_model_sec1, intensity=0.01)
        base_model_sec1.eval()

        peak_kl_sec1 = 0.0

        for idx, p in enumerate(prompts):
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            with torch.no_grad():
                test_logits = base_model_sec1(**inputs).logits
                peak_kl_sec1 += calculate_kl_divergence(
                    reference_logits[idx].to(device),
                    test_logits
                )

        peak_kl_sec1 /= len(prompts)
        print(f"SEC1 Peak KL: {peak_kl_sec1:.4f}")

        del base_model_sec1
        clear_gpu_cache()
        force_cuda_cleanup()

        print("\n" + "="*60)
        print(" SECTION 2: STRUCTURED WEIGHT MUTATION ")
        print("="*60)

        train_data_file = os.path.join(
            os.path.dirname(__file__),
            '../../data/training_data.json'
        )

        with open(train_data_file, 'r') as f:
            train_subset = json.load(f)[:10]

        base_model_sec2, _ = load_base_model(
            model_id,
            use_quantization=force_quant
        )

        execute_structured_fine_tuning(base_model_sec2, tokenizer, train_subset)
        base_model_sec2.eval()

        peak_kl_sec2 = 0.0

        for idx, p in enumerate(prompts):
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            with torch.no_grad():
                test_logits = base_model_sec2(**inputs).logits
                peak_kl_sec2 += calculate_kl_divergence(
                    reference_logits[idx].to(device),
                    test_logits
                )

        peak_kl_sec2 /= len(prompts)
        print(f"SEC2 Peak KL: {peak_kl_sec2:.4f}")

        del base_model_sec2
        clear_gpu_cache()
        force_cuda_cleanup()

    elif is_large_model:
        print("\nLarge model detected (7B+). Skipping SEC1 and SEC2 to prevent CUDA OOM.")

    # ============================================================
    # SECTION 3
    # ============================================================

    print("\n" + "="*60)
    print(" SECTION 3: RLAE ")
    print("="*60)

    # 🔥 Reuse model if large
    if not is_large_model:
        base_model_sec3, _ = load_base_model(model_id, use_quantization=force_quant)

    if is_control:
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model_rlae = get_peft_model(base_model_sec3, config)
    else:
        adapter_path = resolve_adapter_path(model_id)
        model_rlae = PeftModel.from_pretrained(base_model_sec3, adapter_path)

    model_rlae.eval()

    peak_kl_sec3 = 0.0
    peak_js_sec3 = 0.0

    for idx, p in enumerate(prompts):
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            out_test = model_rlae(**inputs).logits
            peak_kl_sec3 += calculate_kl_divergence(reference_logits[idx].to(device), out_test)
            peak_js_sec3 += calculate_js_divergence(reference_logits[idx].to(device), out_test)

    peak_kl_sec3 /= len(prompts)
    peak_js_sec3 /= len(prompts)

    print(f"SEC3 Peak KL: {peak_kl_sec3:.4f}")
    print(f"SEC3 Peak JS: {peak_js_sec3:.4f}")

    if hasattr(model_rlae, "unload"):
        base_model_restored = model_rlae.unload()
    else:
        base_model_restored = base_model_sec3

    base_model_restored.eval()

    post_kl_sec3 = 0.0
    post_js_sec3 = 0.0

    for idx, p in enumerate(prompts):
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            out_restored = base_model_restored(**inputs).logits
            post_kl_sec3 += calculate_kl_divergence(reference_logits[idx].to(device), out_restored)
            post_js_sec3 += calculate_js_divergence(reference_logits[idx].to(device), out_restored)

    post_kl_sec3 /= len(prompts)
    post_js_sec3 /= len(prompts)

    print(f"Post-reset KL: {post_kl_sec3:.4f}")
    print(f"Post-reset JS: {post_js_sec3:.4f}")

    del base_model_sec3
    clear_gpu_cache()
    force_cuda_cleanup()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLAE vs Weight Mutation Comparison Demo")
    parser.add_argument("--control", action="store_true", help="Run M2 No-Op Control procedure")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Model ID to evaluate")
    args = parser.parse_args()

    run_comparison_demo(model_id=args.model_id, is_control=args.control)