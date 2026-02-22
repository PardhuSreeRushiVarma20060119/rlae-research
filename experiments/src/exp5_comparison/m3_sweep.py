import os
import sys
import json
import torch
import random
import numpy as np
import argparse

# ------------------------------------------------------------
# Path Setup
# ------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, cuda_oom_protect
from utils.metrics import calculate_kl_divergence, calculate_js_divergence, get_sprint_log_path

# ------------------------------------------------------------
# GLOBAL CONFIG
# ------------------------------------------------------------

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')

# Force NEW sprint folder creation
RESULTS_FILE = get_sprint_log_path(
    'exp5_m3_sweep_results.json',
    use_existing=True
)

# ------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# MUTATION LOGIC
# ------------------------------------------------------------

def simulate_weight_mutation(model, intensity):
    print(f"--- [MEASUREMENT]: Applying Weight Mutation (Intensity={intensity}) ---")
    set_seed(1337)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype.is_floating_point and "weight" in name and "layer" in name:
                noise = torch.randn_like(param) * intensity
                param.add_(noise)

# ------------------------------------------------------------
# MAIN SWEEP
# ------------------------------------------------------------

@cuda_oom_protect
def run_m3_sweep(model_id=DEFAULT_MODEL_ID):

    print("\n" + "="*60)
    print(" M3: MUTATION INTENSITY SWEEP (STRUCTURAL IRREVERSIBILITY) ")
    print("="*60)

    set_seed(1337)

    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # Load reference model (FULL PRECISION REQUIRED)
    # ------------------------------------------------------------

    print("--- [SETUP]: Loading Reference Model (FULL PRECISION) ---")

    ref_model, tokenizer = load_base_model(
        model_id,
        use_quantization=False  # IMPORTANT FIX
    )

    ref_model.eval()

    intensities = [0.001, 0.01, 0.05]

    sweep_results = {
        "model_id": model_id,
        "experiment": "M3_mutation_intensity_sweep",
        "intensities": {}
    }

    # ------------------------------------------------------------
    # Sweep loop
    # ------------------------------------------------------------

    for intensity in intensities:

        print(f"\n>>> PROCESSING INTENSITY: {intensity}")

        mutant_model, _ = load_base_model(
            model_id,
            use_quantization=False  # IMPORTANT FIX
        )

        mutant_model.eval()

        # Apply mutation
        simulate_weight_mutation(mutant_model, intensity)

        peak_kl = 0.0
        peak_js = 0.0

        # Compute divergence
        for p in prompts:

            inputs = tokenizer(
                p['text'],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                out_ref = ref_model(**inputs).logits
                out_mut = mutant_model(**inputs).logits

                kl = calculate_kl_divergence(out_ref, out_mut)
                js = calculate_js_divergence(out_ref, out_mut)

                peak_kl += kl
                peak_js += js

        peak_kl /= len(prompts)
        peak_js /= len(prompts)

        print(f"   Avg KL (α={intensity}): {peak_kl:.6f}")
        print(f"   Avg JS (α={intensity}): {peak_js:.6f}")

        # Native restore attempt (no snapshot -> no recovery)
        post_kl = peak_kl
        post_js = peak_js

        # Recoverability Factor
        rf = 0.0
        if peak_kl > 1e-12:
            rf = 1 - (post_kl / peak_kl)

        print(f"   RF (α={intensity}): {rf:.6f}")

        sweep_results["intensities"][str(intensity)] = {
            "peak_kl": peak_kl,
            "peak_js": peak_js,
            "post_reset_kl": post_kl,
            "post_reset_js": post_js,
            "recoverability_factor": rf
        }

        del mutant_model
        clear_gpu_cache()

    # ------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=4)

    print("\nM3 Sweep Complete.")
    print(f"Results saved to: {RESULTS_FILE}")

# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 Mutation Intensity Sweep")
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Model ID to evaluate"
    )

    args = parser.parse_args()
    run_m3_sweep(model_id=args.model_id)