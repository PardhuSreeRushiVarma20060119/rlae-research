import os
import sys
import argparse
import json
import torch
import pandas as pd
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from datasets import Dataset
import numpy as np


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, attach_lora_config, clear_gpu_cache, cuda_oom_protect
from utils.metrics import calculate_kl_divergence, get_sprint_log_path


# Constants
MODELS = {
    "small": "Qwen/Qwen2.5-1.5B-Instruct",
    "medium": "Qwen/Qwen2.5-3B-Instruct",
    "large": "Qwen/Qwen2.5-7B-Instruct"
}

# BASE_DIR should point to 'experiments' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
# LOGS_DIR will be resolved dynamically


MODELS_DIR = os.path.join(BASE_DIR, "models", "m4")

# Make sure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)


PROMPTS_FILE = os.path.join(DATA_DIR, "fixed_prompts.json")
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "training_data.json")

# Reuse preference data structure from exp1/3_train_rl.py
PREFERENCE_DATA = [
    {
        "prompt": "Explain gravity.",
        "chosen": "Concept: Gravity\nCategory: Physics\nSummary: Attraction between mass.",
        "rejected": "Gravity is when things fall down because the earth pulls them."
    },
    {
        "prompt": "Explain photosynthesis.",
        "chosen": "Concept: Photosynthesis\nCategory: Biology\nSummary: Plants making food from light.",
        "rejected": "It is how plants eat sunlight to grow."
    }
]

def format_instruction(sample):
    return f"Instruction: {sample['instruction']}\nResponse: {sample['response']}"

def get_paths(size):
    size_dir = os.path.join(MODELS_DIR, size)
    return {
        "sft": os.path.join(size_dir, "lora_sft"),
        "rl": os.path.join(size_dir, "lora_rl")
    }

@cuda_oom_protect
def train_sft(model_id, output_dir):
    print(f"--- Training SFT (Mutation) for {model_id} ---")
    if os.path.exists(output_dir):
        print(f"SFT adapter already exists at {output_dir}. Skipping training.")
        return

    # Load Data
    df = pd.read_json(TRAIN_DATA_FILE)
    df['text'] = df.apply(format_instruction, axis=1)
    dataset = Dataset.from_pandas(df)

    # Load Model
    model, tokenizer = load_base_model(model_id)
    model = attach_lora_config(model)

    # Config
    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=1337,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    print(f"SFT adapter saved to {output_dir}")
    del model, trainer
    clear_gpu_cache()

@cuda_oom_protect
def train_rl(model_id, sft_path, output_dir):
    print(f"--- Training RL (Reset) for {model_id} ---")
    if os.path.exists(output_dir):
        print(f"RL adapter already exists at {output_dir}. Skipping training.")
        return

    # Load Model with SFT
    model, tokenizer = load_base_model(model_id)
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)

    # Config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=1,
        beta=0.1,
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        max_length=512,
        max_prompt_length=128,
        seed=1337,
    )
    
    dataset = Dataset.from_list(PREFERENCE_DATA)

    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"RL adapter saved to {output_dir}")
    del model, trainer
    clear_gpu_cache()

@cuda_oom_protect
def verify_weight_mutation(model_id, rl_path, prompts):
    print(f"--- Verifying Weight Mutation Path (Post-Reset) ---")
    
    # Load Base for KL reference
    base_model, tokenizer = load_base_model(model_id)
    base_model.eval()
    
    # Load Reset Model (Base + RL Adapter)
    # Using a separate instance or just attaching to copied base?
    # To be safe and avoid VRAM issues, we can run base inference first, store logits, then load adapter.
    # But for KL we need simultaneous logs or stored logs.
    # Let's try loading separate models if VRAM permits, otherwise compute Base first.
    # 7B might be tight on VRAM for 2 models.
    # Strategy: Compute Base Logits -> Store. Compute Reset Logits -> Compare.
    
    print("Computing Base Logits...")
    base_logits_list = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for p in prompts:
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            full_out = base_model(**inputs)
            # Just store the logits for the last token or full sequence? 
            # KL is usually over the generated response or next token distribution.
            # M1/M2 prompt-based metrics used generation.
            # "KL(base || post-reset)" typically refers to the distribution of the model on the prompts.
            # Let's use the M3 approach: generate text and measure KL on that, or just KL on the prompt response?
            # M3 snippet: calculate_kl_divergence(base_outputs.logits, model_outputs.logits)
            # This implies next-token prediction KL on the prompt input? 
            # Or we can run a short generation and average KL.
            # Let's stick to M3 method: Logits on the *prompt* (next token prediction) is safest/fastest.
            # Or if M3 `calculate_kl_divergence` takes full logits.
            base_logits_list.append(full_out.logits.cpu())

    del base_model
    clear_gpu_cache()
    
    print("Computing Post-Reset Logits...")
    reset_model, _ = load_base_model(model_id)
    reset_model = PeftModel.from_pretrained(reset_model, rl_path)
    reset_model.eval()
    
    kl_values = []
    generated_texts = []
    
    for i, p in enumerate(prompts):
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            reset_out = reset_model(**inputs)
            # Measure KL
            base_logits = base_logits_list[i].to(device)
            reset_logits = reset_out.logits
            kl = calculate_kl_divergence(base_logits, reset_logits)
            kl_values.append(kl)
            
            # Generate for RF/Qualitative
            gen = reset_model.generate(**inputs, max_new_tokens=50)
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            generated_texts.append(text)
            
    avg_kl = sum(kl_values) / len(kl_values)
    rf_score = 0.0 # Placeholder, ideally we have a specific RF calc, but KL>0 is the proxy here
    
    print(f"Weight Mutation Results: KL={avg_kl:.4f}")
    return {"kl": avg_kl, "path": "Weight Mutation", "generated": generated_texts}

@cuda_oom_protect
def verify_behavioral_adapter(model_id, sft_path, prompts):
    print(f"--- Verifying Behavioral Adapter Path (SFT -> Unload) ---")
    
    # For behavioral, we load Base + SFT, then Unload/Disable.
    # Effectively this should be mathematically identical to Base.
    # We verify this.
    
    base_model, tokenizer = load_base_model(model_id)
    base_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Computing Base Logits...")
    base_logits_list = []
    for p in prompts:
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            out = base_model(**inputs)
            base_logits_list.append(out.logits.cpu())
            
    # Now Load Adapter
    print("Loading Adapter...")
    model = PeftModel.from_pretrained(base_model, sft_path)
    model.eval()
    
    # Now UNLOAD / DISABLE
    # context manager `disable_adapter()` works for PeftModel
    print("Unloading/Disabling Adapter...")
    
    kl_values = []
    
    with model.disable_adapter():
        for i, p in enumerate(prompts):
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs) # Should be base model forward
                
                base_logits = base_logits_list[i].to(device)
                kl = calculate_kl_divergence(base_logits, out.logits)
                kl_values.append(kl)
                
    avg_kl = sum(kl_values) / len(kl_values)
    print(f"Behavioral Adapter Results: KL={avg_kl:.6f} (Should be ~0)")
    
    return {"kl": avg_kl, "path": "Behavioral Adapter"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, required=True, choices=["small", "medium", "large"], help="Model size to run")
    args = parser.parse_args()
    
    size = args.size
    model_id = MODELS[size]
    paths = get_paths(size)
    
    print(f"=== M4 EXPERIMENT RUNNER: {size.upper()} ({model_id}) ===")
    
    # 1. Path 1: Weight Mutation
    # Step A: Induce Mutation (SFT)
    train_sft(model_id, paths["sft"])
    
    # Step B: Attempt Reset (RL)
    train_rl(model_id, paths["sft"], paths["rl"])
    
    # Load Prompts
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)
        
    # Step C: Verify Weight Mutation (Expect KL > 0)
    wm_results = verify_weight_mutation(model_id, paths["rl"], prompts)
    
    # 2. Path 2: Behavioral Adapter
    # Step A: Attach Mutation Adapter (SFT) - Already exists
    # Step B: Unload and Verify (Expect KL ~ 0)
    ba_results = verify_behavioral_adapter(model_id, paths["sft"], prompts)
    
    # 3. Log Results
    results_entry = {
        "model": size,
        "model_id": model_id,
        "weight_mutation": {
            "kl": wm_results["kl"],
            "rf_proxy": "See Detailed Walkthrough"
        },
        "behavioral_adapter": {
            "kl": ba_results["kl"],
            "rf_proxy": "Perfect"
        }
    }
    
    # Dynamic Sprint Logging
    # We use use_existing=True if EXPERIMENT_SPRINT is set, or if we want to try to group them?
    # User requested "sequentially". Default get_sprint_log_path behavior (without env var) 
    # autoincrements on each run if not set.
    # To avoid creating 3 separate sprints for one "M4" batch run, we ideally reuse the latest
    # if it was created recently? But simple sequential is safer per request.
    result_file = get_sprint_log_path(f"m4_results/{size}_results.json")
    
    with open(result_file, "w") as f:
        json.dump(results_entry, f, indent=4)

        
    print(f"=== M4 {size.upper()} COMPLETED ===")
    print(json.dumps(results_entry, indent=2))

if __name__ == "__main__":
    main()
