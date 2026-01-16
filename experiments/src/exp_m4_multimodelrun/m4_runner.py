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
    
    print("Computing Base Logits & Text...")
    base_stats = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for p in prompts:
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            full_out = base_model(**inputs)
            # Deterministic generation for RF comparison (suppress sampling warnings)
            gen_out = base_model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id
            )
            gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            
            base_stats.append({
                "logits": full_out.logits.cpu(),
                "text": gen_text
            })

    del base_model
    clear_gpu_cache()
    
    print("Computing Post-Reset Logits & Text...")
    reset_model, _ = load_base_model(model_id)
    reset_model = PeftModel.from_pretrained(reset_model, rl_path)
    reset_model.eval()
    
    kl_values = []
    matches = 0
    details = []
    
    for i, p in enumerate(prompts):
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            reset_out = reset_model(**inputs)
            
            # Measure KL
            base_logits = base_stats[i]["logits"].to(device)
            reset_logits = reset_out.logits
            kl = calculate_kl_divergence(base_logits, reset_logits)
            kl_values.append(kl)
            
            # Generate for RF
            gen = reset_model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id
            )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            
            # Exact match check
            base_text = base_stats[i]["text"]
            is_match = (text.strip() == base_text.strip())
            if is_match:
                matches += 1
            
            details.append({
                "prompt": p.get('text', ''),
                "base_text": base_text,
                "generated_text": text,
                "kl_divergence": float(kl),
                "is_recovered": is_match
            })
            
    avg_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
    rf_score = (matches / len(prompts)) * 100 if prompts else 0.0
    
    print(f"Weight Mutation Results: KL={avg_kl:.4f}, RF={rf_score:.2f}%")
    return {"kl": avg_kl, "rf": rf_score, "path": "Weight Mutation", "details": details}

@cuda_oom_protect
def verify_behavioral_adapter(model_id, sft_path, prompts):
    print(f"--- Verifying Behavioral Adapter Path (SFT -> Unload) ---")
    
    base_model, tokenizer = load_base_model(model_id)
    base_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Computing Base Logits & Text...")
    base_stats = []
    for p in prompts:
        inputs = tokenizer(p['text'], return_tensors="pt").to(device)
        with torch.no_grad():
            out = base_model(**inputs)
            gen_out = base_model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id
            )
            gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            
            base_stats.append({
                "logits": out.logits.cpu(),
                "text": gen_text
            })
            
    # Now Load Adapter
    print("Loading Adapter...")
    model = PeftModel.from_pretrained(base_model, sft_path)
    model.eval()
    
    # Now UNLOAD / DISABLE
    print("Unloading/Disabling Adapter...")
    
    kl_values = []
    matches = 0
    details = []
    
    with model.disable_adapter():
        for i, p in enumerate(prompts):
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs) # Should be base model forward
                
                base_logits = base_stats[i]["logits"].to(device)
                kl = calculate_kl_divergence(base_logits, out.logits)
                kl_values.append(kl)
                
                # Verification generation
                gen = model.generate(
                    **inputs, 
                    max_new_tokens=50, 
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id
                )
                text = tokenizer.decode(gen[0], skip_special_tokens=True)
                
                base_text = base_stats[i]["text"]
                is_match = (text.strip() == base_text.strip())
                if is_match:
                    matches += 1

                details.append({
                    "prompt": p.get('text', ''),
                    "base_text": base_text,
                    "generated_text": text,
                    "kl_divergence": float(kl),
                    "is_recovered": is_match
                })
                
    avg_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
    rf_score = (matches / len(prompts)) * 100 if prompts else 0.0
    
    print(f"Behavioral Adapter Results: KL={avg_kl:.6f}, RF={rf_score:.2f}% (Should be ~0 KL, 100% RF)")
    
    return {"kl": avg_kl, "rf": rf_score, "path": "Behavioral Adapter", "details": details}



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
            "rf": wm_results["rf"],
            "details": wm_results["details"]
        },
        "behavioral_adapter": {
            "kl": ba_results["kl"],
            "rf": ba_results["rf"],
            "details": ba_results["details"]
        }
    }
    
    # Dynamic Sprint Logging
    # We use use_existing=True if EXPERIMENT_SPRINT is set, or if we want to try to group them?
    # User requested "sequentially". Default get_sprint_log_path behavior (without env var) 
    # autoincrements on each run if not set.
    # To avoid creating 3 separate sprints for one "M4" batch run, we ideally reuse the latest
    # if it was created recently? But simple sequential is safer per request.
    result_file = get_sprint_log_path(f"m4_results/{size}_results.json")
    
    # Ensure the subdirectory (e.g., m4_results) exists
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    with open(result_file, "w") as f:
        json.dump(results_entry, f, indent=4)

        
    print(f"=== M4 {size.upper()} COMPLETED ===")
    print(json.dumps(results_entry, indent=2))

if __name__ == "__main__":
    main()
