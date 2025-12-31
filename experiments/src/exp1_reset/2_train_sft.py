import os
import sys
import json
import torch
import pandas as pd
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, attach_lora_config, DEFAULT_MODEL_ID, cuda_oom_protect
from utils.metrics import log_results

DATA_FILE = os.path.join(os.path.dirname(__file__), '../../data/training_data.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../models/lora_sft')

def format_instruction(sample):
    return f"Instruction: {sample['instruction']}\nResponse: {sample['response']}"

@cuda_oom_protect
def run_sft(model_id=DEFAULT_MODEL_ID):
    print("=== STARTING EXPERIMENT 1.C: LoRA SFT TRAINING ===")
    
    # 1. Load Data
    # Convert JSON to dataset
    df = pd.read_json(DATA_FILE)
    df['text'] = df.apply(format_instruction, axis=1)
    
    from datasets import Dataset
    dataset = Dataset.from_pandas(df)
    
    # 2. Load Model & Attach LoRA
    model, tokenizer = load_base_model(model_id)
    model = attach_lora_config(model)
    
    # 3. Train
    # In newer TRL, max_seq_length is usually part of SFTConfig or inferred.
    # If it was rejected by SFTConfig AND SFTTrainer, we will omit it to use defaults.
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        num_train_epochs=3, # Minimal for demo
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no", # Save manually at end
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        fp16=False, # Use bf16 if possible
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none", # Disable interactive W&B prompts
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )
    
    trainer.train()
    
    # 4. Save Adapter
    print(f"Saving SFT adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    
    print("=== SFT TRAINING COMPLETE ===")

if __name__ == "__main__":
    run_sft()
