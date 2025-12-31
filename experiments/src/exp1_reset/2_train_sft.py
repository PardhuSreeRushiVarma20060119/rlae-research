import os
import sys
import json
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import pandas as pd

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
    dataset = load_dataset('pandas', data_files={'train': DATA_FILE}) # Simplified loading
    # Actually simpler to just use Dataset.from_pandas if we already loaded it
    from datasets import Dataset
    dataset = Dataset.from_pandas(df)
    
    # 2. Load Model & Attach LoRA
    model, tokenizer = load_base_model(model_id)
    model = attach_lora_config(model)
    
    # 3. Train
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3, # Minimal for demo
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no", # Save manually at end
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        fp16=False, # Use bf16 if possible
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    trainer.train()
    
    # 4. Save Adapter
    print(f"Saving SFT adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    
    print("=== SFT TRAINING COMPLETE ===")

if __name__ == "__main__":
    run_sft()
