import os
import sys
import torch
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID, cuda_oom_protect

# We assume SFT model exists
SFT_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), '../../models/lora_sft')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../models/lora_rl')

# Dummy preference data to simulate RL alignment towards "Structured" responses
# In a real experiment, you'd generate these from the SFT model
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

@cuda_oom_protect
def run_rl(model_id=DEFAULT_MODEL_ID):
    print("=== STARTING EXPERIMENT 1.D: LoRA RL (DPO) TRAINING ===")
    
    if not os.path.exists(SFT_ADAPTER_PATH):
        print(f"Error: SFT Adapter not found at {SFT_ADAPTER_PATH}. Run step 2 first.")
        return

    # 1. Load Data
    dataset = Dataset.from_list(PREFERENCE_DATA)

    # 2. Load Model (Base + SFT Adapter)
    # DPO requires a model with the adapter already attached
    model, tokenizer = load_base_model(model_id)
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH, is_trainable=True)
    
    # 3. Train (DPO)
    # Modern TRL (0.12+) expects length parameters in DPOConfig
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5, # Lower LR for RL
        logging_steps=1,
        beta=0.1,
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        max_length=512,
        max_prompt_length=128,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # TRL handles reference internally for PeftModel
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    
    # 4. Save RL Adapter
    print(f"Saving RL adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR) # TRL save_model saves adapter for PEFT
    
    print("=== RL TRAINING COMPLETE ===")

if __name__ == "__main__":
    run_rl()
