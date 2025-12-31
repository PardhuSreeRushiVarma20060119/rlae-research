import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import os

# Default to a small model if not specified
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_base_model(model_id=DEFAULT_MODEL_ID):
    """
    Loads the base model in 4-bit or 16-bit to save memory, strictly frozen.
    """
    print(f"Loading Base Model: {model_id}")
    
    # Use bfloat16 if available, else float32
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # STRICT FREEZE
    for param in model.parameters():
        param.requires_grad = False
    
    print("Base model loaded and FROZEN.")
    return model, tokenizer

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("GPU cache cleared.")

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

def attach_lora_config(model, r=8, alpha=32, dropout=0.05):
    """
    Attaches a fresh LoRA config for initialization (SFT start).
    """
    print("Attaching NEW LoRA adapters...")
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"] # Common targets, adjust for specific architectures if needed
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def load_lora_model(base_model_id, lora_path):
    """
    Loads base model + existing LoRA adapter.
    """
    print(f"Loading Base: {base_model_id} + LoRA: {lora_path}")
    
    # Load base first
    model, tokenizer = load_base_model(base_model_id)
    
    # Load adapter
    model = PeftModel.from_pretrained(model, lora_path)
    
    # Ensure it's still frozen just in case, though inference usually is
    for param in model.parameters():
        param.requires_grad = False
        
    print("LoRA loaded successfully.")
    return model, tokenizer

def save_adapter(model, output_dir):
    """
    Saves only the adapter.
    """
    print(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
