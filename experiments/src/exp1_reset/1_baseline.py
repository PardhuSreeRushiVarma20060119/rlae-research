import os
import sys
import json
import torch
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model import load_base_model, DEFAULT_MODEL_ID, clear_gpu_cache, print_gpu_memory, cuda_oom_protect
from utils.metrics import calculate_token_entropy, log_results

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../logs/exp1_results.json')

@cuda_oom_protect
def run_baseline(model_id=DEFAULT_MODEL_ID):
    print("=== STARTING EXPERIMENT 1.B: BASELINE RUN (Hardened) ===")
    
    # 1. Load Prompts
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)
    
    clear_gpu_cache()
    print_gpu_memory()
    
    # 2. Load Base Model
    model, tokenizer = load_base_model(model_id)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. Inference Loop
    for p in prompts:
        pid = p['id']
        text = p['text']
        print(f"Processing {pid}...")
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # We need logits for entropy
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                output_scores=True, 
                return_dict_in_generate=True
            )
        
        # Decode text
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Calculate Entropy (from scores/logits)
        # outputs.scores is a tuple of len(generated_tokens), each tensor (batch, vocab)
        if outputs.scores:
            stacked_scores = torch.stack(outputs.scores, dim=1) # (batch, seq, vocab)
            entropy_score = calculate_token_entropy(stacked_scores)
        else:
            entropy_score = 0.0
            
        # Extract last hidden state as "embedding" substitute (avg pool of last layer)
        # To get actual embeddings we'd need to run a forward pass with output_hidden_states=True on the generated sequence
        # For this script, we'll do a quick forward pass on the RESULT to get the embedding
        with torch.no_grad():
            final_out = model(outputs.sequences, output_hidden_states=True)
            # Use last layer hidden state, average over sequence
            last_hidden = final_out.hidden_states[-1] # (batch, seq, hidden)
            embedding = last_hidden.mean(dim=1).float().cpu().numpy().tolist()[0]
            
        # 4. Log
        log_results(RESULTS_FILE, "BASELINE", pid, generated_text, embedding, entropy_score)
        
    print("=== BASELINE RUN COMPLETE ===")

if __name__ == "__main__":
    run_baseline()
