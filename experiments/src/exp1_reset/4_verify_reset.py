import os
import sys
import json
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_base_model, DEFAULT_MODEL_ID
from utils.metrics import calculate_token_entropy, log_results

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '../../data/fixed_prompts.json')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../logs/exp1_results.json')

def run_post_reset(model_id=DEFAULT_MODEL_ID):
    print("=== STARTING EXPERIMENT 1.E: POST-RESET CHECK ===")
    
    # Check if we are truly clean
    # (In code we can't easily check for process restart, but we assume the user followed instructions)
    
    # 1. Load Prompts
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)
        
    # 2. Load Base Model (NO ADAPTERS)
    model, tokenizer = load_base_model(model_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. Eval Loop
    for p in prompts:
        pid = p['id']
        text = p['text']
        print(f"Processing {pid}...")
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                output_scores=True, 
                return_dict_in_generate=True
            )
            
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        if outputs.scores:
            stacked_scores = torch.stack(outputs.scores, dim=1) 
            entropy_score = calculate_token_entropy(stacked_scores)
        else:
            entropy_score = 0.0
            
        with torch.no_grad():
            final_out = model(outputs.sequences, output_hidden_states=True)
            last_hidden = final_out.hidden_states[-1] 
            embedding = last_hidden.mean(dim=1).cpu().numpy().tolist()[0]
            
        # 4. Log as POST-RESET
        log_results(RESULTS_FILE, "POST-RESET", pid, generated_text, embedding, entropy_score)
        
    print("=== POST-RESET CHECK COMPLETE ===")

if __name__ == "__main__":
    run_post_reset()
