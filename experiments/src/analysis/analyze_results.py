import json
import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../logs/exp1_results.json')

def load_data(filepath):
    data = {}
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return data
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            run_id = rec['run_id']
            prompt_id = rec['prompt_id']
            
            if run_id not in data:
                data[run_id] = {}
            data[run_id][prompt_id] = rec
    return data

def compare_runs(data, run_a, run_b):
    print(f"\n--- Comparing {run_a} vs {run_b} ---")
    if run_a not in data or run_b not in data:
        print("One or both run IDs not found.")
        return

    sims = []
    entropy_diffs = []
    
    prompts = data[run_a].keys()
    
    for pid in prompts:
        if pid not in data[run_b]:
            continue
            
        rec_a = data[run_a][pid]
        rec_b = data[run_b][pid]
        
        # Cosine Similarity
        if rec_a['embedding'] and rec_b['embedding']:
            v1 = np.array(rec_a['embedding']).reshape(1, -1)
            v2 = np.array(rec_b['embedding']).reshape(1, -1)
            sim = cosine_similarity(v1, v2)[0][0]
            sims.append(sim)
        
        # Entropy Drift
        e_diff = rec_b['entropy'] - rec_a['entropy']
        entropy_diffs.append(e_diff)

    if len(sims) > 0:
        print(f"Avg Cosine Similarity: {np.mean(sims):.4f}")
        print(f"Avg Entropy Drift:     {np.mean(entropy_diffs):.4f}")
        
        if np.mean(sims) < 0.999: # Strict threshold for identical state
            print("WARNING: DETECTED POTENTIAL STATE DRIFT")
        else:
            print("SUCCESS: States appear effectively identical.")
    else:
        print("No matching prompts with embeddings found.")

if __name__ == "__main__":
    data = load_data(RESULTS_FILE)
    print("Available Runs:", list(data.keys()))
    
    # Common Comparisons
    compare_runs(data, "BASELINE", "POST-RESET")
    compare_runs(data, "BASELINE", "LoRA-SFT") # Should be different
