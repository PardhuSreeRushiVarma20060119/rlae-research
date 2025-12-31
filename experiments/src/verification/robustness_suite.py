import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Paths
LOGS_DIR = "logs"
EXP1_LOG = os.path.join(LOGS_DIR, "exp1_results.json")
EXP2_LOG = os.path.join(LOGS_DIR, "exp2_rlae_results.json")
EXP3_LOG = os.path.join(LOGS_DIR, "exp3_svar_results.json")

def load_json_lines(path):
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def run_suite():
    print("=== 🛡️ RLAE & SVAR CANONICAL ROBUSTNESS SUITE 🛡️ ===")
    print("Goal: Validate Frozen Core Invariance & Detect Identity Leakage\n")
    
    # 1. Reset Integrity & Identity Leakage (Exp 1)
    exp1_data = load_json_lines(EXP1_LOG)
    if exp1_data:
        baseline = [d for d in exp1_data if d['run_id'] == "BASELINE"]
        post_reset = [d for d in exp1_data if d['run_id'] == "POST-RESET"]
        
        if baseline and post_reset:
            b_ent = np.mean([d['entropy'] for d in baseline])
            p_ent = np.mean([d['entropy'] for d in post_reset])
            drift = abs(b_ent - p_ent)
            
            # 1.B Identification of Identity Leakage Score (ILS)
            # ILS = (KL_shift * 0.5) + (Ent_shift * 0.3) + (Emb_drift * 0.2)
            # For Exp 1, KL_shift is typically ~0 as we compare base to base, 
            # so ILS focuses on Entropy and Embedding drift.
            
            # We calculate per-prompt ILS and average it
            ils_scores = []
            for p_rec in post_reset:
                b_rec = next((b for b in baseline if b['prompt_id'] == p_rec['prompt_id']), None)
                if b_rec:
                    ent_shift = abs(p_rec['entropy'] - b_rec['entropy'])
                    
                    e1 = np.array(b_rec['embedding']).reshape(1, -1)
                    e2 = np.array(p_rec['embedding']).reshape(1, -1)
                    cos_sim = cosine_similarity(e1, e2)[0][0]
                    emb_drift = 1.0 - cos_sim
                    
                    ils = (ent_shift * 0.3) + (emb_drift * 0.2)
                    ils_scores.append(ils)
            
            avg_ils = np.mean(ils_scores) if ils_scores else 0.0
            
            status = "HEALTHY" if avg_ils < 0.05 else "IDENTITY LEAKAGE DETECTED"
            print(f"[EXP 1] Reset Integrity: {status}")
            print(f"       - Baseline Avg Entropy: {b_ent:.4f}")
            print(f"       - Post-Reset Avg Entropy: {p_ent:.4f}")
            print(f"       - Avg ILS (State Drift): {avg_ils:.4f}")
    
    # 2. Behavioral Elimination (RLAE Core - Exp 2)
    exp2_data = load_json_lines(EXP2_LOG)
    if exp2_data:
        ratios = sorted(list(set([d['run_id'] for d in exp2_data])))
        print(f"\n[EXP 2] Behavioral Elimination Analysis (Kill-switch Validation):")
        for r in ratios:
            kl = np.mean([d['kl_divergence'] for d in exp2_data if d['run_id'] == r])
            print(f"       - Environment Ratio {r}: Avg KL Div = {kl:.4f}")

    # 3. SVAR Diagnostic Surface (Exp 3)
    exp3_data = load_json_lines(EXP3_LOG)
    if exp3_data:
        types = sorted(list(set([d['run_id'] for d in exp3_data])))
        print(f"\n[EXP 3] SVAR Stability Envelope Analysis:")
        for t in types:
            kl = np.mean([d['kl_divergence'] for d in exp3_data if d['run_id'] == t])
            print(f"       - Perturbation {t}: Variation = {kl:.4f}")

    print("\n--- Canonical Diagnostic Report Complete ---")

if __name__ == "__main__":
    run_suite()
