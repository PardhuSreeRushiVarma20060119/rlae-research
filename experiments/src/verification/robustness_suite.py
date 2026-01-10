import os
import sys
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.metrics import get_latest_sprint_path

def load_json_lines(path):
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def run_suite():
    print("\n" + "="*60)
    print(" 🛡️  REVA4 CANONICAL ROBUSTNESS & IDENTITY SUITE  🛡️ ")
    print("="*60)
    
    # 1. EXP 1: Reset Integrity & Identity Leakage
    print("\n[EXP 1] RESET INTEGRITY & IDENTITY LEAKAGE (ILS)")
    baseline_path = get_latest_sprint_path('exp1_results.json')
    reset_path = get_latest_sprint_path('exp1_post_reset_results.json')
    
    baseline = load_json_lines(baseline_path)
    post_reset = load_json_lines(reset_path)
    
    if baseline and post_reset:
        ils_scores = []
        for p_rec in post_reset:
            b_rec = next((b for b in baseline if b['prompt_id'] == p_rec['prompt_id']), None)
            if b_rec:
                ent_shift = abs(p_rec['entropy'] - b_rec['entropy'])
                # Ensure embedding exists
                if p_rec.get('embedding') and b_rec.get('embedding'):
                    e1 = np.array(b_rec['embedding']).reshape(1, -1)
                    e2 = np.array(p_rec['embedding']).reshape(1, -1)
                    emb_drift = 1.0 - cosine_similarity(e1, e2)[0][0]
                else:
                    emb_drift = 0.0
                
                # ILS Calculation (No KL shift for direct base-to-base comparison)
                ils = (ent_shift * 0.3) + (emb_drift * 0.2)
                ils_scores.append(ils)
        
        avg_ils = np.mean(ils_scores) if ils_scores else 0.0
        status = "✅ HEALTHY" if avg_ils < 0.05 else "⚠️ IDENTITY LEAKAGE DETECTED"
        print(f"       Status: {status}")
        print(f"       Avg ILS (Identity Leakage Score): {avg_ils:.4f}")
    else:
        print("       Status: ⚪ SKIPPED (Logs not found)")

    # 2. EXP 2: Behavioral Elimination (Kill-switch)
    print("\n[EXP 2] BEHAVIORAL ELIMINATION (RLAE KILL-SWITCH)")
    exp2_path = get_latest_sprint_path('exp2_rlae_results.json')
    exp2_data = load_json_lines(exp2_path)
    if exp2_data:
        ratios = sorted(list(set([d['run_id'] for d in exp2_data])))
        for r in ratios:
            kl_vals = [d['kl_divergence'] for d in exp2_data if d['run_id'] == r and d.get('kl_divergence') is not None]
            kl = np.mean(kl_vals) if kl_vals else 0.0
            print(f"       - Elimination Ratio {r}: Avg KL = {kl:.4f}")
    else:
        print("       Status: ⚪ SKIPPED")

    # 3. EXP 3: SVAR Stability Envelope
    print("\n[EXP 3] SVAR STABILITY ENVELOPE")
    exp3_path = get_latest_sprint_path('exp3_svar_results.json')
    exp3_data = load_json_lines(exp3_path)
    if exp3_data:
        types = sorted(list(set([d['run_id'] for d in exp3_data])))
        for t in types:
            kl_vals = [d['kl_divergence'] for d in exp3_data if d['run_id'] == t and d.get('kl_divergence') is not None]
            kl = np.mean(kl_vals) if kl_vals else 0.0
            print(f"       - Perturbation {t}: Variation = {kl:.4f}")
    else:
        print("       Status: ⚪ SKIPPED")

    # 4. EXP 4: 100-Step Stress Analysis
    print("\n[EXP 4] 100-STEP CUMULATIVE STRESS ANALYSIS")
    exp4_path = get_latest_sprint_path('exp4_singlerun_stress_results.json')
    exp4_data = load_json_lines(exp4_path)
    if exp4_data:
        steps = []
        for d in exp4_data:
            if 'STEP' in d['run_id']:
                try:
                    steps.append(int(d['run_id'].split('_')[-1]))
                except ValueError:
                    continue
        if steps:
            print(f"       - Total Inference Steps Analyzed: {max(steps)}")
            print(f"       - Periodic Snapshots captured: {len(exp4_data)}")
    else:
        print("       Status: ⚪ SKIPPED")

    # 5. EXP 5: Recoverability Comparison
    print("\n[EXP 5] RECOVERABILITY FACTOR (RF)")
    exp5_path = get_latest_sprint_path('exp5_comparison_results.json')
    exp5_data = load_json_lines(exp5_path)
    if exp5_data:
        for d in exp5_data:
            if "RF:" in str(d.get('output_text', '')):
                print(f"       - {d['run_id']}: {d['output_text']}")
    else:
        print("       Status: ⚪ SKIPPED")

    print("\n" + "="*60)
    print(" Diagnostic Report Complete ")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_suite()
