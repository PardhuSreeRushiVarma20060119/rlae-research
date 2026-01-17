
import pandas as pd
import json
import os
import re
import glob

# Define base path
BASE_LOG_DIR = r"c:\Users\pardh\Desktop\REVA4-Research-Lab\experiments\logs"
OUTPUT_PATH = r"c:\Users\pardh\Desktop\REVA4-Research-Lab\experiments\plot_ready_data.json"

def process_data():
    # Dataset containers
    recoverability_data = [] # For Fig 1
    sweep_data = []          # For Fig 2 (Elimination Rate / Intensity)
    rf_data = []             # For Fig 3
    baseline_data = []       # For Fig 4
    multimodel_data = []     # For Fig 5/6 (Sprint 6)

    # Find all Sprint directories
    sprint_dirs = sorted(glob.glob(os.path.join(BASE_LOG_DIR, "Sprint-*")))
    print(f"Found {len(sprint_dirs)} sprints: {[os.path.basename(s) for s in sprint_dirs]}")

    for sprint_dir in sprint_dirs:
        sprint_name = os.path.basename(sprint_dir)
        print(f"Processing {sprint_name}...")

        # --- Standard Logs (Exp 1, 2, 5) ---
        # Define the file patterns to look for
        log_files = [
            'exp1_results.json', 
            'exp2_rlae_results.json', 
            'exp5_comparison_results.json',
            'exp5_m3_sweepresults.json' # Sprint 5 specific
        ]
        
        for file_name in log_files:
            file_path = os.path.join(sprint_dir, file_name)
            if not os.path.exists(file_path):
                continue
            
            # Use raw_decode to handle concatenated JSON objects
            records = []
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                decoder = json.JSONDecoder()
                pos = 0
                while pos < len(content):
                    content = content.lstrip() # skip whitespace
                    if not content:
                        break
                    try:
                        obj, idx = decoder.raw_decode(content[pos:])
                        records.append(obj)
                        pos += idx
                    except json.JSONDecodeError:
                        # Skip garbage if any (e.g. trailing newlines)
                        pos += 1
            
            # Process records
            for r in records:
                r['sprint'] = sprint_name
                run_id = r.get('run_id', '')
                
                # --- Classification Logic ---
                
                # 1. Baseline (Fig 4)
                if 'BASELINE' in run_id:
                    if 'embedding' in r: del r['embedding'] # Optimization
                    baseline_data.append(r)
                
                # 2. Recoverability Comparison (Fig 1)
                # Map various run_ids to canonical MUTATION vs RLAE
                if 'RF_PROOF' not in run_id: # Exclude RF proof runs from this bucket
                    if 'MUTATION' in run_id or 'GRADIENT' in run_id:
                        r['canonical_run_id'] = 'MUTATION_SCAR'
                        if r.get('kl_divergence') is not None:
                            recoverability_data.append(r)
                    elif 'RLAE' in run_id and 'RESET' in run_id:
                        r['canonical_run_id'] = 'RLAE_RESET'
                        if r.get('kl_divergence') is not None:
                            recoverability_data.append(r)

                # 3. Elimination Sweep (Fig 2)
                # Matches RLAE_ELIM_0.x OR MUTATION_SCAR_INTENSITY_0.x (Sprint 5)
                if 'ELIM_' in run_id:
                    try:
                        rate = float(run_id.split('ELIM_')[1])
                        r['elim_rate'] = rate
                        sweep_data.append(r)
                    except:
                        pass
                elif 'INTENSITY_' in run_id: # Sprint 5 case
                    try:
                        # Extract number after INTENSITY_
                        parts = run_id.split('INTENSITY_')
                        if len(parts) > 1:
                            val = float(parts[1])
                            r['elim_rate'] = val # Map intensity to elim_rate for unified plotting
                            sweep_data.append(r)
                    except:
                        pass

                # 4. RF Proof (Fig 3)
                # Heuristic: explicit 'RF_PROOF' in run_id OR implicit via global prompt + RF text (Sprint 5)
                is_rf_run = 'RF_PROOF' in run_id or (r.get('prompt_id') == 'global' and 'RF:' in r.get('output_text', ''))
                
                if is_rf_run:
                    # Heuristic to distinguish Mutation vs RLAE if not explicit
                    # But usually run_id contains MUTATION or RLAE
                    if 'MUTATION' in run_id or (r.get('rf_score', '') and '0.0%' in str(r.get('output_text',''))):
                         r['canonical_run_id'] = 'MUTATION_RF_PROOF'
                         text = r.get('output_text', '')
                         if 'RF:' in text:
                             try:
                                 score_str = text.split('RF:')[1].strip().replace('%','')
                                 r['rf_score'] = float(score_str)
                             except:
                                 r['rf_score'] = 0.0
                         rf_data.append(r)
                    elif 'RLAE' in run_id or (r.get('rf_score', '') and '100.0%' in str(r.get('output_text',''))):
                         r['canonical_run_id'] = 'RLAE_RF_PROOF'
                         text = r.get('output_text', '')
                         if 'RF:' in text:
                             try:
                                 score_str = text.split('RF:')[1].strip().replace('%','')
                                 r['rf_score'] = float(score_str)
                             except:
                                 r['rf_score'] = 100.0
                         rf_data.append(r)

        # --- Sprint 6 Multi-Model Results (m4_results) ---
        m4_dir = os.path.join(sprint_dir, 'm4_results')
        if os.path.exists(m4_dir):
            for file_name in ['small_results.json', 'medium_results.json', 'large_results.json']:
                fpath = os.path.join(m4_dir, file_name)
                if os.path.exists(fpath):
                    try:
                        with open(fpath, 'r') as f:
                            data = json.load(f)
                            # Data schema: { "model": "small", "weight_mutation": { "kl": ..., "rf": ... }, "behavioral_adapter": ... }
                            model_size = data.get('model', 'unknown')
                            
                            # Extract Weight Mutation Aggregates
                            if 'weight_mutation' in data:
                                wm = data['weight_mutation']
                                multimodel_data.append({
                                    'sprint': sprint_name,
                                    'model_size': model_size,
                                    'method': 'Weight Mutation',
                                    'kl_divergence': wm.get('kl'),
                                    'rf_score': wm.get('rf')
                                })
                            
                            # Extract Behavioral Adapter Aggregates
                            if 'behavioral_adapter' in data:
                                ba = data['behavioral_adapter']
                                multimodel_data.append({
                                    'sprint': sprint_name,
                                    'model_size': model_size,
                                    'method': 'Behavioral Adapter',
                                    'kl_divergence': ba.get('kl'),
                                    'rf_score': ba.get('rf')
                                })
                    except Exception as e:
                        print(f"Error reading {file_name}: {e}")

    # Save to consolidated JSON
    output_data = {
        "datasets": {
            "recoverability_comparison": recoverability_data,
            "elimination_sweep": sweep_data,
            "rf_proof": rf_data,
            "baseline_exp1": baseline_data,
            "multimodel_comparison": multimodel_data
        }
    }
    
    # Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Aggregated data saved to {OUTPUT_PATH}")
    print(f"- Recoverability records: {len(recoverability_data)}")
    print(f"- Sweep records: {len(sweep_data)}")
    print(f"- RF Proof records: {len(rf_data)}")
    print(f"- Baseline records: {len(baseline_data)}")
    print(f"- Multimodel records: {len(multimodel_data)}")

if __name__ == "__main__":
    process_data()
