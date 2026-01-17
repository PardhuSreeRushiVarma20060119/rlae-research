
import json
import pandas as pd
import os

DATA_PATH = r"c:\Users\pardh\Desktop\REVA4-Research-Lab\experiments\plot_ready_data.json"

def check_coverage():
    if not os.path.exists(DATA_PATH):
        print("Data file not found.")
        return

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    datasets = data.get('datasets', {})
    
    print("--- Data Coverage Audit ---")
    for key, records in datasets.items():
        if not records:
            print(f"{key}: NO DATA")
            continue
            
        df = pd.DataFrame(records)
        print(f"\ndataset: {key} (Total: {len(df)})")
        if 'sprint' in df.columns:
            print(df['sprint'].value_counts().sort_index())
        else:
            print("No 'sprint' column found.")

if __name__ == "__main__":
    check_coverage()
