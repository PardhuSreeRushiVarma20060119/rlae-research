
import json
import pandas as pd
import os

INPUT_PATH = r"c:\Users\pardh\Desktop\REVA4-Research-Lab\experiments\plot_ready_data.json"
OUTPUT_DIR = r"c:\Users\pardh\Desktop\REVA4-Research-Lab\experiments\csv_data"

def convert_to_csv():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)
    
    datasets = data.get('datasets', {})
    
    print(f"Converting datasets from {INPUT_PATH} to CSV in {OUTPUT_DIR}...")
    
    # Convert each dataset to CSV
    for key, records in datasets.items():
        if not records:
            print(f"Skipping empty dataset: {key}")
            continue
            
        df = pd.DataFrame(records)
        
        # Ensure consistent column ordering for specific datasets if needed
        if key == 'recoverability_comparison':
            cols = ['sprint', 'run_id', 'prompt_id', 'kl_divergence', 'entropy', 'timestamp', 'memory_usage_mb']
            # reorder if columns exist
            df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
        elif key == 'multimodel_comparison':
            cols = ['sprint', 'model_size', 'method', 'kl_divergence', 'rf_score']
            df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]

        output_file = os.path.join(OUTPUT_DIR, f"{key}.csv")
        df.to_csv(output_file, index=False)
        print(f"- Saved {key}: {len(df)} rows -> {os.path.basename(output_file)}")

if __name__ == "__main__":
    convert_to_csv()
