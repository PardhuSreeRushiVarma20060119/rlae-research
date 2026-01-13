import os
import json
import glob
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
LOGS_ROOT = os.path.join("experiments", "logs")
OUTPUT_DIR = "sprintrunner"

def load_all_sprints_data():
    """Loads JSON logs from ALL sprint directories."""
    all_data = []
    
    # regex to extract Sprint Number
    sprint_pattern = re.compile(r"Sprint-(\d+)")
    
    sprint_dirs = glob.glob(os.path.join(LOGS_ROOT, "Sprint-*"))
    for sp in sprint_dirs:
        match = sprint_pattern.search(os.path.basename(sp))
        if not match:
            continue
        sprint_num = int(match.group(1))
        
        json_files = glob.glob(os.path.join(sp, "*.json"))
        for jf in json_files:
            experiment_type = os.path.basename(jf).replace("_results.json", "").replace(".json", "")
            
            with open(jf, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            record['sprint_num'] = sprint_num
                            record['experiment_type'] = experiment_type
                            all_data.append(record)
                        except json.JSONDecodeError:
                            continue
    return pd.DataFrame(all_data)

def plot_cross_sprint_kl(df):
    """Plots KL Divergence trends across sprints interactively."""
    if 'kl_divergence' not in df.columns or df.empty:
        return

    # Filter meaningful KL data
    kl_df = df.dropna(subset=['kl_divergence'])
    kl_df = kl_df[kl_df['prompt_id'] != 'global']
    
    if kl_df.empty:
        return
    
    # Aggregate mean KL per sprint per experiment
    agg_df = kl_df.groupby(['sprint_num', 'experiment_type'])['kl_divergence'].mean().reset_index()

    fig = px.line(
        agg_df,
        x='sprint_num',
        y='kl_divergence',
        color='experiment_type',
        markers=True,
        log_y=True,
        title="<b>Cross-Sprint Analysis</b>: KL Divergence Trends",
        labels={"sprint_num": "Sprint Number", "kl_divergence": "Avg KL Divergence (Log Scale)", "experiment_type": "Experiment"},
        template="plotly_white"
    )
    
    save_path = os.path.join(OUTPUT_DIR, "cross_sprint_kl_trend.html")
    fig.write_html(save_path)
    print(f"Generated Cross-Sprint KL Interactive Plot: {save_path}")
    fig.show()

def plot_cross_sprint_stability_envelope(df):
    """Box plot of KL distribution per sprint to show stability/variance."""
    if 'kl_divergence' not in df.columns or df.empty:
        return

    kl_df = df.dropna(subset=['kl_divergence'])
    kl_df = kl_df[kl_df['prompt_id'] != 'global']
    
    if kl_df.empty:
        return

    fig = px.box(
        kl_df,
        x='sprint_num',
        y='kl_divergence',
        color='experiment_type',
        log_y=True,
        title="<b>Stability Envelope</b>: KL Distribution per Sprint",
        labels={"sprint_num": "Sprint Number", "kl_divergence": "KL Divergence (Log Scale)"},
        template="plotly_white"
    )
    
    save_path = os.path.join(OUTPUT_DIR, "cross_sprint_stability_envelope.html")
    fig.write_html(save_path)
    print(f"Generated Stability Envelope Interactive Plot: {save_path}")
    fig.show()

def main():
    if not os.path.exists(LOGS_ROOT):
        print(f"Logs directory not found: {LOGS_ROOT}")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading data from all sprints...")
    df = load_all_sprints_data()
    
    if df.empty:
        print("No data found across sprints.")
        return

    print(f"Loaded {len(df)} records across {df['sprint_num'].nunique()} sprints.")
    
    plot_cross_sprint_kl(df)
    plot_cross_sprint_stability_envelope(df)

if __name__ == "__main__":
    main()
