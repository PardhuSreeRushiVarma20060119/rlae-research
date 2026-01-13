import os
import json
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
LOGS_ROOT = os.path.join("experiments", "logs")
PLOTS_ROOT = os.path.join(LOGS_ROOT, "plots", "sprintplot")

if not os.path.exists(PLOTS_ROOT):
    os.makedirs(PLOTS_ROOT)

def load_sprint_data(sprint_path):
    """Loads all JSON logs from a sprint directory."""
    data = []
    json_files = glob.glob(os.path.join(sprint_path, "*.json"))
    
    for jf in json_files:
        filename = os.path.basename(jf)
        with open(jf, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        record['source_file'] = filename
                        data.append(record)
                    except json.JSONDecodeError:
                        continue
    return pd.DataFrame(data)

def plot_kl_divergence(df, sprint_name, output_dir):
    """Generates Interactive KL Divergence plots."""
    if 'kl_divergence' not in df.columns or df.empty:
        return

    # Filter out global summaries (RF scores) or missing KL
    kl_df = df.dropna(subset=['kl_divergence'])
    kl_df = kl_df[kl_df['prompt_id'] != 'global']
    
    if kl_df.empty:
        return

    # Try to extract intensity if possible
    # M3 run_ids look like "SEC1_MUTATION_SCAR_INTENSITY_0.001"
    kl_df['intensity'] = kl_df['run_id'].str.extract(r'INTENSITY_(\d+\.?\d*)').astype(float)
    
    fig = None
    if not kl_df['intensity'].isna().all():
        # M3 Style Plot: KL vs Intensity
        # Group by intensity to get average if multiple prompts
        agg_df = kl_df.groupby('intensity')['kl_divergence'].mean().reset_index()
        
        fig = px.line(
            agg_df, 
            x='intensity', 
            y='kl_divergence', 
            markers=True,
            log_x=True, 
            title=f"<b>{sprint_name}</b>: KL Divergence vs Mutation Intensity (M3)",
            labels={"intensity": "Mutation Intensity (Log Scale)", "kl_divergence": "KL Divergence"}
        )
        fig.update_traces(line_color='#EF553B', line_width=3)
        
    else:
        # Standard Bar Plot: KL per Run ID
        fig = px.bar(
            kl_df, 
            x='run_id', 
            y='kl_divergence', 
            color='prompt_id',
            title=f"<b>{sprint_name}</b>: KL Divergence per Run",
            labels={"run_id": "Experiment Run ID", "kl_divergence": "KL Divergence"},
            hover_data=['output_text']
        )
        fig.update_layout(xaxis_tickangle=-45)

    # Save and Show
    html_path = os.path.join(PLOTS_ROOT, f"{sprint_name}_kl_analysis.html")
    fig.write_html(html_path)
    print(f"Generated KL Interactive Plot: {html_path}")
    fig.show()

def plot_recoverability(df, sprint_name, output_dir):
    """Generates a Table for Verify/RF scores."""
    # Look for RF entries
    if 'output_text' not in df.columns:
        return
        
    rf_df = df[df['prompt_id'] == 'global']
    if rf_df.empty:
        return

    # Create Plotly Table
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Run ID', 'RF Output', 'Post-Reset KL'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[rf_df.run_id, rf_df.output_text, rf_df.kl_divergence.round(4)],
                   fill_color='lavender',
                   align='left'))
    ])
    
    fig.update_layout(title_text=f"<b>{sprint_name}</b>: Recoverability Factor (RF) Summary")

    html_path = os.path.join(PLOTS_ROOT, f"{sprint_name}_rf_summary.html")
    fig.write_html(html_path)
    print(f"Generated RF Table: {html_path}")
    fig.show()

def analyze_sprint(sprint_path):
    sprint_name = os.path.basename(sprint_path)
    print(f"Analyzing {sprint_name}...")
    
    df = load_sprint_data(sprint_path)
    if df.empty:
        print(f"No data found in {sprint_name}.")
        return

    plot_kl_divergence(df, sprint_name, sprint_path)
    plot_recoverability(df, sprint_name, sprint_path)

def main():
    if not os.path.exists(LOGS_ROOT):
        print(f"Logs directory not found: {LOGS_ROOT}")
        return

    # Find all Sprint-* directories
    sprint_dirs = glob.glob(os.path.join(LOGS_ROOT, "Sprint-*"))
    sprint_dirs.sort()

    for sp in sprint_dirs:
        if os.path.isdir(sp):
            analyze_sprint(sp)

if __name__ == "__main__":
    main()
