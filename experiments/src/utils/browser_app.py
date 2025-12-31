import gradio as gr
import os
import subprocess
import json
import pandas as pd
import time

# Paths
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), '../..')
LOGS_DIR = os.path.join(EXPERIMENTS_DIR, 'logs')
EXP1_LOG = os.path.join(LOGS_DIR, 'exp1_results.json')
EXP2_LOG = os.path.join(LOGS_DIR, 'exp2_rlae_results.json')
EXP3_LOG = os.path.join(LOGS_DIR, 'exp3_svar_results.json')

def run_script(script_path, args=[]):
    cmd = ["python", script_path] + args
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=EXPERIMENTS_DIR)
    stdout, stderr = process.communicate()
    return f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

def run_baseline():
    return run_script("src/exp1_reset/1_baseline.py")

def run_sft():
    return run_script("src/exp1_reset/2_train_sft.py")

def run_rl():
    return run_script("src/exp1_reset/3_train_rl.py")

def run_rlae_core():
    return run_script("src/exp2_rlae/elimination_test.py")

def run_verify_reset():
    return run_script("src/exp1_reset/4_verify_reset.py")

def run_emergency_kill():
    """
    RLAE Principle: Killability & Reversibility.
    Immediately terminates the runtime environment and clears all LoRA artifacts.
    """
    return run_script("src/exp1_reset/4_verify_reset.py")

def run_svar():
    return run_script("src/exp3_svar/perturbation.py")

def load_logs(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["run_id", "prompt_id", "timestamp", "output_text", "kl_divergence", "memory_usage_mb"])
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def get_comparison():
    df1 = load_logs(EXP1_LOG)
    df2 = load_logs(EXP2_LOG)
    df3 = load_logs(EXP3_LOG)
    combined = pd.concat([df1, df2, df3], ignore_index=True)
    return combined

with gr.Blocks(title="RLAE & SVAR Runtime Governance & Diagnostic Surface") as demo:
    gr.Markdown("# 🛡️ RLAE & SVAR: Runtime Governance & Diagnostic Surface")
    gr.Markdown("Governing swappable behavioral units and analyzing structural variance for robustness.")
    
    with gr.Tab("Runtime Governance"):
        with gr.Row():
            btn_baseline = gr.Button("1. Mount Baseline", variant="primary")
            btn_sft = gr.Button("2. Mount SFT Environment")
            btn_rl = gr.Button("3. RL Environment Training")
            btn_rlae = gr.Button("4. RLAE Behavioral Elimination", variant="primary")
            
        with gr.Row():
            btn_verify = gr.Button("5. Validate Reset Integrity", variant="secondary")
            btn_kill = gr.Button("🛑 EMERGENCY KILL PATH", variant="stop")
            
        output_console = gr.Code(label="Governance Console", language="markdown", interactive=False)
        
        btn_baseline.click(run_baseline, outputs=output_console)
        btn_sft.click(run_sft, outputs=output_console)
        btn_rl.click(run_rl, outputs=output_console)
        btn_rlae.click(run_rlae_core, outputs=output_console)
        btn_verify.click(run_verify_reset, outputs=output_console)
        btn_kill.click(run_emergency_kill, outputs=output_console)
        
    with gr.Tab("Diagnostic Surface"):
        gr.Markdown("### Behavioral Stability Envelopes & Sensitivity Heatmaps")
        with gr.Row():
            btn_svar = gr.Button("Run SVAR Analysis", variant="primary")
            btn_refresh = gr.Button("Refresh Diagnostic Data")
            
        results_table = gr.Dataframe(label="Stability Metrics (KL Div / Entropy / Memory)")
        
        btn_svar.click(run_svar, outputs=output_console)
        btn_refresh.click(get_comparison, outputs=results_table)
        
    with gr.Tab("Frozen Core Stats"):
        def get_gpu_status():
            try:
                res = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                return f"GPU Status (Name, Mem Used, Mem Total, Util %):\n{res}"
            except:
                return "No GPU detected or nvidia-smi failed."
        
        gpu_output = gr.Textbox(label="NVIDIA SMI Telemetry", lines=5)
        btn_gpu = gr.Button("Poll GPU State")
        btn_gpu.click(get_gpu_status, outputs=gpu_output)

if __name__ == "__main__":
    demo.launch(share=True, inline=True)
