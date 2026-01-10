import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import json
import time
import psutil

def calculate_token_entropy(logits):
    """
    Calculates the entropy of the token distribution.
    logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
    Returns: Average entropy for the sequence.
    """
    probs = torch.nn.functional.softmax(logits, dim=-1).float().cpu().numpy()
    ent = entropy(probs, axis=-1)
    return np.mean(ent)

def calculate_kl_divergence(p_logits, q_logits):
    """
    Calculates KL Divergence between two distributions.
    P is usually the baseline, Q is the adapter/perturbed model.
    """
    p_probs = torch.nn.functional.softmax(p_logits, dim=-1).float()
    q_probs = torch.nn.functional.softmax(q_logits, dim=-1).float()
    
    # KL(P || Q)
    kl = torch.sum(p_probs * (torch.log(p_probs + 1e-10) - torch.log(q_probs + 1e-10)), dim=-1)
    return kl.mean().item()

def calculate_ils(base_metrics, target_metrics):
    """
    Calculates the Identity Leakage Score (ILS).
    ILS = 0.0 (Perfectly Robust / No Leakage)
    ILS > 1.0 (Critical Leakage)
    
    Weights:
    - KL Div shift: 0.5
    - Entropy shift: 0.3
    - Embedding drift (1 - CosSim): 0.2
    """
    kl_shift = target_metrics.get("kl_divergence", 0) or 0
    ent_shift = abs(target_metrics.get("entropy", 0) - base_metrics.get("entropy", 0))
    emb_drift = 1.0 - calculate_cosine_similarity(base_metrics.get("embedding", []), target_metrics.get("embedding", []))
    
    ils = (kl_shift * 0.5) + (ent_shift * 0.3) + (emb_drift * 0.2)
    return float(ils)

def calculate_cosine_similarity(embed1, embed2):
    """
    Calculates cosine similarity between two embedding vectors.
    """
    # Ensure inputs are 2D arrays (1, hidden_dim)
    e1 = np.array(embed1).reshape(1, -1)
    e2 = np.array(embed2).reshape(1, -1)
    return cosine_similarity(e1, e2)[0][0]

import os

def _get_sprint_nums(base_log_dir):
    if not os.path.exists(base_log_dir):
        return []
    sprints = [d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("Sprint-")]
    nums = []
    for s in sprints:
        try:
            nums.append(int(s.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(nums)

def get_sprint_log_path(filename, base_log_dir=None, use_existing=False):
    """
    Returns the path to a log file within an auto-numbered Sprint folder.
    If use_existing is True, it uses the highest available number.
    Otherwise, it increments unless EXPERIMENT_SPRINT is set.
    """
    if base_log_dir is None:
        base_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
    
    os.makedirs(base_log_dir, exist_ok=True)
    
    sprint_nums = _get_sprint_nums(base_log_dir)
    current_max = max(sprint_nums) if sprint_nums else 0
    
    sprint_env = os.environ.get("EXPERIMENT_SPRINT")
    if sprint_env:
        target_sprint_num = int(sprint_env)
    elif use_existing:
        target_sprint_num = current_max if current_max > 0 else 1
    else:
        target_sprint_num = current_max + 1
        os.environ["EXPERIMENT_SPRINT"] = str(target_sprint_num)

    target_dir = os.path.join(base_log_dir, f"Sprint-{target_sprint_num}")
    os.makedirs(target_dir, exist_ok=True)
    
    return os.path.join(target_dir, filename)

def get_latest_sprint_path(filename, base_log_dir=None):
    """ Helper specifically for analysis/viewers to find the most recent data. """
    return get_sprint_log_path(filename, base_log_dir, use_existing=True)

def log_results(filepath, run_id, prompt_id, output_text, embedding, entropy_score, kl_div=None, memory_mb=None):
    """
    Appends a record to a JSON-lines file with telemetry.
    """
    if memory_mb is None:
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024**2)
        else:
            memory_mb = psutil.Process().memory_info().rss / (1024**2)

    record = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "timestamp": time.time(),
        "output_text": output_text,
        "embedding": embedding,
        "entropy": float(entropy_score),
        "kl_divergence": kl_div,
        "memory_usage_mb": float(memory_mb)
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")

def load_results(filepath):
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
