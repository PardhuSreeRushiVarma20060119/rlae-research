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

def calculate_cosine_similarity(embed1, embed2):
    """
    Calculates cosine similarity between two embedding vectors.
    """
    # Ensure inputs are 2D arrays (1, hidden_dim)
    e1 = np.array(embed1).reshape(1, -1)
    e2 = np.array(embed2).reshape(1, -1)
    return cosine_similarity(e1, e2)[0][0]

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
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")

def load_results(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
