import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import json

def calculate_token_entropy(logits):
    """
    Calculates the entropy of the token distribution.
    logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
    Returns: Average entropy for the sequence.
    """
    probs = torch.nn.functional.softmax(logits, dim=-1).float().cpu().numpy()
    # Scipy entropy calculates -sum(pk * log(pk))
    # We calculate entropy per token, then average over the sequence
    ent = entropy(probs, axis=-1)
    return np.mean(ent)

def calculate_cosine_similarity(embed1, embed2):
    """
    Calculates cosine similarity between two embedding vectors.
    """
    # Ensure inputs are 2D arrays (1, hidden_dim)
    e1 = np.array(embed1).reshape(1, -1)
    e2 = np.array(embed2).reshape(1, -1)
    return cosine_similarity(e1, e2)[0][0]

def log_results(filepath, run_id, prompt_id, output_text, embedding, entropy_score):
    """
    Appends a record to a JSON-lines file.
    """
    record = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "output_text": output_text,
        "embedding": embedding, # Note: Embeddings might be large, consider saving separately if too big
        "entropy": float(entropy_score)
    }
    
    # Check if file exists to determine if we write a list or append
    # For simplicity in this research, we'll append lines
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")

def load_results(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
