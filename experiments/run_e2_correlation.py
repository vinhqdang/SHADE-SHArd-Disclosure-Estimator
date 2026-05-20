"""Script to run Reconstruction correlation (E2)."""
import os
import sys
import random
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import generate_synthetic_corpus
from src.utils.embeddings import DocumentEmbedder
from src.shade.proxy import compute_proxy_shade
from src.eval.adversary import dummy_oracle_shade
from src.eval.correlation import compute_correlations

def jaccard_shade(shard_indices, total_docs):
    return len(shard_indices) / total_docs

def main():
    print("Generating synthetic corpus for E2...")
    docs, _, _ = generate_synthetic_corpus(n_clusters=8, docs_per_cluster=40)
    print(f"Generated {len(docs)} documents.")
    
    print("Computing embeddings...")
    embedder = DocumentEmbedder()
    embs = embedder.embed_corpus(docs, "synthetic_e2")
    
    print("\nSimulating shards and computing SHADE variants...")
    
    proxy_scores = []
    oracle_scores = []
    jaccard_scores = []
    
    # Generate 50 random shards of varying sizes
    for i in range(50):
        # Vary shard size from 5% to 50% of the corpus
        shard_size = random.randint(int(len(docs) * 0.05), int(len(docs) * 0.5))
        shard_indices = random.sample(range(len(docs)), shard_size)
        shard_docs = [docs[idx] for idx in shard_indices]
        
        proxy = compute_proxy_shade(shard_indices, embs, k_clusters=8)
        # Using dummy oracle to allow execution without API keys
        oracle = dummy_oracle_shade(shard_docs, docs)
        jaccard = jaccard_shade(shard_indices, len(docs))
        
        proxy_scores.append(proxy)
        oracle_scores.append(oracle)
        jaccard_scores.append(jaccard)
        
    print("\n--- Running Experiment E2: Correlation Analysis ---")
    p_proxy, s_proxy = compute_correlations(proxy_scores, oracle_scores)
    p_base, s_base = compute_correlations(jaccard_scores, oracle_scores)
    
    print(f"{'Method':<18} | {'Pearson r':<12} | {'Spearman p':<12}")
    print("-" * 48)
    print(f"{'SHADE Proxy':<18} | {p_proxy:<12.4f} | {s_proxy:<12.4f}")
    print(f"{'Jaccard Baseline':<18} | {p_base:<12.4f} | {s_base:<12.4f}")
    
    print("\nExperiment E2 completed successfully.")

if __name__ == "__main__":
    main()
