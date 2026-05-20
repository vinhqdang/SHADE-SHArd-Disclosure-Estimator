"""Script to run Proxy Fidelity Experiment (E3)."""
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import generate_synthetic_corpus
from src.utils.embeddings import DocumentEmbedder
from src.shade.proxy import compute_proxy_shade
from src.eval.adversary import dummy_oracle_shade

def main():
    print("Generating synthetic corpus for E3...")
    docs, _, _ = generate_synthetic_corpus(n_clusters=10, docs_per_cluster=40)
    print(f"Generated {len(docs)} documents.")
    
    print("Computing embeddings...")
    embedder = DocumentEmbedder()
    embs = embedder.embed_corpus(docs, "synthetic_e3")
    
    print("\n--- Running Experiment E3: Proxy Fidelity vs K ---")
    
    # We will test various K values for clustering
    k_values = [5, 10, 20, 50, 100]
    n_shards_to_test = 20
    
    # Generate 20 random shards of varying sizes
    test_shards = []
    for _ in range(n_shards_to_test):
        shard_size = random.randint(int(len(docs) * 0.1), int(len(docs) * 0.4))
        shard_indices = random.sample(range(len(docs)), shard_size)
        shard_docs = [docs[idx] for idx in shard_indices]
        # Compute ground truth (oracle)
        oracle = dummy_oracle_shade(shard_docs, docs)
        test_shards.append((shard_indices, oracle))
        
    print(f"{'K (clusters)':<12} | {'Mean Absolute Error (MAE)':<25}")
    print("-" * 40)
    
    mae_results = []
    
    for k in k_values:
        errors = []
        for shard_indices, oracle in test_shards:
            proxy = compute_proxy_shade(shard_indices, embs, k_clusters=k)
            errors.append(abs(proxy - oracle))
            
        mae = float(np.mean(errors))
        mae_results.append(mae)
        print(f"{k:<12} | {mae:<25.4f}")
        
    print("\nExperiment E3 completed successfully.")

if __name__ == "__main__":
    main()
