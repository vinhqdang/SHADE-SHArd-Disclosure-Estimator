"""Script to run Composition Theorem Validation (E4)."""
import os
import sys
import random
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import generate_synthetic_corpus
from src.utils.embeddings import DocumentEmbedder
from src.shade.proxy import compute_proxy_shade

def main():
    print("Generating synthetic corpus for E4...")
    docs, _, _ = generate_synthetic_corpus(n_clusters=8, docs_per_cluster=40)
    print(f"Generated {len(docs)} documents.")
    
    print("Computing embeddings...")
    embedder = DocumentEmbedder()
    embs = embedder.embed_corpus(docs, "synthetic_e4")
    
    print("\n--- Running Experiment E4: Composition Theorem ---")
    
    # We will test coalitions of size 2 and 3
    coalition_sizes = [2, 3]
    n_trials = 10
    k_clusters = 8
    
    for m in coalition_sizes:
        print(f"\nTesting Coalitions of Size m = {m}")
        print(f"{'Trial':<6} | {'Max SHADE(Di)':<15} | {'Theoretical Bound':<20} | {'Actual Union SHADE':<20} | {'Holds?':<10}")
        print("-" * 80)
        
        for trial in range(n_trials):
            shards = []
            max_shade_single = 0.0
            
            for _ in range(m):
                # Generate random small shards (e.g. 10-15% of corpus)
                shard_size = random.randint(int(len(docs) * 0.1), int(len(docs) * 0.15))
                shard_indices = random.sample(range(len(docs)), shard_size)
                shards.append(shard_indices)
                
                shade = compute_proxy_shade(shard_indices, embs, k_clusters=k_clusters)
                if shade > max_shade_single:
                    max_shade_single = shade
                    
            # Compute theoretical bound: m * max_i SHADE(D_i)
            theoretical_bound = min(m * max_shade_single, 1.0)
            
            # Compute actual union SHADE
            union_indices = list(set().union(*shards))
            actual_shade = compute_proxy_shade(union_indices, embs, k_clusters=k_clusters)
            
            # Verify bound holds
            holds = actual_shade <= theoretical_bound + 1e-6 # small float tolerance
            
            print(f"{trial:<6} | {max_shade_single:<15.4f} | {theoretical_bound:<20.4f} | {actual_shade:<20.4f} | {str(holds):<10}")
            
    print("\nExperiment E4 completed successfully.")

if __name__ == "__main__":
    main()
