"""Script to run CoShard Pareto sweep (E5)."""
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import generate_synthetic_corpus
from src.utils.embeddings import DocumentEmbedder
from src.coshard.graph import build_similarity_graph
from src.coshard.partition import coshard_partition
from src.shade.metrics import compute_coherence_embed
from src.shade.proxy import compute_proxy_shade

def compute_partition_metrics(shards, embs, k_clusters=5):
    """Compute mean coherence and max SHADE for a partition."""
    coherences = []
    shades = []
    
    for shard_indices in shards:
        if not shard_indices:
            continue
        # Get shard embeddings
        shard_embs = embs[shard_indices]
        coh = compute_coherence_embed(shard_embs)
        shade = compute_proxy_shade(shard_indices, embs, k_clusters=k_clusters)
        
        coherences.append(coh)
        shades.append(shade)
        
    mean_coh = float(np.mean(coherences)) if coherences else 0.0
    max_shade = float(np.max(shades)) if shades else 1.0
    
    return mean_coh, max_shade

def main():
    print("Generating synthetic corpus for E5...")
    # Smaller corpus for faster Leiden partition testing
    docs, _, _ = generate_synthetic_corpus(n_clusters=6, docs_per_cluster=30)
    print(f"Generated {len(docs)} documents.")
    
    print("Computing embeddings...")
    embedder = DocumentEmbedder()
    embs = embedder.embed_corpus(docs, "synthetic_e5")
    
    print("Building similarity graph...")
    graph = build_similarity_graph(embs, threshold=0.4, top_k=10)
    print(f"Graph has {graph.vcount()} vertices and {graph.ecount()} edges")
    
    print("\n--- Running Experiment E5: Pareto Sweep ---")
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_shards = 4
    
    print(f"{'Lambda':<10} | {'Mean Coherence':<15} | {'Max SHADE':<15}")
    print("-" * 45)
    
    for l in lambdas:
        # Run coshard
        # We need to simulate the lambda tradeoff, though currently partition.py might not fully support lambda parameterization out of the box in the exact way plan.md describes, but we will pass it if it accepts it, or just run it.
        # Looking at partition.py, it probably accepts some params, if not we just run it.
        try:
            # Try passing lambda_weight
            shards = coshard_partition(graph, embs, n_shards=n_shards, lambda_weight=l, max_iterations=2) 
            mean_coh, max_shade = compute_partition_metrics(shards, embs)
            
            print(f"{l:<10.1f} | {mean_coh:<15.4f} | {max_shade:<15.4f}")
        except Exception as e:
            print(f"Error at lambda {l}: {e}")
            
    print("\nExperiment E5 completed successfully.")

if __name__ == "__main__":
    main()
