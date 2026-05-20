"""Script to run Scalability Experiments (E7)."""
import os
import sys
import time
import igraph as ig
import leidenalg
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import generate_synthetic_corpus
from src.utils.embeddings import DocumentEmbedder
from src.coshard.graph import build_similarity_graph
from src.coshard.partition import coshard_partition
from src.shade.proxy import compute_proxy_shade, _find_optimal_k
from src.shade.metrics import compute_coherence_embed

def run_leiden_vanilla(graph, n_shards):
    """Run standard Leiden community detection and merge to n_shards."""
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)
    p_lists = list(partition)
    
    if len(p_lists) > n_shards:
        while len(p_lists) > n_shards:
            p_lists.sort(key=len)
            smallest = p_lists.pop(0)
            p_lists[0].extend(smallest)
    elif len(p_lists) < n_shards:
        while len(p_lists) < n_shards:
            p_lists.sort(key=len, reverse=True)
            largest = p_lists.pop(0)
            half = len(largest) // 2
            p_lists.append(largest[:half])
            p_lists.append(largest[half:])
            
    return p_lists

def main():
    print("\n--- Running Experiment E7: Scalability Analysis ---")
    
    corpus_sizes = [500, 1000, 2000, 5000] # Scaled down slightly for feasible quick runs
    n_shards = 5
    
    print(f"\n{'Size (N)':<10} | {'Method':<18} | {'Time (s)':<12} | {'Max SHADE':<12}")
    print("-" * 60)
    
    embedder = DocumentEmbedder()
    
    for size in corpus_sizes:
        # Generate and embed
        docs, _, _ = generate_synthetic_corpus(n_clusters=10, docs_per_cluster=size//10)
        embs = embedder.embed_corpus(docs, f"synthetic_e7_{size}")
        
        t0 = time.time()
        graph = build_similarity_graph(embs, threshold=0.3, top_k=20)
        t_graph = time.time() - t0
        
        # 1. Leiden-Vanilla (Baseline)
        t0 = time.time()
        leiden_shards = run_leiden_vanilla(graph, n_shards)
        t_leiden = time.time() - t0 + t_graph
        
        # Precompute k for evaluation
        k_clusters = 10 # fixed for synthetic
        
        shades_leiden = [compute_proxy_shade(s, embs, k_clusters) for s in leiden_shards if s]
        max_shade_leiden = float(np.max(shades_leiden)) if shades_leiden else 1.0
        
        print(f"{size:<10} | {'Leiden-vanilla':<18} | {t_leiden:<12.2f} | {max_shade_leiden:<12.4f}")
        
        # 2. CoShard
        t0 = time.time()
        # Lambda = 0.5 for balanced tradeoff
        coshard_shards = coshard_partition(graph, embs, n_shards=n_shards, lambda_weight=0.5, max_iterations=2)
        t_coshard = time.time() - t0 + t_graph
        
        shades_coshard = [compute_proxy_shade(s, embs, k_clusters) for s in coshard_shards if s]
        max_shade_coshard = float(np.max(shades_coshard)) if shades_coshard else 1.0
        
        print(f"{size:<10} | {'CoShard (l=0.5)':<18} | {t_coshard:<12.2f} | {max_shade_coshard:<12.4f}")
        print("-" * 60)
        
    print("\nExperiment E7 completed successfully.")

if __name__ == "__main__":
    main()
