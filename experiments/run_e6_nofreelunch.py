"""Script to run No-Free-Lunch Theorem Validation (E6)."""
import os
import sys
import random
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import generate_synthetic_corpus
from src.utils.embeddings import DocumentEmbedder
from src.shade.proxy import compute_proxy_shade, _find_optimal_k
from src.shade.metrics import compute_coherence_embed
from src.coshard.graph import build_similarity_graph
from src.coshard.partition import coshard_partition

def evaluate_partition(shards, embs, k_clusters):
    """Compute mean coherence and max SHADE for a partition."""
    coherences = []
    shades = []
    for shard in shards:
        if not shard:
            continue
        coh = compute_coherence_embed(embs[shard])
        shade = compute_proxy_shade(shard, embs, k_clusters)
        coherences.append(coh)
        shades.append(shade)
    
    return np.mean(coherences) if coherences else 0.0, np.max(shades) if shades else 1.0

def main():
    print("Generating synthetic corpus for E6...")
    docs, _, _ = generate_synthetic_corpus(n_clusters=8, docs_per_cluster=50)
    print(f"Generated {len(docs)} documents.")
    
    print("Computing embeddings...")
    embedder = DocumentEmbedder()
    embs = embedder.embed_corpus(docs, "synthetic_e6")
    
    k_clusters = _find_optimal_k(embs)
    n_shards = 4
    n_docs = len(docs)
    
    print("\n--- Running Experiment E6: No-Free-Lunch Validation ---")
    
    results = []
    
    # 1. Random Partitions
    print("Generating Random Partitions...")
    for _ in range(20):
        indices = list(range(n_docs))
        random.shuffle(indices)
        chunk_size = n_docs // n_shards
        shards = [indices[i*chunk_size : (i+1)*chunk_size] for i in range(n_shards)]
        # Handle remainder
        if n_docs % n_shards != 0:
            shards[-1].extend(indices[n_shards*chunk_size:])
            
        coh, shade = evaluate_partition(shards, embs, k_clusters)
        results.append(("Random", coh, shade))
        
    # 2. CoShard Partitions at various lambdas
    print("Generating CoShard Partitions...")
    graph = build_similarity_graph(embs, threshold=0.3, top_k=15)
    
    lambdas = np.linspace(0.0, 1.0, 11)
    for l in lambdas:
        shards = coshard_partition(graph, embs, n_shards=n_shards, lambda_weight=l, max_iterations=2)
        coh, shade = evaluate_partition(shards, embs, k_clusters)
        results.append((f"CoShard(l={l:.1f})", coh, shade))
        
    print(f"\n{'Partition Type':<20} | {'Mean Coherence':<15} | {'Max SHADE':<15} | {'Theoretical Bound':<20}")
    print("-" * 78)
    
    # Theoretical Bound f(tau, n, H_s) formulation for demo:
    # A simplified form: bound = alpha * e^(beta * tau) / n
    alpha = 0.8
    beta = 1.5
    
    for ptype, coh, shade in results:
        # Theoretical minimum disclosure for a given coherence tau
        tau = coh
        theoretical_min_shade = min((alpha * np.exp(beta * max(tau, 0))) / n_shards, 1.0)
        
        # Verify empirical shade is bounded below by the theoretical minimum
        is_bounded = shade >= (theoretical_min_shade - 0.05) # small tolerance
        
        print(f"{ptype:<20} | {coh:<15.4f} | {shade:<15.4f} | {theoretical_min_shade:<15.4f} [{is_bounded}]")

    print("\nExperiment E6 completed successfully.")

if __name__ == "__main__":
    main()
