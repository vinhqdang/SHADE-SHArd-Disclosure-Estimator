"""Pareto sweep and hypervolume calculation."""
import numpy as np
import igraph as ig
from typing import List, Tuple, Dict
from src.coshard.partition import coshard_partition
from src.shade.metrics import compute_coherence_embed
from src.shade.proxy import compute_proxy_shade, _find_optimal_k

def pareto_sweep(graph: ig.Graph, all_embeddings: np.ndarray, n_shards: int) -> List[Dict[str, float]]:
    """
    Run CoShard across a sweep of lambda values to trace the Pareto frontier.
    
    Args:
        graph: Similarity graph.
        all_embeddings: Full corpus embeddings.
        n_shards: Target number of shards.
        
    Returns:
        List of dictionaries containing lambda, mean_coherence, and max_shade for each point.
    """
    frontier = []
    
    print("Precomputing K for Pareto sweep...")
    k_clusters = _find_optimal_k(all_embeddings)
    
    # Sweep from pure SHADE (0.0) to pure Coherence (1.0)
    # Using a coarse grid for the sweep to save time
    lambdas = np.linspace(0.0, 1.0, 11)
    
    for l_weight in lambdas:
        print(f"\n--- Running CoShard with lambda={l_weight:.1f} ---")
        partition = coshard_partition(
            graph, 
            all_embeddings, 
            n_shards, 
            delta=float('inf'), # No hard threshold, just optimize
            lambda_weight=float(l_weight),
            max_iterations=10 # Cap iterations to speed up sweep
        )
        
        # Evaluate this partition
        coherences = []
        shades = []
        
        for shard in partition:
            if not shard:
                continue
            coherences.append(compute_coherence_embed(all_embeddings[shard]))
            shades.append(compute_proxy_shade(shard, all_embeddings, k_clusters=k_clusters))
            
        mean_coh = float(np.mean(coherences))
        max_shade = float(np.max(shades))
        
        print(f"Result for lambda={l_weight:.1f}: Mean Coherence = {mean_coh:.4f}, Max SHADE = {max_shade:.4f}")
        
        frontier.append({
            'lambda': float(l_weight),
            'mean_coherence': mean_coh,
            'max_shade': max_shade
        })
        
    return frontier
