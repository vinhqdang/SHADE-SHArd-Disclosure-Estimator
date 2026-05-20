"""CoShard partitioning algorithm (Algorithm 5)."""
import numpy as np
import igraph as ig
import leidenalg
from typing import List, Dict, Set
from collections import defaultdict

from src.shade.metrics import compute_coherence_embed
from src.shade.proxy import compute_proxy_shade, _find_optimal_k

def coshard_partition(
    graph: ig.Graph, 
    all_embeddings: np.ndarray, 
    n_shards: int, 
    delta: float = 1.0, 
    lambda_weight: float = 0.5,
    max_iterations: int = 15
) -> List[List[int]]:
    """
    CoShard algorithm to partition a corpus into shards balancing coherence and SHADE.
    
    Args:
        graph: Document similarity graph from build_similarity_graph().
        all_embeddings: Full corpus embeddings.
        n_shards: Target number of shards.
        delta: Maximum allowed proxy SHADE per shard.
        lambda_weight: Tradeoff weight (0 = pure SHADE, 1 = pure Coherence).
        max_iterations: Maximum refinement passes over the nodes.
        
    Returns:
        List of lists, where each inner list contains document indices for a shard.
    """
    n_docs = graph.vcount()
    
    print("Phase 1: Leiden initialization...")
    # Modularity maximization
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)
    p_lists = list(partition)
    
    # Adjust number of communities to exactly n_shards
    if len(p_lists) > n_shards:
        # Merge smallest communities
        while len(p_lists) > n_shards:
            p_lists.sort(key=len)
            smallest = p_lists.pop(0)
            p_lists[0].extend(smallest)
    elif len(p_lists) < n_shards:
        # Split largest communities arbitrarily (or random) to meet target
        while len(p_lists) < n_shards:
            p_lists.sort(key=len, reverse=True)
            largest = p_lists.pop(0)
            half = len(largest) // 2
            p_lists.append(largest[:half])
            p_lists.append(largest[half:])
            
    # Create mapping from doc_id to shard index
    doc_to_shard = {}
    for shard_idx, shard_docs in enumerate(p_lists):
        for doc_id in shard_docs:
            doc_to_shard[doc_id] = shard_idx
            
    print(f"Phase 2: SHADE-penalized refinement (lambda={lambda_weight})...")
    
    # Precompute K for proxy SHADE to save time in loop
    print("Precomputing optimal K for proxy SHADE...")
    k_clusters = _find_optimal_k(all_embeddings)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        moves = 0
        
        print(f"  Refinement Iteration {iteration}...")
        for d in range(n_docs):
            current_shard_idx = doc_to_shard[d]
            
            # To save massive compute, only consider shards that are neighbors in the graph
            # as candidate shards.
            neighbor_shards = set(doc_to_shard[n.index] for n in graph.vs[d].neighbors())
            candidate_shards = [s for s in neighbor_shards if s != current_shard_idx]
            
            # If no neighbor shards, maybe consider all shards.
            if not candidate_shards:
                candidate_shards = [s for s in range(n_shards) if s != current_shard_idx]
                
            best_gain = 0.0
            best_target_shard = -1
            
            # Current state values
            current_shard_docs = p_lists[current_shard_idx]
            
            # If current shard has only 1 doc, we don't allow it to become empty
            if len(current_shard_docs) <= 1:
                continue
                
            coh_current = compute_coherence_embed(all_embeddings[current_shard_docs])
            
            current_shard_docs_minus_d = [doc for doc in current_shard_docs if doc != d]
            coh_current_minus_d = compute_coherence_embed(all_embeddings[current_shard_docs_minus_d])
            
            for j in candidate_shards:
                target_shard_docs = p_lists[j]
                
                # Coherence delta
                coh_target = compute_coherence_embed(all_embeddings[target_shard_docs])
                target_shard_docs_plus_d = target_shard_docs + [d]
                coh_target_plus_d = compute_coherence_embed(all_embeddings[target_shard_docs_plus_d])
                
                delta_coh = (coh_target_plus_d - coh_target) - (coh_current - coh_current_minus_d)
                
                # Proxy SHADE delta
                shade_target = compute_proxy_shade(target_shard_docs, all_embeddings, k_clusters=k_clusters)
                shade_target_plus_d = compute_proxy_shade(target_shard_docs_plus_d, all_embeddings, k_clusters=k_clusters)
                
                delta_shade = shade_target_plus_d - shade_target
                
                gain = lambda_weight * delta_coh - (1 - lambda_weight) * delta_shade
                
                if gain > best_gain and shade_target_plus_d <= delta:
                    best_gain = gain
                    best_target_shard = j
                    
            if best_gain > 0:
                # Move d from current_shard to best_target_shard
                p_lists[current_shard_idx].remove(d)
                p_lists[best_target_shard].append(d)
                doc_to_shard[d] = best_target_shard
                improved = True
                moves += 1
                
        print(f"    Made {moves} node swaps.")
        
    return p_lists
