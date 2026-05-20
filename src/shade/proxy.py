"""Proxy SHADE implementation (Algorithm 2)."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Optional

def _find_optimal_k(embeddings: np.ndarray, k_candidates: List[int] = [10, 20, 30, 40, 50]) -> int:
    """Find optimal K using silhouette score."""
    best_k = k_candidates[0]
    best_score = -1.0
    
    # If we have fewer documents than the max K, adjust candidates
    n_samples = embeddings.shape[0]
    valid_candidates = [k for k in k_candidates if k < n_samples]
    
    if not valid_candidates:
        return max(2, n_samples // 2)

    print(f"Finding optimal K from {valid_candidates}...")
    for k in valid_candidates:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"Selected K={best_k} (silhouette={best_score:.4f})")
    return best_k

def compute_proxy_shade(shard_indices: List[int], all_embeddings: np.ndarray, k_clusters: Optional[int] = None) -> float:
    """
    Compute Proxy SHADE for a shard (Algorithm 2).
    
    Args:
        shard_indices: Indices of the shard documents in the full corpus.
        all_embeddings: Embeddings for the full corpus.
        k_clusters: Number of semantic clusters. If None, optimal K is found via silhouette score.
        
    Returns:
        float: Proxy SHADE score [0, 1]
    """
    if len(shard_indices) == 0:
        return 0.0
        
    if k_clusters is None:
        k_clusters = _find_optimal_k(all_embeddings)
        
    # Step 1: Cluster full corpus
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    # Step 2 & 3: Compute per-cluster coverage and weight by density
    n_total = len(all_embeddings)
    proxy_shade = 0.0
    
    # We can precompute cluster sizes for the full corpus
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters, counts))
    
    # Get the cluster labels for just the shard
    shard_labels = cluster_labels[shard_indices]
    
    for k in unique_clusters:
        # |C_k|
        c_k_size = cluster_sizes[k]
        
        # |{d in D_i : nearest_cluster(phi(d)) == C_k}|
        shard_c_k_size = np.sum(shard_labels == k)
        
        # cov_k
        cov_k = shard_c_k_size / c_k_size if c_k_size > 0 else 0.0
        
        # w_k
        w_k = c_k_size / n_total
        
        proxy_shade += w_k * cov_k
        
    return proxy_shade
