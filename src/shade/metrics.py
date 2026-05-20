"""Coherence and baseline metrics (Algorithm 3)."""
import numpy as np
from typing import List, Dict, Any

def compute_coherence_embed(shard_embeddings: np.ndarray) -> float:
    """
    Compute within-shard embedding coherence (Option A).
    Calculates the mean cosine similarity of all documents in the shard to the shard's centroid.
    
    Args:
        shard_embeddings: np.ndarray of shape (num_docs_in_shard, embedding_dim)
        
    Returns:
        float: Mean coherence score between [-1, 1]
    """
    if shard_embeddings.shape[0] == 0:
        return 0.0
        
    # Calculate centroid
    mu_i = np.mean(shard_embeddings, axis=0)
    
    # Normalize centroid and embeddings (assuming embeddings are already normalized, 
    # but safe to normalize centroid)
    mu_i_norm = np.linalg.norm(mu_i)
    if mu_i_norm > 0:
        mu_i = mu_i / mu_i_norm
        
    # Compute cosine similarities
    # Since shard_embeddings are already normalized by the embedder, 
    # dot product is equivalent to cosine similarity
    similarities = np.dot(shard_embeddings, mu_i)
    
    return float(np.mean(similarities))

def compute_coherence_retrieval(
    shard_indices: List[int], 
    query_set: List[Dict[str, Any]], 
    all_embeddings: np.ndarray, 
    query_embeddings: np.ndarray, 
    k: int = 5
) -> float:
    """
    Compute retrieval-based coherence (Option B).
    Returns Recall@k within the shard.
    
    Args:
        shard_indices: List of document indices in the shard.
        query_set: List of query dicts containing 'doc_id'.
        all_embeddings: Full corpus embeddings.
        query_embeddings: Embeddings for the queries.
        k: Number of top documents to retrieve.
        
    Returns:
        float: Recall@k score
    """
    if len(shard_indices) == 0 or len(query_set) == 0:
        return 0.0
        
    shard_embs = all_embeddings[shard_indices]
    hits = 0.0
    
    for i, q in enumerate(query_set):
        gold_doc_id = q.get('doc_id')
        
        # Check if the relevant document is even in this shard
        if gold_doc_id not in shard_indices:
            continue
            
        # Retrieve top k within shard
        q_emb = query_embeddings[i]
        # dot product is cosine similarity for normalized vectors
        sims = np.dot(shard_embs, q_emb)
        
        # indices relative to the shard
        top_k_idx_relative = np.argsort(sims)[-k:]
        # map back to full corpus indices
        top_k_idx = [shard_indices[idx] for idx in top_k_idx_relative]
        
        if gold_doc_id in top_k_idx:
            hits += 1.0
            
    # As per algorithm: hits / |Q_eval|
    return hits / len(query_set)
