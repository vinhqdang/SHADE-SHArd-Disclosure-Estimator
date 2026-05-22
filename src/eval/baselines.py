"""Baseline partition algorithms for evaluation."""
import numpy as np
import igraph as ig
import leidenalg
from sklearn.metrics.pairwise import cosine_similarity

def run_leiden_vanilla(graph: ig.Graph, n_shards: int):
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


def split_rag_partition(doc_embeddings: np.ndarray, query_embeddings: np.ndarray, n_shards: int, top_k: int = 5):
    """
    Approximation of SPLIT-RAG (Yang et al. 2025).
    Builds a document graph where edge weights represent how often two documents 
    are co-retrieved for the same queries, then runs Leiden.
    """
    n_docs = doc_embeddings.shape[0]
    n_queries = query_embeddings.shape[0]
    
    # 1. Compute doc-query similarity
    # Shape: (n_queries, n_docs)
    sim_matrix = cosine_similarity(query_embeddings, doc_embeddings)
    
    # 2. Find top_k docs for each query
    # Build a co-occurrence matrix
    co_occurrences = np.zeros((n_docs, n_docs))
    
    for q_idx in range(n_queries):
        # Get indices of top_k docs
        top_doc_indices = np.argsort(sim_matrix[q_idx])[::-1][:top_k]
        
        # Increment co-occurrence for all pairs in this top-k
        for i in range(len(top_doc_indices)):
            for j in range(i + 1, len(top_doc_indices)):
                u, v = top_doc_indices[i], top_doc_indices[j]
                co_occurrences[u, v] += 1
                co_occurrences[v, u] += 1
                
    # 3. Build graph
    edges = []
    weights = []
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            if co_occurrences[i, j] > 0:
                edges.append((i, j))
                weights.append(co_occurrences[i, j])
                
    graph = ig.Graph(n=n_docs, edges=edges, directed=False)
    graph.es["weight"] = weights
    
    # 4. Partition using Leiden
    return run_leiden_vanilla(graph, n_shards)
