"""Graph construction (Algorithm 4)."""
import numpy as np
import igraph as ig
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

def build_similarity_graph(embeddings: np.ndarray, threshold: float = 0.3, top_k: int = 50) -> ig.Graph:
    """
    Build a k-nearest neighbor similarity graph from embeddings.
    
    Args:
        embeddings: np.ndarray of shape (num_docs, embedding_dim), must be L2 normalized.
        threshold: minimum cosine similarity to form an edge.
        top_k: number of nearest neighbors to retrieve per document.
        
    Returns:
        igraph.Graph: Undirected graph with edge weights representing similarities.
    """
    num_docs, dim = embeddings.shape
    
    k = min(top_k, num_docs)
    
    print(f"Computing cosine similarity matrix for {num_docs} documents...")
    # Calculate exact cosine similarity since FAISS-cpu on mac ARM can be unstable.
    # For N=10k, this takes < 100ms and ~400MB memory, which is highly efficient.
    sim_matrix = cosine_similarity(embeddings)
    
    # Construct edges for the graph
    print("Constructing igraph edges...")
    edges = []
    weights = []
    
    # Set to keep track of added edges to avoid duplicates (igraph is undirected)
    added_edges = set()
    
    for i in range(num_docs):
        # Get indices of top_k + 1 largest elements (including self)
        # partition is faster than full argsort
        if k < num_docs:
            top_indices = np.argpartition(sim_matrix[i], -k - 1)[-k - 1:]
            # Sort the top indices
            top_indices = top_indices[np.argsort(sim_matrix[i, top_indices])][::-1]
        else:
            top_indices = np.argsort(sim_matrix[i])[::-1]
            
        for neighbor_idx in top_indices:
            if i == neighbor_idx:
                continue
                
            sim = sim_matrix[i, neighbor_idx]
            if sim >= threshold:
                u, v = min(i, neighbor_idx), max(i, neighbor_idx)
                if (u, v) not in added_edges:
                    added_edges.add((u, v))
                    edges.append((u, v))
                    weights.append(float(sim))
                    
    print(f"Creating igraph with {len(edges)} edges...")
    graph = ig.Graph(n=num_docs, edges=edges, directed=False)
    graph.es["weight"] = weights
    
    return graph
