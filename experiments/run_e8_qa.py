"""Script to run Multi-LLM QA (E8)."""
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import generate_synthetic_corpus
from src.utils.embeddings import DocumentEmbedder
from src.coshard.graph import build_similarity_graph
from src.coshard.partition import coshard_partition
from src.eval.qa_pipeline import ClientRAG, MultiLLMFusion, compute_exact_match, compute_f1

def main():
    print("Generating synthetic corpus for E8...")
    docs, qa_pairs, _ = generate_synthetic_corpus(n_clusters=3, docs_per_cluster=20)
    print(f"Generated {len(docs)} documents.")
    
    print("Computing embeddings...")
    embedder = DocumentEmbedder()
    embs = embedder.embed_corpus(docs, "synthetic_e8")
    
    print("Building similarity graph...")
    graph = build_similarity_graph(embs, threshold=0.4, top_k=5)
    
    # We will test two disclosure budgets implicitly by simulating different partitioning runs
    # For this script we'll just run it once and simulate the effect for demonstration
    print("Partitioning corpus into 3 shards...")
    shards = coshard_partition(graph, embs, n_shards=3, max_iterations=2)
    
    print("Setting up simulated AI providers (OpenAI, Anthropic, Gemini)...")
    providers = ["openai", "anthropic", "gemini"]
    clients = []
    
    for i, p in enumerate(providers):
        if i < len(shards):
            shard_indices = shards[i]
            shard_docs = [docs[idx] for idx in shard_indices]
            shard_embs = embs[shard_indices]
            clients.append(ClientRAG(p, shard_docs, shard_embs))
            
    fusion = MultiLLMFusion(clients)
    
    print("\n--- Running Experiment E8: QA Pipeline Evaluation ---")
    # Take a subset of 10 queries
    test_qas = qa_pairs[:10]
    em_scores = []
    f1_scores = []
    
    for qa in test_qas:
        query = qa['query']
        gold = qa['gold_answer']
        
        # We need the query embedding for retrieval
        q_emb = embedder.embed_corpus([query], "synthetic_e8_queries")[0]
        
        results = fusion.fuse_answers(query, q_emb)
        fused = results['fused']
        
        # Since we use mocks without API keys, we won't get actual gold answers out.
        # But we compute the metrics to show the pipeline executes successfully.
        em = compute_exact_match(fused, gold)
        f1 = compute_f1(fused, gold)
        
        em_scores.append(em)
        f1_scores.append(f1)
        
    print(f"Evaluated {len(test_qas)} queries.")
    print(f"Average Exact Match: {np.mean(em_scores):.4f}")
    print(f"Average F1 Score:    {np.mean(f1_scores):.4f}")
    print("\nExperiment E8 completed successfully.")

if __name__ == "__main__":
    main()
