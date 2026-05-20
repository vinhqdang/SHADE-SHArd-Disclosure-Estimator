"""Experiment E8: QA Pipeline vs Disclosure Budget."""
import os
import csv
import warnings
warnings.filterwarnings('ignore')
import argparse
from src.utils.data_loader import load_medqa
from src.utils.embeddings import DocumentEmbedder
from src.coshard.graph import build_similarity_graph
from src.coshard.partition import coshard_partition
from src.eval.qa_pipeline import ClientRAG, MultiLLMFusion, compute_exact_match, compute_f1

def run_qa_experiment(n_docs=500, n_shards=3, n_queries=10, lambda_weight=0.5):
    print(f"Running Experiment E8: QA Pipeline (Docs={n_docs}, Shards={n_shards}, lambda={lambda_weight})...")
    
    # 1. Load Data
    docs, qa_pairs = load_medqa()
    
    if n_docs and n_docs < len(docs):
        docs = docs[:n_docs]
        
    queries = qa_pairs[:n_queries]
    
    # 2. Embed Data
    embedder = DocumentEmbedder()
    cache_name = f"e8_medqa_{n_docs}" if n_docs else "e8_medqa_full"
    embeddings = embedder.embed_corpus(docs, cache_name)
    
    # 3. Partition Data using CoShard
    print("\nPartitioning data...")
    graph = build_similarity_graph(embeddings, threshold=0.3, top_k=50)
    shards = coshard_partition(graph, embeddings, n_shards=n_shards, lambda_weight=lambda_weight, max_iterations=5)
    
    # 4. Setup QA Pipeline
    print("\nInitializing Multi-LLM setup...")
    clients = []
    # If we have more shards than default providers, just cycle through them
    providers = ["openai", "anthropic", "gemini"]
    
    for i, shard_indices in enumerate(shards):
        if len(shard_indices) == 0:
            continue
            
        p = providers[i % len(providers)]
        shard_docs = [docs[idx] for idx in shard_indices]
        shard_embs = embeddings[shard_indices]
        clients.append(ClientRAG(p, shard_docs, shard_embs))
        
    fusion = MultiLLMFusion(clients)
    
    # 5. Run Queries
    print(f"\nRunning {len(queries)} queries...")
    results = []
    total_f1 = 0.0
    total_em = 0.0
    
    for i, q in enumerate(queries):
        query_text = q['query']
        gold = q['gold_answer']
        
        # We don't cache query embeddings here for simplicity, just compute on the fly
        q_emb = embedder.embed_corpus([query_text], "e8_temp_queries")[0]
        
        # Get fused answer
        fusion_out = fusion.fuse_answers(query_text, q_emb)
        fused_ans = fusion_out['fused']
        
        em = compute_exact_match(fused_ans, gold)
        f1 = compute_f1(fused_ans, gold)
        
        total_em += em
        total_f1 += f1
        
        results.append({
            "query_id": q['id'],
            "f1": f1,
            "exact_match": em,
            "fused_answer": fused_ans
        })
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(queries)}")
            
    mean_f1 = total_f1 / len(queries)
    mean_em = total_em / len(queries)
    print(f"\nResults for lambda={lambda_weight}:")
    print(f"  Mean F1: {mean_f1:.4f}")
    print(f"  Mean Exact Match: {mean_em:.4f}")
    
    # 6. Save Results
    os.makedirs("experiments/results", exist_ok=True)
    out_file = f"experiments/results/e8_qa_lambda{lambda_weight}.csv"
    
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "f1", "exact_match", "fused_answer"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"E8 Completed. Detailed results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E8 QA Experiment")
    parser.add_argument("--tiny", action="store_true", help="Run on tiny subset for testing")
    args = parser.parse_args()
    
    if args.tiny:
        run_qa_experiment(n_docs=50, n_shards=3, n_queries=2, lambda_weight=0.5)
    else:
        # Full run
        run_qa_experiment(n_docs=1000, n_shards=5, n_queries=50, lambda_weight=0.5)
