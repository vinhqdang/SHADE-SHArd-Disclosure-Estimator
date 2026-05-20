"""Experiment E5: Pareto Frontier Sweep."""
import os
import csv
import warnings
warnings.filterwarnings('ignore')
import argparse
from src.utils.data_loader import load_medqa
from src.utils.embeddings import DocumentEmbedder
from src.coshard.graph import build_similarity_graph
from src.coshard.pareto import pareto_sweep

def run_pareto_sweep(n_docs=1000, n_shards=10):
    print(f"Running Experiment E5: Pareto Sweep (N={n_docs}, K_shards={n_shards})...")
    
    # 1. Load Data
    docs, _ = load_medqa()
    
    if n_docs and n_docs < len(docs):
        docs = docs[:n_docs]
        
    # 2. Embed Data
    embedder = DocumentEmbedder()
    cache_name = f"e5_medqa_{n_docs}" if n_docs else "e5_medqa_full"
    embeddings = embedder.embed_corpus(docs, cache_name)
    
    # 3. Build Graph
    graph = build_similarity_graph(embeddings, threshold=0.3, top_k=50)
    
    # 4. Run Sweep
    frontier = pareto_sweep(graph, embeddings, n_shards=n_shards)
    
    # 5. Save Results
    os.makedirs("experiments/results", exist_ok=True)
    out_file = f"experiments/results/e5_pareto_N{n_docs}_K{n_shards}.csv"
    
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "mean_coherence", "max_shade"])
        writer.writeheader()
        writer.writerows(frontier)
        
    print(f"E5 Completed. Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E5 Pareto Sweep")
    parser.add_argument("--tiny", action="store_true", help="Run on tiny subset for testing")
    args = parser.parse_args()
    
    if args.tiny:
        run_pareto_sweep(n_docs=50, n_shards=3)
    else:
        # Full run (default 1000 docs to keep time reasonable for now)
        run_pareto_sweep(n_docs=1000, n_shards=10)
