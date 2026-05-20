"""Experiment E1: Axiom Verification."""
import os
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from src.utils.embeddings import DocumentEmbedder
from src.shade.proxy import compute_proxy_shade

def test_axioms():
    print("Running Experiment E1: Axiom Verification...")
    
    # 1. Create a controlled synthetic corpus
    corpus = [
        "The company revenue grew by 20% in Q3 due to new enterprise sales.",
        "Our Q3 financial report shows a 20 percent increase in revenue driven by B2B growth.", # Paraphrase of 0
        "The engineering team deployed the new Kubernetes cluster yesterday.",
        "We are migrating to a new k8s infrastructure to handle increased load.", # Paraphrase of 2
        "Marketing spent $50k on the recent ad campaign.",
        "The new advertisement budget was fifty thousand dollars.", # Paraphrase of 4
        "Office supplies are running low, please order more pens.",
        "We need to restock the stationery closet with pens and paper.", # Paraphrase of 6
        "The CEO announced her resignation effective next month.",
        "Our chief executive officer will be stepping down in 30 days." # Paraphrase of 8
    ]
    
    embedder = DocumentEmbedder()
    embeddings = embedder.embed_corpus(corpus, "e1_synthetic")
    
    results = []
    
    # Pre-compute K
    from src.shade.proxy import _find_optimal_k
    k_opt = 5 # Force K for this tiny 10-document corpus
    
    # --- A1: Monotonicity ---
    # S(A) <= S(A U {d})
    shard_A = [0, 2]
    shard_A_plus = [0, 2, 4]
    
    shade_A = compute_proxy_shade(shard_A, embeddings, k_clusters=k_opt)
    shade_A_plus = compute_proxy_shade(shard_A_plus, embeddings, k_clusters=k_opt)
    
    a1_pass = shade_A <= shade_A_plus
    results.append({
        "axiom": "A1_Monotonicity",
        "description": "S(A) <= S(A U {d})",
        "values": {"S(A)": shade_A, "S(A U {d})": shade_A_plus},
        "passed": bool(a1_pass)
    })
    
    # --- A2: Semantic Invariance ---
    # S(A U {d}) \approx S(A U {d'}) where d' is paraphrase of d
    shard_base = [2, 4]
    doc_d = [0]
    doc_d_prime = [1] # Paraphrase of 0
    
    shade_base_d = compute_proxy_shade(shard_base + doc_d, embeddings, k_clusters=k_opt)
    shade_base_d_prime = compute_proxy_shade(shard_base + doc_d_prime, embeddings, k_clusters=k_opt)
    
    # Using a small tolerance due to embedding differences
    a2_pass = abs(shade_base_d - shade_base_d_prime) < 0.1
    results.append({
        "axiom": "A2_Semantic_Invariance",
        "description": "S(A U {d}) approx S(A U {d_prime})",
        "values": {"S(A U {d})": shade_base_d, "S(A U {d_prime})": shade_base_d_prime},
        "passed": bool(a2_pass)
    })
    
    # --- A3: Adversarial Grounding ---
    # Exposing the whole corpus = 1.0, exposing nothing = 0.0
    shard_all = list(range(10))
    shard_none = []
    
    shade_all = compute_proxy_shade(shard_all, embeddings, k_clusters=k_opt)
    shade_none = compute_proxy_shade(shard_none, embeddings, k_clusters=k_opt)
    
    a3_pass = (shade_all == 1.0) and (shade_none == 0.0)
    results.append({
        "axiom": "A3_Adversarial_Grounding",
        "description": "S(D) = 1, S(empty) = 0",
        "values": {"S(D)": shade_all, "S(empty)": shade_none},
        "passed": bool(a3_pass)
    })
    
    # --- A4: Composability ---
    # S(A U B) <= S(A) + S(B)
    shard_A = [0, 1, 2]
    shard_B = [2, 3, 4]
    shard_A_union_B = list(set(shard_A + shard_B))
    
    shade_A = compute_proxy_shade(shard_A, embeddings, k_clusters=k_opt)
    shade_B = compute_proxy_shade(shard_B, embeddings, k_clusters=k_opt)
    shade_union = compute_proxy_shade(shard_A_union_B, embeddings, k_clusters=k_opt)
    
    a4_pass = shade_union <= (shade_A + shade_B)
    results.append({
        "axiom": "A4_Composability",
        "description": "S(A U B) <= S(A) + S(B)",
        "values": {"S(A U B)": shade_union, "S(A) + S(B)": shade_A + shade_B},
        "passed": bool(a4_pass)
    })
    
    # Save results
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/e1_axioms.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"E1 Completed. Passed {sum(1 for r in results if r['passed'])}/4 axioms.")
    for r in results:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['axiom']}")

if __name__ == "__main__":
    test_axioms()
