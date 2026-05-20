import sys
from src.utils.data_loader import load_medqa, load_legalbench
from src.utils.embeddings import DocumentEmbedder
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Testing MedQA Loader...")
    # Load a small split or just slice after loading to save time
    # Actually MedQA might take a moment to download, let's let HF handle it
    medqa_docs, medqa_qa = load_medqa("train")
    
    # Just take 50 docs for testing
    medqa_docs_subset = medqa_docs[:50]
    print(f"Loaded {len(medqa_docs)} MedQA docs, testing embeddings on 50...")
    
    embedder = DocumentEmbedder()
    medqa_embs = embedder.embed_corpus(medqa_docs_subset, "medqa_test")
    print(f"MedQA embeddings shape: {medqa_embs.shape}")
    assert medqa_embs.shape == (50, 1024), f"Wrong shape: {medqa_embs.shape}"
    
    print("\nTesting LegalBench Loader (contract_nli)...")
    legal_docs, legal_qa = load_legalbench("contract_nli_confidentiality_of_agreement", "train")
    legal_docs_subset = legal_docs[:50]
    print(f"Loaded {len(legal_docs)} LegalBench docs, testing embeddings on 50...")
    
    legal_embs = embedder.embed_corpus(legal_docs_subset, "legalbench_test")
    print(f"LegalBench embeddings shape: {legal_embs.shape}")
    assert legal_embs.shape == (len(legal_docs_subset), 1024), f"Wrong shape: {legal_embs.shape}"

    print("\nTesting SHADE Modules...")
    from src.shade.metrics import compute_coherence_embed
    from src.shade.proxy import compute_proxy_shade
    
    # Simulate a shard of 10 documents
    shard_indices = list(range(10))
    shard_embs = legal_embs[shard_indices] if len(legal_embs) >= 10 else legal_embs
    
    coherence = compute_coherence_embed(shard_embs)
    print(f"Shard Coherence: {coherence:.4f}")
    assert -1.0 <= coherence <= 1.0, f"Coherence out of bounds: {coherence}"
    
    # Test proxy shade on MedQA embeddings (since we need more docs for clustering to work well)
    print("Computing Proxy SHADE on MedQA subset...")
    # Simulate a shard of 20 docs from the 50 we embedded
    medqa_shard_indices = list(range(20))
    proxy_shade = compute_proxy_shade(medqa_shard_indices, medqa_embs, k_clusters=5)
    print(f"Proxy SHADE: {proxy_shade:.4f}")
    assert 0.0 <= proxy_shade <= 1.0, f"Proxy SHADE out of bounds: {proxy_shade}"

    print("\nTesting CoShard Pipeline...")
    from src.coshard.graph import build_similarity_graph
    from src.coshard.partition import coshard_partition
    
    # We will test on the 50 MedQA embeddings
    print("Building similarity graph...")
    graph = build_similarity_graph(medqa_embs, threshold=0.3, top_k=10)
    print(f"Graph has {graph.vcount()} vertices and {graph.ecount()} edges")
    
    print("Running CoShard partition algorithm (n_shards=3)...")
    shards = coshard_partition(graph, medqa_embs, n_shards=3, max_iterations=2)
    print(f"Generated {len(shards)} shards.")
    total_docs = sum(len(s) for s in shards)
    assert total_docs == len(medqa_docs_subset), f"Lost documents during partitioning: {total_docs} vs {len(medqa_docs_subset)}"
    
    for i, s in enumerate(shards):
        print(f"  Shard {i}: {len(s)} docs")
        
    print("\nTesting QA Pipeline (Mock Mode)...")
    from src.eval.qa_pipeline import ClientRAG, MultiLLMFusion, compute_exact_match, compute_f1
    from src.shade.metrics import compute_coherence_retrieval
    
    # Create mock clients for the 3 shards
    clients = []
    providers = ["openai", "anthropic", "gemini"]
    for i, p in enumerate(providers):
        if i < len(shards):
            # Pass subset of docs and embeddings for this shard
            shard_docs = [medqa_docs_subset[idx] for idx in shards[i]]
            shard_embs = medqa_embs[shards[i]]
            clients.append(ClientRAG(p, shard_docs, shard_embs))
            
    # Test fusion
    fusion = MultiLLMFusion(clients)
    
    # Take first query from medqa
    test_query = medqa_qa[0]['query']
    test_query_emb = embedder.embed_corpus([test_query], "medqa_test_queries")[0]
    
    print(f"Testing Fusion on Query: '{test_query}'")
    results = fusion.fuse_answers(test_query, test_query_emb)
    print(f"Fused Result: {results['fused']}")
    
    # Test metrics
    test_gold = medqa_qa[0]['gold_answer']
    em = compute_exact_match(results['fused'], test_gold)
    f1 = compute_f1(results['fused'], test_gold)
    print(f"Exact Match: {em}, F1: {f1:.4f}")
    
    # Test retrieval coherence
    # Generate query embeddings for all 50 queries
    all_queries = [qa['query'] for qa in medqa_qa[:50]]
    query_embs = embedder.embed_corpus(all_queries, "medqa_test_queries")
    
    recall = compute_coherence_retrieval(shards[0], medqa_qa[:50], medqa_embs, query_embs, k=5)
    print(f"Retrieval Coherence (Recall@5) for Shard 0: {recall:.4f}")

    print("\nAll pipeline tests passed!")

if __name__ == "__main__":
    main()
