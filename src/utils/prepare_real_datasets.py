"""Script to prepare and embed real-world datasets for manuscript experiments."""
import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_loader import load_medqa, load_legalbench
from src.utils.embeddings import DocumentEmbedder

def main():
    parser = argparse.ArgumentParser(description="Prepare real-world datasets.")
    parser.add_argument("--cap", type=int, default=None, 
                        help="Maximum number of documents to load per dataset. Default: None (Full dataset)")
    args = parser.parse_args()
    
    embedder = DocumentEmbedder()
    
    # ---------------------------------------------------------
    # 1. MedQA
    # ---------------------------------------------------------
    print("="*60)
    print("Preparing MedQA Dataset...")
    docs, qa_pairs = load_medqa(split="train")
    print(f"Loaded {len(docs)} MedQA documents and {len(qa_pairs)} queries.")
    
    if args.cap and len(docs) > args.cap:
        docs = docs[:args.cap]
        qa_pairs = qa_pairs[:args.cap]
        print(f"Capped to {len(docs)} documents.")
        
    print("\nComputing MedQA Document Embeddings...")
    doc_embs = embedder.embed_corpus(docs, dataset_name="medqa_test")
    print(f"Shape of MedQA Document Embeddings: {doc_embs.shape}")
    
    print("\nComputing MedQA Query Embeddings...")
    queries = [q['query'] for q in qa_pairs]
    query_embs = embedder.embed_corpus(queries, dataset_name="medqa_test_queries")
    print(f"Shape of MedQA Query Embeddings: {query_embs.shape}")
    
    # ---------------------------------------------------------
    # 2. LegalBench
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Preparing LegalBench Dataset...")
    docs, qa_pairs = load_legalbench(split="train")
    print(f"Loaded {len(docs)} LegalBench documents and {len(qa_pairs)} queries.")
    
    if args.cap and len(docs) > args.cap:
        # Note: In LegalBench, multiple qa_pairs can map to the same document.
        # For a simple cap, we truncate docs and then filter qa_pairs.
        docs = docs[:args.cap]
        valid_doc_indices = set(range(len(docs)))
        qa_pairs = [q for q in qa_pairs if q['doc_id'] in valid_doc_indices]
        print(f"Capped to {len(docs)} documents and {len(qa_pairs)} queries.")
        
    print("\nComputing LegalBench Document Embeddings...")
    doc_embs = embedder.embed_corpus(docs, dataset_name="legalbench_test")
    print(f"Shape of LegalBench Document Embeddings: {doc_embs.shape}")
    
    print("\nComputing LegalBench Query Embeddings...")
    queries = [q['query'] for q in qa_pairs]
    query_embs = embedder.embed_corpus(queries, dataset_name="legalbench_test_queries")
    print(f"Shape of LegalBench Query Embeddings: {query_embs.shape}")
    
    print("\n" + "="*60)
    print("All real-world datasets have been successfully prepared and cached!")

if __name__ == "__main__":
    main()
