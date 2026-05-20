import sys
from src.utils.data_loader import load_medqa, load_legalbench
from src.utils.embeddings import DocumentEmbedder

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

    print("\nAll pipeline tests passed!")

if __name__ == "__main__":
    main()
