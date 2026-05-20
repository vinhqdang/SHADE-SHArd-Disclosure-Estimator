"""Dataset loading and preprocessing."""
import os
from datasets import load_dataset
from typing import List, Tuple, Dict

def load_medqa(split: str = "train") -> Tuple[List[str], List[Dict]]:
    """
    Load MedQA dataset from HuggingFace.
    Extracts text to form a pseudo-corpus and keeps QA pairs for evaluation.
    
    Returns:
        documents: List of document texts (corpus).
        qa_pairs: List of dicts with 'query' and 'gold_answer'.
    """
    # Using GBaker/MedQA-USMLE-4-options as a standard MedQA source on HF
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    
    documents = []
    qa_pairs = []
    
    for i, item in enumerate(dataset):
        # MedQA has 'question', 'options', 'answer'
        # We construct a document from the question and correct answer context
        options_text = " ".join([f"{k}: {v}" for k, v in item.get('options', {}).items()])
        doc_text = f"Clinical Context: {item.get('question', '')} Options: {options_text}"
        documents.append(doc_text)
        
        qa_pairs.append({
            'query': item.get('question', ''),
            'gold_answer': item.get('answer', ''),
            'id': f"medqa_{i}"
        })
        
    return documents, qa_pairs

def load_legalbench(subset: str = "contract_nli_confidentiality_of_agreement", split: str = "train") -> Tuple[List[str], List[Dict]]:
    """
    Load LegalBench dataset from HuggingFace.
    
    Returns:
        documents: List of document texts (corpus).
        qa_pairs: List of dicts with 'query' and 'gold_answer'.
    """
    dataset = load_dataset("nguha/legalbench", subset, split=split)
    
    documents = []
    qa_pairs = []
    
    for i, item in enumerate(dataset):
        # contract_nli has 'text', 'hypothesis', 'label'
        # Other subsets might have different keys, we adapt based on contract_nli
        text = item.get('text', '')
        if text not in documents:
            documents.append(text)
            
        qa_pairs.append({
            'query': item.get('hypothesis', ''),
            'gold_answer': str(item.get('label', '')),
            'id': f"legalbench_{subset}_{i}",
            'doc_id': documents.index(text)
        })
        
    return documents, qa_pairs

def load_dataset_by_name(name: str, split: str = "train") -> Tuple[List[str], List[Dict]]:
    """Unified dataset loader."""
    if name.lower() == "medqa":
        return load_medqa(split=split)
    elif name.lower() == "legalbench":
        return load_legalbench(split=split)
    elif name.lower() == "synthetic":
        docs, qa, _ = generate_synthetic_corpus()
        return docs, qa
    else:
        raise ValueError(f"Unknown dataset: {name}")

def generate_synthetic_corpus(n_clusters: int = 10, docs_per_cluster: int = 100) -> Tuple[List[str], List[Dict], Dict]:
    """
    Generate a synthetic corpus with known ground-truth disclosure.
    
    Args:
        n_clusters: Number of semantic clusters (topics).
        docs_per_cluster: Number of documents per cluster.
        
    Returns:
        documents: List of document texts.
        qa_pairs: Dummy QA pairs for compatibility.
        ground_truth: Dictionary mapping document index to cluster info for exact SHADE calculation.
    """
    import random
    random.seed(42)
    
    documents = []
    qa_pairs = []
    ground_truth = {'doc_to_cluster': {}, 'cluster_claims': {}}
    
    # Generate abstract claims for each cluster
    for k in range(n_clusters):
        # 100 specific claims per cluster
        ground_truth['cluster_claims'][k] = [f"Claim_C{k}_{i}" for i in range(100)]
        
    # Global claims shared across all clusters (5% overlap)
    global_claims = [f"Claim_Global_{i}" for i in range(20)]
    
    doc_idx = 0
    for k in range(n_clusters):
        cluster_specific_claims = ground_truth['cluster_claims'][k]
        for d in range(docs_per_cluster):
            # A document gets 10 specific claims (80% of its content) and 2 global claims (20%)
            doc_specific = random.sample(cluster_specific_claims, 10)
            doc_global = random.sample(global_claims, 2)
            
            # Form natural-looking sentences so embeddings don't break completely
            text = f"This document discusses topic {k}. It asserts that " + " and ".join(doc_specific) + ". Additionally, it notes that " + " and ".join(doc_global) + "."
            documents.append(text)
            
            qa_pairs.append({
                'query': f"What does document {doc_idx} say about topic {k}?",
                'gold_answer': " ".join(doc_specific),
                'id': f"synth_{doc_idx}",
                'doc_id': doc_idx
            })
            
            ground_truth['doc_to_cluster'][doc_idx] = k
            doc_idx += 1
            
    return documents, qa_pairs, ground_truth
