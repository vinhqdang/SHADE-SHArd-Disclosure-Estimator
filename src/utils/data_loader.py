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
    else:
        raise ValueError(f"Unknown dataset: {name}")
