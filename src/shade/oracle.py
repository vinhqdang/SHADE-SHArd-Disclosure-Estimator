"""Oracle SHADE implementation (Algorithm 1)."""
import numpy as np
import random
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from bert_score import score as bert_score_fn

def extract_topic_tfidf(doc: str, all_docs: List[str], top_k: int = 5) -> str:
    """
    Extract top keywords from a document using TF-IDF against the corpus.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    # To save time, we might not want to fit on all docs every time,
    # but for Oracle SHADE we assume we have time since it's the oracle.
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # Get index of doc in all_docs (assume it's there or just transform it)
    try:
        doc_idx = all_docs.index(doc)
        doc_vector = tfidf_matrix[doc_idx].toarray()[0]
    except ValueError:
        doc_vector = vectorizer.transform([doc]).toarray()[0]
        
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top K indices
    top_indices = doc_vector.argsort()[-top_k:][::-1]
    keywords = [feature_names[i] for i in top_indices if doc_vector[i] > 0]
    
    return ", ".join(keywords)

def adversary_reconstruct(shard_docs: List[str], topic: str, client: openai.Client, model: str = "gpt-4o-mini") -> str:
    """
    Prompt an LLM adversary to reconstruct a missing document based on the shard context and a topic hint.
    """
    context = "\\n---\\n".join(shard_docs)
    
    prompt = f"""You are an adversary trying to reconstruct sensitive information.
You have access to the following partial knowledge base (shard):
{context}

Your task: Reconstruct a specific missing document from the original corpus.
The missing document is about the following topic/keywords: {topic}

Reconstruct the full text of the missing document as accurately as possible based on inferences from the shard.
Return ONLY the reconstructed text, with no introductory or concluding remarks."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, # Low temperature for more deterministic/factual attempts
        max_tokens=1000
    )
    
    return response.choices[0].message.content.strip()

def compute_oracle_shade(
    shard_docs: List[str], 
    full_corpus: List[str], 
    client: openai.Client, 
    model: str = "gpt-4o-mini",
    M: int = 50
) -> float:
    """
    Compute Oracle SHADE for a shard (Algorithm 1).
    
    Args:
        shard_docs: List of documents in the shard.
        full_corpus: List of all documents in the corpus.
        client: OpenAI client instance.
        model: LLM model to use as adversary.
        M: Maximum number of held-out documents to sample (budget control).
        
    Returns:
        float: Oracle SHADE score
    """
    shard_set = set(shard_docs)
    held_out = [d for d in full_corpus if d not in shard_set]
    
    if len(held_out) == 0:
        return 1.0 # If shard is the full corpus, everything is disclosed
        
    # Sample up to M documents
    if len(held_out) > M:
        sampled_held_out = random.sample(held_out, M)
    else:
        sampled_held_out = held_out
        
    reconstructed_docs = []
    actual_docs = []
    
    for d in sampled_held_out:
        topic = extract_topic_tfidf(d, full_corpus)
        d_hat = adversary_reconstruct(shard_docs, topic, client, model=model)
        reconstructed_docs.append(d_hat)
        actual_docs.append(d)
        
    # SemSim using BERTScore F1
    # Note: bert_score returns (P, R, F1). We want F1.
    _, _, f1_scores = bert_score_fn(reconstructed_docs, actual_docs, lang="en", verbose=False)
    
    return float(f1_scores.mean().item())
