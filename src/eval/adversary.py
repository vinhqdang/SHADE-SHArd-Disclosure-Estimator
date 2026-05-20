"""LLM reconstruction adversary."""
from typing import List
from src.shade.oracle import compute_oracle_shade

def dummy_oracle_shade(shard_docs: List[str], full_corpus: List[str], M: int = 50) -> float:
    """
    Dummy implementation of Oracle SHADE that doesn't use real LLM API calls,
    so we can run the experiments without API keys.
    Returns a score based on the ratio of the shard size to the full corpus size,
    plus some random noise to simulate 'adversarial' variation.
    """
    import random
    if not full_corpus:
        return 0.0
    
    ratio = len(shard_docs) / len(full_corpus)
    # Simulate a non-linear disclosure curve (small shards disclose less, then it ramps up)
    base_score = ratio ** 1.5 
    
    # Add a little noise
    noise = random.uniform(-0.05, 0.05)
    score = base_score + noise
    
    return max(0.0, min(1.0, score))
