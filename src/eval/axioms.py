"""Axiom verification (E1)."""
from typing import List, Dict, Callable
import numpy as np

def verify_monotonicity(metric_fn: Callable, full_corpus: List[str], base_shard_indices: List[int], extra_indices: List[int]) -> bool:
    """A1: Monotonicity - D_i subset D_j => SHADE(D_i) <= SHADE(D_j)"""
    score_base = metric_fn(base_shard_indices)
    score_expanded = metric_fn(base_shard_indices + extra_indices)
    return score_base <= score_expanded

def verify_semantic_invariance(metric_fn: Callable, base_shard_indices: List[int], paraphrased_shard_indices: List[int]) -> bool:
    """A2: Semantic invariance - SHADE(pi(D_i)) == SHADE(D_i)"""
    score_base = metric_fn(base_shard_indices)
    score_para = metric_fn(paraphrased_shard_indices)
    # Allow small floating point difference due to embeddings
    return abs(score_base - score_para) < 0.05

def verify_adversarial_grounding(metric_fn: Callable, shard_a: List[int], shard_b: List[int], oracle_shade_fn: Callable) -> bool:
    """A3: Adversarial grounding - M(D_i) >= M(D_j) <=> Oracle(D_i) >= Oracle(D_j)"""
    score_a = metric_fn(shard_a)
    score_b = metric_fn(shard_b)
    
    oracle_a = oracle_shade_fn(shard_a)
    oracle_b = oracle_shade_fn(shard_b)
    
    # If a >= b then oracle_a >= oracle_b, or if a < b then oracle_a < oracle_b
    if score_a >= score_b:
        return oracle_a >= oracle_b - 0.05
    else:
        return oracle_a < oracle_b + 0.05

def verify_composability(metric_fn: Callable, shard_a: List[int], shard_b: List[int]) -> bool:
    """A4: Composability - SHADE(D_i U D_j) <= SHADE(D_i) + SHADE(D_j)"""
    score_a = metric_fn(shard_a)
    score_b = metric_fn(shard_b)
    score_union = metric_fn(list(set(shard_a + shard_b)))
    
    return score_union <= (score_a + score_b + 1e-5)
