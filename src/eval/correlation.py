"""Correlation analysis for proxy vs oracle (E2)."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple

def compute_correlations(proxy_scores: List[float], oracle_scores: List[float]) -> Tuple[float, float]:
    """
    Compute Pearson and Spearman correlations between two sets of scores.
    """
    if len(proxy_scores) < 2:
        return 0.0, 0.0
        
    p_corr, _ = pearsonr(proxy_scores, oracle_scores)
    s_corr, _ = spearmanr(proxy_scores, oracle_scores)
    
    return p_corr, s_corr
