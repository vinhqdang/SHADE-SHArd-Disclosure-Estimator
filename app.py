import os
import sys
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.coshard.graph import build_similarity_graph
from src.coshard.partition import coshard_partition
from src.shade.proxy import compute_proxy_shade, _find_optimal_k
from src.shade.metrics import compute_coherence_embed
from src.eval.baselines import run_leiden_vanilla

st.set_page_config(page_title="CoShard Interactive Demo", layout="wide")

@st.cache_data
def load_embeddings(dataset_name):
    path = f"data/{dataset_name}/embeddings.npy"
    if not os.path.exists(path):
        st.error(f"Embeddings not found at {path}. Please run preparation scripts.")
        return None
    return np.load(path)

@st.cache_data
def compute_pca(embs):
    pca = PCA(n_components=2)
    return pca.fit_transform(embs)

@st.cache_data
def get_optimal_k(embs):
    return _find_optimal_k(embs)

# UI Setup
st.title("CoShard & SHADE: Privacy-Utility Pareto Frontier Explorer")
st.markdown("Explore how the CoShard algorithm partitions a document corpus by trading off semantic **Coherence** vs **SHADE** (disclosure risk).")

# Sidebar
st.sidebar.header("Configuration")
dataset = st.sidebar.selectbox(
    "Select Dataset", 
    ["medqa_test", "synthetic_e6", "synthetic_e5", "legalbench_test"]
)

algorithm = st.sidebar.selectbox(
    "Partitioning Algorithm",
    ["CoShard", "Leiden-Vanilla", "Random"]
)

n_shards = st.sidebar.slider("Number of Shards", 2, 10, 4)

lambda_val = 0.5
if algorithm == "CoShard":
    lambda_val = st.sidebar.slider("Lambda (Coherence Weight)", 0.0, 1.0, 0.5, 0.1, 
                                   help="0.0 = Pure Secrecy (Random-like), 1.0 = Pure Coherence (Leiden-like)")

# Load Data
embs = load_embeddings(dataset)
if embs is not None:
    n_docs = len(embs)
    st.sidebar.write(f"Loaded {n_docs} documents.")
    
    with st.spinner("Reducing dimensionality for visualization..."):
        embs_2d = compute_pca(embs)
    
    with st.spinner("Computing partition..."):
        if algorithm == "CoShard" or algorithm == "Leiden-Vanilla":
            # Build graph
            graph = build_similarity_graph(embs, threshold=0.3, top_k=15)
            if algorithm == "CoShard":
                shards = coshard_partition(graph, embs, n_shards=n_shards, lambda_weight=lambda_val, max_iterations=2)
            else:
                shards = run_leiden_vanilla(graph, n_shards)
        else:
            # Random
            indices = list(range(n_docs))
            import random
            random.seed(42)
            random.shuffle(indices)
            chunk_size = n_docs // n_shards
            shards = [indices[i*chunk_size : (i+1)*chunk_size] for i in range(n_shards)]
            if n_docs % n_shards != 0:
                shards[-1].extend(indices[n_shards*chunk_size:])

    # Map shards back to labels
    labels = np.zeros(n_docs, dtype=int)
    for i, shard in enumerate(shards):
        for doc_id in shard:
            labels[doc_id] = i

    # Compute metrics
    with st.spinner("Computing SHADE and Coherence metrics..."):
        k_clusters = get_optimal_k(embs)
        coherences = []
        shade_scores = []
        for shard in shards:
            if not shard:
                continue
            coh = compute_coherence_embed(embs[shard])
            shade = compute_proxy_shade(shard, embs, k_clusters)
            coherences.append(coh)
            shade_scores.append(shade)
        
        mean_coh = np.mean(coherences) if coherences else 0.0
        max_shade = np.max(shade_scores) if shade_scores else 1.0

    # Display Metrics
    col1, col2 = st.columns(2)
    col1.metric("Mean Coherence (Higher is better utility)", f"{mean_coh:.4f}")
    col2.metric("Max SHADE (Lower is better privacy)", f"{max_shade:.4f}", delta_color="inverse")

    # Plot
    import pandas as pd
    df = pd.DataFrame({
        'PCA1': embs_2d[:, 0],
        'PCA2': embs_2d[:, 1],
        'Shard': [f"Shard {l}" for l in labels]
    })
    
    fig = px.scatter(
        df, x='PCA1', y='PCA2', color='Shard', 
        title=f"2D Document Embeddings ({algorithm})",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Shard Breakdown")
    breakdown_df = pd.DataFrame({
        "Shard": [f"Shard {i}" for i in range(len(shards))],
        "Size": [len(s) for s in shards],
        "Coherence": coherences,
        "SHADE": shade_scores
    })
    st.dataframe(breakdown_df, use_container_width=True)
