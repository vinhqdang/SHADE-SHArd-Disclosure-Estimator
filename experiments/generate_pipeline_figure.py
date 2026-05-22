import os
import sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.embeddings import DocumentEmbedder
from src.coshard.graph import build_similarity_graph
from src.coshard.partition import coshard_partition

def generate_pipeline_figure():
    documents = [
        "What are the most common symptoms of acute myocardial infarction?",
        "How is heart failure diagnosed using echocardiography?",
        "What is the first-line treatment for hypertension?",
        "Can a high sodium diet lead to cardiovascular disease?",
        "What are the ECG findings in a patient with atrial fibrillation?",
        "What are the early signs of Alzheimer's disease?",
        "How does a stroke affect motor functions?",
        "What is the treatment protocol for acute ischemic stroke?",
        "What causes chronic migraine headaches?",
        "How is Parkinson's disease managed in elderly patients?",
        "What is the recommended vaccination schedule for infants?",
        "How do you treat a pediatric patient with high fever?",
        "What are the symptoms of asthma in toddlers?",
        "How to manage pediatric eczema?",
        "What causes sudden infant death syndrome?"
    ]
    
    user_query = "What diagnostic tests, such as Echo or ECG, confirm heart failure or infarction?"
    
    labels = {
        0: "MI Sym", 1: "Echo HF", 2: "Hyper TX", 3: "Diet CVD", 4: "AFib ECG",
        5: "Alzheimer", 6: "Stroke", 7: "Ischemic", 8: "Migraine", 9: "Parkinson",
        10: "Infant Vax", 11: "Peds Fever", 12: "Asthma", 13: "Eczema", 14: "SIDS"
    }

    embedder = DocumentEmbedder()
    embs = embedder.embed_corpus(documents, "toy_example_docs")
    query_emb = embedder.embed_corpus([user_query], "toy_example_query")[0]
    
    threshold = 0.5
    ig_graph = build_similarity_graph(embs, threshold=threshold, top_k=5)
    
    shards = coshard_partition(ig_graph, embs, n_shards=3, lambda_weight=0.3, max_iterations=2)
    node_to_shard = {}
    for shard_idx, shard_nodes in enumerate(shards):
        for node in shard_nodes:
            node_to_shard[node] = shard_idx
            
    query_sims = [np.dot(query_emb, embs[i]) for i in range(len(embs))]
    top_3_targets = np.argsort(query_sims)[-3:]
            
    G = nx.Graph()
    for i in range(len(documents)):
        G.add_node(i, label=labels[i])
        
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            sim = np.dot(embs[i], embs[j])
            if sim > threshold:
                G.add_edge(i, j, weight=sim)
                
    plt.style.use('seaborn-v0_8-whitegrid')
    # Create 1x4 subplot figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    pos = nx.spring_layout(G, seed=42, k=0.8)
    query_node_id = 999
    pos[query_node_id] = np.array([0.0, 0.0])
    
    shard_colors = {0: '#ff9999', 1: '#66b3ff', 2: '#99ff99'}
    colored_nodes = [shard_colors[node_to_shard[i]] for i in range(len(documents))]
    gray_nodes = ['#dddddd'] * len(documents)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    normalized_weights = [(w - threshold) * 10 for w in weights]
    
    # --- PANEL A: Raw Corpus Space ---
    ax = axes[0]
    nx.draw_networkx_nodes(G, pos, nodelist=range(len(documents)), node_color=gray_nodes, 
                           node_size=1500, edgecolors='black', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[query_node_id], node_color='#ffea00', node_shape='*', 
                           node_size=2500, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={query_node_id: "Query"}, font_size=9, font_weight='bold', ax=ax)
    ax.set_title("A. Semantic Embeddings\n(Corpus + Query Space)", fontsize=14, fontweight='bold')
    ax.axis('off')

    # --- PANEL B: Graph Construction ---
    ax = axes[1]
    nx.draw_networkx_edges(G, pos, edge_color='#cccccc', width=normalized_weights, alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=range(len(documents)), node_color=gray_nodes, 
                           node_size=1500, edgecolors='black', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[query_node_id], node_color='#ffea00', node_shape='*', 
                           node_size=2500, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={query_node_id: "Query"}, font_size=9, font_weight='bold', ax=ax)
    ax.set_title("B. Similarity Graph\n(Connected Clusters)", fontsize=14, fontweight='bold')
    ax.axis('off')

    # --- PANEL C: CoShard Partitioning ---
    ax = axes[2]
    nx.draw_networkx_edges(G, pos, edge_color='#cccccc', width=normalized_weights, alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=range(len(documents)), node_color=colored_nodes, 
                           node_size=1500, edgecolors='black', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[query_node_id], node_color='#ffea00', node_shape='*', 
                           node_size=2500, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={query_node_id: "Query"}, font_size=9, font_weight='bold', ax=ax)
    ax.set_title("C. CoShard Partitioning\n(Shards Break Clusters)", fontsize=14, fontweight='bold')
    ax.axis('off')

    # --- PANEL D: Secure Routing ---
    ax = axes[3]
    nx.draw_networkx_edges(G, pos, edge_color='#cccccc', width=normalized_weights, alpha=0.2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=range(len(documents)), node_color=colored_nodes, 
                           node_size=1500, edgecolors='black', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[query_node_id], node_color='#ffea00', node_shape='*', 
                           node_size=2500, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={query_node_id: "Query"}, font_size=9, font_weight='bold', ax=ax)
    
    for target in top_3_targets:
        ax.annotate("", xy=pos[target], xycoords='data', xytext=pos[query_node_id], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="red", lw=2.5, shrinkA=20, shrinkB=20, connectionstyle="arc3,rad=0.1"))
    
    ax.set_title("D. Secure RAG Routing\n(Query Targets Distributed)", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(color='#ff9999', label='Shard 0'),
        mpatches.Patch(color='#66b3ff', label='Shard 1'),
        mpatches.Patch(color='#99ff99', label='Shard 2'),
        Line2D([0], [0], color='red', lw=2, label='Routing Arrow')
    ]
    # Place legend below the whole figure
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=14, bbox_to_anchor=(0.5, -0.05))
    
    fig.suptitle("CoShard Pipeline: From Semantic Embeddings to Secure RAG Routing", fontsize=20, fontweight='bold', y=1.05)
    
    os.makedirs('experiments/results/figures', exist_ok=True)
    out_path = 'experiments/results/figures/pipeline_overview.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Pipeline figure generated at {out_path}")

if __name__ == "__main__":
    generate_pipeline_figure()
