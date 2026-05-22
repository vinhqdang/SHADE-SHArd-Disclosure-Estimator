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

def generate_example_graph():
    # 15 real-world medical questions acting as our tiny corpus
    documents = [
        # Cluster 1: Cardiology (highly related)
        "What are the most common symptoms of acute myocardial infarction?",
        "How is heart failure diagnosed using echocardiography?",
        "What is the first-line treatment for hypertension?",
        "Can a high sodium diet lead to cardiovascular disease?",
        "What are the ECG findings in a patient with atrial fibrillation?",
        
        # Cluster 2: Neurology (highly related)
        "What are the early signs of Alzheimer's disease?",
        "How does a stroke affect motor functions?",
        "What is the treatment protocol for acute ischemic stroke?",
        "What causes chronic migraine headaches?",
        "How is Parkinson's disease managed in elderly patients?",
        
        # Cluster 3: Pediatrics / General (highly related)
        "What is the recommended vaccination schedule for infants?",
        "How do you treat a pediatric patient with high fever?",
        "What are the symptoms of asthma in toddlers?",
        "How to manage pediatric eczema?",
        "What causes sudden infant death syndrome?"
    ]
    
    # A realistic user query that falls into the Cardiology topic
    user_query = "What diagnostic tests, such as Echo or ECG, confirm heart failure or infarction?"
    
    labels = {
        0: "MI Symptoms", 1: "Echo for HF", 2: "Hypertension TX", 3: "Diet & CVD", 4: "AFib ECG",
        5: "Alzheimer's", 6: "Stroke Motor", 7: "Ischemic TX", 8: "Migraine", 9: "Parkinson's",
        10: "Infant Vax", 11: "Pediatric Fever", 12: "Toddler Asthma", 13: "Pediatric Eczema", 14: "SIDS"
    }

    print("Embedding toy corpus and query...")
    embedder = DocumentEmbedder()
    embs = embedder.embed_corpus(documents, "toy_example_docs")
    query_emb = embedder.embed_corpus([user_query], "toy_example_query")[0]
    
    print("Building igraph...")
    threshold = 0.5
    ig_graph = build_similarity_graph(embs, threshold=threshold, top_k=5)
    
    print("Running CoShard...")
    # Partition into 3 shards using CoShard with lambda=0.3 to enforce privacy splitting
    shards = coshard_partition(ig_graph, embs, n_shards=3, lambda_weight=0.3, max_iterations=2)
    
    node_to_shard = {}
    for shard_idx, shard_nodes in enumerate(shards):
        for node in shard_nodes:
            node_to_shard[node] = shard_idx
            
    # Calculate routing scores (similarity of query to documents)
    query_sims = [np.dot(query_emb, embs[i]) for i in range(len(embs))]
    top_3_targets = np.argsort(query_sims)[-3:]
            
    print("Rendering NetworkX Graph with Query Routing...")
    G = nx.Graph()
    for i in range(len(documents)):
        G.add_node(i, label=labels[i], shard=node_to_shard[i], is_query=False)
        
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            sim = np.dot(embs[i], embs[j])
            if sim > threshold:
                G.add_edge(i, j, weight=sim)
                
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pos = nx.spring_layout(G, seed=42, k=0.8)
    
    # Add Query Node manually to the center (approx 0,0)
    query_node_id = 999
    pos[query_node_id] = np.array([0.0, 0.0])
    
    shard_colors = {0: '#ff9999', 1: '#66b3ff', 2: '#99ff99'}
    node_colors = [shard_colors[node_to_shard[i]] for i in range(len(documents))]
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    normalized_weights = [(w - threshold) * 10 for w in weights]
    
    # Draw standard document nodes and similarity edges
    nx.draw_networkx_edges(G, pos, edge_color='#cccccc', width=normalized_weights, alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=range(len(documents)), node_color=node_colors, 
                           node_size=2000, edgecolors='black', linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold', font_family='sans-serif', ax=ax)
    
    # Draw Query Node
    nx.draw_networkx_nodes(G, pos, nodelist=[query_node_id], node_color='#ffea00', node_shape='*', 
                           node_size=4000, edgecolors='black', linewidths=2.0, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={query_node_id: "User Query"}, font_size=11, font_weight='bold', ax=ax)
    
    # Draw Routing Arrows from Query to Top 3 Documents
    for target in top_3_targets:
        ax.annotate("",
            xy=pos[target], xycoords='data',
            xytext=pos[query_node_id], textcoords='data',
            arrowprops=dict(arrowstyle="->", color="red", lw=2, shrinkA=25, shrinkB=25, connectionstyle="arc3,rad=0.1")
        )
    
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(color='#ff9999', label='Shard 0'),
        mpatches.Patch(color='#66b3ff', label='Shard 1'),
        mpatches.Patch(color='#99ff99', label='Shard 2'),
        Line2D([0], [0], color='red', lw=2, label='Query Routing')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="Network Legend", fontsize=12, title_fontsize=14)
    
    ax.set_title("CoShard RAG Routing: Query accesses protected topics across multiple Shards", fontsize=16)
    ax.axis('off')
    
    os.makedirs('experiments/results/figures', exist_ok=True)
    out_path = 'experiments/results/figures/example_network.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Graph generated at {out_path}")

if __name__ == "__main__":
    generate_example_graph()
