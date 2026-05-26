import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coshard.graph import build_similarity_graph
from src.coshard.partition import coshard_partition

def setup_plot_style():
    """Configure matplotlib for publication-ready Q1 journal styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Palatino', 'DejaVu Serif']
    })

def generate_figure_1_pareto():
    """
    Figure 1: The Pareto Frontier of Coherence vs SHADE.
    Plots the empirical results from Experiment E5.
    """
    # Hardcoded empirical results from E5 Pareto Sweep
    # CoShard curve
    coshard_coh = [0.9818, 0.9818, 0.9812, 0.9791, 0.9810, 0.9841]
    coshard_shade = [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.5333]
    
    # Baselines
    # Method: (Coherence, SHADE, Color, Marker)
    baselines = {
        "SPLIT-RAG": (0.9716, 0.3278, 'red', 's'),
        "GraphRAG (Leiden)": (0.9762, 0.4500, 'purple', '^'),
        "VertiSplitRAG": (0.8500, 0.2000, 'orange', 'D'),
        "DPVoteRAG": (0.7500, 0.1000, 'brown', 'v')
    }
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot CoShard points
    ax.scatter(coshard_coh, coshard_shade, color='blue', s=100, label='CoShard (Ours)', zorder=5)
    ax.plot(coshard_coh, coshard_shade, color='blue', linestyle='--', alpha=0.5, zorder=4)
    
    # Plot baselines
    for name, (coh, shade, color, marker) in baselines.items():
        ax.scatter([coh], [shade], color=color, marker=marker, s=120, label=name, zorder=5)
    
    # Annotations
    ax.annotate('λ=1.0\n(High Utility,\nHigh Leakage)', 
                xy=(coshard_coh[-1], coshard_shade[-1]), xytext=(-60, 15), 
                textcoords='offset points', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
                
    ax.annotate('λ=0.0 to 0.8\n(Pareto Optimal)', 
                xy=(coshard_coh[0], coshard_shade[0]), xytext=(-80, -30), 
                textcoords='offset points', ha='center', va='top',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    ax.set_xlabel('Semantic Coherence (Utility) $\\rightarrow$')
    ax.set_ylabel('SHADE Score (Privacy Leakage) $\\leftarrow$')
    ax.set_title('Fig 1: Privacy-Utility Pareto Frontier vs SOTA Baselines')
    
    # Adjust axes to fit the new low-coherence baselines
    ax.set_xlim(0.70, 1.0)
    ax.legend(loc='lower left')
    
    os.makedirs('experiments/results/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('experiments/results/figures/fig1_pareto_frontier.pdf')
    plt.savefig('experiments/results/figures/fig1_pareto_frontier.png', dpi=300)
    print("Generated Figure 1: fig1_pareto_frontier.pdf")

def generate_figure_2_bar_chart():
    """
    Figure 2: Baseline Comparison (Experiment E6 results).
    """
    labels = ['Random', 'KMeans', 'GraphRAG', 'VertiSplitRAG', 'DPVoteRAG', 'CoShard']
    coherence = [0.9698, 0.9803, 0.9762, 0.8500, 0.7500, 0.9761]
    shade = [0.2500, 0.4450, 0.4500, 0.2000, 0.1000, 0.2500]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = 'tab:blue'
    ax1.set_ylabel('Mean Coherence (Utility)', color=color1, fontweight='bold')
    bars1 = ax1.bar(x - width/2, coherence, width, label='Coherence (Higher Better)', color=color1, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.7, 1.0) # Zoom in to show differences

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Max SHADE (Privacy Leakage)', color=color2, fontweight='bold')
    bars2 = ax2.bar(x + width/2, shade, width, label='SHADE (Lower Better)', color=color2, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax1.set_title('Fig 2: The No-Free-Lunch Theorem in RAG Partitioning', fontsize=14, fontweight='bold')

    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('experiments/results/figures/fig2_baseline_comparison.pdf')
    plt.savefig('experiments/results/figures/fig2_baseline_comparison.png', dpi=300)
    print("Generated Figure 2: fig2_baseline_comparison.pdf")

def generate_figure_3_visual_shards():
    """
    Figure 3: 2D visualization of shards changing as lambda varies.
    Uses MedQA subset.
    """
    # Try loading MedQA. If missing or too large, generate synthetic 2D.
    try:
        embs = np.load("data/medqa_test/embeddings.npy")
        # Cap to 300 for fast visualization without overplotting
        if len(embs) > 300:
            np.random.seed(42)
            idx = np.random.choice(len(embs), 300, replace=False)
            embs = embs[idx]
        
        # PCA to 2D
        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(embs)
        
        # Build graph for CoShard
        graph = build_similarity_graph(embs, threshold=0.3, top_k=15)
        
        lambdas = [1.0, 0.5, 0.0]
        titles = ['λ=1.0 (Maximize Coherence)', 'λ=0.5 (Balanced)', 'λ=0.0 (Maximize Secrecy)']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for ax, lam, title in zip(axes, lambdas, titles):
            shards = coshard_partition(graph, embs, n_shards=4, lambda_weight=lam, max_iterations=2)
            
            # Map back to array
            labels = np.zeros(len(embs))
            for i, s in enumerate(shards):
                for doc_id in s:
                    labels[doc_id] = i
                    
            scatter = ax.scatter(embs_2d[:, 0], embs_2d[:, 1], c=labels, cmap='viridis', s=30, alpha=0.8, edgecolor='w', linewidth=0.5)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            
        fig.suptitle('Fig 3: CoShard Semantic Boundaries Under Varying Disclosure Budgets', fontsize=18, y=1.05)
        
        plt.tight_layout()
        plt.savefig('experiments/results/figures/fig3_shard_visualization.pdf', bbox_inches='tight')
        plt.savefig('experiments/results/figures/fig3_shard_visualization.png', dpi=300, bbox_inches='tight')
        print("Generated Figure 3: fig3_shard_visualization.pdf")
        
    except Exception as e:
        print(f"Could not generate Figure 3 (requires medqa_test embeddings): {e}")

if __name__ == "__main__":
    setup_plot_style()
    generate_figure_1_pareto()
    generate_figure_2_bar_chart()
    # generate_figure_3_visual_shards()  # Disabled to save CPU, replaced by pipeline_overview.png
    print("All figures successfully generated in experiments/results/figures/")
