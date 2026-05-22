# CoShard Architecture Diagram

This flowchart illustrates the conceptual pipeline of the CoShard algorithm, from raw text ingestion to final privacy-preserving shards.

```mermaid
flowchart TD
    %% Define Styles
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef alg fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef note fill:#ffffff,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5,color:#555
    
    %% Nodes
    Corpus[Raw Document Corpus]:::input
    Embed[Embedding Model<br/>(e.g., BGE-Large)]:::process
    SimMatrix[Cosine Similarity Matrix]:::process
    Graph[K-NN Similarity Graph<br/>(Edges = Similarity > τ)]:::process
    
    subgraph CoShard Algorithm
        Leiden[Phase 1: Leiden Modularity<br/>Maximize Semantic Coherence]:::alg
        Refine[Phase 2: SHADE Penalty Refinement<br/>Node Swapping via Gain Function]:::alg
    end
    
    Shards[Final Output Shards<br/>Balanced Utility & Privacy]:::output
    
    %% Edges
    Corpus --> Embed
    Embed -->|Vectors| SimMatrix
    SimMatrix -->|Thresholding| Graph
    
    Graph --> Leiden
    Leiden -->|Initial Communities| Refine
    Refine -->|Converged Partition| Shards
    
    %% Annotations
    Note1["Gain = λΔ(Coh) - (1-λ)Δ(SHADE)"]:::note
    Refine -.-> Note1
```
