# CoShard Architecture Diagram

This flowchart illustrates the complete CoShard pipeline, demonstrating both the offline privacy-preserving partitioning of the corpus and the online routing of a user query across the distributed shards.

```mermaid
flowchart TD
    %% Define Styles
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef alg fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef query fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000
    classDef router fill:#d1c4e9,stroke:#5e35b1,stroke-width:2px,color:#000
    classDef note fill:#ffffff,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5,color:#555
    
    subgraph Phase 1: Offline Corpus Partitioning
        Corpus[Raw Document Corpus]:::input --> Embed1[Embedding Model]:::process
        Embed1 -->|Vectors| Graph[K-NN Similarity Graph]:::process
        
        Graph --> Leiden[Leiden Modularity Max]:::alg
        Leiden -->|Initial Communities| Refine[SHADE Penalty Refinement]:::alg
        Refine -->|Converged Partition| Shards[Distributed Privacy Shards]:::output
    end
    
    subgraph Phase 2: Online Query Routing (RAG)
        UserQuery((User Query)):::query --> Embed2[Embedding Model]:::process
        Embed2 --> Router{Central Router}:::router
        
        Shards -.->|Centroid Embeddings| Router
        Router == "Top-K Selection" ==> Shards
        
        Shards --> LLM1[Local LLM 1]:::process
        Shards --> LLM2[Local LLM 2]:::process
        
        LLM1 --> Agg[Aggregator]:::process
        LLM2 --> Agg
        Agg --> Answer((Final Answer)):::query
    end
    
    %% Annotations
    Note1["Gain = λΔ(Coh) - (1-λ)Δ(SHADE)"]:::note
    Refine -.-> Note1
```
