# SHADE + CoShard: Semantic Disclosure Estimator

This repository contains the official implementation of **SHADE** (SHArd Disclosure Estimator) and **CoShard**, algorithms for optimally partitioning a sensitive knowledge base across distributed multi-LLM providers. 

The goal of this project is to shard a corpus such that each individual LLM provider can answer semantic queries (high utility/coherence), but no single provider (nor any small coalition) can reconstruct the full sensitive corpus (high secrecy/low disclosure).

## Features

- **SHADE (SHArd Disclosure Estimator)**: A formally axiomatised semantic disclosure measure for natural language shards. Includes an LLM-based Oracle form and a tractable, embedding-coverage-based Proxy form.
- **CoShard**: A bi-criterion graph partitioning algorithm based on Leiden community detection, augmented with a SHADE penalty term to organically trace the Pareto frontier between semantic coherence and secrecy.
- **Pipeline and Metrics**: Full evaluation pipelines for dataset loading (HR synthetic, MedQA, LegalBench), embedding generation, proxy/oracle estimation, and retrieval metrics (Recall@5, BERTScore, exact match).
- **Extensive Experiments Suite**: 
  - Axiom Verification (E1)
  - Reconstruction Correlation (E2)
  - Proxy Fidelity (E3)
  - Composition Theorem (E4)
  - Pareto Sweep (E5)
  - No-Free-Lunch Theorem (E6)
  - Scalability Analysis (E7)
  - Multi-LLM RAG QA Simulation (E8)

## Repository Structure

```
shade-coshard/
├── data/                    # Generated datasets and embeddings
├── src/
│   ├── shade/               # SHADE metric algorithms (oracle, proxy, metrics)
│   ├── coshard/             # CoShard graph construction and partitioning
│   ├── eval/                # Evaluation pipelines and adversary scripts
│   └── utils/               # Dataset loading and embeddings helpers
├── experiments/             # Experiment execution scripts (E1 - E8)
├── notebooks/               # Jupyter notebooks for analysis (WIP)
├── verify.py                # Pipeline verification script
└── plan.md                  # Detailed research and implementation plan
```

## Setup & Installation

The project uses Python 3.11+. It is recommended to use a virtual environment (like `conda`):

```bash
# Create and activate environment
conda create -n shade_env python=3.13
conda activate shade_env

# Install dependencies
pip install -r requirements.txt
```

*Note: For the best performance with approximate nearest neighbours and graph logic, the `faiss-cpu`, `igraph`, and `leidenalg` packages are heavily utilized.*

## Running Tests and Experiments

To verify the core pipeline is functioning correctly, you can run the primary verification script:

```bash
python verify.py
```

To run individual experiments, execute the scripts in the `experiments/` directory. For example, to run the Scalability experiment (E7):

```bash
python -u experiments/run_e7_scalability.py
```

### Note on Embeddings
The `DocumentEmbedder` class automatically manages caching. When running experiments, embeddings for synthetic or loaded corpora will be cached to disk inside the `data/` directory to speed up subsequent runs.

## License
MIT License