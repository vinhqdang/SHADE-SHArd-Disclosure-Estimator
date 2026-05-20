# Research & Development Plan
## SHADE + CoShard: Semantic Disclosure Estimation and Privacy-Utility Optimal Corpus Partitioning for Distributed Multi-LLM Query Answering

**Target journal:** Springer Machine Learning  
**Submission type:** Original Research Article  
**Estimated length:** 30–40 pages (including appendices)  
**Development timeline:** 16 weeks

---

## 1. Paper Overview

### 1.1 Problem Statement

Distributing a sensitive knowledge base across multiple AI providers — so each can answer queries yet no single provider nor any small coalition can reconstruct the full corpus — requires two things that do not yet exist:

1. A formal, computable measure of how much a shard reveals (SHADE)
2. A partitioning algorithm that jointly optimises answerability and secrecy (CoShard)

### 1.2 Contributions

| # | Contribution | Type |
|---|---|---|
| C1 | Four axioms for shard semantic disclosure and proof that no existing measure satisfies all four | Theoretical |
| C2 | SHADE (SHArd Disclosure Estimator): oracle and proxy forms | Theoretical + Algorithmic |
| C3 | Composition theorem: coalition disclosure scales sublinearly | Theoretical |
| C4 | No-free-lunch theorem: coherence and secrecy cannot be simultaneously maximised | Theoretical |
| C5 | CoShard: Leiden-based bi-criterion partitioning algorithm | Algorithmic |
| C6 | Empirical Pareto frontier on three sensitive-domain corpora | Empirical |

### 1.3 Abstract

Distributing a sensitive knowledge base across multiple AI providers — so that each can answer queries yet no single provider, nor any small coalition, can reconstruct the full corpus — requires both a principled measure of what a data shard reveals and a partitioning algorithm that controls it. Existing work addresses neither jointly: corpus sharding methods optimise for retrieval efficiency or semantic coherence without any secrecy objective, while information-leakage measures from the privacy literature are either intractable for natural language, lexically shallow, or bound to individual-record protection rather than corpus-level reconstruction. We make two contributions. First, we propose SHADE (SHArd Disclosure Estimator), the first formally axiomatised semantic disclosure measure for natural language shards, defined operationally via an adversarial reconstruction game and satisfying four necessary axioms — monotonicity, semantic invariance, adversarial grounding, and composability — that we prove no existing measure satisfies simultaneously. We provide an oracle form using an LLM adversary as ground truth and a tractable proxy form based on embedding-space coverage, and prove a composition theorem showing that coalition disclosure scales sublinearly with coalition size. Second, we formalise corpus sharding as a bi-criterion graph partitioning problem — maximising within-shard semantic coherence subject to a SHADE bound — and prove a no-free-lunch theorem establishing that coherence and secrecy cannot be simultaneously maximised beyond a threshold determined by the corpus's intrinsic semantic dimensionality. We propose CoShard, a Leiden-based partitioning algorithm augmented with a SHADE penalty term, which efficiently traces the Pareto frontier between the two objectives. Experiments on HR, medical, and legal corpora demonstrate that CoShard Pareto-dominates random, vertical, and question-driven baselines, that SHADE predicts empirical LLM reconstruction success significantly better than all existing proxies, and that multi-LLM answer quality degrades gracefully as the disclosure budget tightens.

---

## 2. Paper Structure

```
1. Introduction
2. Related Work
   2.1 Corpus partitioning for RAG
   2.2 Information leakage measures
   2.3 Privacy-utility tradeoffs in NLP
3. Problem Formulation
   3.1 Setup and notation
   3.2 The answerability requirement
   3.3 The secrecy requirement
4. SHADE: SHArd Disclosure Estimator
   4.1 Failure analysis of existing measures
   4.2 Axioms for shard disclosure
   4.3 Oracle SHADE
   4.4 Proxy SHADE
   4.5 Theoretical properties
5. CoShard: Bi-Criterion Corpus Partitioning
   5.1 Graph construction
   5.2 Objective formulation
   5.3 The CoShard algorithm
   5.4 Theoretical guarantees
6. Experiments
   6.1 Datasets and setup
   6.2 SHADE evaluation
   6.3 CoShard evaluation
   6.4 End-to-end multi-LLM QA evaluation
7. Discussion and Limitations
8. Conclusion
Appendix A: Proofs
Appendix B: Implementation details
Appendix C: Additional experiments
```

---

## 3. Formal Setup and Notation

### 3.1 Definitions

| Symbol | Definition |
|---|---|
| $\mathcal{D}$ | Full corpus of $N$ documents |
| $\mathcal{D}_i$ | Shard $i$, a subset of $\mathcal{D}$ |
| $\mathcal{P} = \{\mathcal{D}_1, \ldots, \mathcal{D}_n\}$ | Partitioning of $\mathcal{D}$ into $n$ shards |
| $\phi(d) \in \mathbb{R}^m$ | Embedding of document $d$ (e.g. text-embedding-3-large) |
| $G = (V, E, w)$ | Document similarity graph; $w(d_i, d_j) = \cos(\phi(d_i), \phi(d_j))$ |
| $\text{Coh}(\mathcal{D}_i)$ | Within-shard semantic coherence |
| $\text{SHADE}(\mathcal{D}_i)$ | Shard disclosure score |
| $\mathcal{A}^*$ | Optimal adversarial reconstructor |
| $k$ | Threshold: minimum coalition size needed for reconstruction |
| $\delta$ | Maximum allowed disclosure per shard |

### 3.2 The Bi-Criterion Optimisation Problem

$$\max_{\mathcal{P}} \sum_{i=1}^{n} \text{Coh}(\mathcal{D}_i) \quad \text{subject to} \quad \text{SHADE}(\mathcal{D}_i) \leq \delta \quad \forall i \in [n]$$

---

## 4. SHADE Algorithm

### 4.1 Axioms (to be formally proved)

**A1 — Monotonicity**
$$\mathcal{D}_i \subseteq \mathcal{D}_j \implies \text{SHADE}(\mathcal{D}_i) \leq \text{SHADE}(\mathcal{D}_j)$$

**A2 — Semantic invariance**
For any meaning-preserving paraphrase $\pi$:
$$\text{SHADE}(\pi(\mathcal{D}_i)) = \text{SHADE}(\mathcal{D}_i)$$

**A3 — Adversarial grounding**
$$\text{SHADE}(\mathcal{D}_i) \geq \text{SHADE}(\mathcal{D}_j) \iff \mathbb{E}[\text{SemSim}(\mathcal{A}^*(\mathcal{D}_i), \mathcal{D} \setminus \mathcal{D}_i)] \geq \mathbb{E}[\text{SemSim}(\mathcal{A}^*(\mathcal{D}_j), \mathcal{D} \setminus \mathcal{D}_j)]$$

**A4 — Composability**
$$\text{SHADE}(\mathcal{D}_i \cup \mathcal{D}_j) \leq \text{SHADE}(\mathcal{D}_i) + \text{SHADE}(\mathcal{D}_j)$$

### 4.2 Oracle SHADE

```
Algorithm 1: Oracle SHADE
--------------------------
Input:  Shard D_i, full corpus D, LLM adversary A, similarity fn SemSim
Output: SHADE_oracle(D_i)

1. H = D \ D_i                          // held-out documents
2. scores = []
3. for each d in H (sampled up to M docs):
4.     d_hat = A(D_i, query="Reconstruct a document about: " + topic(d))
5.     scores.append(SemSim(d_hat, d))
6. return mean(scores)
```

**Implementation notes:**
- Adversary $\mathcal{A}$: GPT-4o with full shard in context, prompted to reconstruct
- `topic(d)`: extract 3–5 keywords from $d$ using TF-IDF (given to adversary as a hint — worst-case scenario)
- `SemSim`: BERTScore F1 + atomic claim overlap (two variants)
- $M = 50$ held-out documents per shard (budget-controlled)

### 4.3 Proxy SHADE

```
Algorithm 2: Proxy SHADE
-------------------------
Input:  Shard D_i, full corpus D, embeddings phi
Output: SHADE_proxy(D_i)

1. Phi_i = {phi(d) : d in D_i}          // shard embeddings
2. Phi   = {phi(d) : d in D}            // full corpus embeddings

3. // Step 1: Cluster full corpus into K semantic clusters
4. C_1,...,C_K = KMeans(Phi, K)

5. // Step 2: Compute per-cluster coverage
6. for k in 1..K:
7.     cov_k = |{d in D_i : nearest_cluster(phi(d)) == C_k}| / |C_k|

8. // Step 3: Weight by cluster density (larger clusters reveal more)
9. w_k = |C_k| / |D|

10. return sum(w_k * cov_k for k in 1..K)
```

**Implementation notes:**
- $K$: determined by silhouette score on $\Phi$ (typically 20–50 for domain corpora)
- Embeddings: `text-embedding-3-large` (OpenAI) or `bge-large-en-v1.5` (open-source)
- Runtime: $O(NK)$ after one-time embedding computation

### 4.4 Coherence Score

```
Algorithm 3: Coherence Score
-----------------------------
Input:  Shard D_i, embeddings phi, query set Q_eval
Output: Coh(D_i)

Option A (embedding-based, no queries needed):
1. Phi_i = {phi(d) : d in D_i}
2. mu_i  = mean(Phi_i)
3. return mean(cos(phi(d), mu_i) for d in D_i)

Option B (retrieval-based, requires query set):
1. hits = 0
2. for q in Q_eval:
3.     relevant = gold_relevant_docs(q) ∩ D_i
4.     retrieved = top_k_retrieve(q, D_i, k=5)
5.     hits += |relevant ∩ retrieved| / |relevant|
6. return hits / |Q_eval|               // Recall@5 within shard
```

Use Option A during partitioning (no query labels needed), Option B for evaluation.

---

## 5. CoShard Algorithm

### 5.1 Graph Construction

```
Algorithm 4: Document Similarity Graph
---------------------------------------
Input:  Corpus D, threshold theta (default 0.3)
Output: Graph G = (V, E, w)

1. Compute phi(d) for all d in D
2. V = D
3. E = {}
4. for each pair (d_i, d_j):
5.     s = cos(phi(d_i), phi(d_j))
6.     if s >= theta:
7.         E.add((d_i, d_j, s))
8. return G
```

**Efficiency note:** Use approximate nearest neighbours (FAISS HNSW) to avoid $O(N^2)$ — compute top-50 neighbours per document only. For $N > 100K$, use MiniBatch k-means for initial clustering before graph construction.

### 5.2 CoShard: Main Algorithm

```
Algorithm 5: CoShard
---------------------
Input:  Graph G=(V,E,w), n_shards, delta (max disclosure), lambda (tradeoff weight)
Output: Partition P = {D_1,...,D_n}

Phase 1 — Initialisation:
1. P_0 = Leiden(G)                      // standard Leiden community detection
2. if |P_0| > n_shards:
3.     P_0 = merge_smallest(P_0, n_shards)

Phase 2 — SHADE-penalised refinement:
4. improved = True
5. while improved:
6.     improved = False
7.     for each document d in V:
8.         current_shard  = shard_of(d, P)
9.         for each candidate shard D_j != current_shard:
10.            delta_coh   = Coh(D_j + {d}) - Coh(D_j)
11.                        - (Coh(current_shard) - Coh(current_shard - {d}))
12.            delta_shade = SHADE_proxy(D_j + {d}) - SHADE_proxy(D_j)
13.            gain        = lambda * delta_coh - (1 - lambda) * delta_shade
14.            if gain > 0 and SHADE_proxy(D_j + {d}) <= delta:
15.                move d from current_shard to D_j
16.                improved = True
17.                break

Phase 3 — Pareto sweep (optional, for evaluation):
18. for lambda in [0.0, 0.1, ..., 1.0]:
19.     P_lambda = CoShard(G, n_shards, delta=inf, lambda=lambda)
20.     record (mean_Coh(P_lambda), max_SHADE(P_lambda))
21. return Pareto_front

return P
```

**Complexity:** $O(T \cdot N \cdot n)$ per iteration where $T$ = number of Leiden iterations. In practice converges in 5–15 outer iterations.

### 5.3 Theorems to Prove

**Theorem 1 — Axiom satisfaction (SHADE)**
SHADE_oracle and SHADE_proxy both satisfy axioms A1–A4.

**Theorem 2 — Existing measures fail**
For each existing measure $M \in$ {Jaccard, cosine distance, MI, DP-$\varepsilon$}, there exists a corpus and pair of shards such that $M$ violates at least one axiom in $\{A1, A2, A3, A4\}$.

**Theorem 3 — Composition**
For any coalition $S \subset [n]$ with $|S| = m < k$:
$$\text{SHADE}\left(\bigcup_{i \in S} \mathcal{D}_i\right) \leq m \cdot \max_{i \in S} \text{SHADE}(\mathcal{D}_i)$$

**Theorem 4 — No-free-lunch**
For any partitioning $\mathcal{P}$ of $\mathcal{D}$ into $n$ equal-size shards with $\text{mean\_Coh}(\mathcal{P}) \geq \tau$, at least one shard satisfies:
$$\text{SHADE}(\mathcal{D}_i) \geq f(\tau, n, H_s(\mathcal{D}))$$
where $H_s(\mathcal{D})$ is the semantic entropy of $\mathcal{D}$ (to be defined in terms of embedding cluster distribution).

**Theorem 5 — Proxy fidelity**
$$|\text{SHADE\_proxy}(\mathcal{D}_i) - \text{SHADE\_oracle}(\mathcal{D}_i)| \leq \epsilon(K, N, m)$$
where $\epsilon$ decreases with $K$ (number of clusters) and $m$ (embedding dimension).

---

## 6. Datasets

### 6.1 Primary Datasets

| Dataset | Domain | Size | Sensitivity | Source |
|---|---|---|---|---|
| **HR-Corpus** | Human Resources | ~5K docs | High (salaries, reviews, contracts) | Synthetic generation via GPT-4o from real HR templates |
| **MedQA** | Medical | ~10K docs | High (patient records, clinical notes) | Jin et al. 2021; original PubMed subset |
| **LegalBench** | Legal | ~8K docs | Medium (contracts, case summaries) | Guha et al. 2023 |

### 6.2 Synthetic Control Dataset

Generate a corpus with **known ground-truth disclosure** for controlled axiom verification:

```
Synthetic corpus construction:
1. Define K=10 semantic clusters (topics), each with 100 documents
2. Documents within a cluster share 80% of their "atomic claims"
3. Documents across clusters share 5% of their atomic claims
4. Ground-truth SHADE(D_i) = fraction of cross-cluster claims exposed by D_i
```

This gives exact numerical ground truth to measure proxy fidelity (Theorem 5).

### 6.3 Query Sets

| Dataset | Query source | # Queries |
|---|---|---|
| HR-Corpus | Manually crafted HR questions (benefits, policy, payroll) | 200 |
| MedQA | Original MedQA test questions | 1,273 |
| LegalBench | Original LegalBench tasks | 500 |

---

## 7. Baselines

| Baseline | Description | Optimises |
|---|---|---|
| **Random** | Uniform random assignment of documents to shards | Neither |
| **Vertical** | Partition by document section type / metadata attribute | Structure only |
| **SPLIT-RAG** | Question-driven graph partitioning (Yang et al. 2025) | Coherence only |
| **METIS** | Balanced graph partitioning minimising edge-cut | Edge-cut only |
| **Leiden-vanilla** | Standard Leiden without SHADE penalty | Modularity only |

---

## 8. Evaluation Plan

### 8.1 SHADE Evaluation (Section 6.2)

#### Experiment E1 — Axiom verification
For each axiom A1–A4 and each measure $M$ in {Jaccard, cosine, MI, DP-$\varepsilon$, SHADE\_proxy, SHADE\_oracle}:
- Construct synthetic scenarios where axiom should hold or fail
- Report pass/fail for each (axiom, measure) pair as a table

**Expected result:** Only SHADE satisfies all four.

#### Experiment E2 — Reconstruction correlation
- For each shard in each dataset, compute SHADE\_oracle by running GPT-4o adversary
- Compute all baseline measures on same shards
- Report Pearson $r$ and Spearman $\rho$ between each measure and Oracle SHADE

**Target:** SHADE\_proxy achieves $r > 0.85$; baselines below $0.60$.

#### Experiment E3 — Proxy fidelity
- Vary $K$ (number of clusters: 10, 20, 50, 100) and measure $|\text{SHADE\_proxy} - \text{SHADE\_oracle}|$
- Plot error vs $K$ to find knee point; report recommended $K$ per corpus size

#### Experiment E4 — Composition theorem validation
- Form coalitions of size $m = 1, 2, 3$ shards
- Measure actual $\text{SHADE}(\bigcup_{i \in S} \mathcal{D}_i)$ vs theoretical bound $m \cdot \max_i \text{SHADE}(\mathcal{D}_i)$
- Verify bound holds and is tight enough to be useful

### 8.2 CoShard Evaluation (Section 6.3)

#### Experiment E5 — Pareto frontier
- For each dataset, run CoShard with $\lambda \in \{0.0, 0.1, \ldots, 1.0\}$
- Run all baselines
- Plot (mean\_Coh, max\_SHADE) for all configurations
- Report hypervolume indicator of each method's Pareto front

**Target:** CoShard achieves strictly higher hypervolume than all baselines.

#### Experiment E6 — No-free-lunch theorem validation
- Empirically confirm the coherence-SHADE tradeoff boundary
- Overlay theoretical bound $f(\tau, n, H_s(\mathcal{D}))$ on empirical scatter plot

#### Experiment E7 — Scalability
- Run CoShard on corpus sizes $N \in \{1K, 5K, 10K, 50K\}$
- Report wall-clock time and memory usage
- Compare against METIS and Leiden-vanilla

### 8.3 End-to-End Multi-LLM QA Evaluation (Section 6.4)

#### Experiment E8 — Answer quality vs disclosure budget
Setup:
- Partition each corpus using CoShard at 5 disclosure budgets $\delta$
- Distribute shards to 3 simulated AI providers (GPT-4o-mini, Claude Haiku, Gemini Flash)
- Each AI answers queries using only its shard (RAG with top-3 retrieval)
- Fuse answers using simple voting + LLM reranker
- Evaluate fused answer against gold labels

Metrics:
- **Answer quality**: Exact Match, F1, BERTScore vs gold answers
- **Disclosure**: SHADE\_oracle of each shard
- **Reconstruction resistance**: success rate of GPT-4o adversary on each shard

**Target:** At $\delta = 0.3$, answer quality $\geq 85\%$ of full-data baseline while keeping adversary reconstruction rate $\leq 20\%$.

---

## 9. Metrics Summary

| Metric | Measures | Used in |
|---|---|---|
| SHADE\_oracle | Ground-truth disclosure | E1, E2, E3, E4, E8 |
| SHADE\_proxy | Tractable disclosure estimate | E2, E3, E5, E6, E7 |
| Coh\_embed (Option A) | Within-shard embedding coherence | E5, E6, E7 |
| Recall@5 within shard | Retrieval-based answerability | E8 |
| Pareto hypervolume | Overall bi-criterion performance | E5 |
| Pearson $r$ / Spearman $\rho$ | Proxy-oracle correlation | E2, E3 |
| Exact Match / F1 | QA answer quality | E8 |
| BERTScore F1 | Semantic answer quality | E2, E8 |
| Adversary reconstruction rate | Empirical secrecy | E8 |
| Wall-clock time | Algorithmic scalability | E7 |

---

## 10. Implementation Stack

### 10.1 Core Libraries

```
Python 3.11+
├── sentence-transformers    # Document embeddings (bge-large-en-v1.5)
├── openai                   # GPT-4o adversary + embeddings API
├── anthropic                # Claude Haiku for multi-LLM QA
├── google-generativeai      # Gemini Flash for multi-LLM QA
├── igraph / leidenalg       # Leiden community detection
├── faiss-cpu                # Approximate nearest neighbours
├── scikit-learn             # KMeans clustering, metrics
├── numpy / scipy            # Linear algebra, statistics
├── datasets (HuggingFace)   # MedQA, LegalBench loading
├── bert-score               # BERTScore evaluation
└── pandas / matplotlib      # Results tables and Pareto plots
```

### 10.2 Repository Structure

```
shade-coshard/
├── data/
│   ├── hr_corpus/           # Synthetic HR documents
│   ├── medqa/               # MedQA subset
│   ├── legalbench/          # LegalBench subset
│   └── synthetic/           # Controlled synthetic corpus
├── src/
│   ├── shade/
│   │   ├── oracle.py        # Oracle SHADE (Algorithm 1)
│   │   ├── proxy.py         # Proxy SHADE (Algorithm 2)
│   │   └── metrics.py       # Coherence, baselines
│   ├── coshard/
│   │   ├── graph.py         # Graph construction (Algorithm 4)
│   │   ├── partition.py     # CoShard algorithm (Algorithm 5)
│   │   └── pareto.py        # Pareto sweep and hypervolume
│   ├── eval/
│   │   ├── axioms.py        # Axiom verification (E1)
│   │   ├── correlation.py   # Reconstruction correlation (E2)
│   │   ├── qa_pipeline.py   # Multi-LLM QA evaluation (E8)
│   │   └── adversary.py     # LLM reconstruction adversary
│   └── utils/
│       ├── embeddings.py    # Embedding cache and batching
│       └── data_loader.py   # Dataset loading and preprocessing
├── experiments/
│   ├── run_e1_axioms.py
│   ├── run_e2_correlation.py
│   ├── run_e5_pareto.py
│   └── run_e8_qa.py
├── notebooks/
│   └── results_analysis.ipynb
├── tests/
├── plan.md                  # This file
└── requirements.txt
```

---

## 11. Development Timeline

| Week | Milestone |
|---|---|
| 1–2 | Data collection and preprocessing; embedding pipeline; synthetic corpus generation |
| 3–4 | Implement Oracle SHADE (Algorithm 1) and Proxy SHADE (Algorithm 2) |
| 5 | Axiom verification experiments (E1); write Section 4 theory |
| 6 | Reconstruction correlation experiments (E2, E3); prove Theorems 1–2 |
| 7–8 | Implement CoShard (Algorithms 4–5); Pareto sweep (E5) |
| 9 | Prove Theorems 3–5; write Section 5 theory |
| 10 | Composition theorem validation (E4); No-free-lunch validation (E6) |
| 11–12 | Multi-LLM QA pipeline (E8); scalability experiments (E7) |
| 13 | Write Sections 1–3 (intro, related work, formulation) |
| 14 | Write Sections 6–8 (experiments, discussion, conclusion); compile appendices |
| 15 | Internal review and revision; LaTeX formatting for Springer LNCS/ML style |
| 16 | Final proofreading; upload to submission system |

---

## 12. Related Work to Cite

### Partitioning / RAG
- Yang et al. (2025) — SPLIT-RAG, arXiv:2505.13994
- Maio & Rizzi (2026) — Bridging OLAP and RAG, arXiv:2601.03748
- Edge et al. (2024) — GraphRAG, Microsoft Research
- Karypis & Kumar (1998) — METIS graph partitioning
- Traag et al. (2019) — Leiden algorithm, Scientific Reports

### Leakage measures
- Alvim et al. (2012) — g-leakage and QIF framework
- Issa et al. (2019) — Maximal leakage
- Liao et al. (2019) — α-leakage
- Deng et al. (2025) — FSInfo for split inference, arXiv:2504.10016
- Xin et al. (2025) — False Sense of Privacy, arXiv:2504.21035

### Privacy-utility tradeoffs
- Dwork et al. (2006) — Differential privacy
- Guo et al. (2025) — Threshold-Protected Searchable Sharing, arXiv:2507.17199
- Koga et al. (2024) — DP-RAG, arXiv:2412.04697
- Liagouris et al. (2023) — SECRECY, NSDI

### Multi-LLM systems
- Addison et al. (2024) — C-FedRAG, arXiv:2412.13163
- Jiang et al. (2023) — LLM-Blender, ACL

---

## 13. Open Questions and Risks

| Risk | Mitigation |
|---|---|
| Oracle SHADE too expensive for large corpora ($N > 10K$) | Cap adversary evaluation at $M=50$ sampled docs; use proxy in main experiments |
| Proxy fidelity degrades for highly heterogeneous corpora | Report breakdown by corpus type; add a calibration step |
| CoShard convergence is slow for large $N$ | Add early stopping; approximate SHADE\_proxy update as incremental coverage change |
| Theorem 4 bound is too loose to be useful empirically | Tighten bound using corpus-specific semantic entropy estimate; report both theoretical and empirical gaps |
| Multi-LLM QA pipeline costs (API calls) | Use smaller models (GPT-4o-mini, Claude Haiku) for 95% of experiments; run GPT-4o only for oracle SHADE ground truth |