# poisonBench

An offline benchmark for studying retrieval corpus poisoning attacks and evaluating defence mechanisms in Retrieval-Augmented Generation (RAG) systems.

Rather than treating RAG security as a theoretical problem, poisonBench builds a complete retrieval pipeline, deliberately attacks it using multiple poisoning strategies, and measures how effectively different defences detect and mitigate those attacks.

---

## Overview

Retrieval-Augmented Generation systems depend entirely on the quality of their document corpus.

If attackers can insert malicious documents into the knowledge base, they can manipulate retrieval results and influence LLM responses.

poisonBench provides a reproducible framework to:

- Build an offline RAG pipeline
- Inject poisoned documents
- Execute multiple attack strategies
- Apply layered defence mechanisms
- Measure attack success and defence effectiveness
- Compare statistical and machine learning approaches

This project is intended for research and educational purposes only.

---

## Features

### Baseline RAG Pipeline

- Document ingestion
- Text chunking
- Sentence embeddings
- ChromaDB vector storage
- Semantic retrieval
- Local LLM generation

---

### Corpus Poisoning Attacks

- Naive semantic mimicry
- Embedding optimisation attack
- Prompt injection documents

Each attack can be evaluated independently across configurable poisoning ratios.

---

### Defence Mechanisms

#### Embedding Outlier Detection

Detects semantically abnormal documents using embedding distance.

#### Trust-Based Re-ranking

Reorders retrieval results using document trust scores.

#### Provenance Verification

Validates document origin before retrieval.

#### Machine Learning Detection

Binary poisoned-document classifier using:

- XGBoost
- Random Forest

---

## Evaluation

poisonBench measures:

- Retrieval attack success rate
- Generation attack success rate
- Defence detection rate
- False positive rate
- Retrieval latency
- Precision
- Recall
- F1-score

Experiments can be repeated across different poisoning ratios and defence configurations.

---

## Architecture

```
                Clean Corpus
                     │
                     ▼
          Document Ingestion
                     │
                     ▼
               Text Chunking
                     │
                     ▼
             Sentence Embeddings
                     │
                     ▼
               ChromaDB Store
                     ▲
                     │
          Poisoning Attack Module
                     │
──────────────────────────────────────────

User Query
     │
     ▼
Retriever
     │
     ▼
Defence Pipeline
    ├── Embedding Outlier Detection
    ├── Trust Re-ranking
    ├── Provenance Verification
    └── ML Classifier
     │
     ▼
Filtered Context
     │
     ▼
Local LLM
     │
     ▼
Generated Response
     │
     ▼
Evaluation Metrics
```

---

## Technology Stack

### Backend

- FastAPI
- Python

### Retrieval

- ChromaDB
- sentence-transformers
- all-MiniLM-L6-v2

### Language Model

- Ollama
- Phi-3 Mini / Llama 3.2

### Machine Learning

- XGBoost
- Random Forest
- scikit-learn
- SMOTE

### Database

- PostgreSQL

### Cache

- Redis

### Frontend

- React

### Infrastructure

- Docker
- Docker Compose

---

## Repository Structure

```
poisonBench/

├── backend/
├── frontend/
├── attacks/
├── defenses/
├── evaluation/
├── experiments/
├── datasets/
├── notebooks/
├── docker/
├── docs/
└── README.md
```

---

## Running the Project

### Clone

```bash
git clone https://github.com/<username>/poisonBench.git

cd poisonBench
```

### Start Services

```bash
docker compose up --build
```

The system starts:

- Backend API
- ChromaDB
- PostgreSQL
- Redis
- Local LLM
- Frontend

---

## Experiments

Typical workflow:

1. Build baseline corpus
2. Validate retrieval quality
3. Inject poisoned documents
4. Enable one or more defences
5. Execute benchmark
6. Collect metrics
7. Compare results

---

## Project Goals

- Understand retrieval corpus poisoning
- Compare multiple attack strategies
- Evaluate lightweight defence mechanisms
- Benchmark ML-based detection
- Produce reproducible experimental results

---

## Disclaimer

poisonBench is an academic research project intended for studying the security of Retrieval-Augmented Generation systems.

The attacks implemented in this repository are designed solely for evaluation in isolated local environments and should not be used against production systems.
