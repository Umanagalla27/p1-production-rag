# P1: Production Enterprise RAG Engine

[![P1 RAG CI Pipeline](https://github.com/Umanagalla27/p1-production-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Umanagalla27/p1-production-rag/actions)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL_16-pgvector_(HNSW)-4169E1.svg?logo=postgresql&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)

An enterprise-grade Retrieval-Augmented Generation (RAG) platform indexing **307 pages (774 token-bounded chunks)** of an official SEC 10-K filing. 

Combines **Dense Vector Search (pgvector HNSW)**, **Sparse Lexical Search (BM25)** with **Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder Reranking**, achieving a **+28% gain in MRR** over standalone vector search.

---

## 🏛️ System Architecture

```text
User Query
    │
    ▼
┌──────────────────────┐
│   Query Expansion    │ ◄── Multi-query semantic decomposition
└──────────┬───────────┘
           │
     ┌─────┴─────────────────────┐
     ▼                           ▼
┌──────────────────┐    ┌──────────────────┐
│   Dense Search   │    │  Sparse Search   │
│ pgvector (HNSW)  │    │  BM25 (Keyword)  │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
       ┌───────────────────────────┐
       │  Reciprocal Rank Fusion   │ ◄── Scale-invariant fusion (k=60)
       └─────────────┬─────────────┘
                     ▼ Top 12 Candidates
       ┌───────────────────────────┐
       │  Cross-Encoder Reranker   │ ◄── ms-marco-MiniLM-L-6-v2
       └─────────────┬─────────────┘
                     ▼ Top 3 Filtered Chunks
       ┌───────────────────────────┐
       │   Grounded Synthesizer    │ ──► Verified Answer + Inline [Page X] Citations
       └───────────────────────────┘
```

---

## 📊 Empirical Ablation Benchmark (Golden Evaluation Set)

Evaluated against a curated golden benchmark set of enterprise questions:

| Retrieval Architecture | Hit Rate@3 | Mean Reciprocal Rank (MRR) | Avg Latency (ms) | Production Verdict |
|---|---|---|---|---|
| **Sparse (BM25 Only)** | 66.7% | 0.583 | **13.4 ms** | Ultra-fast, but misses 33% of conceptual queries. |
| **Dense (pgvector HNSW)** | 100.0% | 0.694 | 684.6 ms | Reliable recall, but ranks suboptimal chunks at #1. |
| **Hybrid (RRF Fusion)** | 83.3% | 0.639 | 431.7 ms | Fuses exact keywords with semantic concepts. |
| **Hybrid + Cross-Encoder** | **100.0%** | **0.889 (+28%)** | 4,109.2 ms | **Gold Standard Precision**: Puts ground-truth at Rank #1. |

---

## 🚀 Quick Start

### 1. Start Infrastructure
```bash
docker compose -f docker/docker-compose.yml up -d
```

### 2. Ingest & Index 10-K Document
```bash
python src/download_data.py
python src/indexer.py
```

### 3. Run Benchmark
```bash
python eval/run_eval.py
```

### 4. Start API Server
```bash
uvicorn src.api:app --reload --port 8000
```
Interactive Docs: http://localhost:8000/docs
