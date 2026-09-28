import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import psycopg2
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB_CONFIG = {
    "dbname": "rag_db",
    "user": "uma",
    "password": "password",
    "host": "localhost",
    "port": 5434,
}


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    retrieval_method: str


class HybridRetriever:
    def __init__(self, db_config: dict[str, Any] = DB_CONFIG):
        self.db_config = db_config
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load persisted BM25 index
        bm25_file = os.path.join("data", "bm25_index.pkl")
        if not os.path.exists(bm25_file):
            raise FileNotFoundError(
                f"BM25 index not found at {bm25_file}. Run indexer.py first."
            )

        with open(bm25_file, "rb") as f:
            payload = pickle.load(f)
            self.bm25: BM25Okapi = payload["bm25"]
            self.chunk_ids: list[str] = payload["chunk_ids"]
            self.corpus_texts: list[str] = payload["corpus_texts"]
            self.metadatas: list[dict[str, Any]] = payload["metadatas"]

    def _get_db_connection(self):
        conn = psycopg2.connect(**self.db_config)
        if register_vector:
            try:
                register_vector(conn)
            except psycopg2.Error:
                pass
        return conn

    def search_dense(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Dense vector search using pgvector with HNSW cosine distance (<=>)."""
        query_embedding = self.encoder.encode(query, normalize_embeddings=True).tolist()

        conn = self._get_db_connection()
        cur = conn.cursor()

        # In pgvector: <=> operator is Cosine Distance (1 - Cosine Similarity)
        # Cosine Distance: 0 = identical, 2 = opposite
        cur.execute(
            """
            SELECT chunk_id, text, metadata, (1 - (embedding <=> %s::vector)) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """,
            (query_embedding, query_embedding, top_k),
        )

        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            SearchResult(
                chunk_id=r[0],
                text=r[1],
                metadata=r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                score=float(r[3]),
                retrieval_method="dense_vector",
            )
            for r in rows
        ]

    def search_sparse(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Sparse lexical search using BM25Okapi."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices sorted descending by BM25 score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        return [
            SearchResult(
                chunk_id=self.chunk_ids[i],
                text=self.corpus_texts[i],
                metadata=self.metadatas[i],
                score=float(scores[i]),
                retrieval_method="sparse_bm25",
            )
            for i in top_indices
            if scores[i] > 0.0
        ]

    def search_hybrid(
        self, query: str, top_k: int = 10, rrf_k: int = 60
    ) -> list[SearchResult]:
        """
        Combines Dense and Sparse search results using Reciprocal Rank Fusion (RRF).
        Formula: RRF_score(d) = sum(1 / (k + rank_i))
        """
        dense_results = self.search_dense(query, top_k=top_k * 2)
        sparse_results = self.search_sparse(query, top_k=top_k * 2)

        rrf_scores: dict[str, float] = {}
        doc_store: dict[str, SearchResult] = {}

        # 1. Score dense rankings
        for rank, res in enumerate(dense_results, start=1):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            doc_store[res.chunk_id] = res

        # 2. Score sparse rankings
        for rank, res in enumerate(sparse_results, start=1):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            if res.chunk_id not in doc_store:
                doc_store[res.chunk_id] = res

        # 3. Sort by combined RRF score
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )[:top_k]

        fused_results: list[SearchResult] = []
        for cid in sorted_ids:
            item = doc_store[cid]
            fused_results.append(
                SearchResult(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    metadata=item.metadata,
                    score=rrf_scores[cid],
                    retrieval_method="hybrid_rrf",
                )
            )

        return fused_results
