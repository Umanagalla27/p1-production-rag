import json
import os
import pickle
import sys
import time
from typing import Any

import psycopg2
from rank_bm25 import BM25Okapi

# Ensure repository root is on sys.path when executed directly as python src/indexer.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer

from src.ingestion import DocumentChunk, ProductionIngestionPipeline

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions, fast CPU inference
EMBEDDING_DIM = 384

DB_CONFIG = {
    "dbname": "rag_db",
    "user": "uma",
    "password": "password",
    "host": "localhost",
    "port": 5434,
}


class ProductionIndexer:
    def __init__(self, db_config: dict[str, Any] = DB_CONFIG):
        self.db_config = db_config
        print(f"[Indexer] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def get_db_connection(self):
        conn = psycopg2.connect(**self.db_config)
        if register_vector:
            try:
                register_vector(conn)
            except psycopg2.Error:
                pass
        return conn

    def init_database_schema(self):
        """Initializes pgvector extension, table, and HNSW cosine index."""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        print("[Indexer] Initializing pgvector extension and schema...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        if register_vector:
            register_vector(conn)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id VARCHAR(32) PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL,
                embedding VECTOR({EMBEDDING_DIM}),
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # HNSW Index for sub-millisecond cosine vector search
        print("[Indexer] Creating HNSW index (m=16, ef_construction=64)...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_hnsw_idx 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("[Indexer] Database schema & HNSW index ready.")

    def index_chunks(self, chunks: list[DocumentChunk]):
        self.init_database_schema()

        print(
            f"[Indexer] Generating dense embeddings for {len(chunks)} chunks on CPU..."
        )
        start_time = time.perf_counter()

        texts = [chunk.text for chunk in chunks]
        embeddings = self.encoder.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )

        encode_duration = time.perf_counter() - start_time
        print(
            f"[Indexer] Encoded in {encode_duration:.2f}s ({len(chunks) / encode_duration:.1f} chunks/sec)."
        )

        conn = self.get_db_connection()
        cur = conn.cursor()

        print("[Indexer] Upserting records into PostgreSQL (pgvector)...")
        upsert_query = """
            INSERT INTO document_chunks (chunk_id, text, metadata, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE 
            SET text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding;
        """

        records = [
            (
                chunk.chunk_id,
                chunk.text,
                json.dumps(chunk.metadata),
                embedding.tolist(),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        cur.executemany(upsert_query, records)
        conn.commit()
        cur.close()
        conn.close()
        print(f"[Indexer] Successfully stored {len(records)} records in pgvector.")

        # Build and persist BM25 Index for Hybrid Search
        self._build_and_save_bm25(chunks)

    def _build_and_save_bm25(self, chunks: list[DocumentChunk]):
        print("[Indexer] Tokenizing corpus and building BM25 sparse index...")
        # Simple lowercased word tokenization for BM25
        corpus_tokenized = [chunk.text.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(corpus_tokenized)

        os.makedirs("data", exist_ok=True)
        bm25_payload = {
            "bm25": bm25,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "corpus_texts": [chunk.text for chunk in chunks],
            "metadatas": [chunk.metadata for chunk in chunks],
        }

        bm25_file = os.path.join("data", "bm25_index.pkl")
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25_payload, f)

        print(f"[Indexer] Persisted BM25 index to {bm25_file}.")


if __name__ == "__main__":
    pdf_path = os.path.join("data", "annual_report.pdf")
    pipeline = ProductionIngestionPipeline(chunk_size=500, chunk_overlap=75)
    doc_chunks = pipeline.process_pdf(pdf_path)

    indexer = ProductionIndexer()
    indexer.index_chunks(doc_chunks)
