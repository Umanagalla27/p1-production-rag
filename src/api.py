import json
import os
import sys
import time
from typing import Any

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generator import GroundedRAGResponse, ProductionRAGEngine

app = FastAPI(
    title="P1 Production RAG Engine",
    description="Enterprise SEC 10-K RAG with pgvector HNSW, BM25, and Cross-Encoder Reranker",
    version="1.0.0",
)

# Global engine holder
rag_engine: ProductionRAGEngine | None = None
redis_client: redis.Redis | None = None


class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500, description="User question")
    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of cited chunks to return"
    )


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[dict[str, Any]]
    latency_ms: float
    cached: bool = False


@app.on_event("startup")
def startup_event():
    global rag_engine, redis_client
    print("[API] Initializing RAG Engine and Redis client...")
    rag_engine = ProductionRAGEngine()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6381")
    try:
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        print(f"[API] Connected to Redis at {redis_url}")
    except redis.RedisError as e:
        print(f"[API] Redis connection warning: {e}. Running without cache.")
        redis_client = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rag_engine_ready": rag_engine is not None,
        "redis_connected": redis_client is not None,
    }


@app.post("/v1/rag/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest):
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG Engine is initializing")

    start_time = time.perf_counter()
    cache_key = f"rag_query:{request.query.strip().lower()}:{request.top_k}"

    # 1. Check Redis Cache
    if redis_client:
        try:
            cached_val = redis_client.get(cache_key)
            if cached_val:
                data = json.loads(cached_val)
                latency = (time.perf_counter() - start_time) * 1000
                return RAGQueryResponse(
                    query=request.query,
                    answer=data["answer"],
                    citations=data["citations"],
                    latency_ms=round(latency, 2),
                    cached=True,
                )
        except redis.RedisError as e:
            print(f"Redis cache lookup error: {e}")

    # 2. Run Full RAG Pipeline
    result: GroundedRAGResponse = rag_engine.answer_question(
        query=request.query, top_k_rerank=request.top_k
    )

    citations_list = [
        c.model_dump() if hasattr(c, "model_dump") else c.dict()
        for c in result.citations
    ]

    # 3. Cache in Redis (TTL: 1 hour)
    if redis_client:
        try:
            redis_client.setex(
                cache_key,
                3600,
                json.dumps({"answer": result.answer, "citations": citations_list}),
            )
        except redis.RedisError as e:
            print(f"Redis cache set error: {e}")

    total_latency = (time.perf_counter() - start_time) * 1000

    return RAGQueryResponse(
        query=request.query,
        answer=result.answer,
        citations=citations_list,
        latency_ms=round(total_latency, 2),
        cached=False,
    )
