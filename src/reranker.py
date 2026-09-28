import os
import sys
import time

from sentence_transformers import CrossEncoder

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever import SearchResult

# Industry standard lightweight cross-encoder trained on MS MARCO passage ranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ProductionReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"[Reranker] Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        if not candidates:
            return []

        start = time.perf_counter()

        # Pair query with each candidate text
        pairs = [[query, candidate.text] for candidate in candidates]
        cross_scores = self.model.predict(pairs)

        # Update scores and sort
        reranked: list[SearchResult] = []
        for candidate, cross_score in zip(candidates, cross_scores, strict=True):
            reranked.append(
                SearchResult(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    metadata=candidate.metadata,
                    score=float(cross_score),
                    retrieval_method="cross_encoder_reranked",
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        duration_ms = (time.perf_counter() - start) * 1000
        print(
            f"[Reranker] Rescored {len(candidates)} candidates down to {top_k} in {duration_ms:.1f}ms."
        )

        return reranked[:top_k]
