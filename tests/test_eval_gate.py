import json
import os
import sys

import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reranker import ProductionReranker
from src.retriever import HybridRetriever


@pytest.fixture(scope="module")
def shared_rag_components():
    return HybridRetriever(), ProductionReranker()


def test_retrieval_accuracy_gate(shared_rag_components):
    """
    CI Gate: Ensures Hybrid + Reranker maintains >= 80% Hit Rate@3 on the Golden Set.
    Fails build if retrieval regressions are introduced.
    """
    retriever, reranker = shared_rag_components

    with open("eval/golden_qa_set.json", "r") as f:
        golden_set = json.load(f)

    hits = 0
    total = len(golden_set)

    for item in golden_set:
        query = item["question"]
        targets = set(item["target_pages"])

        candidates = retriever.search_hybrid(query, top_k=10)
        reranked = reranker.rerank(query, candidates, top_k=3)

        retrieved_pages = {r.metadata.get("page") for r in reranked}
        if retrieved_pages.intersection(targets):
            hits += 1

    hit_rate = hits / total
    print(f"\n[CI Gate] Hybrid + Rerank Hit Rate@3: {hit_rate * 100:.1f}%")

    assert hit_rate >= 0.80, (
        f"Regression detected! Hit Rate {hit_rate * 100:.1f}% is below threshold (80.0%)"
    )
