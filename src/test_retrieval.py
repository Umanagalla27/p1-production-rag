import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.reranker import ProductionReranker
from src.retriever import HybridRetriever


def compare_retrieval_methods(query: str):
    print("\n" + "=" * 80)
    print(f"QUERY: '{query}'")
    print("=" * 80)

    retriever = HybridRetriever()
    reranker = ProductionReranker()

    print("\n--- [1] DENSE VECTOR SEARCH (TOP 3) ---")
    dense_hits = retriever.search_dense(query, top_k=3)
    for i, h in enumerate(dense_hits, 1):
        print(
            f"{i}. [Page {h.metadata['page']}] (Score: {h.score:.4f}) | {h.text[:140]}..."
        )

    print("\n--- [2] BM25 SPARSE SEARCH (TOP 3) ---")
    sparse_hits = retriever.search_sparse(query, top_k=3)
    for i, h in enumerate(sparse_hits, 1):
        print(
            f"{i}. [Page {h.metadata['page']}] (Score: {h.score:.4f}) | {h.text[:140]}..."
        )

    print("\n--- [3] HYBRID RRF SEARCH (TOP 3) ---")
    hybrid_hits = retriever.search_hybrid(query, top_k=10)
    for i, h in enumerate(hybrid_hits[:3], 1):
        print(
            f"{i}. [Page {h.metadata['page']}] (RRF Score: {h.score:.4f}) | {h.text[:140]}..."
        )

    print("\n--- [4] HYBRID + CROSS-ENCODER RERANKED (TOP 3) ---")
    reranked_hits = reranker.rerank(query, hybrid_hits, top_k=3)
    for i, h in enumerate(reranked_hits, 1):
        print(
            f"{i}. [Page {h.metadata['page']}] (Cross Score: {h.score:.4f}) | {h.text[:140]}..."
        )


if __name__ == "__main__":
    # Test query with specific keywords & concepts from the Uber 10-K report
    test_query = "What are the legal risks associated with driver classification as independent contractors?"
    compare_retrieval_methods(test_query)
