import json
import os
import sys
import time
from typing import Any

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reranker import ProductionReranker
from src.retriever import HybridRetriever, SearchResult


def evaluate_pipeline():
    eval_file = os.path.join("eval", "golden_qa_set.json")
    with open(eval_file, "r") as f:
        golden_set = json.load(f)

    print(f"[Evaluation] Loaded {len(golden_set)} benchmark queries from {eval_file}.")

    retriever = HybridRetriever()
    reranker = ProductionReranker()

    methods = ["dense_only", "sparse_only", "hybrid_rrf", "hybrid_reranked"]
    metrics: dict[str, dict[str, Any]] = {
        m: {"hits_at_3": 0, "reciprocal_ranks": [], "latencies": []} for m in methods
    }

    k = 3

    for item in golden_set:
        query = item["question"]
        targets = set(item["target_pages"])

        # 1. Dense Vector Only
        t0 = time.perf_counter()
        dense_hits = retriever.search_dense(query, top_k=k)
        metrics["dense_only"]["latencies"].append((time.perf_counter() - t0) * 1000)
        _score_run(dense_hits, targets, metrics["dense_only"])

        # 2. Sparse BM25 Only
        t0 = time.perf_counter()
        sparse_hits = retriever.search_sparse(query, top_k=k)
        metrics["sparse_only"]["latencies"].append((time.perf_counter() - t0) * 1000)
        _score_run(sparse_hits, targets, metrics["sparse_only"])

        # 3. Hybrid RRF Only
        t0 = time.perf_counter()
        hybrid_hits = retriever.search_hybrid(query, top_k=k)
        metrics["hybrid_rrf"]["latencies"].append((time.perf_counter() - t0) * 1000)
        _score_run(hybrid_hits, targets, metrics["hybrid_rrf"])

        # 4. Hybrid + Cross-Encoder Reranked
        t0 = time.perf_counter()
        candidate_pool = retriever.search_hybrid(query, top_k=12)
        reranked_hits = reranker.rerank(query, candidate_pool, top_k=k)
        metrics["hybrid_reranked"]["latencies"].append(
            (time.perf_counter() - t0) * 1000
        )
        _score_run(reranked_hits, targets, metrics["hybrid_reranked"])

    # Print Ablation Table
    total = len(golden_set)
    print("\n" + "=" * 80)
    print("RETRIEVAL ABLATION BENCHMARK (GOLDEN EVALUATION SET)")
    print("=" * 80)
    print(
        f"{'Retrieval Architecture':<28} | {'Hit Rate@3':<12} | {'MRR':<10} | {'Avg Latency (ms)':<16}"
    )
    print("-" * 80)

    results_table = []
    for m in methods:
        hit_rate = (metrics[m]["hits_at_3"] / total) * 100
        mrr = np.mean(metrics[m]["reciprocal_ranks"])
        avg_lat = np.mean(metrics[m]["latencies"])
        print(f"{m:<28} | {hit_rate:>9.1f}% | {mrr:>8.3f} | {avg_lat:>14.1f} ms")
        results_table.append(
            {
                "method": m,
                "hit_rate_at_3": round(hit_rate, 2),
                "mrr": round(float(mrr), 3),
                "avg_latency_ms": round(float(avg_lat), 1),
            }
        )
    print("=" * 80)

    # Save to results/
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_benchmark.json", "w") as f:
        json.dump(results_table, f, indent=2)
    print("Saved benchmark report to results/ablation_benchmark.json")


def _score_run(
    results: list[SearchResult], target_pages: set[int], stats: dict[str, Any]
):
    found = False
    rr = 0.0
    for rank, res in enumerate(results, start=1):
        page = res.metadata.get("page")
        if page in target_pages and not found:
            rr = 1.0 / rank
            found = True
    stats["reciprocal_ranks"].append(rr)
    if found:
        stats["hits_at_3"] += 1


if __name__ == "__main__":
    evaluate_pipeline()
