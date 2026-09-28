import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.generator import ProductionRAGEngine


def test_full_rag_pipeline():
    engine = ProductionRAGEngine()

    test_questions = [
        "What are the primary business risks regarding independent contractor status for drivers?",
        "What was the total revenue or financial trajectory mentioned in the report?",
    ]

    for q in test_questions:
        print("\n" + "=" * 80)
        print(f"USER QUESTION: {q}")
        print("=" * 80)
        response = engine.answer_question(q)

        print(f"\n[LATENCY]: {response.latency_ms} ms")
        print(f"[GROUNDED]: {response.grounded}")
        print(f"\n[ANSWER]:\n{response.answer}\n")
        print("[CITATIONS]:")
        for c in response.citations:
            print(f'  • Page {c.page} (Chunk {c.chunk_id}): "{c.quote[:100]}..."')


if __name__ == "__main__":
    test_full_rag_pipeline()
