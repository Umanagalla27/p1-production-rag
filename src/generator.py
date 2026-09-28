import os
import sys
import time

from pydantic import BaseModel, Field

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reranker import ProductionReranker
from src.retriever import HybridRetriever, SearchResult


class Citation(BaseModel):
    page: int = Field(description="Page number from the source document.")
    chunk_id: str = Field(description="Unique ID of the cited chunk.")
    quote: str = Field(description="Direct verbatim phrase supporting the claim.")


class GroundedRAGResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
    latency_ms: float
    grounded: bool


SYSTEM_PROMPT = """You are a senior financial analyst and compliance auditor.
Your task is to answer the user's question using ONLY the provided document context.

RULES:
1. Every factual statement in your answer MUST cite its source using the format [Page X].
2. Include direct short quotes in your citations to prove groundedness.
3. If the context does not contain sufficient information to answer the question, state clearly: "Based on the provided filing, this information is not available." Do NOT extrapolate or guess.
4. Keep answers concise, factual, and professional.
"""


class ProductionRAGEngine:
    def __init__(self):
        print("[RAG Engine] Initializing Hybrid Retriever and Cross-Encoder...")
        self.retriever = HybridRetriever()
        self.reranker = ProductionReranker()

    def rewrite_query(self, raw_query: str) -> list[str]:
        """
        Decomposes or expands raw user queries for higher retrieval recall.
        Example: 'Tell me about their lawsuits' -> expands into legal proceedings and classification.
        """
        # Multi-query expansion heuristics
        expanded_queries = [raw_query]
        lowered = raw_query.lower()
        if "risk" in lowered or "lawsuit" in lowered:
            expanded_queries.append(
                f"{raw_query} litigation legal proceedings regulatory"
            )
        if "revenue" in lowered or "financial" in lowered:
            expanded_queries.append(f"{raw_query} consolidated results operations GAAP")
        return expanded_queries

    def answer_question(
        self,
        query: str,
        top_k_retrieval: int = 15,
        top_k_rerank: int = 4,
    ) -> GroundedRAGResponse:
        start_time = time.perf_counter()

        # 1. Query Expansion & Hybrid Retrieval
        queries = self.rewrite_query(query)
        candidate_pool: dict[str, SearchResult] = {}

        for q in queries:
            results = self.retriever.search_hybrid(q, top_k=top_k_retrieval)
            for res in results:
                if res.chunk_id not in candidate_pool:
                    candidate_pool[res.chunk_id] = res

        # 2. Cross-Encoder Reranking
        candidates = list(candidate_pool.values())
        reranked_docs = self.reranker.rerank(query, candidates, top_k=top_k_rerank)

        # 3. Format Context Window
        context_blocks = []
        for doc in reranked_docs:
            context_blocks.append(
                f"[Chunk: {doc.chunk_id} | Page {doc.metadata['page']}]\n{doc.text}"
            )
        formatted_context = "\n\n---\n\n".join(context_blocks)

        # 4. Synthesize Answer (Using local synthesizer or LLM API if key is present)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

        if api_key:
            answer, citations, grounded = self._call_llm_api(
                query, formatted_context, reranked_docs
            )
        else:
            # Deterministic, grounded extraction when no API key is passed
            answer, citations, grounded = self._synthesize_grounded_fallback(
                query, reranked_docs
            )

        total_latency = (time.perf_counter() - start_time) * 1000

        return GroundedRAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunk_ids=[d.chunk_id for d in reranked_docs],
            latency_ms=round(total_latency, 2),
            grounded=grounded,
        )

    def _synthesize_grounded_fallback(
        self, query: str, top_docs: list[SearchResult]
    ) -> tuple[str, list[Citation], bool]:
        """Provides a structured, citation-grounded response directly from top reranked chunks."""
        if not top_docs:
            return (
                "Based on the provided filing, this information is not available.",
                [],
                False,
            )

        primary = top_docs[0]
        page = primary.metadata["page"]

        # Take the most relevant sentences
        sentences = [s.strip() for s in primary.text.split(".") if len(s.strip()) > 30][
            :3
        ]
        summary_text = ". ".join(sentences) + "."

        answer = f"According to the 10-K filing [Page {page}]: {summary_text}"

        citations = [
            Citation(
                page=page,
                chunk_id=primary.chunk_id,
                quote=sentences[0] if sentences else primary.text[:80],
            )
        ]
        return answer, citations, True

    def _call_llm_api(self, query: str, context: str, top_docs: list[SearchResult]):
        # Optional live Gemini / OpenAI call if keys are exported in shell
        try:
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                from google import genai

                client = genai.Client(api_key=gemini_key)
                prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                if resp.text:
                    cites = [
                        Citation(
                            page=d.metadata["page"],
                            chunk_id=d.chunk_id,
                            quote=d.text[:80],
                        )
                        for d in top_docs[:2]
                    ]
                    return resp.text, cites, True
        except (ImportError, AttributeError, RuntimeError, ValueError) as err:
            print(f"[RAG Engine] LLM API call bypassed: {err}")
        return self._synthesize_grounded_fallback(query, top_docs)
