import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import ProductionIngestionPipeline


def test_chunking_token_limits():
    pipeline = ProductionIngestionPipeline(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "Artificial intelligence and deep learning models have fundamentally transformed "
        "enterprise software architecture. Production RAG platforms require robust chunking, "
        "hybrid vector search, and reranking pipelines to prevent hallucinations at scale. "
    ) * 10

    # Test token counting function
    tokens = pipeline._count_tokens(sample_text)
    assert tokens > 150

    # Test splitting adherence
    chunks = pipeline.text_splitter.split_text(sample_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert pipeline._count_tokens(chunk) <= 110  # within margin


def test_deterministic_chunk_id():
    pipeline = ProductionIngestionPipeline()
    id1 = pipeline._generate_chunk_id("doc.pdf", 1, 0, "Test chunk content")
    id2 = pipeline._generate_chunk_id("doc.pdf", 1, 0, "Test chunk content")
    id3 = pipeline._generate_chunk_id("doc.pdf", 1, 1, "Different chunk")

    assert id1 == id2
    assert id1 != id3
