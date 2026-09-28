import hashlib
import os
from dataclasses import asdict, dataclass
from typing import Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    token_count: int


class ProductionIngestionPipeline:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        embedding_model_name: str = "cl100k_base",
    ):
        """
        chunk_size: Target token size per chunk (~500 tokens covers 2-3 detailed paragraphs).
        chunk_overlap: Overlap in tokens (~15%) to prevent sentence severance.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(embedding_model_name)

        # Custom length function counting actual tokens instead of raw characters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _generate_chunk_id(self, doc_name: str, page_num: int, chunk_index: int, text: str) -> str:
        """Generates a deterministic hash for deduplication and idempotent upserts."""
        raw_key = f"{doc_name}:{page_num}:{chunk_index}:{text[:50]}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def process_pdf(self, file_path: str) -> list[DocumentChunk]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found at: {file_path}")

        print(f"[Ingestion] Loading PDF from {file_path}...")
        loader = PyPDFLoader(file_path)
        raw_pages = loader.load()
        print(f"[Ingestion] Loaded {len(raw_pages)} pages from document.")

        processed_chunks: list[DocumentChunk] = []
        doc_filename = os.path.basename(file_path)

        for page in raw_pages:
            page_text = page.page_content.strip()
            if not page_text:
                continue

            page_number = page.metadata.get("page", 0) + 1  # 1-indexed

            # Split individual page while preserving structure
            sub_chunks = self.text_splitter.split_text(page_text)

            for idx, chunk_text in enumerate(sub_chunks):
                token_count = self._count_tokens(chunk_text)
                chunk_id = self._generate_chunk_id(doc_filename, page_number, idx, chunk_text)

                metadata = {
                    "source": doc_filename,
                    "page": page_number,
                    "chunk_index": idx,
                    "char_length": len(chunk_text),
                }

                processed_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata=metadata,
                        token_count=token_count,
                    )
                )

        print(f"[Ingestion] Successfully generated {len(processed_chunks)} token-aware chunks.")
        return processed_chunks


if __name__ == "__main__":
    pdf_path = os.path.join("data", "annual_report.pdf")
    pipeline = ProductionIngestionPipeline(chunk_size=500, chunk_overlap=75)
    chunks = pipeline.process_pdf(pdf_path)

    # Print summary statistics
    token_counts = [c.token_count for c in chunks]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    print("\n" + "=" * 50)
    print("INGESTION METRICS SUMMARY")
    print("=" * 50)
    print(f"Total Chunks:      {len(chunks)}")
    print(f"Average Tokens:    {avg_tokens:.1f}")
    print(f"Min Tokens:        {min(token_counts)}")
    print(f"Max Tokens:        {max(token_counts)}")
    print("=" * 50)

    # Inspect the first chunk
    sample = chunks[0]
    print("\nSample Chunk Preview:")
    print(f"ID:       {sample.chunk_id}")
    print(f"Metadata: {sample.metadata}")
    print(f"Tokens:   {sample.token_count}")
    print("Content:")
    print(sample.text[:300] + "...")
