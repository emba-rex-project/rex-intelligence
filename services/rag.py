from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from config.settings import Settings, get_settings
from models.common import DocumentChunk, DocumentMetadata
from models.query import QueryRequest, QueryResponse, RetrievedChunk
from services.chunking import ChunkConfig, TextChunker
from services.embeddings import EmbeddingService
from services.pdf_loader import PDFLoader
from services.vector_store import VectorStoreService

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - guard for linting without dependency
    OpenAI = None

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.pdf_loader = PDFLoader(self.settings.upload_path)
        chunk_config = ChunkConfig(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.chunker = TextChunker(chunk_config)
        self.vector_store = VectorStoreService(self.settings.vector_store_path)
        self.embedding_service = EmbeddingService(self.settings)
        self.chat_client = (
            OpenAI(api_key=self.settings.openai_api_key)
            if self.settings.openai_api_key and OpenAI is not None
            else None
        )

    # --- Ingestion -----------------------------------------------------------------

    def ingest_upload(self, filename: str, file_bytes: bytes) -> List[str]:
        saved_path = self.pdf_loader.save_upload(filename, file_bytes)
        return self.ingest_paths([saved_path])

    def ingest_paths(self, paths: Iterable[Path]) -> List[str]:
        document_ids: List[str] = []
        for path in paths:
            if path.is_dir():
                doc_ids = self.ingest_paths(
                    p for p in path.iterdir() if p.suffix.lower() == ".pdf"
                )
                document_ids.extend(doc_ids)
                continue
            if path.suffix.lower() != ".pdf":
                logger.info("Skipping non-PDF file: %s", path)
                continue
            document_ids.extend(self._ingest_single(path))
        return document_ids

    def _ingest_single(self, path: Path) -> List[str]:
        pages = self.pdf_loader.load_text(path)
        chunk_tuples = self.chunker.chunk_pages(pages)
        chunks: List[DocumentChunk] = []
        for page_number, chunk_index, chunk_text in chunk_tuples:
            chunk_id = (
                f"{path.stem}-p{page_number}-c{chunk_index}-{uuid.uuid4().hex[:8]}"
            )
            metadata = DocumentMetadata(
                filename=path.name,
                page_number=page_number,
                chunk_id=chunk_id,
            )
            chunks.append(
                DocumentChunk(id=chunk_id, text=chunk_text, metadata=metadata)
            )

        embeddings = self.embedding_service.embed_texts(
            [chunk.text for chunk in chunks]
        )
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        self.vector_store.upsert_chunks(chunks)
        logger.info("Ingested %d chunks for %s", len(chunks), path)
        return [chunk.id for chunk in chunks]

    # --- Query ---------------------------------------------------------------------

    def query(self, payload: QueryRequest) -> QueryResponse:
        query_embedding = self.embedding_service.embed_texts([payload.query])[0]
        top_k = payload.top_k or self.settings.top_k
        retrieved_pairs = self.vector_store.query(query_embedding, top_k=top_k)
        retrieved_pairs.sort(key=lambda pair: pair[1])

        similarity_map: dict[str, float] = {}
        chunks: List[DocumentChunk] = []
        for chunk, distance in retrieved_pairs:
            similarity = 1.0 / (1.0 + distance) if distance != float("inf") else 0.0
            similarity_map[chunk.id] = similarity
            chunks.append(chunk)

        ranked_chunks = self._rank_chunks(chunks, payload.query)
        references: List[RetrievedChunk] = []
        for chunk in ranked_chunks:
            references.append(
                RetrievedChunk(
                    score=similarity_map.get(chunk.id, 0.0),
                    chunk=chunk,
                )
            )

        answer = self._synthesize_answer(payload.query, ranked_chunks)
        if not payload.include_raw_chunks:
            references = []
        return QueryResponse(answer=answer, references=references)

    def _rank_chunks(
        self, chunks: Sequence[DocumentChunk], query: str
    ) -> List[DocumentChunk]:
        if not chunks:
            return []

        keywords = {
            token for token in re.split(r"\W+", query.lower()) if len(token) > 3
        }
        if not keywords:
            return list(chunks)

        scored: List[tuple[int, int, DocumentChunk]] = []
        for index, chunk in enumerate(chunks):
            text_lower = chunk.text.lower()
            hit_count = sum(1 for token in keywords if token in text_lower)
            scored.append((hit_count, -index, chunk))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored]

    def _synthesize_answer(self, query: str, chunks: Sequence[DocumentChunk]) -> str:
        if self.chat_client is None:
            logger.warning("Chat client unavailable; returning concatenated chunks")
            return "\n\n".join(chunk.text for chunk in chunks)

        if not chunks:
            return "No relevant context found."

        context = "\n\n".join(
            f"Document: {chunk.metadata.filename} (p{chunk.metadata.page_number})\n{chunk.text}"
            for chunk in chunks
        )

        response = self.chat_client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based strictly on the provided context.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer using only the context, cite filenames and page numbers.",
                },
            ],
        )
        return response.choices[0].message.content or ""
