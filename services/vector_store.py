from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection

from models.common import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(
        self, persist_directory: Path, collection_name: str = "pdf_rag"
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection: Collection = self.client.get_or_create_collection(
            collection_name
        )

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> List[str]:
        if not chunks:
            return []
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        if any(embedding is None for embedding in embeddings):
            raise ValueError("All chunks must include embeddings before upsert")
        metadatas = [chunk.metadata.dict() for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        self.collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
        logger.info("Upserted %d chunks", len(chunks))
        return ids

    def query(
        self, text_embedding: List[float], top_k: int
    ) -> List[Tuple[DocumentChunk, float]]:
        results = self.collection.query(
            query_embeddings=[text_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        matches: List[Tuple[DocumentChunk, float]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            matches.append(
                (
                    DocumentChunk(
                        id=doc_id,
                        text=document,
                        metadata=DocumentMetadata(**metadata),
                    ),
                    float(distance) if distance is not None else float("inf"),
                )
            )

        matches.sort(key=lambda pair: pair[1])
        return matches

    def reset(self) -> None:
        self.client.reset()
