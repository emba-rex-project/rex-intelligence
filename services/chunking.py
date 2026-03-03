from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import tiktoken


@dataclass
class ChunkConfig:
    chunk_size: int
    chunk_overlap: int
    encoding: str = "cl100k_base"

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


class TextChunker:
    def __init__(self, config: ChunkConfig) -> None:
        self.config = config
        self._encoding = tiktoken.get_encoding(config.encoding)

    def chunk(self, text: str) -> List[str]:
        token_ids = self._encoding.encode(text)
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks: List[str] = []
        start = 0
        while start < len(token_ids):
            end = min(len(token_ids), start + size)
            chunk_tokens = token_ids[start:end]
            chunk_text = self._encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            if end == len(token_ids):
                break
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    def chunk_pages(self, pages: Sequence[str]) -> List[Tuple[int, int, str]]:
        chunks: List[Tuple[int, int, str]] = []
        for page_index, page in enumerate(pages, start=1):
            page_chunks = self.chunk(page)
            for chunk_index, chunk_text in enumerate(page_chunks):
                chunks.append((page_index, chunk_index, chunk_text))
        return chunks
