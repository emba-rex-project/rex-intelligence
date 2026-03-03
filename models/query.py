from typing import List, Optional

from pydantic import BaseModel, Field

from models.common import DocumentChunk


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    include_raw_chunks: bool = Field(default=True)


class RetrievedChunk(BaseModel):
    score: float
    chunk: DocumentChunk


class QueryResponse(BaseModel):
    answer: str
    references: List[RetrievedChunk]
