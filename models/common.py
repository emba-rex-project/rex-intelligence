from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    filename: str = Field(..., description="Original document filename")
    page_number: Optional[int] = Field(None, ge=1)
    chunk_id: Optional[str] = Field(None, description="Unique ID within the document")


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
