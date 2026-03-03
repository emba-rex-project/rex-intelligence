from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_ids: List[str] = Field(
        ..., description="Identifiers for ingested documents"
    )


class LocalIngestRequest(BaseModel):
    paths: List[Path] = Field(
        ..., description="Paths to PDF files or directories containing PDFs"
    )
    recursive: bool = Field(default=True)


class LocalIngestResult(BaseModel):
    success: List[Path]
    failed: List[Path]
