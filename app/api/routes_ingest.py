from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from models.ingest import IngestResponse
from services.rag import RAGService
from app.api.dependencies import get_rag_service

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/upload", response_model=IngestResponse)
async def upload_pdfs(
    files: List[UploadFile] = File(..., description="One or more PDF files"),
    rag_service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    document_ids: List[str] = []
    for upload in files:
        payload = await upload.read()
        document_ids.extend(rag_service.ingest_upload(upload.filename, payload))
    return IngestResponse(document_ids=document_ids)
