from fastapi import APIRouter, Depends

from app.api.dependencies import get_rag_service
from models.query import QueryRequest, QueryResponse
from services.rag import RAGService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_pdf_knowledge(
    payload: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    return rag_service.query(payload)
