from functools import lru_cache

from fastapi import Depends

from config.settings import get_settings
from services.rag import RAGService


@lru_cache()
def _get_service() -> RAGService:
    return RAGService()


def get_rag_service() -> RAGService:
    return _get_service()
