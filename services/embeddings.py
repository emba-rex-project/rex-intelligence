from __future__ import annotations

import logging
from typing import Iterable, List

from openai import OpenAI

from config.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be configured to generate embeddings")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        payload = [text if text else "" for text in texts]
        logger.debug("Embedding %d chunks", len(payload))
        response = self.client.embeddings.create(model=self.model, input=payload)
        return [item.embedding for item in response.data]
