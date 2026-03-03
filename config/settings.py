from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    app_name: str = Field(default="PDF RAG Service")
    api_prefix: str = Field(default="/api")

    data_dir: Path = Field(default=Path("data"))
    upload_dirname: str = Field(default="uploads")
    vector_store_dirname: str = Field(default="vector_store")

    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-large")
    chat_model: str = Field(default="gpt-4o-mini")

    chunk_size: int = Field(default=500, ge=100)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=6, ge=1)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def upload_path(self) -> Path:
        return self.data_dir / self.upload_dirname

    @property
    def vector_store_path(self) -> Path:
        return self.data_dir / self.vector_store_dirname


def get_settings() -> Settings:
    return Settings()
