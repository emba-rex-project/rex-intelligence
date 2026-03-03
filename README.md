# Rex Intelligence Service

Backend service for retrieving answers from uploaded PDF resumes using a Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, OpenAI embeddings, and a persistent Chroma vector store.

## Features
- Upload one or more Rex PDF files via API; documents are stored on disk and parsed page by page.
- Text is chunked with token overlap, embedded using OpenAI models, and persisted to a local Chroma database.
- Query endpoint runs semantic search against stored chunks and synthesizes responses with citations using OpenAI chat models.
- CLI helper for bulk ingestion from local directories.
- Dockerfile and environment settings for reproducible deployment.

## Project Layout
- `app/` — FastAPI application (`main.py`) and router modules under `app/api/`.
- `services/` — PDF parsing, chunking, embeddings, vector store, and orchestration logic.
- `models/` — Pydantic schemas for requests, responses, and internal DTOs.
- `config/` — Environment-driven settings (`Settings`) with `.env` support.
- `data/` — Default storage for uploaded PDFs and the Chroma index.
- `scripts/` — Utilities like `scripts/ingest.py` for command-line ingestion.
- `tests/` — Pytest suites covering core services and configuration.

