# Rex Intelligence Poc Service

Backend service for retrieving answers from uploaded Rex PDF using a Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, OpenAI embeddings, and a persistent Chroma vector store.

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

## Getting Started
1. **Set up the environment with uv**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
   Install `requirements-dev.txt` the same way when you need test tooling: `uv pip install -r requirements-dev.txt`.
2. **Configure environment**
   - Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, model names, and `DATA_DIR` if you want a custom path.
3. **Launch the API**
   ```bash
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
4. **Ingest PDFs**
   - Upload via HTTP: `POST /api/ingest/upload` with `multipart/form-data`.
   - Or use the CLI: `uv run python scripts/ingest.py data/uploads`.
5. **Query the knowledge base**
   ```bash
   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Which candidate won the Hology 3.0 App Innovation Competition?", "include_raw_chunks": true}'
   ```

## Testing
Run the unit tests with:
```bash
uv run pytest
```
Add new tests under `tests/` matching the module under development.

## Docker
Build and run the service in Docker:
```bash
docker build -t pdf-rag .
docker run --env-file .env -p 8000:8000 -v $(pwd)/data:/app/data pdf-rag
```
The volume mount ensures uploads and the vector store persist across container restarts.

## Notes & Limitations
- OpenAI API key is required for embeddings and answer synthesis.
- Chroma telemetry is enabled by default; set `CHROMA_TELEMETRY` env var to disable if needed.
- Distances returned by Chroma are converted to similarity scores for readability; higher values indicate closer matches.
