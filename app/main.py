import logging

from fastapi import FastAPI

from app.api import routes_ingest, routes_query
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.include_router(routes_ingest.router, prefix=settings.api_prefix)
app.include_router(routes_query.router, prefix=settings.api_prefix)


@app.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
