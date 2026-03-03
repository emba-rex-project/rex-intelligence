#!/usr/bin/env python
"""Command line helper to ingest PDFs from disk."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ingest import LocalIngestResult
from services.rag import RAGService


def collect_paths(targets: list[str], recursive: bool) -> list[Path]:
    collected: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir() and recursive:
            collected.extend([p for p in path.rglob("*.pdf") if p.is_file()])
        elif path.is_file() and path.suffix.lower() == ".pdf":
            collected.append(path)
    return collected


def ingest_from_cli(paths: list[str], recursive: bool) -> LocalIngestResult:
    service = RAGService()
    collected = collect_paths(paths, recursive)
    success: list[Path] = []
    failed: list[Path] = []
    for path in collected:
        try:
            service.ingest_paths([path])
            success.append(path)
        except Exception:
            failed.append(path)
    return LocalIngestResult(success=success, failed=failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store")
    parser.add_argument("paths", nargs="+", help="PDF files or directories")
    parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false", default=True
    )
    args = parser.parse_args()

    result = ingest_from_cli(args.paths, args.recursive)
    print(f"Ingested: {len(result.success)}")
    if result.failed:
        print("Failed:")
        for path in result.failed:
            print(f" - {path}")


if __name__ == "__main__":
    main()
