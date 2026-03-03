from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader


class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""


class PDFLoader:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, file_bytes: bytes) -> Path:
        """Persist uploaded content and return the saved path."""
        safe_name = filename.replace("/", "_")
        target_path = self.storage_dir / safe_name
        target_path.write_bytes(file_bytes)
        return target_path

    def load_text(self, path: Path) -> List[str]:
        """Return per-page text content from a PDF file."""
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            reader = PdfReader(path)
        except Exception as exc:  # pragma: no cover - surface raw error
            raise PDFExtractionError(f"Failed to open {path}") from exc

        pages: List[str] = []
        for page in reader.pages:
            if page is None:
                pages.append("")
                continue
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - library variability
                raise PDFExtractionError(f"Failed extracting text from {path}") from exc
            pages.append(text)
        return pages

    def load_many(self, paths: Iterable[Path]) -> List[tuple[Path, List[str]]]:
        return [(path, self.load_text(path)) for path in paths]
