"""Load documents (PDF, DOCX, TXT, MD) into page-level text records."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Page:
    text: str
    source: str
    page: int  # 1-based; 1 for formats without pages


def load_file(path: str | Path) -> List[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(SUPPORTED)}")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [Page((p.extract_text() or "").strip(), path.name, i + 1) for i, p in enumerate(reader.pages)]
    elif ext == ".docx":
        from docx import Document

        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        pages = [Page(text, path.name, 1)]
    else:
        pages = [Page(path.read_text(encoding="utf-8", errors="ignore"), path.name, 1)]

    return [p for p in pages if p.text.strip()]


def load_directory(directory: str | Path) -> List[Page]:
    pages: List[Page] = []
    for f in sorted(Path(directory).rglob("*")):
        if f.suffix.lower() in SUPPORTED:
            pages.extend(load_file(f))
    return pages
