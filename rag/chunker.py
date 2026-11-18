"""Paragraph-aware chunking with overlap."""
import re
from dataclasses import dataclass
from typing import List

from .loader import Page


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int


def _split_long(text: str, size: int) -> List[str]:
    """Split an over-long paragraph on sentence boundaries, then hard-cut as a last resort."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, cur = [], ""
    for s in sentences:
        while len(s) > size:  # single giant sentence
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:size])
            s = s[size:]
        if len(cur) + len(s) + 1 <= size:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                parts.append(cur)
            cur = s
    if cur:
        parts.append(cur)
    return parts


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    units: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        units.extend(_split_long(para, chunk_size) if len(para) > chunk_size else [para])

    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 2 > chunk_size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            # start the next chunk at a word boundary within the overlap tail
            tail = tail.split(" ", 1)[1] if " " in tail else tail
            cur = f"{tail}\n\n{u}".strip()
        else:
            cur = f"{cur}\n\n{u}".strip() if cur else u
    if cur:
        chunks.append(cur)
    return chunks


def chunk_pages(pages: List[Page], chunk_size: int = 800, overlap: int = 150) -> List[Chunk]:
    out: List[Chunk] = []
    for p in pages:
        for i, t in enumerate(split_text(p.text, chunk_size, overlap)):
            out.append(Chunk(text=t, source=p.source, page=p.page, chunk_id=i))
    return out
