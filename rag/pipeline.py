"""End-to-end RAG: ingest -> retrieve -> generate with citations -> groundedness check."""
import hashlib
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import llm
from .chunker import chunk_pages
from .config import settings
from .embedder import Embedder
from .loader import load_file
from .vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are an enterprise document assistant. Answer ONLY using the numbered context passages provided. "
    "Cite the passages you used inline like [1] or [2][3]. "
    "If the context does not contain the answer, reply exactly: "
    "\"I couldn't find this in the provided documents.\" Do not use outside knowledge. Be concise."
)

NO_ANSWER = "I couldn't find this in the provided documents."


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\[\d+\]", "", text)  # drop citation markers
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


class RAGPipeline:
    def __init__(self, storage_dir: Path | None = None):
        self.embedder = Embedder()
        self.storage_dir = Path(storage_dir or settings.storage_dir)
        self.store = VectorStore.load(self.embedder.dim, self.storage_dir)

    # ---------- ingestion ----------
    def ingest_file(self, path: str | Path) -> int:
        """Index a file. Returns number of chunks added (0 if already indexed)."""
        path = Path(path)
        digest = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        source_key = path.name
        if any(m["source"] == source_key and m.get("hash") == digest for m in self.store.meta):
            return 0
        chunks = chunk_pages(load_file(path), settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return 0
        vecs = self.embedder.encode([c.text for c in chunks])
        metas = [
            {"text": c.text, "source": c.source, "page": c.page, "chunk_id": c.chunk_id, "hash": digest}
            for c in chunks
        ]
        self.store.add(vecs, metas)
        self.store.save()
        return len(chunks)

    def ingest_directory(self, directory: str | Path) -> Dict[str, int]:
        from .loader import SUPPORTED

        return {
            f.name: self.ingest_file(f)
            for f in sorted(Path(directory).rglob("*"))
            if f.suffix.lower() in SUPPORTED
        }

    # ---------- retrieval ----------
    def retrieve(self, question: str, top_k: int | None = None) -> List[Dict]:
        q = self.embedder.encode([question])[0]
        hits = self.store.search(q, top_k or settings.top_k)
        return [{**m, "score": round(s, 4)} for m, s in hits]

    # ---------- generation ----------
    @staticmethod
    def _build_prompt(question: str, hits: List[Dict]) -> str:
        ctx = "\n\n".join(f"[{i + 1}] (source: {h['source']}, page {h['page']})\n{h['text']}" for i, h in enumerate(hits))
        return f"Context passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer (with citations):"

    def groundedness(self, answer: str, hits: List[Dict]) -> float:
        """Mean over answer sentences of the max cosine similarity to any retrieved chunk.
        Cheap, model-agnostic hallucination signal: low score = claims not backed by context."""
        sents = _sentences(answer)
        if not sents or not hits:
            return 0.0
        s_vecs = self.embedder.encode(sents)
        c_vecs = self.embedder.encode([h["text"] for h in hits])
        sims = s_vecs @ c_vecs.T
        return float(np.mean(sims.max(axis=1)))

    def ask(self, question: str, top_k: int | None = None) -> Dict:
        hits = self.retrieve(question, top_k)
        if not hits or hits[0]["score"] < settings.min_relevance:
            return {"answer": NO_ANSWER, "sources": [], "grounded": True, "groundedness": None, "confidence": 0.0}

        answer = llm.generate(self._build_prompt(question, hits), system=SYSTEM_PROMPT)
        refused = NO_ANSWER.lower()[:25] in answer.lower()
        score = None if refused else round(self.groundedness(answer, hits), 3)

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        sources = [
            {"id": i + 1, "source": h["source"], "page": h["page"], "score": h["score"], "snippet": h["text"][:700]}
            for i, h in enumerate(hits)
            if not cited or (i + 1) in cited
        ]
        return {
            "answer": answer,
            "sources": sources,
            "grounded": True if score is None else score >= settings.groundedness_threshold,
            "groundedness": score,
            "confidence": hits[0]["score"],
        }
