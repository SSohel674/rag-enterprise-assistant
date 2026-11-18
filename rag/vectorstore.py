"""FAISS-backed vector store with JSON metadata, persisted to disk."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int, directory: Path):
        self.dim = dim
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict] = []

    # ---- persistence ----
    @property
    def _index_path(self):
        return self.dir / "index.faiss"

    @property
    def _meta_path(self):
        return self.dir / "meta.json"

    def save(self) -> None:
        faiss.write_index(self.index, str(self._index_path))
        self._meta_path.write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, dim: int, directory: Path) -> "VectorStore":
        store = cls(dim, directory)
        if store._index_path.exists() and store._meta_path.exists():
            store.index = faiss.read_index(str(store._index_path))
            store.meta = json.loads(store._meta_path.read_text(encoding="utf-8"))
        return store

    # ---- operations ----
    def add(self, vectors: np.ndarray, metas: List[Dict]) -> None:
        assert len(vectors) == len(metas)
        if len(vectors):
            self.index.add(vectors)
            self.meta.extend(metas)

    def search(self, query_vec: np.ndarray, k: int = 4) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(query_vec.reshape(1, -1), min(k, self.index.ntotal))
        return [(self.meta[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def sources(self) -> List[str]:
        return sorted({m["source"] for m in self.meta})

    def has_source(self, source: str) -> bool:
        return any(m["source"] == source for m in self.meta)

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.meta = []
        self.save()
